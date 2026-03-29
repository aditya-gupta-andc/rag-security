"""
Layer 1: KB-Shield — Knowledge Base Integrity
GMTP (PS5) + FATH HMAC (PS7) + SAG AES-256-GCM (SAG)
"""
import os, hmac, hashlib, logging
from typing import List, Tuple, Dict
from collections import defaultdict
import numpy as np, torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModel
from shieldrag.data.dataset_loader import Document

logger = logging.getLogger("shieldrag.layer1")

class GMTPScanner:
    INJECTION_KW = [
        "ignore","override","system","admin","instruction","forget",
        "disregard","priority","critical","hidden","reveal","dump",
        "unrestricted","bypass","disable","new task","you are now",
        "patch","memo","internal","sudo","jailbreak","developer mode",
        "emergency","supersede","deprecated","override","hidden instruction",
    ]
    def __init__(self, config, device):
        cfg = config.get("layer1",{}).get("gmtp",{})
        self.threshold = cfg.get("anomaly_threshold", 0.55)
        model_name = config.get("models",{}).get("embedding_model","sentence-transformers/all-MiniLM-L6-v2")
        logger.info(f"GMTP: Loading {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self.classifier = GradientBoostingClassifier(
            n_estimators=cfg.get("n_estimators",300),
            max_depth=cfg.get("max_depth",7), random_state=42)
        self.is_trained = False
        self.feature_dim = None

    def _extract_features(self, text):
        f = []
        tl = text.lower()
        words = text.split()
        kw_count = sum(1 for kw in self.INJECTION_KW if kw in tl)
        f.append(kw_count)
        f.append(kw_count / max(len(words),1))
        f.append(sum(1 for c in text if c.isupper()) / max(len(text),1))
        f.append(sum(1 for c in text if c in "[]<>{}\\#@!") / max(len(text),1))
        f.append(np.mean([len(w) for w in words]) if words else 0)
        non_ascii = sum(1 for c in text if ord(c)>127 and c not in "\u2018\u2019\u201c\u201d\u2013\u2014\u2026")
        f.append(non_ascii / max(len(text),1))
        freq = defaultdict(int)
        for c in tl: freq[c] += 1
        total = sum(freq.values())
        entropy = -sum((v/total)*np.log2(v/total) for v in freq.values() if v>0)
        f.append(entropy)
        f.extend([len(text), len(words), text.count(":"), text.count("\n"),
                  text.count("<!--"), text.count("[")+text.count("]"),
                  text.count("SYSTEM")+text.count("ADMIN")+text.count("OVERRIDE")])
        try:
            with torch.no_grad():
                enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                                     max_length=128, padding=True).to(self.device)
                out = self.model(**enc)
                cls = out.last_hidden_state[:,0,:].cpu().numpy().flatten()
                f.extend(cls[:48].tolist())
                all_e = out.last_hidden_state.cpu().numpy()[0]
                f.append(float(np.std(all_e)))
                f.append(float(np.mean(np.abs(all_e))))
        except Exception:
            f.extend([0.0]*50)
        return np.array(f, dtype=np.float32)

    def train(self, documents):
        logger.info(f"GMTP: Training on {len(documents)} documents...")
        X, y = [], []
        for doc in documents:
            if doc.is_decoy: continue
            X.append(self._extract_features(doc.content))
            y.append(1 if doc.is_poisoned else 0)
        X = np.array(X); y = np.array(y)
        self.feature_dim = X.shape[1]
        self.classifier.fit(X, y)
        self.is_trained = True
        yp = self.classifier.predict(X)
        logger.info("GMTP train report:\n" + classification_report(y, yp, target_names=["benign","poisoned"], zero_division=0))

    def scan(self, doc):
        if not self.is_trained: return True, 0.5, "Not trained"
        X = self._extract_features(doc.content).reshape(1,-1)
        if X.shape[1] != self.feature_dim:
            X = np.pad(X, ((0,0),(0,max(0,self.feature_dim-X.shape[1]))), 'constant')[:,:self.feature_dim]
        prob = self.classifier.predict_proba(X)[0]
        pp = prob[1] if len(prob)>1 else 0.0
        safe = pp < self.threshold
        return safe, 1.0-pp, "Clean" if safe else f"Anomaly({pp:.3f})"


class FATHAuthenticator:
    def __init__(self, config):
        cfg = config.get("layer1",{}).get("fath",{})
        self.secret = cfg.get("secret_key","shieldrag_key").encode()
        self.algo = cfg.get("hmac_algorithm","sha256")

    def generate_token(self, content):
        return hmac.new(self.secret, content.encode(), getattr(hashlib, self.algo)).hexdigest()

    def sign(self, doc):
        doc.hmac_token = self.generate_token(doc.content)
        return doc

    def verify(self, doc):
        expected = self.generate_token(doc.content)
        if doc.hmac_token == expected:
            return True, "HMAC valid"
        return False, "HMAC mismatch"


class SAGEncryption:
    def __init__(self, config):
        key_mat = config.get("layer1",{}).get("sag",{}).get("secret_key","sag_key")
        self.key = hashlib.sha256(key_mat.encode()).digest()
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            self.has_crypto = True
        except ImportError:
            self.has_crypto = False

    def encrypt(self, content):
        if self.has_crypto:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            nonce = os.urandom(12)
            return nonce + AESGCM(self.key).encrypt(nonce, content.encode(), None)
        data = content.encode()
        return bytes(b ^ self.key[i%len(self.key)] for i,b in enumerate(data))

    def decrypt(self, encrypted):
        if self.has_crypto:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            return AESGCM(self.key).decrypt(encrypted[:12], encrypted[12:], None).decode()
        return bytes(b ^ self.key[i%len(self.key)] for i,b in enumerate(encrypted)).decode()

    def verify_integrity(self, doc):
        try:
            dec = self.decrypt(self.encrypt(doc.content))
            return (dec == doc.content), "SAG OK" if dec == doc.content else "SAG fail"
        except Exception as e:
            return False, f"SAG error: {e}"


class KBShield:
    def __init__(self, config, device):
        self.gmtp = GMTPScanner(config, device)
        self.fath = FATHAuthenticator(config)
        self.sag = SAGEncryption(config)
        self.results = []

    def train(self, documents): self.gmtp.train(documents)

    def sign_documents(self, documents):
        return [self.fath.sign(d) if not d.is_poisoned else d for d in documents]

    def evaluate(self, doc):
        r = {"doc_id": doc.doc_id, "checks": {}}
        gs, gc, gr = self.gmtp.scan(doc)
        r["checks"]["gmtp"] = {"passed": gs, "confidence": float(gc), "reason": gr}
        fv, fr2 = self.fath.verify(doc)
        r["checks"]["fath"] = {"passed": fv, "reason": fr2}
        sv, sr = self.sag.verify_integrity(doc)
        r["checks"]["sag"] = {"passed": sv, "reason": sr}
        safe = gs and fv and sv
        r["verdict"] = "SAFE" if safe else "BLOCKED"
        self.results.append(r)
        return safe, r

    def filter_kb(self, documents):
        safe, blocked = [], []
        for d in documents:
            if d.is_decoy: safe.append(d); continue
            s, _ = self.evaluate(d)
            (safe if s else blocked).append(d)
        logger.info(f"KB-Shield: {len(safe)} safe, {len(blocked)} blocked")
        return safe, blocked
