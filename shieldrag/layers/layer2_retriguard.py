"""
Layer 2: RetriGuard — Retrieval Hardening
Multi-retriever ensemble (PS8) + Cross-encoder reranking + Semantic isolation (PS6)
"""
import re, math, logging
from typing import List, Tuple, Dict
from collections import defaultdict, Counter
import numpy as np, torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from shieldrag.data.dataset_loader import Document

logger = logging.getLogger("shieldrag.layer2")

class DenseRetriever:
    def __init__(self, model_name, device, name="dense"):
        self.name = name; self.device = device
        logger.info(f"DenseRetriever[{name}]: Loading {model_name}")
        self.model = SentenceTransformer(model_name, device=str(device))
        self.documents = []; self.embeddings = None; self.index = None

    def index_documents(self, documents, batch_size=64):
        import faiss
        self.documents = documents
        texts = [d.content for d in documents]
        logger.info(f"DenseRetriever[{self.name}]: Encoding {len(texts)} docs...")
        self.embeddings = self.model.encode(texts, batch_size=batch_size,
            show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings.astype(np.float32))
        logger.info(f"DenseRetriever[{self.name}]: FAISS index built ({dim}d, {len(documents)} vecs)")

    def retrieve(self, query, top_k=10):
        qe = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        scores, indices = self.index.search(qe, top_k)
        return [(self.documents[i], float(s)) for s, i in zip(scores[0], indices[0])
                if 0 <= i < len(self.documents)]


class BM25Retriever:
    def __init__(self, name="bm25", k1=1.5, b=0.75):
        self.name=name; self.k1=k1; self.b=b; self.documents=[]; self.N=0
        self.doc_freqs={}; self.tf_cache=[]; self.doc_lens=[]; self.avg_dl=1.0

    def _tok(self, text): return re.findall(r'\w+', text.lower())

    def index_documents(self, documents):
        self.documents=documents; self.N=len(documents)
        self.doc_freqs=defaultdict(int); self.tf_cache=[]; self.doc_lens=[]
        for d in documents:
            tokens=self._tok(d.content); self.doc_lens.append(len(tokens))
            tf=Counter(tokens); self.tf_cache.append(tf)
            for t in set(tokens): self.doc_freqs[t]+=1
        self.avg_dl=np.mean(self.doc_lens) if self.doc_lens else 1.0
        logger.info(f"BM25[{self.name}]: Indexed {len(documents)} docs")

    def _score(self, qt, di):
        tf=self.tf_cache[di]; dl=self.doc_lens[di]; s=0.0
        for t in qt:
            if t not in tf: continue
            df=self.doc_freqs.get(t,0)
            idf=math.log((self.N-df+0.5)/(df+0.5)+1.0)
            tfn=(tf[t]*(self.k1+1))/(tf[t]+self.k1*(1-self.b+self.b*dl/self.avg_dl))
            s+=idf*tfn
        return s

    def retrieve(self, query, top_k=10):
        qt=self._tok(query)
        scores=[(i,self._score(qt,i)) for i in range(self.N)]
        scores.sort(key=lambda x:x[1], reverse=True)
        return [(self.documents[i],s) for i,s in scores[:top_k] if s>0]


class SemanticIsolator:
    MARKERS = [
        "ignore all previous","ignore previous instructions","override",
        "new task","forget your","disregard","you are now","hidden instruction",
        "priority override","admin mode","developer mode","jailbreak","sudo",
        "emergency override","begin override","unrestricted",
        "security filters disabled","system update","critical patch",
        "admin notice","internal memo","SYSTEM OVERRIDE","ADMIN",
        "INSTRUCTION:","HIDDEN","SECRET_DIRECTIVE","developer_note",
    ]
    def __init__(self, config):
        cfg=config.get("layer2",{}).get("isolation",{})
        self.enabled=cfg.get("enabled",True)

    def isolate(self, doc):
        if not self.enabled: return doc.content, False, "Disabled"
        cl = doc.content
        found=[m for m in self.MARKERS if m.lower() in cl.lower()]
        if not found: return doc.content, False, "Clean"
        safe=doc.content
        for m in found:
            safe=re.sub(re.escape(m),"[REDACTED]",safe,flags=re.IGNORECASE)
        return f"[ISOLATED] {safe[:300]}", True, f"{len(found)} markers"


class RetriGuard:
    def __init__(self, config, device):
        self.device=device
        cfg=config.get("layer2",{})
        cc=cfg.get("consensus",{})
        self.min_votes=cc.get("min_votes",2)
        self.final_top_k=cc.get("final_top_k",5)
        rcfgs=cfg.get("retrievers",[])
        self.retrievers=[]
        loaded={}
        for rc in rcfgs:
            rt=rc.get("type","dense"); rn=rc.get("name",rt)
            if rt=="bm25":
                self.retrievers.append(BM25Retriever(name=rn))
            else:
                mn=rc.get("model",config["models"]["embedding_model"])
                if mn not in loaded:
                    loaded[mn]=DenseRetriever(mn,device,name=rn)
                    self.retrievers.append(loaded[mn])
                else:
                    self.retrievers.append(DenseRetriever(mn,device,name=rn))
        if not self.retrievers:
            em=config.get("models",{}).get("embedding_model","sentence-transformers/all-MiniLM-L6-v2")
            self.retrievers=[DenseRetriever(em,device,"dense1"),BM25Retriever("bm25"),
                             DenseRetriever(em,device,"dense2")]
        # Cross-encoder reranker
        self.reranker=None
        rr_cfg=cfg.get("reranker",{})
        if rr_cfg.get("enabled",False):
            rr_model=config.get("models",{}).get("reranker_model","cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info(f"RetriGuard: Loading reranker {rr_model}")
            self.reranker=CrossEncoder(rr_model, device=str(device))
            self.rerank_top_k=rr_cfg.get("top_k",5)
        self.isolator=SemanticIsolator(config)

    def index(self, documents):
        for r in self.retrievers:
            if isinstance(r, DenseRetriever): r.index_documents(documents)
            else: r.index_documents(documents)

    def retrieve_with_consensus(self, query, top_k=None):
        top_k=top_k or self.final_top_k
        votes=defaultdict(lambda:{"doc":None,"votes":0,"scores":[]})
        for r in self.retrievers:
            for doc,score in r.retrieve(query, top_k=top_k+5):
                votes[doc.doc_id]["doc"]=doc
                votes[doc.doc_id]["votes"]+=1
                votes[doc.doc_id]["scores"].append(score)
        consensus=[]
        for did,info in votes.items():
            if info["votes"]>=self.min_votes:
                consensus.append((info["doc"], float(np.mean(info["scores"]))))
        consensus.sort(key=lambda x:x[1], reverse=True)
        candidates=consensus[:top_k*2]
        # Cross-encoder reranking
        if self.reranker and candidates:
            pairs=[(query, d.content[:512]) for d,_ in candidates]
            re_scores=self.reranker.predict(pairs)
            reranked=sorted(zip(candidates,re_scores), key=lambda x:x[1], reverse=True)
            candidates=[(doc,float(rs)) for (doc,_),rs in reranked]
        return candidates[:top_k]

    def apply_isolation(self, documents):
        return [self.isolator.isolate(d) for d in documents]
