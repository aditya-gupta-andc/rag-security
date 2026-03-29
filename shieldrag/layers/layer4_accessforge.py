"""
Layer 4: AccessForge — Infrastructure Governance
RAGFort ACL (PS10) + PA-RAG Isolation + Audit Logging
"""
import json, re, time, hashlib, logging
from typing import List, Tuple, Dict
from collections import defaultdict
from pathlib import Path
from shieldrag.data.dataset_loader import Document

logger = logging.getLogger("shieldrag.layer4")

EXTRACT_KW=["list all","show everything","dump","all documents","entire",
            "full content","every document","raw contents","word for word",
            "repeat all","complete listing","enumerate all","export the full",
            "comprehensive listing","wildcard","knowledge base","retrieval corpus",
            "complete export","full text of every"]

ACCESS_VIOLATION_KW=[
    "all tenants","across all tenants","every tenant","every namespace",
    "all tenant data","bypass controls","bypass access","override access",
    "ignoring restrictions","ignoring access control","cross-tenant",
    "merge all tenant","combined tenant","classified documents from",
    "across all namespace","override.*permission","bypass.*permission",
]

class RAGFortACL:
    def __init__(self, config):
        c=config.get("layer4",{}).get("ragfort",{})
        self.threshold=c.get("extraction_threshold",4)
        self.perms={}; self.counter=defaultdict(int); self.history=defaultdict(list)

    def register_tenant(self, tid, ns): self.perms[tid]=ns

    def check_access(self, tid, doc):
        if doc.tenant_id=="system": return True,"System"
        if tid not in self.perms: return False,f"Unknown: {tid}"
        return (doc.tenant_id in self.perms[tid], "OK" if doc.tenant_id in self.perms[tid] else "Denied")

    def monitor_extraction(self, tid, query):
        if any(kw in query.lower() for kw in EXTRACT_KW):
            self.counter[tid]+=1
            self.history[tid].append({"q":query[:80],"t":time.time()})
        if self.counter[tid]>=self.threshold:
            return False,f"Extraction detected ({self.counter[tid]} hits)"
        return True,"OK"

    def monitor_access_violation(self, tid, query):
        """Detect cross-tenant and access-violation patterns in a single query."""
        ql = query.lower()
        # Check keyword-based access violation patterns; use regex only for those that need it
        for kw in ACCESS_VIOLATION_KW:
            if '.*' in kw:
                if re.search(kw, ql):
                    return False, f"Access violation pattern: '{kw}'"
            else:
                if kw in ql:
                    return False, f"Access violation keyword: '{kw}'"
        # Check if the query explicitly references a different tenant's namespace
        for other_tid in self.perms:
            if other_tid != tid:
                pattern = r'\b' + re.escape(other_tid.lower()) + r'\b'
                if re.search(pattern, ql):
                    return False, f"Cross-tenant reference to '{other_tid}'"
        return True, "OK"

    def check_decoy(self, docs):
        d=[x.doc_id for x in docs if x.is_decoy]
        return (bool(d), d)


class PARAGIsolation:
    def __init__(self):
        self.keys={}
    def register(self, tid):
        self.keys[tid]=hashlib.sha256(f"parag_{tid}".encode()).digest()
    def can_access(self, req, doc_tid):
        return doc_tid=="system" or req==doc_tid
    def filter_docs(self, tid, docs):
        return [d for d in docs if self.can_access(tid, d.tenant_id)]


class AuditLogger:
    def __init__(self, config):
        c=config.get("layer4",{}).get("audit",{})
        self.enabled=c.get("enabled",True)
        self.log_file=c.get("log_file","results/audit.log")
        self.entries=[]
        if self.enabled: Path(self.log_file).parent.mkdir(parents=True,exist_ok=True)

    def log(self, event):
        event["ts"]=time.time(); self.entries.append(event)
        if self.enabled:
            try:
                with open(self.log_file,"a") as f: f.write(json.dumps(event)+"\n")
            except Exception: pass


class AccessForge:
    def __init__(self, config):
        self.ragfort=RAGFortACL(config)
        self.parag=PARAGIsolation()
        self.audit=AuditLogger(config)

    def setup_tenants(self, tids=None):
        for t in (tids or ["tenant_A","tenant_B","tenant_C"]):
            self.ragfort.register_tenant(t, {t})
            self.parag.register(t)
        logger.info(f"AccessForge: tenants configured")

    def evaluate(self, tid, query, docs):
        r={"checks":[]}
        es, er = self.ragfort.monitor_extraction(tid, query)
        r["checks"].append({"check":"extraction","passed":es,"reason":er})
        if not es:
            self.audit.log({"event":"extraction_blocked","tenant":tid})
            r["verdict"]="BLOCKED"; return False, r, []
        avs, avr = self.ragfort.monitor_access_violation(tid, query)
        r["checks"].append({"check":"access_violation","passed":avs,"reason":avr})
        if not avs:
            self.audit.log({"event":"access_violation_blocked","tenant":tid,"reason":avr})
            r["verdict"]="BLOCKED"; return False, r, []
        filtered=[]
        for d in docs:
            ok,_=self.ragfort.check_access(tid,d)
            if ok and self.parag.can_access(tid, d.tenant_id): filtered.append(d)
        r["docs_allowed"]=len(filtered); r["docs_blocked"]=len(docs)-len(filtered)
        da, dids = self.ragfort.check_decoy(filtered)
        r["checks"].append({"check":"decoy","triggered":da,"ids":dids})
        if da: self.audit.log({"event":"decoy","tenant":tid,"ids":dids})
        self.audit.log({"event":"query","tenant":tid,"allowed":len(filtered)})
        r["verdict"]="SAFE"; return True, r, filtered
