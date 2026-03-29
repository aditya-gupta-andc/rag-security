"""ShieldRAG Pipeline — 4-layer composition with proper train/test split."""
import time, logging
from typing import List, Dict
import torch
from shieldrag.data.dataset_loader import Document, Query, DatasetLoader
from shieldrag.layers.layer1_kb_shield import KBShield
from shieldrag.layers.layer2_retriguard import RetriGuard
from shieldrag.layers.layer3_gensafe import GenSafe
from shieldrag.layers.layer4_accessforge import AccessForge
from shieldrag.utils.helpers import get_device, set_seed

logger = logging.getLogger("shieldrag.pipeline")

class ShieldRAGPipeline:
    def __init__(self, config):
        self.config=config
        self.device=get_device(config.get("device","auto"))
        set_seed(config.get("seed",42))
        logger.info("="*70); logger.info("Initialising ShieldRAG v2"); logger.info("="*70)
        self.layer1=KBShield(config, self.device)
        self.layer2=RetriGuard(config, self.device)
        self.layer3=GenSafe(config, self.device)
        self.layer4=AccessForge(config)
        self.all_docs=[]; self.safe_docs=[]; self.blocked_docs=[]
        self.is_ready=False; self.query_log=[]

    def load_and_prepare(self, documents=None, train_queries=None):
        loader=DatasetLoader(self.config)
        all_docs, train_q, val_q, test_q = loader.load_all()
        self.all_docs=all_docs
        # L1: sign + train + filter
        all_docs=self.layer1.sign_documents(all_docs)
        self.layer1.train(all_docs)
        self.safe_docs, self.blocked_docs = self.layer1.filter_kb(all_docs)
        # L2: index
        self.layer2.index(self.safe_docs)
        # L3: train on TRAINING queries only (proper split!)
        self.layer3.train(all_docs, train_q)
        # L4: tenants
        self.layer4.setup_tenants()
        self.is_ready=True
        logger.info("ShieldRAG v2 pipeline ready!")
        return all_docs, train_q, val_q, test_q

    def process_query(self, query):
        if not self.is_ready: raise RuntimeError("Call load_and_prepare() first")
        t0=time.time()
        r={"query_id":query.query_id,"query_text":query.text,
           "is_adversarial":query.is_adversarial,
           "attack_type":query.attack_type or "benign",
           "attack_difficulty":getattr(query,"attack_difficulty",""),
           "tenant_id":query.tenant_id,"layers":{},"response":"",
           "blocked":False,"blocked_by":"","latency_ms":0}
        # L3a: query check
        t1=time.time()
        gs, gr = self.layer3.check_query(query.text)
        r["layers"]["L3_query"]=gr; r["layers"]["L3_query"]["latency_ms"]=round((time.time()-t1)*1000,1)
        if not gs:
            r["blocked"]=True; r["blocked_by"]="Layer 3 (GenSafe)"
            r["response"]="Blocked: injection detected."
            r["latency_ms"]=round((time.time()-t0)*1000,1)
            self.query_log.append(r); return r
        # L2: retrieve
        t1=time.time()
        candidates=self.layer2.retrieve_with_consensus(query.text, top_k=5)
        cdocs=[d for d,_ in candidates]
        r["layers"]["L2_retrieval"]={"docs":len(candidates),"latency_ms":round((time.time()-t1)*1000,1)}
        # L4: access
        t1=time.time()
        a4s, a4r, adocs = self.layer4.evaluate(query.tenant_id, query.text, cdocs)
        r["layers"]["L4_access"]=a4r; r["layers"]["L4_access"]["latency_ms"]=round((time.time()-t1)*1000,1)
        if not a4s:
            r["blocked"]=True; r["blocked_by"]="Layer 4 (AccessForge)"
            r["response"]="Blocked: extraction/access violation."
            r["latency_ms"]=round((time.time()-t0)*1000,1)
            self.query_log.append(r); return r
        # L2: isolation
        iso_results=self.layer2.apply_isolation(adocs)
        contexts=[c for c,_,_ in iso_results if c]
        any_iso=any(i for _,i,_ in iso_results)
        r["layers"]["L2_isolation"]={"docs":len(contexts),"isolated":any_iso}
        # L3: generate
        t1=time.time()
        if not contexts:
            response="No relevant documents found."; out_safe=True; gen_r={}
        else:
            response, out_safe, gen_r = self.layer3.generate_response(query.text, contexts)
        r["layers"]["L3_gen"]={**gen_r,"latency_ms":round((time.time()-t1)*1000,1)}
        if not out_safe:
            r["blocked"]=True; r["blocked_by"]="Layer 3 (Output Verifier)"
            r["response"]="Blocked: output leak detected."
        else:
            r["response"]=response
        r["latency_ms"]=round((time.time()-t0)*1000,1)
        self.query_log.append(r); return r

    def get_stats(self):
        total=len(self.query_log); blocked=sum(1 for r in self.query_log if r["blocked"])
        adv=sum(1 for r in self.query_log if r["is_adversarial"])
        ab=sum(1 for r in self.query_log if r["is_adversarial"] and r["blocked"])
        ben=total-adv; bp=sum(1 for r in self.query_log if not r["is_adversarial"] and not r["blocked"])
        return {"total_queries":total,"total_blocked":blocked,"adversarial_total":adv,
                "adversarial_blocked":ab,"benign_total":ben,"benign_passed":bp,
                "kb_total":len(self.all_docs),"kb_safe":len(self.safe_docs),
                "kb_blocked":len(self.blocked_docs),
                "avg_latency_ms":round(sum(r["latency_ms"] for r in self.query_log)/max(total,1),1)}
