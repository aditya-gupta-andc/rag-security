"""ShieldRAG v2 — CLI entry point."""
import os, sys, argparse, logging, time
from shieldrag.utils.helpers import load_config, setup_logging
from shieldrag.pipeline import ShieldRAGPipeline
from shieldrag.data.dataset_loader import Query
from shieldrag.evaluation.metrics import build_results_df, compute_metrics, generate_charts, save_results, print_report

logger = logging.getLogger("shieldrag.main")

def run_evaluate(config):
    print("\n"+"="*70+"\n  ShieldRAG v2 — Full Evaluation\n"+"="*70)
    pipe=ShieldRAGPipeline(config)
    all_docs, train_q, val_q, test_q = pipe.load_and_prepare()
    print(f"\n  Evaluating on TEST SET ({len(test_q)} queries, unseen during training)\n")
    results=[]
    for i,q in enumerate(test_q):
        r=pipe.process_query(q)
        results.append(r)
        if (i+1)%5==0 or i<3:
            st="BLOCKED" if r["blocked"] else "OK"
            diff=r.get("attack_difficulty","")
            print(f"  [{i+1:3d}/{len(test_q)}] {q.query_id:15s} {q.attack_type or 'benign':18s} {diff:10s} -> {st:7s} ({r['latency_ms']:5.0f}ms)")
    df=build_results_df(results)
    metrics=compute_metrics(df)
    out=config.get("evaluation",{}).get("output_dir","results")
    print_report(metrics)
    print("  Generating charts...")
    charts=generate_charts(metrics, df, out)
    for c in charts: print(f"    -> {c}")
    save_results(metrics, df, out)
    stats=pipe.get_stats()
    print(f"\n  KB: {stats['kb_safe']} safe / {stats['kb_blocked']} blocked")
    print(f"  Attacks blocked: {stats['adversarial_blocked']}/{stats['adversarial_total']}")
    print(f"  Benign passed: {stats['benign_passed']}/{stats['benign_total']}")
    print(f"  Avg latency: {stats['avg_latency_ms']}ms")
    print(f"\n  Results in: {out}/\n")

def run_demo(config):
    print("\n"+"="*70+"\n  ShieldRAG v2 — Interactive Demo\n"+"="*70)
    pipe=ShieldRAGPipeline(config)
    pipe.load_and_prepare()
    print("\n  Type queries to test. 'quit' to exit.\n")
    while True:
        try: inp=input("  [ShieldRAG] > ").strip()
        except (EOFError, KeyboardInterrupt): print("\n  Bye!"); break
        if not inp or inp.lower()=="quit": print("  Bye!"); break
        q=Query(f"DEMO-{int(time.time())}", inp, tenant_id="tenant_A")
        r=pipe.process_query(q)
        if r["blocked"]: print(f"\n  BLOCKED by {r['blocked_by']}\n  -> {r['response']}")
        else: print(f"\n  RESPONSE:\n  -> {r['response'][:300]}")
        print(f"  [{r['latency_ms']}ms]\n")

def run_web(config):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from web_ui.app import create_app
    pipe=ShieldRAGPipeline(config); pipe.load_and_prepare()
    app=create_app(pipe, config)
    h=config.get("web",{}).get("host","0.0.0.0"); p=config.get("web",{}).get("port",5000)
    print(f"\n  Web UI: http://{h}:{p}\n")
    app.run(host=h, port=p, debug=config.get("web",{}).get("debug",False))

def main():
    parser=argparse.ArgumentParser(description="ShieldRAG v2")
    parser.add_argument("--mode", choices=["evaluate","demo","web"], default="evaluate")
    parser.add_argument("--config", default="configs/config.yaml")
    args=parser.parse_args()
    os.makedirs("results", exist_ok=True)
    setup_logging()
    config=load_config(args.config)
    if args.mode=="evaluate": run_evaluate(config)
    elif args.mode=="demo": run_demo(config)
    elif args.mode=="web": run_web(config)

if __name__=="__main__": main()
