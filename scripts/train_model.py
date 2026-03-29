"""
ShieldRAG v2 — One-time training script
========================================
Downloads real datasets (SQuAD + deepset/prompt-injections), trains all
security classifiers and builds retrieval indexes with high-quality settings,
then persists every artifact to the configured model_cache_dir (default: models/).

Subsequent calls to ShieldRAGPipeline.load_and_prepare() will load from cache
instead of re-training — making cold-start essentially instant.

Usage
-----
    python scripts/train_model.py [--config configs/config.yaml] [--model-dir models/]

The saved artifacts are:
    models/gmtp_classifier.joblib      — Layer 1 anomaly detector (sklearn)
    models/injecguard.joblib           — Layer 3 injection classifier (sklearn)
    models/verifier.joblib             — Layer 3 output verifier (sklearn)
    models/retriever_dense_mpnet.*     — Layer 2 FAISS index + doc list (mpnet)
    models/retriever_sparse_bm25.pkl   — Layer 2 BM25 index
    models/retriever_dense_minilm.*    — Layer 2 FAISS index + doc list (minilm)
"""

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shieldrag.utils.helpers import load_config, setup_logging, set_seed
from shieldrag.pipeline import ShieldRAGPipeline


def parse_args():
    p = argparse.ArgumentParser(description="ShieldRAG v2 — one-time model trainer")
    p.add_argument("--config", default="configs/config.yaml",
                   help="Path to YAML config file (default: configs/config.yaml)")
    p.add_argument("--model-dir", default=None,
                   help="Override model_cache_dir from config")
    p.add_argument("--force", action="store_true",
                   help="Re-train and overwrite existing cached models")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()
    config = load_config(args.config)

    if args.model_dir:
        config["model_cache_dir"] = args.model_dir

    model_dir = config.get("model_cache_dir", "models")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # If --force, remove existing model artifacts so they are rebuilt.
    if args.force:
        removed = []
        for pattern in ["*.joblib", "*.faiss", "*.pkl"]:
            for f in glob.glob(os.path.join(model_dir, pattern)):
                os.remove(f)
                removed.append(f)
        if removed:
            print(f"  [force] Removed {len(removed)} cached artifact(s)")

    set_seed(config.get("seed", 42))

    print("\n" + "=" * 70)
    print("  ShieldRAG v2 — Model Training")
    print(f"  Real datasets : SQuAD (benign) + deepset/prompt-injections (attack)")
    print(f"  Model cache   : {os.path.abspath(model_dir)}")
    print("=" * 70 + "\n")

    t0 = time.time()
    pipe = ShieldRAGPipeline(config)
    all_docs, train_q, val_q, test_q = pipe.load_and_prepare()
    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print("  Training complete!")
    print(f"  Elapsed     : {elapsed:.1f}s")
    print(f"  Documents   : {len(all_docs)}")
    print(f"  Train queries: {len(train_q)}")
    print(f"  Val queries : {len(val_q)}")
    print(f"  Test queries: {len(test_q)}")
    print(f"\n  Saved artifacts in: {os.path.abspath(model_dir)}/")
    for fname in sorted(os.listdir(model_dir)):
        fpath = os.path.join(model_dir, fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {fname:<45s} {size_kb:8.1f} KB")
    print("=" * 70)
    print("\n  Next time ShieldRAGPipeline runs it will load from cache — no re-training.\n")


if __name__ == "__main__":
    main()
