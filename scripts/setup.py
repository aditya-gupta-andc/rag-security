"""Setup — check deps, download models."""
import os, sys

def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            name=torch.cuda.get_device_name(0)
            vram=torch.cuda.get_device_properties(0).total_memory/1e9
            print(f"  [OK] GPU: {name} ({vram:.1f} GB VRAM)")
        else: print("  [!!] No GPU — CPU mode")
    except ImportError: print("  [!!] PyTorch not installed"); return False
    return True

def download_models():
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
    for n,d in [("sentence-transformers/all-mpnet-base-v2","Embedding"),
                ("sentence-transformers/all-MiniLM-L6-v2","Secondary")]:
        print(f"  Downloading {d}: {n}"); SentenceTransformer(n); print("    [OK]")
    print("  Downloading reranker..."); CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2"); print("    [OK]")
    print("  Downloading generator..."); AutoTokenizer.from_pretrained("google/flan-t5-large")
    AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large"); print("    [OK]")
    print("  Downloading GMTP model..."); AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2"); print("    [OK]")

def main():
    print("="*60+"\n  ShieldRAG v2 Setup\n"+"="*60)
    print("\n1. GPU:"); check_gpu()
    print("\n2. Dependencies:")
    for d in ["torch","transformers","sentence_transformers","faiss","sklearn","flask","yaml","matplotlib","seaborn","cryptography"]:
        try: __import__(d); print(f"  [OK] {d}")
        except: print(f"  [!!] {d}")
    print("\n3. Models:"); download_models()
    os.makedirs("results",exist_ok=True)
    print("\n"+"="*60+"\n  Ready! Run: python -m shieldrag.main --mode evaluate\n"+"="*60)

if __name__=="__main__": main()
