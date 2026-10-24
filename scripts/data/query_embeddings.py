"""Builds the BGE-M3 query embeddings and the pairwise cosine-similarity matrix of DATA_DIR (query_embeddings_bgem3.npy, sim_matrix_bgem3.npy).

Same model, record order and computation as tiered_cache_model.precompute_similarity_matrix: SentenceTransformer("BAAI/bge-m3") encodes the
query of every record of tiered_cache_model.build_query_records (base questions and paraphrases, in that order); the embeddings are saved as
encoded, and the similarity matrix is E_n @ E_n.T in float32 with L2-normalised rows. precompute_similarity_matrix needs CUDA devices; this
script runs on one GPU when available and on the CPU otherwise (the sample takes a few minutes on a CPU). The model is downloaded from the
Hugging Face Hub on first use. Existing files are kept unless --force is given.
Usage: python scripts/data/query_embeddings.py [--force] [--batch-size 64] [--device cpu|cuda]"""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import argparse, os, pathlib, sys
REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
sys.path.insert(0, str(REPO_ROOT / "freshcache/method")); sys.path.insert(0, str(REPO_ROOT))
ap = argparse.ArgumentParser(add_help=False); ap.add_argument("--force", action="store_true"); ap.add_argument("--batch-size", type=int, default=64)
ap.add_argument("--device", default=None); a = ap.parse_args()
import numpy as np, torch                     # noqa: E402
import tiered_cache_model as exp              # noqa: E402
if exp.SIM_MATRIX_CACHE.exists() and exp.EMBEDDINGS_CACHE.exists() and not a.force:
    print(f"kept existing {exp.EMBEDDINGS_CACHE.name} and {exp.SIM_MATRIX_CACHE.name} in {exp.DATA_DIR}"); sys.exit(0)
records = exp.build_query_records(exp.load_jsonl(exp.QUERIES_FILE), exp.load_jsonl(exp.MANIFEST_FILE),
                                  exp.load_jsonl(exp.PARAPHRASE_FILE) if exp.PARAPHRASE_FILE.exists() else [])
queries = [r["query"] for r in records]
device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
from sentence_transformers import SentenceTransformer   # noqa: E402
print(f"encoding {len(queries)} queries with BAAI/bge-m3 on {device}", flush=True)
model = SentenceTransformer("BAAI/bge-m3", device=device)
embeddings = model.encode(queries, batch_size=a.batch_size, show_progress_bar=False, convert_to_numpy=True)
np.save(str(exp.EMBEDDINGS_CACHE), embeddings)
E = torch.tensor(embeddings, dtype=torch.float32, device=device); E = E / E.norm(dim=1, keepdim=True)
S = (E @ E.T).cpu().numpy().astype(np.float32)
np.save(str(exp.SIM_MATRIX_CACHE), S)
print(f"wrote {exp.EMBEDDINGS_CACHE} {embeddings.shape} and {exp.SIM_MATRIX_CACHE} {S.shape}")
