"""Unseen-query preparation: BGE-M3 embeddings (sentence_transformers, same model as data/sim_matrix_bgem3.npy; normalised) for the base
queries whose clusters are in NEITHER the validation nor the held-out split (never used anywhere). Writes untouched_queries.jsonl +
emb_untouched.npz here only."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import json, os, pathlib, sys, numpy as np; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); import importlib.util as _ilu; _ds = _ilu.spec_from_file_location("dataset_sizes", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/dataset_sizes.py")); DS = _ilu.module_from_spec(_ds); _ds.loader.exec_module(DS)
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); O = ((OUTPUT_ROOT / "unseen_queries"))
s = json.load(open((OUTPUT_ROOT / "validation_split/split.json"))); used = set(s["validation_clusters"]) | set(s["test_clusters"])
q = [json.loads(l) for l in open((DATA_ROOT / "queries.jsonl"))]; un = [x for x in q if x["query_id"] not in used]; assert DS.check("unseen_queries", len(un))
man = {}
for l in open((DATA_ROOT / "url_manifest.jsonl")):
    x = json.loads(l)
    if x["run_id"] == "run_00": man.setdefault(x["query_id"], []).append((x.get("rank", 99), x["url_hash"], x["url"]))
for x in un: x["run00_urls"] = [{"url_hash": u, "url": w} for _, u, w in sorted(man.get(x["query_id"], []))]
with open(O / "untouched_queries.jsonl", "w", encoding="utf-8") as f:
    for x in un: f.write(json.dumps(x, ensure_ascii=False) + "\n")
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("BAAI/bge-m3", device="cuda:0"); E = m.encode([x["query"] for x in un], batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
# self-consistency gate: re-embed 200 stream queries and compare with the frozen embedding file
Z = np.load((OUTPUT_ROOT / "l2_verifier/out/emb_queries.npz"), allow_pickle=True); keys = list(Z["keys"])[:200]
R = m.encode(keys, batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype(np.float32); F = Z["emb"][:200]; F = F / np.linalg.norm(F, axis=1, keepdims=True)
gate = float(np.abs((R * F).sum(1) - 1).max()); np.savez(O / "emb_untouched.npz", keys=np.array([x["query"] for x in un]), ids=np.array([x["query_id"] for x in un]), emb=E)
json.dump({"n": len(un), "dim": int(E.shape[1]), "max_|cos(re-embedded, frozen) - 1|_over_200_stream_queries": gate, "model": "BAAI/bge-m3 (sentence_transformers, normalised)"}, open(O / "work/embed_gate.json", "w"), indent=1); print("gate", gate)
