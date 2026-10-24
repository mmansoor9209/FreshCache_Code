"""Decision-path lookup cost with a real vector index instead of the precomputed similarity matrix. For policy D's held-out
stream, each non-REAL_TIME request searches the answer-cache index (top-16 by inner product on normalized BGE-M3 query embeddings) and,
on an answer-cache miss, the URL-list index (top-16); every answer-cache write and URL-list write adds the embedding to its index. Timed
per request (search + add), single thread, for an exact index (IndexFlatIP) and an HNSW index (M=32, efSearch=64). Policy B needs the
answer-cache index only; the D-B difference is the URL-list index cost. Writes outputs/vector_index_timing.json."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import json, sys, time, pathlib, pickle
import numpy as np, faiss
faiss.omp_set_num_threads(1)
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); O = (OUTPUT_ROOT / "composition_checks"); sys.path.insert(0, str(ROOT / "freshcache/replay_engine")); import engine_loader as C
E = C.load("reported"); H = C.streams(E)["heldout"]
ST = pickle.load(open((OUTPUT_ROOT / "composition/work/composition_ablation_replay.pkl"), "rb"))["policies"]
Z = np.load((OUTPUT_ROOT / "l2_verifier/out/emb_queries.npz"), allow_pickle=True); KI = {k: i for i, k in enumerate(Z["keys"])}
V = Z["emb"].astype(np.float32); V /= np.linalg.norm(V, axis=1, keepdims=True)
def run(kind, with_l2):
    mk = (lambda: faiss.IndexFlatIP(1024)) if kind == "flat" else (lambda: faiss.IndexHNSWFlat(1024, 32, faiss.METRIC_INNER_PRODUCT))
    ia, il = mk(), mk()
    if kind == "hnsw": ia.hnsw.efSearch = il.hnsw.efSearch = 64
    sig = ST["D"]["sig"]; tl = []
    for _, r in H:
        tier = sig[r["query_id"]][0]
        if tier == "REAL_TIME": continue
        x = V[KI[r["query"]]][None, :]; t0 = time.perf_counter()
        if ia.ntotal: ia.search(x, min(16, ia.ntotal))
        if tier != "L1":
            if with_l2:
                if il.ntotal: il.search(x, min(16, il.ntotal))
                il.add(x)
            ia.add(x)
        tl.append((time.perf_counter() - t0) * 1e3)
    return {"mean_ms_per_request": round(float(np.mean(tl)), 4), "p95_ms": round(float(np.percentile(tl, 95)), 4), "answer_index_size": ia.ntotal, "url_list_index_size": il.ntotal}
R = {}
for kind in ("flat", "hnsw"):
    b = run(kind, False); d = run(kind, True)
    R[kind] = {"B_answer_index_only": b, "D_answer_and_url_list_index": d, "D_minus_B_ms_per_request": round(d["mean_ms_per_request"] - b["mean_ms_per_request"], 4)}
    print(kind, R[kind], flush=True)
json.dump(R, open(O / "outputs/vector_index_timing.json", "w"), indent=1)
