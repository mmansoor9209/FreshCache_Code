"""Index parity: does top-16 candidate retrieval (as in the vector_index_timing index runs) reproduce the frozen admission decisions?
Replays B and D on the held-out stream with the frozen engine (composition_engine_topk_index.py = composition_engine.py + candidate-restriction hooks only) under
(i) full scan (must reproduce the frozen trace composition_ablation_replay.pkl exactly), (ii) exact top-16 (sorted by the frozen similarity matrix, identical to the
inner product of the BGE-M3 embeddings up to float16 precision), (iii) HNSW top-16 (FAISS IndexHNSWFlat, M=32, efSearch=64, incremental adds,
as in vector_index_timing.py). Admission rules are unchanged: among retrieved candidates, the same similarity floor, TTL eligibility and argmax.
Compares each request's (route, served evidence, generation question) with the frozen trace. Writes outputs/vector_index_parity.json."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import json, pickle, sys, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import numpy as np, faiss
faiss.omp_set_num_threads(1)
O = ((OUTPUT_ROOT / "changed_population")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/composition_study_support")); sys.path.insert(0, str(REPO_ROOT / "freshcache/composition"))
import output_workspace; output_workspace.use("changed_population"); import composition_study_helpers as X, composition_engine_topk_index as RE   # noqa: E402
E = X.engine(); exp = E["exp"]; SIM, IDX = exp._SIM_MATRIX, exp._QUERY_TO_IDX; INV = {i: q for q, i in IDX.items()}
H = X.C.streams(E)["heldout"]; TH, KA = X.ca_params(0.0, 1.0)
Z = np.load((OUTPUT_ROOT / "l2_verifier/out/emb_queries.npz"), allow_pickle=True); KI = {k: i for i, k in enumerate(Z["keys"])}
V = Z["emb"].astype(np.float32); V /= np.linalg.norm(V, axis=1, keepdims=True)
FROZEN = pickle.load(open(X.RCD / "work/composition_ablation_replay.pkl", "rb"))["policies"]
K = 16
class Exact:
    def __init__(s): pass
    def _top(s, sims): return np.argsort(-sims, kind="stable")[:K] if sims.size > K else np.arange(sims.size)
    def l1(s, qi, n, sims): return s._top(sims)
    def l2(s, qi, n, sims): return s._top(sims)
    def add1(s, qi): pass
    def add2(s, qi): pass
class HNSW:
    def __init__(s):
        s.i1, s.i2 = (faiss.IndexHNSWFlat(1024, 32, faiss.METRIC_INNER_PRODUCT) for _ in range(2))
        s.i1.hnsw.efSearch = s.i2.hnsw.efSearch = 64
    def _q(s, idx, qi):
        if idx.ntotal == 0: return np.array([], int)
        _, I = idx.search(V[KI[INV[qi]]][None, :], min(K, idx.ntotal)); return I[0][I[0] >= 0]
    def l1(s, qi, n, sims): return s._q(s.i1, qi)
    def l2(s, qi, n, sims): return s._q(s.i2, qi)
    def add1(s, qi): s.i1.add(V[KI[INV[qi]]][None, :])
    def add2(s, qi): s.i2.add(V[KI[INV[qi]]][None, :])
POL = {"B": dict(l2_on=False), "D": dict()}
def sig(row): return (row["tier"], tuple(tuple(x) for x in row["ev"]), row.get("gen_query"))
R = {}
for p, kw in POL.items():
    fz = FROZEN[p]["sig"]
    for mode, mk in (("full_scan", lambda: None), ("exact_top16", Exact), ("hnsw_top16", HNSW)):
        m, rows = RE.replay(E, H, l1="ca", floor=0.80, theta_by_class=TH, kappa_by_class=KA, retr=mk(), **kw)
        diff = [q for q in rows if sig(rows[q]) != tuple((fz[q][0], tuple(tuple(x) for x in fz[q][1]), fz[q][2]))]
        tiers = {}
        for q in diff: k = f"{fz[q][0]}->{rows[q]['tier']}"; tiers[k] = tiers.get(k, 0) + 1
        R[f"{p}/{mode}"] = {"search_saved_pct": round(m["search_saved_pct"], 2), "l1_hits": m["l1_hits"], "l2_hits": m["l2_hits"], "fetches": m["fetches"],
                            "requests_differing_from_frozen": len(diff), "pct_differing": round(100 * len(diff) / len(rows), 3), "route_changes": tiers}
        print(p, mode, R[f"{p}/{mode}"], flush=True)
json.dump(R, open(O / "outputs/vector_index_parity.json", "w"), indent=1)
