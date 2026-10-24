"""Learned L2 compatibility gate on unseen queries, step 1: frozen gate scores (Youden tau) for ALL live L2 candidates, with the exact serving-time
features of l2_gate_heldout_evaluation 'live' (cached-path texts FreshCache would hold = L3 versions or live fetch of the cached URLs; incoming-query embedding;
cached-entry metadata). The independent-search arm is never read. Also times feature extraction + inference per candidate (online inspection overhead)."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import os; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import hashlib, json, pickle, re, sys, time, pathlib
GATE = ((OUTPUT_ROOT / "learned_l2_gate")); T2 = ((OUTPUT_ROOT / "unseen_queries")); O = ((OUTPUT_ROOT / "composition"))
sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/admission_controls")); sys.dont_write_bytecode = True
import l2_gate_features as Lb   # noqa: E402  (read-only; no writes on import)
import numpy as np   # noqa: E402
GZ = pickle.load(open(GATE / "gate_frozen.pkl", "rb")); FEATS, TAU, MODEL = GZ["features"], GZ["tau"], GZ["model"]
pscore = lambda f: float(MODEL.predict_proba(np.array([[f[k] for k in FEATS]], float))[:, 1][0])
FZ = json.load(open(T2 / "sample_frozen.json"))["cases"]; MV = {json.loads(l)["query_id"]: json.loads(l) for l in open(T2 / "cache_state_l2_matching.jsonl")}
UZ = np.load(T2 / "emb_untouched.npz", allow_pickle=True); UQ = {k: i for i, k in enumerate(UZ["keys"])}
from sentence_transformers import SentenceTransformer   # noqa: E402
enc = SentenceTransformer("BAAI/bge-m3", device="cuda:0")
uid = lambda u: hashlib.sha1(u.encode()).hexdigest()
def live_text(u):
    for d in (T2 / "raw/pages_pass2", T2 / "raw/pages"):
        p = d / f"{uid(u)}.json"
        if p.exists():
            x = json.load(open(p, encoding="utf-8"))
            if x.get("ok"): return re.sub(r"\s+", " ", x.get("extracted_text") or "").strip()[:2000]
    return ""
OUT = []; enc.encode(["warm-up"], normalize_embeddings=True)
for z in FZ:
    l3t = MV[z["query_id"]].get("l3_cache_time", {}); texts = [Lb.page_text(uh, l3t[uh]) if uh in l3t else live_text(u) for uh, u in zip(z["cached_url_hashes"], z["cached_urls"])]
    t0 = time.perf_counter()
    qv = UZ["emb"][UQ[z["incoming_query"]]]; pe = enc.encode([t for t in texts if t], normalize_embeddings=True) if any(texts) else []; sims = [float(qv @ e) for e in pe]
    f = Lb.features_core(z["incoming_query"], sims, texts, len(texts), MV[z["query_id"]]["entry_creator_query"], z["similarity"], z["freshness_class"], MV[z["query_id"]]["entry_age_s"], z["entry_class"])
    p = pscore(f); ms = (time.perf_counter() - t0) * 1e3
    OUT.append({"case_id": z["case_id"], "similarity": z["similarity"], "gate_p": p, "gate_accept": p >= TAU, "inspect_ms": round(ms, 2), "n_cached_pages_with_text": sum(1 for t in texts if t)})
old = {r["case_id"]: r["gate_p"] for r in json.load(open(GATE / "gate_live_cases.json"))["cases"]}
G = {"reproduces the frozen 150-case gate scores": all(abs(old[r["case_id"]] - r["gate_p"]) < 1e-6 for r in OUT if r["case_id"] in old), "n_overlap": sum(1 for r in OUT if r["case_id"] in old)}
json.dump({"tau": TAU, "gates": G, "cases": OUT, "inspect_ms_mean": round(float(np.mean([r["inspect_ms"] for r in OUT])), 2)}, open(O / "outputs/gate_scores.json", "w"), indent=1)
print(G, "accepted", sum(r["gate_accept"] for r in OUT), "of", len(OUT), "mean inspect ms", round(float(np.mean([r["inspect_ms"] for r in OUT])), 2))
