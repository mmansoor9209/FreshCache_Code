"""B (CategoryAware + L3) vs D (+ L2/L3) and FreshCache primary on the held-out sub-streams of Table 5, each replayed alone from empty caches.
Policies are frozen; per stream, reports the share of requests with an earlier request (same sub-stream) within cosine 0.75 (frozen BGE-M3
query embeddings). Tests the savings decomposition: predicted savings = neighbour share x reuse rate, with the reuse rate calibrated ONCE on
the full held-out stream and applied unchanged to every sub-stream (out-of-sample check). Writes outputs/substream_replay.json."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os; import collections, json, sys, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import numpy as np
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); O = (OUTPUT_ROOT / "composition_checks"); sys.path.insert(0, str(ROOT / "scripts/experiments/main")); sys.path.insert(0, str(ROOT / "freshcache/composition"))
import composition_helpers as X, composition_engine as RE   # noqa: E402
E = X.engine(); H = X.C.streams(E)["heldout"]; TH, KA = X.ca_params(0.0, 1.0)
SRC = {json.loads(l)["query_id"]: json.loads(l)["source"] for l in open((DATA_ROOT / "queries.jsonl"))}
REC = {r["query_id"]: r for r in E["recs"]}; src = lambda q: SRC.get(REC[q].get("cluster_id") or q, SRC.get(q, "unknown")); taq = lambda r: src(r["query_id"]) == "temporal_align_qa"
Z = np.load(X.OV / "emb_queries.npz", allow_pickle=True); KI = {k: i for i, k in enumerate(Z["keys"])}; V = Z["emb"].astype(np.float32); V /= np.linalg.norm(V, axis=1, keepdims=True)
def neighbour_share(Hs):
    Xs = V[[KI[r["query"]] for _, r in Hs]]; n = len(Hs); has = np.zeros(n, bool)
    for s in range(0, n, 1024):
        e = min(n, s + 1024); S = Xs[s:e] @ Xs[:e].T
        for i in range(s, e):
            if i: has[i] = S[i - s, :i].max() >= 0.75
    return float(has.mean())
STREAMS = {"all": H, "without TAQ": [x for x in H if not taq(x[1])], "originals": [x for x in H if not x[1].get("is_paraphrase")],
           "TAQ originals": [x for x in H if not x[1].get("is_paraphrase") and taq(x[1])], "other originals": [x for x in H if not x[1].get("is_paraphrase") and not taq(x[1])]}
R = {}
for name, Hs in STREAMS.items():
    S = E["support_of"](Hs, E["rounds"]); fc, _, _ = X.C.metrics(E, Hs, "FreshCache", S)
    mB, _ = RE.replay(E, Hs, l1="ca", floor=0.80, theta_by_class=TH, kappa_by_class=KA, l2_on=False)
    mD, _ = RE.replay(E, Hs, l1="ca", floor=0.80, theta_by_class=TH, kappa_by_class=KA)
    n = len(Hs); sh = neighbour_share(Hs)
    R[name] = {"n": n, "neighbour_share_pct": round(100 * sh, 2), "FreshCache_saved_pct": round(fc["search_saved_pct"], 2), "B_saved_pct": round(mB["search_saved_pct"], 2), "D_saved_pct": round(mD["search_saved_pct"], 2),
               "L2_increment_pp": round(mD["search_saved_pct"] - mB["search_saved_pct"], 2), "B_fetch_per_1k": round(1000 * mB["fetches"] / n, 1), "D_fetch_per_1k": round(1000 * mD["fetches"] / n, 1)}
    print(name, R[name], flush=True)
rate_fc = R["all"]["FreshCache_saved_pct"] / R["all"]["neighbour_share_pct"]; rate_inc = R["all"]["L2_increment_pp"] / R["all"]["neighbour_share_pct"]
for name in R:
    R[name]["FreshCache_predicted_pct"] = round(rate_fc * R[name]["neighbour_share_pct"], 2); R[name]["L2_increment_predicted_pp"] = round(rate_inc * R[name]["neighbour_share_pct"], 2)
json.dump({"calibration": {"FreshCache_rate_per_neighbour_share": round(rate_fc, 4), "L2_increment_rate_per_neighbour_share": round(rate_inc, 4), "calibrated_on": "full held-out stream"}, "streams": R},
          open(O / "outputs/substream_replay.json", "w"), indent=1)
for k, v in R.items(): print(k, "share", v["neighbour_share_pct"], "FC meas/pred", v["FreshCache_saved_pct"], v["FreshCache_predicted_pct"], "L2 inc meas/pred", v["L2_increment_pp"], v["L2_increment_predicted_pp"])
