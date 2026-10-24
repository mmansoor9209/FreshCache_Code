"""Stratifies the changed-request audit of composition D versus B by the depth of the URL list D served and by D's route.
Uses the audited requests of the eligible D!=B requests (frozen two-judge labels already computed); depth = number of URLs in D's
evidence list. Recomputes the stream-level scaling with the matching eligible share (eligible / held-out requests) and with the full changed share
(changed / held-out requests). Read-only use of the existing audit pools. Writes outputs/changed_audit_by_list_depth.json."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, json, math, sys, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
CHECKS_OUT = (OUTPUT_ROOT / "composition_checks"); O = (OUTPUT_ROOT / "composition_summary")
sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/composition_study_support")); sys.path.insert(0, str(REPO_ROOT / "freshcache/composition")); import output_workspace; output_workspace.use("composition_checks"); import composition_study_helpers as X   # noqa: E402  (composition-checks helpers: their pools contain the changed-request audit answers/judgments)
E = X.engine(); p2 = E["p2"]; hl = X.Held(); jl = X.jl
GOLD = json.load(open(p2.GOLD, encoding="utf-8"))
def goldof(qid, qt):
    g = GOLD.get(qid) or GOLD.get(qt) or GOLD.get((qt or "").strip().lower()); return g.get("answer") if isinstance(g, dict) else g
REC = {r["query_id"]: r for r in E["recs"]}; F = json.load(open(CHECKS_OUT / "work/changed_sample.json"))
lab = {}
for q in F["requests"]:
    for p in ("B", "D"):
        r = F["rows"][q][p]; k, _, _ = X.key_for({"ev": r["ev"], "gen_query": r["gen_query"]}, REC[q]["query"])
        lab[(q, p)] = hl.labels(REC[q]["query"], goldof(q, REC[q]["query"]), k)[0] or "UNRESOLVED"
def wil(k, n, z=1.96):
    if not n: return None
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return [round(100 * (c - m) / d, 1), round(100 * (c + m) / d, 1)]
def stat(ids):
    bc = [q for q in ids if lab[(q, "B")] == "CORRECT"]; lost = sum(1 for q in bc if lab[(q, "D")] != "CORRECT"); gained = sum(1 for q in ids if lab[(q, "B")] != "CORRECT" and lab[(q, "D")] == "CORRECT")
    return {"n": len(ids), "B_correct": len(bc), "lost": lost, "gained": gained, "loss_rate_changed_pct": round(100 * lost / len(ids), 2) if ids else None, "ci": wil(lost, len(ids)),
            "loss_given_B_correct_pct": round(100 * lost / len(bc), 1) if bc else None, "ci_given_B_correct": wil(lost, len(bc))}
depth = {q: len(F["rows"][q]["D"]["ev"]) for q in F["requests"]}; tier = {q: F["rows"][q]["D"]["tier"] for q in F["requests"]}
R = {"all": stat(F["requests"]), "by_D_route": {t: stat([q for q in F["requests"] if tier[q] == t]) for t in sorted(set(tier.values()))},
     "by_D_list_depth": {d: stat([q for q in F["requests"] if (str(depth[q]) if depth[q] <= 2 else ">=3") == d]) for d in ("0", "1", "2", ">=3")},
     "depth_counts": dict(collections.Counter(depth.values()))}
L2 = [q for q in F["requests"] if tier[q] == "L2"]
R["L2_served_by_depth"] = {d: stat([q for q in L2 if (str(depth[q]) if depth[q] <= 2 else ">=3") == d]) for d in ("1", "2", ">=3")}
loss = R["all"]["lost"]; n = R["all"]["n"]; ci = wil(loss, n)
for nm, share in (("eligible_share_2580", F["frame"] / F["stream"]), ("changed_share_2785", F["changed_requests"] / F["stream"])):
    R[f"stream_cost_{nm}"] = {"share_pct": round(100 * share, 2), "pp": round(100 * share * loss / n, 2), "ci": [round(share * x, 2) for x in ci]}
# depth distribution of cached lists in the held-out stream (all L2-served D requests)
import pickle
ST = pickle.load(open(X.RCD / "work/composition_ablation_replay.pkl", "rb"))["policies"]["D"]["sig"]
dd = collections.Counter(min(len(v[1]), 3) for v in ST.values() if v[0] == "L2"); tot = sum(dd.values())
R["heldout_L2_served_depth_share_pct"] = {("%d" % k if k < 3 else ">=3"): round(100 * c / tot, 1) for k, c in sorted(dd.items())}
json.dump(R, open(O / "outputs/changed_audit_by_list_depth.json", "w"), indent=1); print(json.dumps(R, indent=1))
