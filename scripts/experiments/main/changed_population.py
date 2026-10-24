"""Changed-population map for B (CA + L3) vs D (CA + L2/L3) on the held-out stream (outputs/composition/work/composition_ablation_replay.pkl, frozen trace).
'Changed' = the request's served route, served evidence (URL list and page versions), or generation question differs between B and D; the
signature sig[q] = (tier, evidence versions, generation question) is recorded per request by the replay, so later answer-cache hits whose stored
answer was generated from different earlier evidence are included (their served evidence differs). Writes outputs/changed_population.json."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, json, pickle, sys, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
O = ((OUTPUT_ROOT / "changed_population")); CHECKS_OUT = ((OUTPUT_ROOT / "composition_checks")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/composition_study_support")); sys.path.insert(0, str(REPO_ROOT / "freshcache/composition"))
import output_workspace; output_workspace.use("composition_checks"); import composition_study_helpers as X   # noqa: E402
E = X.engine(); p2 = E["p2"]; REC = {r["query_id"]: r for r in E["recs"]}
GOLD = json.load(open(p2.GOLD, encoding="utf-8"))
def goldof(qid, qt):
    g = GOLD.get(qid) or GOLD.get(qt) or GOLD.get((qt or "").strip().lower()); return g.get("answer") if isinstance(g, dict) else g
ST = pickle.load(open(X.RCD / "work/composition_ablation_replay.pkl", "rb"))["policies"]; B, D = ST["B"]["sig"], ST["D"]["sig"]
N = len(B); M = [q for q in B if B[q] != D[q]]
why = collections.Counter(); Me, Mu = [], []
for q in M:
    rt = REC[q]["freshness_class"] == "REAL_TIME"; ng = not goldof(q, REC[q]["query"])
    if rt or ng: Mu.append(q); why["REAL_TIME" if rt else "no reference answer"] += 1
    else: Me.append(q)
route = lambda S, q: S[q][0]
dist = {"B_route_x_D_route": collections.Counter(f"{route(B,q)}->{route(D,q)}" for q in M)}
dist["D_L1_hits_with_different_stored_evidence"] = sum(1 for q in M if route(B, q) == "L1" and route(D, q) == "L1")
samp = json.load(open(CHECKS_OUT / "work/changed_sample.json")); assert samp["changed_requests"] == len(M) and samp["frame"] == len(Me)
gen = [d["query_id"] for d in json.load(open((OUTPUT_ROOT / "general_audit/sample.json")))["requests"]]
gen_changed = [q for q in gen if q in set(M)]
R = {"N_heldout": N, "M_changed": len(M), "M_e_eligible": len(Me), "M_u_ineligible": len(Mu), "M_u_reasons": dict(why), "n_changed_sample": samp["n"],
     "changed_definition": "served route, served evidence (URL list and page versions), or generation question differs between B and D; includes answer-cache hits whose stored answer was generated from different earlier evidence",
     "route_transitions_B_to_D": dict(dist["B_route_x_D_route"].most_common()), "answer_cache_hits_in_both_with_different_stored_evidence": dist["D_L1_hits_with_different_stored_evidence"],
     "general_audit_n": len(gen), "general_audit_changed": len(gen_changed), "general_audit_changed_eligible": sum(1 for q in gen_changed if q in set(Me)),
     "overlap_general_audit_and_changed_sample": len(set(gen) & set(samp["requests"])), "unchanged_requests": N - len(M),
     "unchanged_note": "unchanged requests receive identical route, evidence and generation question under B and D, hence identical answers (deterministic greedy generation): their quality difference is zero by construction"}
json.dump(R, open(O / "outputs/changed_population.json", "w"), indent=1); print(json.dumps(R, indent=1))
