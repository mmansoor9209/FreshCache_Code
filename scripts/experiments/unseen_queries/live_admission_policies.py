"""Frozen L2 admission controls as complete live policies with fallback on the unseen-query live candidates.
Fixed-state counterfactual composition of recorded live paths: accept -> FreshCache service path C (cached list, L3 page reuse), reject -> the
incoming query's own live search path A at matched depth. Policies: always-search; primary L2 (theta 0.75: every candidate is an L2 hit by
construction); stricter L2 (theta 0.80 on the recorded similarity); learned L2 compatibility gate (frozen model, Youden tau, serving-time features;
outputs/gate_scores.json). Operational outcomes over all candidates; quality over those with a reference (answers and two-judge labels
exist for every path); paired vs always-search with McNemar. Cost per candidate (measured live coefficients of component_costs.json, ms): searches x c_s + fetches x
c_f + c_g generation + c_embed query embedding + inspection (L2 lookup; gate feature extraction + inference measured).
Inspection is charged whether the candidate is accepted or rejected."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import os; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import collections, json, math, sys, pathlib
T2 = ((OUTPUT_ROOT / "unseen_queries")); O = ((OUTPUT_ROOT / "composition"))
src = open((REPO_ROOT / "scripts/experiments/unseen_queries/unseen_query_scoring.py")).read(); cut = src.index("PAIR = ["); g = {"__name__": "t2s", "__file__": str((REPO_ROOT / "scripts/experiments/unseen_queries/unseen_query_scoring.py"))}; sys.argv = ["unseen_query_scoring.py", "score"]
exec(compile(src[:cut], "<unseen_query_scoring.py functions, read-only>", "exec"), g); C, label, suff = g["C"], g["label"], g["suff"]
GS = {r["case_id"]: r for r in json.load(open(O / "outputs/gate_scores.json"))["cases"]}
CS, CF, CG, CE = [(c["c_search_ms"]["live_p50"], c["c_fetch_ms"]["live_p50"], c["c_gen_ms"]["value"], c["c_embed_ms"]["value"]) for c in [json.load(open(OUTPUT_ROOT / "concurrency_cost/component_costs.json"))["coefficients"]]][0]   # component_costs.json (serving_cost_summary.py)
def mcn(b, c): n = b + c; return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
POL = {"Always search": lambda c: False, "Primary L2 (theta 0.75)": lambda c: True, "Stricter L2 (theta 0.80)": lambda c: GS[c["case_id"]]["similarity"] >= 0.80,
       "Learned gate (Youden tau)": lambda c: GS[c["case_id"]]["gate_accept"]}
INSPECT = {"Always search": 0.0, "Primary L2 (theta 0.75)": 0.1, "Stricter L2 (theta 0.80)": 0.1, "Learned gate (Youden tau)": None}
R = {}; SERVED = {}
for name, acc in POL.items():
    rows = []
    for c in C:
        a = acc(c); path = "C_freshcache" if a else "A_matched"
        fetches = (len(c["cached_urls"]) - c.get("l3_served_pages_C", 0)) if a else len(c["paths"]["A_matched"].get("fetched", [])) or c["k"]
        insp = GS[c["case_id"]]["inspect_ms"] if INSPECT[name] is None else INSPECT[name]
        cost = (0 if a else 1) * CS + fetches * CF + CG + CE + insp
        usable = c["cached_avail"] if a else c["fresh_avail"]
        lab = label(c, path) if c["reference"] else "NO_REFERENCE"; sf = suff(c, path)
        rows.append({"case_id": c["case_id"], "accepted": a, "path": path, "cost_ms": cost, "usable": usable, "search_failed": (not a) and not c["search_ok"], "label": lab, "suff": sf})
    SERVED[name] = {r["case_id"]: r for r in rows}
    n = len(rows); ref = [r for r in rows if r["label"] != "NO_REFERENCE"]
    R[name] = {"candidates": n, "accepted": sum(r["accepted"] for r in rows), "rejected": sum(not r["accepted"] for r in rows), "acceptance_pct": round(100 * sum(r["accepted"] for r in rows) / n, 2),
               "searches_avoided_per_candidate_pct": round(100 * sum(r["accepted"] for r in rows) / n, 2), "mean_cost_ms": round(sum(r["cost_ms"] for r in rows) / n, 1),
               "fallback_successful": sum(1 for r in rows if not r["accepted"] and r["usable"]), "fallback_failed_unusable_pages": sum(1 for r in rows if not r["accepted"] and not r["usable"]),
               "accepted_usable": sum(1 for r in rows if r["accepted"] and r["usable"]), "accepted_unusable": sum(1 for r in rows if r["accepted"] and not r["usable"]),
               "availability_failures_total": sum(1 for r in rows if not r["usable"]), "adjudicable_with_reference": len(ref), "unresolved": sum(1 for r in ref if r["label"] == "UNRESOLVED"),
               "sufficient": sum(1 for r in rows if r["suff"] == "SUFFICIENT"), "sufficient_pct": round(100 * sum(1 for r in rows if r["suff"] == "SUFFICIENT") / n, 2),
               "correct": sum(1 for r in ref if r["label"] == "CORRECT"), "accuracy_pct": round(100 * sum(1 for r in ref if r["label"] == "CORRECT") / len(ref), 2),
               "wrong": sum(1 for r in ref if r["label"] == "WRONG"), "abstain": sum(1 for r in ref if r["label"] == "ABSTAIN"),
               "usable_evidence_subset": {"n": sum(1 for r in ref if r["usable"]), "sufficient": sum(1 for r in ref if r["usable"] and r["suff"] == "SUFFICIENT"), "correct": sum(1 for r in ref if r["usable"] and r["label"] == "CORRECT")}}
base = SERVED["Always search"]
for name in POL:
    if name == "Always search": continue
    S = SERVED[name]; ids = [i for i in S if S[i]["label"] != "NO_REFERENCE"]
    sl = sum(1 for i in S if base[i]["suff"] == "SUFFICIENT" and S[i]["suff"] != "SUFFICIENT"); sg = sum(1 for i in S if base[i]["suff"] != "SUFFICIENT" and S[i]["suff"] == "SUFFICIENT")
    cl = sum(1 for i in ids if base[i]["label"] == "CORRECT" and S[i]["label"] != "CORRECT"); cg = sum(1 for i in ids if base[i]["label"] != "CORRECT" and S[i]["label"] == "CORRECT")
    R[name]["vs_always_search"] = {"SUFF->not": sl, "not->SUFF": sg, "suff_p": mcn(sl, sg), "suff_diff_pp": round(100 * (sg - sl) / len(S), 2), "CORRECT->not": cl, "not->CORRECT": cg, "correct_p": mcn(cl, cg),
                                   "correct_diff_pp": round(100 * (cg - cl) / len(ids), 2), "cost_saving_pct": round(100 * (1 - R[name]["mean_cost_ms"] / R["Always search"]["mean_cost_ms"]), 2)}
json.dump({"semantics": "fixed-state counterfactual composition of recorded live paths (no later cache-state changes)", "fallback_path": "A_matched (incoming-query live search, cached-list depth)", "policies": R},
          open(O / "outputs/live_policies.json", "w"), indent=1)
for k, v in R.items(): print(k, json.dumps(v))
