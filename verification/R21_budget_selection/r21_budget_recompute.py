"""Recompute the validation budget selection and the held-out check of the frozen triple from saved row-level records. Read-only."""
import json, csv, itertools, collections, pathlib, datetime
S = pathlib.Path("<PROJECT_ROOT>/validation_V3/parameter_calibration"); OUT = pathlib.Path(__file__).parent
pre = json.load(open(S/"04_budgets/prereg.json")); sw = json.load(open(S/"04_budgets/validation_sweep.json")); fz = json.load(open(S/"04_budgets/frozen_config.json"))
sp = pre["search_space"]; grid = list(itertools.product(sp["eps_answer"], sp["eps_url_list"], sp["eps_content"]))
R = {"utc": datetime.datetime.utcnow().isoformat()+"Z", "grid": {"eps_L1": sp["eps_answer"], "eps_L2": sp["eps_url_list"], "eps_L3": sp["eps_content"], "product": f"{len(sp['eps_answer'])}x{len(sp['eps_url_list'])}x{len(sp['eps_content'])}={len(grid)}", "cells_in_sweep": len(sw["cells"]), "grid_equals_sweep_keys": {f"{a}_{u}_{c}" for a,u,c in grid} == set(sw["cells"])}}
pub = sw["published_validation"]
C1, C2, C3 = pub["drift_pct"] + 0.25, pub["l1_hits"], 0.95*pub["coverage_pct"]
R["constraints_recomputed"] = {"C1_drift_max": round(C1,4), "C2_l1_max": C2, "C3_coverage_min": round(C3,6), "recorded": sw["constraints"], "published_validation": pub}
cells = list(sw["cells"].values())
feas = [v for v in cells if v["drift_pct"] is not None and v["drift_pct"] <= C1 and v["l1_hits"] <= C2 and v["coverage_pct"] >= C3]
key = lambda v: (-v["search_saved_pct"], v["drift_pct"], v["l1_hits"], v["eps_answer"], v["eps_url_list"], v["eps_content"])
ranked = sorted(feas, key=key); sel = ranked[0]
R["selection"] = {"n_feasible": len(feas), "recorded_n_feasible": sw["n_feasible"], "selected": (sel["eps_answer"], sel["eps_url_list"], sel["eps_content"]), "recorded_selected": (sw["selected"]["eps_answer"], sw["selected"]["eps_url_list"], sw["selected"]["eps_content"]), "frozen": tuple(fz["selected_eps"].values()),
   "selected_metrics": {k: sel[k] for k in ("search_saved_pct","drift_pct","coverage_pct","l1_hits","l2_hits","l3_hits")},
   "feasible_ranked": [{"eps": (v["eps_answer"], v["eps_url_list"], v["eps_content"]), "saved": v["search_saved_pct"], "drift": v["drift_pct"], "cov": v["coverage_pct"], "L1": v["l1_hits"]} for v in ranked],
   "tie_on_primary_objective": [ (v["eps_answer"], v["eps_url_list"], v["eps_content"]) for v in feas if v["search_saved_pct"] == sel["search_saved_pct"]],
   "objective_and_tiebreak_as_coded": "sweep_validation.py:122-124 sort key (-saved, drift, l1_hits, eps_L1, eps_L2, eps_L3)"}
# 48-row table
with open(OUT/"candidate_grid_48.csv","w",newline="") as fh:
    w = csv.writer(fh); w.writerow(["eps_L1","eps_L2","eps_L3","k_L1","k_L2","k_L3","val_search_saved_pct","val_drift_pct","val_coverage_pct","L1_hits","L2_hits","L3_hits","C1_drift<=%.4f"%C1,"C2_L1<=%d"%C2,"C3_cov>=%.4f"%C3,"eligible","rank_among_eligible"])
    rank = {(v["eps_answer"],v["eps_url_list"],v["eps_content"]): i+1 for i,v in enumerate(ranked)}
    for a,u,c in grid:
        v = sw["cells"][f"{a}_{u}_{c}"]; ok1, ok2, ok3 = v["drift_pct"] <= C1, v["l1_hits"] <= C2, v["coverage_pct"] >= C3
        w.writerow([a,u,c,v["k_answer"],v["k_url_list"],v["k_content"],v["search_saved_pct"],v["drift_pct"],v["coverage_pct"],v["l1_hits"],v["l2_hits"],v["l3_hits"],ok1,ok2,ok3,ok1 and ok2 and ok3, rank.get((a,u,c),"")])
# held-out: drift and coverage of C from per-request outcomes; savings not recomputable from this file
rows = list(csv.DictReader(open(S/"05_heldout/heldout_per_request.csv")))
hc = {}
for cfg in ("A_original","C_budget_only"):
    c = collections.Counter(r[cfg] for r in rows); det = c["CHANGED"]+c["UNCHANGED"]
    hc[cfg] = {"N": len(rows), "outcomes": dict(c), "observable_n": det, "coverage_pct": round(100*det/len(rows),4), "drift_pct": round(100*c["CHANGED"]/det,4)}
hr = json.load(open(S/"05_heldout/heldout_results.json"))["configs"]
R["heldout"] = {"recomputed": hc, "recorded_C": {k: hr["C_budget_only"][k] for k in ("search_saved_pct","l1_hits","l2_hits","l3_hits","observable_n","coverage_pct","drift_pct","drift_ci95")}, "recorded_A": {k: hr["A_original"][k] for k in ("search_saved_pct","l1_hits","l2_hits","coverage_pct","drift_pct")}, "savings_note": "search_saved_pct is not recomputable from heldout_per_request.csv (outcomes only); value is the study's record, anchored by config A reproducing 60.5776/806/12,429"}
aq = list(csv.DictReader(open(S/"07_answer_quality/answer_quality_rows.csv")))
wai = {cfg: [r["query_id"] for r in aq if r["fresh"]=="CORRECT" and r[cfg]=="WRONG"] for cfg in ("A_original","C_budget_only")}
R["wai"] = {"n": len(aq), "A": len(wai["A_original"]), "C": len(wai["C_budget_only"]), "C_ids": wai["C_budget_only"], "shared_ids": len(set(wai["A_original"]) & set(wai["C_budget_only"])), "C_tiers": dict(collections.Counter(r["C_budget_only_tier"] for r in aq)), "C_wai_tiers": dict(collections.Counter(r["C_budget_only_tier"] for r in aq if r["query_id"] in wai["C_budget_only"]))}
json.dump(R, open(OUT/"R21_budget_results.json","w"), indent=1, default=str); print(json.dumps(R, indent=1, default=str))
