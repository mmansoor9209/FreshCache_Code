"""Budget selection rerun under the replay engine with the original pre-registered protocol
(configs/protocols/budget_selection_protocol.json, scripts/experiments/ablations/risk_budget_validation_sweep.py): 48 cells, constraints relative to the
recorded configuration's validation run, maximise savings, tie-breaks lower drift, fewer L1 hits, lexicographic eps.
Held-out configurations A-E and paired contrasts as in the parameter-calibration held-out scripts/evaluation
(m_content = the recorded validation MLE from parameter_calibration/heldout/heldout_results.json; it is fitted on page labels and does not depend on the replay)."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import sys, json, math, random, itertools, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); import importlib.util as _ilu; _ds = _ilu.spec_from_file_location("dataset_sizes", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/dataset_sizes.py")); DS = _ilu.module_from_spec(_ds); _ds.loader.exec_module(DS)
from collections import defaultdict
from fractions import Fraction
sys.path.insert(0, str(REPO_ROOT / "freshcache/replay_engine")); import engine_loader as C
ENG = sys.argv[1]; E = C.load(ENG); exp, me, ea = E["exp"], E["me"], E["ea"]; rich = ea._rich_feats()
PR = json.load(open(C.ROOT / "configs/protocols/budget_selection_protocol.json")); sp = PR["search_space"]
MHAT = json.load(open((OUTPUT_ROOT / "parameter_calibration/heldout/heldout_results.json")))["m_content_mle"]
ST = C.streams(E); VAL, TEST = ST["validation"], ST["heldout"]; assert DS.check("validation_requests", len(VAL)) and DS.check("heldout_requests", len(TEST))
cid = {r["query_id"]: (r.get("cluster_id") or r["query_id"]) for r in E["recs"]}
base_eps = (exp.EPS_ANSWER, exp.EPS_URL_LIST, exp.EPS_CONTENT); base_m = dict(exp.TIER_MULT)
def run_val(a, u, c):
    exp.EPS_ANSWER, exp.EPS_URL_LIST, exp.EPS_CONTENT = a, u, c
    try: m, per = me.replay(VAL, E["rounds"], "FreshCache", rich=rich)
    finally: exp.EPS_ANSWER, exp.EPS_URL_LIST, exp.EPS_CONTENT = base_eps
    det = sum(1 for _, o in per if o in (me.CHANGED, me.UNCHANGED)); chg = sum(1 for _, o in per if o == me.CHANGED)
    return {"eps_answer": a, "eps_url_list": u, "eps_content": c, "search_saved_pct": round(m["search_saved_pct"], 4), "l1_hits": m["l1_hits"], "l2_hits": m["l2_hits"],
            "coverage_pct": round(100*det/len(per), 4), "drift_pct": round(100*chg/det, 4) if det else None}
pub = run_val(*base_eps); cells = [run_val(a, u, c) for a, u, c in itertools.product(sp["eps_answer"], sp["eps_url_list"], sp["eps_content"])]
C1, C2, C3 = pub["drift_pct"] + 0.25, pub["l1_hits"], 0.95*pub["coverage_pct"]
feas = [v for v in cells if v["drift_pct"] is not None and v["drift_pct"] <= C1 and v["l1_hits"] <= C2 and v["coverage_pct"] >= C3]
sel = sorted(feas, key=lambda v: (-v["search_saved_pct"], v["drift_pct"], v["l1_hits"], v["eps_answer"], v["eps_url_list"], v["eps_content"]))[0]
seps = (sel["eps_answer"], sel["eps_url_list"], sel["eps_content"])
print("validation recorded", pub, "\nfeasible", len(feas), "selected", seps, sel, flush=True)
CONFIGS = {"A_original": (base_m, base_eps, "FreshCache"), "B_mult_only": ({**base_m, "content": MHAT}, base_eps, "FreshCache"),
           "C_budget_only": (base_m, seps, "FreshCache"), "D_full": ({**base_m, "content": MHAT}, seps, "FreshCache"),
           "E_equivalent_ttl": ({**base_m, "content": MHAT}, seps, "EquivalentTTL")}
out, perc = {}, {}
try:
    for name, (mult, eps, var) in CONFIGS.items():
        exp.TIER_MULT.update(mult); exp.EPS_ANSWER, exp.EPS_URL_LIST, exp.EPS_CONTENT = eps
        m, per = me.replay(TEST, E["rounds"], var, rich=rich)
        det = [(q, o) for q, o in per if o in (me.CHANGED, me.UNCHANGED)]; chg = sum(1 for _, o in det if o == me.CHANGED)
        byc = defaultdict(list)
        for q, o in det: byc[cid[q]].append(o)
        ks = list(byc); rng = random.Random(42); ds = []
        for _ in range(10_000):
            fl = [x for _ in range(len(ks)) for x in byc[ks[rng.randrange(len(ks))]]]; ds.append(100*sum(1 for x in fl if x == me.CHANGED)/len(fl))
        ds.sort()
        out[name] = {"eps": list(eps), "tier_mult": dict(exp.TIER_MULT), "search_saved_pct": round(m["search_saved_pct"], 4), "l1_hits": m["l1_hits"], "l2_hits": m["l2_hits"], "l3_hits": m["l3_hits"],
                     "observable_n": len(det), "coverage_pct": round(100*len(det)/len(per), 4), "drift_pct": round(100*chg/len(det), 4), "drift_ci95": [round(ds[250], 4), round(ds[9750], 4)]}
        perc[name] = dict(per); print(name, out[name], flush=True)
        exp.TIER_MULT.clear(); exp.TIER_MULT.update(base_m); exp.EPS_ANSWER, exp.EPS_URL_LIST, exp.EPS_CONTENT = base_eps
finally:
    exp.TIER_MULT.clear(); exp.TIER_MULT.update(base_m); exp.EPS_ANSWER, exp.EPS_URL_LIST, exp.EPS_CONTENT = base_eps
def mcn(b, c):
    n = b + c; return 1.0 if not n else min(1.0, float(sum(Fraction(math.comb(n, i), 2**n) for i in range(min(b, c)+1))*2))
pc = {}
for y in list(CONFIGS)[1:]:
    ids = [q for q in perc["A_original"] if perc["A_original"][q] in (me.CHANGED, me.UNCHANGED) and perc[y].get(q) in (me.CHANGED, me.UNCHANGED)]
    cx = sum(perc["A_original"][q] == me.CHANGED for q in ids); cy = sum(perc[y][q] == me.CHANGED for q in ids)
    b01 = sum(perc["A_original"][q] == me.UNCHANGED and perc[y][q] == me.CHANGED for q in ids); b10 = sum(perc["A_original"][q] == me.CHANGED and perc[y][q] == me.UNCHANGED for q in ids)
    byc = defaultdict(list)
    for q in ids: byc[cid[q]].append(q)
    ks = list(byc); rng = random.Random(42); ds = []
    for _ in range(10_000):
        fl = [x for _ in range(len(ks)) for x in byc[ks[rng.randrange(len(ks))]]]
        ds.append(100*sum(perc[y][x] == me.CHANGED for x in fl)/len(fl) - 100*sum(perc["A_original"][x] == me.CHANGED for x in fl)/len(fl))
    ds.sort(); pc[f"A_vs_{y}"] = {"n": len(ids), "drift_A_pct": round(100*cx/len(ids), 4), "drift_other_pct": round(100*cy/len(ids), 4), "delta_pp": round(100*(cy-cx)/len(ids), 4),
                                   "ci95": [round(ds[250], 4), round(ds[9750], 4)], "A_clean_other_drift": b01, "A_drift_other_clean": b10, "mcnemar_p": mcn(b01, b10)}
    print(y, pc[f"A_vs_{y}"], flush=True)
json.dump({"engine": ENG, "validation_recorded": pub, "constraints": {"C1_drift_max": C1, "C2_l1_max": C2, "C3_coverage_min": C3}, "cells": cells, "n_feasible": len(feas),
           "selected": sel, "heldout": out, "paired_contrasts": pc}, open(C.W/"results"/f"budget_{ENG}.json", "w"), indent=1)
