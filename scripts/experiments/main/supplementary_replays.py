"""Additional comparisons on one engine, following the original definitions of each measurement:
 - L3Only (mixed_age_trace_construction.py), predicted-class arm (drift on S_fixed),
 - FreshCache-L1Only held-out (exact_ttl_l1_only_policies.register), SemanticTTL held-out points (heldout_split_evaluation.py module patches),
 - schedule table: gate-off minus gate-on on the common-determinate set, paired cluster bootstrap B=500 default_rng(42),
 - pairwise common-determinate drift over all requests (mixed_age_consistency_analysis.py part_d),
 - held-out matched drift vs SemanticTTL points, paired cluster bootstrap B=1000 default_rng(42) (as in heldout_split_evaluation.py)."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import sys, json, copy, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve(); import importlib.util as _ilu; _rs = _ilu.spec_from_file_location("reference_values", os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "scripts/analysis/reference_values.py")); REF = _ilu.module_from_spec(_rs); _rs.loader.exec_module(REF)
import numpy as np
sys.path.insert(0, str(REPO_ROOT / "freshcache/replay_engine")); import engine_loader as C
ENG = sys.argv[1]; E = C.load(ENG); me, exp, ea, sc = E["me"], E["exp"], E["ea"], E["sc"]
me.VARIANTS.setdefault("L3Only", ("L3Only", "risk"))
ea.CONFIG.setdefault("FreshCache_L1Only", dict(l1="eqgate_risk", l2="none", l3="none", l2_reg_on_hit=False, ttl=None, online=False)); me.VARIANTS.setdefault("FreshCache_L1Only", ("FreshCache_L1Only", "risk"))
OUT = {}
def codes(stream, per): return np.array([0 if per.get(r["query_id"]) is None else (1 if per[r["query_id"]] == me.CHANGED else (2 if per[r["query_id"]] == me.UNCHANGED else 0)) for _, r in stream], np.int8)
def searches(stream, tr): return np.array([1 if tr[r["query_id"]]["tier"] in ("miss", "REAL_TIME_BYPASS") else 0 for _, r in stream], np.int64)
def run(stream, v, S):
    row, per, tr = C.metrics(E, stream, v, S); return row, codes(stream, per), searches(stream, tr)
def paired_boot(stream, a, b, both, B, seed):
    cl = sorted({(r.get("cluster_id") or r["query_id"]) for _, r in stream}); ci = {c: i for i, c in enumerate(cl)}
    rq = np.array([ci[(r.get("cluster_id") or r["query_id"])] for _, r in stream]); pc = lambda v: np.bincount(rq, weights=v, minlength=len(cl))
    rng = np.random.default_rng(seed); cnt = np.zeros((B, len(cl)), np.int64)
    for i in range(B): cnt[i] = np.bincount(rng.integers(0, len(cl), len(cl)), minlength=len(cl))
    (as_, ao), (bs, bo) = a, b; N = cnt @ pc(np.ones(len(stream)))
    sav = 100*(1 - (cnt @ pc(bs.astype(float)))/N) - 100*(1 - (cnt @ pc(as_.astype(float)))/N)
    f = lambda o, k: pc((((o == 1) | (o == 2)) & both).astype(float)) if k == "d" else pc(((o == 1) & both).astype(float))
    with np.errstate(invalid="ignore", divide="ignore"): dd = 100*(cnt @ f(bo, "c"))/(cnt @ f(bo, "d")) - 100*(cnt @ f(ao, "c"))/(cnt @ f(ao, "d"))
    dd = dd[np.isfinite(dd)]
    return {"drift_ci95": [round(float(np.percentile(dd, 2.5)), 2), round(float(np.percentile(dd, 97.5)), 2)], "saved_ci95": [round(float(np.percentile(sav, 2.5)), 2), round(float(np.percentile(sav, 97.5)), 2)]}
def cd(a_o, b_o, mask):
    both = mask & (a_o > 0) & (b_o > 0); r = lambda o: 100*((o == 1) & both).sum()/both.sum() if both.sum() else None
    return both, {"shared_n": int(both.sum()), "a_drift_pct": round(float(r(a_o)), 4), "b_drift_pct": round(float(r(b_o)), 4)}
ST = C.streams(E); P = ST["primary"]; SP = E["support_of"](P, E["rounds"])
# L3Only
OUT["primary/L3Only"] = run(P, "L3Only", SP)[0]
# predicted classes (drift on the dataset-class support S_fixed)
preds = [json.loads(l) for l in open((OUTPUT_ROOT / "class_predictor/predicted_classes.jsonl"))]
assert [p["query_id"] for p in preds] == [r["query_id"] for r in E["recs"]]
prs = sc.build_stream([dict(r, freshness_class=p["pred_class"]) for r, p in zip(E["recs"], preds)], "zipf_uniform", 42)
OUT["primary/FreshCache_predicted_class"] = run(prs, "FreshCache", SP)[0]
# Table 9 schedules
OUT["table9"] = {}
for sched in ("zipf_uniform", "uniform", "bursty_poisson"):
    s = sc.build_stream(E["recs"], sched, 42); S = E["support_of"](s, E["rounds"]); ins = np.array([r["query_id"] in S for _, r in s])
    on, ao, asr = run(s, "FreshCache", S); off, bo, bsr = run(s, "FreshCache_AlwaysPassTemporal", S)
    both, c = cd(ao, bo, ins)
    OUT["table9"][sched] = {"delta_drift_pp": round(c["b_drift_pct"] - c["a_drift_pct"], 2), "delta_saved_pp": round(off["search_saved_pct"] - on["search_saved_pct"], 2), **c,
                            **paired_boot(s, (asr, ao), (bsr, bo), both, 500, 42), "gate_on": on, "gate_off": off}
    print("table9", sched, {k: OUT["table9"][sched][k] for k in ("delta_drift_pp", "drift_ci95", "delta_saved_pp", "shared_n")}, flush=True)
# pairwise comparison on the primary stream (all requests, no S restriction)
fc_row, fc_o, _ = run(P, "FreshCache", SP); allm = np.ones(len(P), bool); OUT["pairwise_5_2"] = {}
for b in ("L3Only", "SCALM", "SemanticTTL", "TieredFixedTTL"):
    _, bo, _ = run(P, b, SP); OUT["pairwise_5_2"][b] = cd(fc_o, bo, allm)[1]
    print("pairwise", b, OUT["pairwise_5_2"][b], flush=True)
# Held-out: L1Only, SemanticTTL operating points, matched drift with bootstrap B=1000
H = ST["heldout"]; SH = E["support_of"](H, E["rounds"]); inh = np.array([r["query_id"] in SH for _, r in H])
OUT["heldout/FreshCache_L1Only"] = run(H, "FreshCache_L1Only", SH)[0]
fcr, fco, fcs = run(H, "FreshCache", SH); OUT["heldout/FreshCache"] = fcr
ORIG = {"FIXED_TTL": copy.deepcopy(exp.FIXED_TTL), "L1T": exp.L1_SIM_THRESHOLD}; OUT["heldout_semanticttl"] = {}
for th, ka, name in ((0.60, 0.5, "quality_selected_0.60_1/2"), (0.40, 0.0625, "drift_selected_0.40_1/16")):
    exp.L1_SIM_THRESHOLD = th; exp.FIXED_TTL.clear(); exp.FIXED_TTL.update({c: v*ka for c, v in ORIG["FIXED_TTL"].items()})
    try: r, bo, bs = run(H, "SemanticTTL", SH)
    finally: exp.FIXED_TTL.clear(); exp.FIXED_TTL.update(ORIG["FIXED_TTL"]); exp.L1_SIM_THRESHOLD = ORIG["L1T"]
    both, c = cd(fco, bo, inh)
    OUT["heldout_semanticttl"][name] = {"replay": r, "matched_vs_freshcache": {**c, **paired_boot(H, (fcs, fco), (bs, bo), both, 1000, 42)}}
    print("heldout sttl", name, r["search_saved_pct"], c, flush=True)
json.dump(OUT, open(C.W/"results"/f"extras_{ENG}.json", "w"), indent=1, default=str)
if ENG == "reported": REF.record("reported_engine/heldout/SemanticTTL_0.60_k0.5/search_saved_pct", OUT["heldout_semanticttl"]["quality_selected_0.60_1/2"]["replay"]["search_saved_pct"])
print("DONE", flush=True)
