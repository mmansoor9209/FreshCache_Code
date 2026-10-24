"""Held-out replays of the list-level controls inside composition D on the common sample.
Gates: composition_engine_admission_hooks with no hooks reproduces the D trace (and B) request for request; D+gamma (floor 0.75, gamma 0.55, frozen support
scores) reproduces the H55 operation counts. D+gate: frozen learned L2 compatibility gate (frozen, Youden tau) on L2 candidates at floor
0.75, features from serving-time information (the learned-gate feature code, read-only), inspection time measured.
Writes work/list_control_traces.pkl, outputs/list_controls_replay.json."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import json, pickle, sys, time, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
O = ((OUTPUT_ROOT / "natural_workload")); CS_DIR = ((OUTPUT_ROOT / "common_sample")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/composition_study_support")); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/live_streams")); sys.path.insert(0, str(REPO_ROOT / "freshcache/composition"))
import output_workspace; output_workspace.use("natural_workload"); import composition_study_helpers as X, composition_engine_admission_hooks as RV   # noqa: E402
import numpy as np   # noqa: E402
E = X.engine(); TH, KA = X.ca_params(0.0, 1.0); H = X.C.streams(E)["heldout"]; n = len(H); CS, CF, CG, CE = [(c["c_search_ms"]["live_p50"], c["c_fetch_ms"]["live_p50"], c["c_gen_ms"]["value"], c["c_embed_ms"]["value"]) for c in [json.load(open(OUTPUT_ROOT / "concurrency_cost/component_costs.json"))["coefficients"]]][0]   # component_costs.json (serving_cost_summary.py)
sig = lambda r: (r["tier"], tuple(tuple(x) for x in r["ev"]), r.get("gen_query"))
TRACES = pickle.load(open(CS_DIR / "work/traces.pkl", "rb"))
kw = dict(theta_by_class=TH, kappa_by_class=KA)
def run(**k):
    s = RV.State(E, n, **kw, **k); rows = {}
    for t, r in H: rows[r["query_id"]] = s.step(t, r)
    return s.metrics(), rows
G = {}
mD, rD = run(floor=0.80); G["D_equals_reference_trace"] = all(sig(rD[q]) == sig(TRACES["D"][q]) for q in rD)
# gamma verifier: frozen support scores (precomputed BGE-M3 query/page embeddings) as in the CategoryAware-style + FreshCache composition
import page_version_replay as ma   # noqa: E402
SUP = RV.__dict__.get("Support") or __import__("composition_engine").Support
sup = SUP(X.OV, ma); gms = []
def gamma_check(q, ev, meta):
    t0 = time.perf_counter(); v = sup.score(q, ev); gms.append((time.perf_counter() - t0) * 1e3); return v is None or v >= 0.55   # composition: unscored candidates are accepted
mG, rG = run(floor=0.75, l2_check=gamma_check)
import csv; H55 = {k: float(v) for k, v in next(r for r in csv.DictReader(open(OUTPUT_ROOT / "categoryaware_freshcache/ca_composition_heldout_results.csv")) if r["method"] == "H55").items() if k != "method"}   # categoryaware_composition.py
G["Dgamma_equals_composition_H55_counts"] = mG["l2_hits"] == H55["l2_hits"] and round(1000 * mG["search_calls"] / n, 1) == H55["search_per_1k"] and round(1000 * mG["fetches"] / n, 1) == H55["fetch_per_1k"]
# learned gate (frozen), learned-gate feature code read-only
GATE = (OUTPUT_ROOT / "learned_l2_gate"); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/admission_controls")); import l2_gate_features as Lb   # noqa: E402
GZ = pickle.load(open(GATE / "gate_frozen.pkl", "rb")); FEATS, TAU, MODEL = GZ["features"], GZ["tau"], GZ["model"]   # frozen gate (Youden tau) of learned_l2_gate
gtm = []
def gate_check(q, ev, meta):
    t0 = time.perf_counter(); f = Lb.features(q, [tuple(x) for x in ev], meta["eq"], meta["sim"], meta["fc"], meta["age"], meta["efc"])
    p = float(MODEL.predict_proba(np.array([[f[k] for k in FEATS]], float))[:, 1][0]); gtm.append((time.perf_counter() - t0) * 1e3); return p >= TAU
mT, rT = run(floor=0.75, l2_check=gate_check)
assert all(G.values()), G
nonrt = sum(1 for _, r in H if r["freshness_class"] != "REAL_TIME") / n
def cost(m, insp_ms): return round((m["search_calls"] * CS + m["fetches"] * CF + m["generations"] * CG) / n + nonrt * CE + insp_ms, 1)
R = {"gate": G, "n": n, "policies": {}}
for nm, m, ms in (("D", mD, []), ("Dgamma", mG, gms), ("Dgate", mT, gtm)):
    insp = sum(ms) / n
    R["policies"][nm] = dict(m, search_per_1k=round(1000 * m["search_calls"] / n, 1), fetch_per_1k=round(1000 * m["fetches"] / n, 1), gen_per_1k=round(1000 * m["generations"] / n, 1),
                             inspection_ms_per_request=round(insp, 2), inspection_ms_per_check=round(sum(ms) / len(ms), 2) if ms else 0, cost_ms=cost(m, insp))
pickle.dump({"Dgamma": rG, "Dgate": rT}, open(O / "work/list_control_traces.pkl", "wb")); json.dump(R, open(O / "outputs/list_controls_replay.json", "w"), indent=1); print(json.dumps(R, indent=1))
