"""Replay the four composition policies A-D (frozen engine copy composition_engine.py, each policy with its own cache state from empty caches;
identical switches to the main composition run) on
  (a) the held-out stream, primary zipf_uniform schedule (reference; must reproduce the main composition table),
  (b) the validation stream (zipf_uniform) -- note: the CategoryAware answer cache and theta_L2 were tuned on it,
  (c) the held-out stream under the uniform and bursty_poisson arrival schedules (seed 42, request_schedules.py, read-only),
  (d) the held-out stream beneath the stricter CategoryAware cost-matched point (+0.075 thresholds, TTL x1.0).
Modeled cost per request = n_s*c_s + n_f*c_f + n_g*c_g + c_embed per non-REAL_TIME request (live-measured coefficients, component_costs.json);
the sub-ms decision path is omitted (reported separately). Operation counts only; no answers are generated. Writes outputs/composition_operation_replications.json."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import json, sys, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
O = (OUTPUT_ROOT / "composition_summary"); sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/composition_study_support")); sys.path.insert(0, str(REPO_ROOT / "freshcache/composition"))
import output_workspace; output_workspace.use("composition_summary"); import composition_study_helpers as X, composition_engine as RE   # noqa: E402
E = X.engine()
POL = {"A": dict(l2l3=False), "B": dict(l2_on=False), "C": dict(l3_on=False), "D": dict()}
CS, CF, CG, CE = [(c["c_search_ms"]["live_p50"], c["c_fetch_ms"]["live_p50"], c["c_gen_ms"]["value"], c["c_embed_ms"]["value"]) for c in [json.load(open(OUTPUT_ROOT / "concurrency_cost/component_costs.json"))["coefficients"]]][0]   # component_costs.json (serving_cost_summary.py)
def run(stream, dtheta=0.0):
    TH, KA = X.ca_params(dtheta, 1.0); out = {}
    nonrt = sum(1 for _, r in stream if r.get("freshness_class") != "REAL_TIME") / len(stream)
    for p, kw in POL.items():
        m, _ = RE.replay(E, stream, l1="ca", floor=0.80, theta_by_class=TH, kappa_by_class=KA, **kw); n = m["n_requests"]
        cost = (m["search_calls"] * CS + m["fetches"] * CF + m["generations"] * CG) / n + nonrt * CE
        out[p] = {"n": n, "saved_pct": round(m["search_saved_pct"], 2), "search_per_1k": round(1000 * m["search_calls"] / n, 1), "fetch_per_1k": round(1000 * m["fetches"] / n, 1),
                  "gen_per_1k": round(1000 * m["generations"] / n, 1), "l1_hits": m["l1_hits"], "l2_hits": m["l2_hits"], "cost_ms": round(cost, 1)}
    out["increment_BD_pp"] = round(out["D"]["saved_pct"] - out["B"]["saved_pct"], 2)
    out["cost_cut_AB_pct"] = round(100 * (1 - out["B"]["cost_ms"] / out["A"]["cost_ms"]), 1); out["cost_cut_BD_pct"] = round(100 * (1 - out["D"]["cost_ms"] / out["B"]["cost_ms"]), 1)
    out["cost_cut_AD_pct"] = round(100 * (1 - out["D"]["cost_ms"] / out["A"]["cost_ms"]), 1)
    return out
R = {}
S0 = X.C.streams(E)
R["heldout_zipf_uniform"] = run(S0["heldout"]); print("heldout", R["heldout_zipf_uniform"]["increment_BD_pp"], flush=True)
import pickle; FZ = pickle.load(open(X.RCD / "work/composition_ablation_replay.pkl", "rb"))["policies"]   # main composition table (composition_ablation_replay.py)
assert all(R["heldout_zipf_uniform"][p]["saved_pct"] == round(FZ[p]["metrics"]["search_saved_pct"], 2) for p in ("B", "D")), "reference must reproduce the main composition table"
R["validation_zipf_uniform"] = run(S0["validation"]); print("validation", R["validation_zipf_uniform"]["increment_BD_pp"], flush=True)
for sch in ("uniform", "bursty_poisson"):
    R[f"heldout_{sch}"] = run(X.C.streams(E, schedule=sch)["heldout"]); print(sch, R[f"heldout_{sch}"]["increment_BD_pp"], flush=True)
R["heldout_strict_CA_+0.075"] = run(S0["heldout"], 0.075); print("strict CA", R["heldout_strict_CA_+0.075"]["increment_BD_pp"], flush=True)
json.dump(R, open(O / "outputs/composition_operation_replications.json", "w"), indent=1)
for k, v in R.items(): print(k, {p: (v[p]["saved_pct"], v[p]["fetch_per_1k"], v[p]["cost_ms"]) for p in "ABCD"}, "inc", v["increment_BD_pp"], "cuts", v["cost_cut_AB_pct"], v["cost_cut_BD_pct"])
