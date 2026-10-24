"""Measured search latency for the paraphrase live stream from the real search timestamps (interval between consecutive searches; an
upper bound because the interval also includes the online policy step and fetch submission; fetch durations were not logged there), and
the implied measured search cost per policy (modeled fetch and generation unchanged)."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import json, datetime, statistics, pickle, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
O = ((OUTPUT_ROOT / "natural_workload")); CS_DIR = ((OUTPUT_ROOT / "common_sample"))
S = sorted((json.load(open(f)) for f in (CS_DIR / "raw/c/search").glob("*.json")), key=lambda s: s["order"])
ts = [datetime.datetime.fromisoformat(s["searched_utc"]) for s in S]; gaps = [(b - a).total_seconds() for a, b in zip(ts, ts[1:])]
med = statistics.median(gaps); M = pickle.load(open(CS_DIR / "work/c_rows.pkl", "rb"))["metrics"]; n = len(S); out = {}
CS, CF, CG, CE = [(c["c_search_ms"]["live_p50"], c["c_fetch_ms"]["live_p50"], c["c_gen_ms"]["value"], c["c_embed_ms"]["value"]) for c in [json.load(open(OUTPUT_ROOT / "concurrency_cost/component_costs.json"))["coefficients"]]][0]   # serving_cost_summary.py
for cfg, k in (("P", 2), ("S", 5)):
    for p in ("B", "Be", "D", "Dp"):
        m = M[f"{k}|{p}"]; modeled = (m["search_calls"] * CS + m["fetches"] * CF + m["generations"] * CG) / n + CE
        meas = (m["search_calls"] * med * 1000 + m["fetches"] * CF + m["generations"] * CG) / n + CE
        out[f"{cfg}|{p}"] = {"modeled_ms": round(modeled, 1), "with_measured_search_upper_bound_ms": round(meas, 1)}
R = {"searches": n, "median_interval_s": round(med, 3), "mean_interval_s": round(statistics.mean(gaps), 3), "p90_interval_s": round(sorted(gaps)[int(0.9 * len(gaps))], 3), "per_policy": out,
     "D_vs_B_cut_modeled_pct": round(100 * (1 - out["P|D"]["modeled_ms"] / out["P|B"]["modeled_ms"]), 1),
     "D_vs_B_cut_with_measured_search_pct": round(100 * (1 - out["P|D"]["with_measured_search_upper_bound_ms"] / out["P|B"]["with_measured_search_upper_bound_ms"]), 1)}
json.dump(R, open(O / "outputs/paraphrase_stream_search_latency.json", "w"), indent=1); print(json.dumps(R, indent=1))
