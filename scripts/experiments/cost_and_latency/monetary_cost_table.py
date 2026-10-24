"""Recompute the monetary-cost table from its declared coefficients (outputs/write_controls/outputs/monetary_cost.json written by monetary_cost.py, sha 04bd02b6).
Per scenario: dollars per 1,000 requests for A, B, D, D'; D's fetch saving vs. B; whether it covers the index overhead; and the
break-even search price (search-only when the fetch price is zero). Writes outputs/monetary_cost_table.json."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import json, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
O = ((OUTPUT_ROOT / "cross_workload")); M = json.load(open((OUTPUT_ROOT / "write_controls/outputs/monetary_cost.json")))
ops, pr, ov = M["ops_per_1k"], M["prices"], {k: float(v) for k, v in M["index_overhead_usd_per_1k_requests"].items()}
R = {"source": "monetary_cost.json", "index_overhead_usd_per_1k_requests": ov, "scenarios": {}}
ds = ops["B"]["s"] - ops["D"]["s"]; df = ops["B"]["f"] - ops["D"]["f"]
for sc in ("low", "mid", "high"):
    ps, pf, pg = pr["search_per_1k_calls"][sc], pr["fetch_per_1k"][sc], pr["generation_per_1k_answers"][sc]
    usd = {p: (o["s"] * ps + o["f"] * pf + o["g"] * pg) / 1000 for p, o in ops.items()}
    fetch_saving = df * pf / 1000
    be = {k: max(0.0, (v - fetch_saving) * 1000 / ds) for k, v in ov.items()}
    R["scenarios"][sc] = {"usd_per_1k_requests": {p: round(v, 4) for p, v in usd.items()}, "D_vs_B_pct": round(100 * (usd["D"] / usd["B"] - 1), 2),
                          "fetch_price_per_1k": pf, "D_fetch_saving_usd_per_1k_requests": fetch_saving,
                          "fetch_saving_covers_overhead": {k: fetch_saving >= v for k, v in ov.items()},
                          "break_even_search_price_per_1k_calls": be, "break_even_kind": "search-only (fetch price zero)" if pf == 0 else "fetch saving covers overhead; any nonnegative search price" if all(fetch_saving >= v for v in ov.values()) else "search and fetch"}
R["searches_avoided_per_1k_requests"] = round(ds, 1); R["fetches_avoided_per_1k_requests"] = round(df, 1)
json.dump(R, open(O / "outputs/monetary_cost_table.json", "w"), indent=1); print(json.dumps(R, indent=1))
