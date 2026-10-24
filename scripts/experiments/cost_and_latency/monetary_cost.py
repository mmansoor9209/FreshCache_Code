"""Dollars per 1,000 requests for A, B, D (and D-prime) from the frozen operation counts, at stated price points.
Prices are illustrative public list-price levels, not quotes: search API $0.30 / $1.00 / $5.00 per 1,000 calls; page fetch (bandwidth +
compute) $0.00 / $0.05 / $0.20 per 1,000 fetches; generation $0.02 (self-hosted 3B) / $0.30 (small API model) / $3.00 (large API model) per
1,000 answers. Index and decision overhead converted at $0.05 per vCPU-hour. Break-even: search price per 1,000 calls above which D is cheaper
than B. Writes outputs/monetary_cost.json."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import json, pathlib; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
O = ((OUTPUT_ROOT / "write_controls"))
T = json.load(open(((OUTPUT_ROOT / "composition/outputs/composition_cost.json"))))["policies"]; DP = json.load(open(O / "outputs/dprime_replay.json"))["heldout"]["Dprime"]
ops = {p: {"s": 1000 * T[p]["n_s"], "f": 1000 * T[p]["n_f"], "g": 1000 * T[p]["n_g"]} for p in "ABD"}
ops["Dprime"] = {"s": DP["search_per_1k"], "f": DP["fetch_per_1k"], "g": DP["gen_per_1k"]}
SEARCH, FETCH, GEN = {"low": 0.30, "mid": 1.00, "high": 5.00}, {"low": 0.0, "mid": 0.05, "high": 0.20}, {"low": 0.02, "mid": 0.30, "high": 3.00}
vcpu_h = 0.05   # price parameter ($ per vCPU-hour)
_DEC = json.load(open(OUTPUT_ROOT / "composition/outputs/composition_cost.json"))["B_vs_D"]["d_overhead_ms_conservative"]   # D-B decision path (composition_cost_model.py)
_ANN = json.load(open(OUTPUT_ROOT / "composition_checks/outputs/vector_index_timing.json"))   # vector_index_timing.py
idx_ms = {"exact": round(_DEC + _ANN["flat"]["D_minus_B_ms_per_request"], 3), "hnsw": round(_DEC + _ANN["hnsw"]["D_minus_B_ms_per_request"], 3)}   # D-B decision path + index, ms per request
over = {k: 1000 * v / 3.6e6 * vcpu_h for k, v in idx_ms.items()}   # $ per 1,000 requests
R = {"ops_per_1k": {p: {k: round(v, 1) for k, v in o.items()} for p, o in ops.items()}, "prices": {"search_per_1k_calls": SEARCH, "fetch_per_1k": FETCH, "generation_per_1k_answers": GEN},
     "index_overhead_usd_per_1k_requests": {k: f"{v:.2e}" for k, v in over.items()}, "scenarios": {}}
for lvl in ("low", "mid", "high"):
    c = {p: round((o["s"] * SEARCH[lvl] + o["f"] * FETCH[lvl] + o["g"] * GEN[lvl]) / 1000, 4) for p, o in ops.items()}
    c["D_saving_vs_B_pct"] = round(100 * (1 - c["D"] / c["B"]), 1); c["D_saving_vs_A_pct"] = round(100 * (1 - c["D"] / c["A"]), 1)
    R["scenarios"][lvl] = c
dS = (ops["B"]["s"] - ops["D"]["s"]) / 1000; dF = (ops["B"]["f"] - ops["D"]["f"]) / 1000
R["break_even_search_price_per_1k_calls"] = {f"fetch_{fl}_{ix}": round(max(0.0, (over[ix] / 1000 - dF * FETCH[fl] / 1000) / dS * 1000), 6) for fl in FETCH for ix in over}
json.dump(R, open(O / "outputs/monetary_cost.json", "w"), indent=1); print(json.dumps(R, indent=1))
