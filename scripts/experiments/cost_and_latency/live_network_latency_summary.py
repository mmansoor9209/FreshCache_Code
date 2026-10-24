"""Multi-window live-network latency summary for compositions A, B and D (the additional windows from live_network_latency_windows.py plus
the fixed window of live_network_latency.py). Writes outputs/composition_summary/outputs/live_latency_summary.json (read by break_even_analysis.py)."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import os; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import json, pathlib, random, statistics
O6 = ((OUTPUT_ROOT / "composition_summary")); O = ((OUTPUT_ROOT / "composition_checks")); RC = ((OUTPUT_ROOT / "composition/outputs"))
ld = lambda p: json.load(open(p))
# ---- live latency (multi-window)
W = sorted(O.glob("outputs/live_latency_window_[0-9]*.json")); old = ld(RC / "runtime_live.json")
LS = {"windows": [[12000, 12300]] + [ld(f)["window"] for f in W], "per_window_mean_ms": {"12000": {p: old["policies"][p]["mean_ms"] for p in "ABD"}}}
REQ = []
for f in W:
    d = ld(f); REQ += d["requests"]; LS["per_window_mean_ms"][str(d["window"][0])] = {p: round(statistics.mean(r[p]["ms"] for r in d["requests"]), 1) for p in "ABD"}
    LS.setdefault("serper_calls_total_windows", d["serper_calls_total_windows"])
    LS["serper_calls_total_windows"] = max(LS["serper_calls_total_windows"], d["serper_calls_total_windows"])
if REQ:
    miss = [r for r in REQ if r["A"]["tier"] != "L1"]; rng = random.Random(7); B = 10000
    def boot(f, rows):
        v = sorted(f([rows[rng.randrange(len(rows))] for _ in rows]) for _ in range(B)); return [round(v[int(.025 * B)], 1), round(v[int(.975 * B) - 1], 1)]
    LS["new_windows_requests"] = len(REQ); LS["miss_requests"] = len(miss)
    LS["miss_only"] = {p: {"mean_ms": round(statistics.mean(r[p]["ms"] for r in miss), 1), "ci95": boot(lambda rs: statistics.mean(r[p]["ms"] for r in rs), miss),
                           "p50_ms": round(statistics.median(r[p]["ms"] for r in miss), 1)} for p in "ABD"}
    for a, b in (("B", "D"), ("A", "B"), ("A", "D")):
        LS["miss_only"][f"{a}-{b}_ms"] = {"mean": round(statistics.mean(r[a]["ms"] - r[b]["ms"] for r in miss), 1), "ci95": boot(lambda rs: statistics.mean(r[a]["ms"] - r[b]["ms"] for r in rs), miss)}
    LS["all_requests_new_windows"] = {p: {"mean_ms": round(statistics.mean(r[p]["ms"] for r in REQ), 1), "p50_ms": round(statistics.median(r[p]["ms"] for r in REQ), 1)} for p in "ABD"}
    (O6 / "outputs").mkdir(parents=True, exist_ok=True); json.dump(LS, open(O6 / "outputs/live_latency_summary.json", "w"), indent=1)
    print(json.dumps(LS["miss_only"], indent=1))
