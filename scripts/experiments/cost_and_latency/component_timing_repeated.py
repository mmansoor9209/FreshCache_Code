"""Dedicated CPU timing (nothing else of this experiment running): 3 sequential passes of the gated replay; pass 1 = cold caches,
passes 2-3 = warm. A component is STABLE if its warm-pass p50 and mean agree within 20%; otherwise reported as MISSING (not stable)."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import json, statistics, sys, time, pathlib, os; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.dont_write_bytecode = True
W = ((OUTPUT_ROOT / "systems_timing")); sys.path.insert(0, str(REPO_ROOT / "freshcache/replay_engine")); sys.path.insert(0, str(REPO_ROOT / "freshcache/simulation"))
import engine_loader as C, reference_replay as R
E = C.load("reported"); P = C.streams(E)["primary"]; out = {"load_avg_before": os.getloadavg()}
def summ(v): v = sorted(v); n = len(v); return {"n": n, "mean_ms": statistics.mean(v), "p50_ms": v[n // 2], "p95_ms": v[int(.95 * n)]}
passes, tiers0 = [], None
for i in range(3):
    t0 = time.time(); m, tiers, T, _ = R.Replay(E, timers=True).run(P); passes.append({k: summ(v) for k, v in T.items() if k != "l3_pages_per_request"}); passes[-1]["wall_s"] = time.time() - t0
    tiers0 = tiers0 or tiers; assert tiers == tiers0
    print(i + 1, {k: round(v["p50_ms"], 4) for k, v in passes[-1].items() if k != "wall_s"}, round(passes[-1]["wall_s"], 1), os.getloadavg(), flush=True)
st = {}
for k in passes[0]:
    if k == "wall_s": continue
    a, b = passes[1][k], passes[2][k]
    st[k] = {"cold_pass1": passes[0][k], "warm_pass2": a, "warm_pass3": b,
             "stable": abs(a["p50_ms"] - b["p50_ms"]) <= 0.2 * max(a["p50_ms"], b["p50_ms"]) and abs(a["mean_ms"] - b["mean_ms"]) <= 0.2 * max(a["mean_ms"], b["mean_ms"])}
out.update({"components": st, "wall_s": [p["wall_s"] for p in passes], "load_avg_after": os.getloadavg()})
json.dump(out, open(W / "intermediate/cpu_timing_dedicated.json", "w"), indent=1)
print({k: v["stable"] for k, v in st.items()})
