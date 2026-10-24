"""Fresh-process decision-path timing (new process, engine load, one cold timed pass, two warm passes) of policies A/B/C/D on
the held-out stream. Decision-only: answer-cache, L2 and L3 lookups and in-memory cache updates (no network, no generation).
Usage: decision_path_timing.py A|B|C|D [parallel]   ("parallel": the run in which the four policies are timed concurrently, written to
work/time_parallel_<policy>.json and used by composition_cost_model.py as the conservative alternative)"""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import json, sys, time, pathlib
sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/main")); sys.path.insert(0, str(REPO_ROOT / "freshcache/composition"))
import composition_helpers as X, composition_engine as RE
POL = {"A": dict(l2l3=False), "B": dict(l2_on=False), "C": dict(l3_on=False), "D": dict()}
p = sys.argv[1]; TAG = "parallel_" if sys.argv[2:3] == ["parallel"] else ""; E = X.engine(); H = X.C.streams(E)["heldout"]; TH, KA = X.ca_params(0.0, 1.0); v = []
for _ in range(3):
    t0 = time.perf_counter(); RE.replay(E, H, l1="ca", floor=0.80, theta_by_class=TH, kappa_by_class=KA, **POL[p]); v.append((time.perf_counter() - t0) * 1e3)
json.dump({"policy": p, "n": len(H), "total_ms": v, "cold_ms_per_request": round(v[0] / len(H), 4), "warm_ms_per_request": round((v[1] + v[2]) / 2 / len(H), 4)}, open(X.O / f"work/time_{TAG}{p}.json", "w"), indent=1)
print(p, round(v[0] / len(H), 4), round((v[1] + v[2]) / 2 / len(H), 4))
