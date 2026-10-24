"""Fresh-process cold timing of one hybrid's decision-only replay (new process, engine load, one timed cold pass, two warm passes).
Usage: categoryaware_composition_timing.py VARIANT STREAM   (VARIANT in H75 H80 H55; STREAM in validation heldout)"""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import json, os, pathlib, sys, time
sys.path.insert(0, str(REPO_ROOT / "scripts/experiments/main")); sys.path.insert(0, str(REPO_ROOT / "freshcache/composition"))
import categoryaware_composition_helpers as X, categoryaware_freshcache_engine as HE
V, S = sys.argv[1], sys.argv[2]; E = X.engine(); ma = sys.modules["page_version_replay"]; st = X.C.streams(E)[S]
TH, KA = X.ca_params(0.0, 1.0); SUP = HE.Support(X.OV, ma) if V == "H55" else None
CFG = {"H75": (0.75, None), "H80": (0.80, None), "H55": (0.75, 0.55)}[V]
v = []
for _ in range(3):
    t0 = time.perf_counter(); HE.replay(E, st, l1="ca", floor=CFG[0], gamma=CFG[1], support=SUP, theta_by_class=TH, kappa_by_class=KA); v.append((time.perf_counter() - t0) * 1e3)
json.dump({"variant": V, "stream": S, "n": len(st), "total_ms": v, "cold_ms_per_request": round(v[0] / len(st), 4), "warm_ms_per_request": round((v[1] + v[2]) / 2 / len(st), 4),
           "protocol": "fresh process; engine load; pass 1 cold (used); passes 2-3 warm; decision-only (no drift scoring)"}, open(X.O / f"work/time_{V}_{S}.json", "w"), indent=1)
print(V, S, round(v[0] / len(st), 4))
