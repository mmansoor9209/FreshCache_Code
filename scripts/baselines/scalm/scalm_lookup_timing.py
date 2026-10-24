"""SCALM-style baseline: our reimplementation of SCALM's admission policy (Li et al., IWQoS 2024), not the complete recorded
system; it omits SCALM's hierarchical cache structure and its eviction policy.

Fresh-process cold lookup timing using the protocol of scripts/experiments/cost_and_latency/cold_lookup_timing.py.
New process, engine load, ONE timed decision-only pass over the frozen held-out stream with the drift scorer `outcome` stubbed
(cold, reported), then two warm passes.
Usage: scalm_lookup_timing.py SCALM | FreshCache_control
  SCALM              SCALM_FCadmit_noTTL through the in-memory baseline engine copy (scripts/baselines/matched_admission/matched_admission_engine_patch.py), same policy as the baseline replay
  FreshCache_control frozen FreshCache through the frozen engine: same-session environment control only (Table 3 reports the cold_lookup_timing.py FreshCache value)"""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import json, os, pathlib, sys, time; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.dont_write_bytecode = True
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); O = (OUTPUT_ROOT / "baselines/scalm"); M = sys.argv[1]
sys.path.insert(0, str(ROOT / "freshcache/replay_engine")); sys.path.insert(0, str(ROOT / "scripts/baselines/matched_admission"))
import engine_loader as C   # noqa: E402
E = C.load("reported"); me, ea, RT = E["me"], E["ea"], E["rounds"]; H = C.streams(E)["heldout"]; n = len(H)
if M == "SCALM":
    import matched_admission_engine_patch as PE   # read-only import of the earlier baseline's engine patcher
    eng = PE.build(me); ea.CONFIG.setdefault("SCALM_FCadmit_noTTL", dict(ea.CONFIG["SCALM"], l1="scalm_eqgate")); eng.VARIANTS["SCALM_FCadmit_noTTL"] = ("SCALM_FCadmit_noTTL", "risk")
    eng.NO_TTL_VARIANTS.add("SCALM_FCadmit_noTTL"); VAR = "SCALM_FCadmit_noTTL"
else:
    eng = me; VAR = "FreshCache"
_o = eng.outcome; v = []; res = None
for _ in range(3):
    eng.outcome = lambda uh, tc, tr, rounds: eng.UNOBS
    try: t0 = time.perf_counter(); res = eng.replay(H, RT, VAR); v.append((time.perf_counter() - t0) * 1e3)
    finally: eng.outcome = _o
m = res[0]
out = {"method": M, "variant": VAR, "total_ms": v, "cold_ms_per_request": round(v[0] / n, 4), "warm_ms_per_request_mean": round(sum(v[1:]) / 2 / n, 4), "n": n,
       "ops_during_timed_pass": {k: m[k] for k in ("search_calls", "fetches", "generations", "l1_hits", "l2_hits", "l3_hits") if k in m}, "pid": os.getpid(), "loadavg_at_end": os.getloadavg(),
       "protocol": "cost-model cold_lookup_timing.py: fresh process; engine load; pass 1 timed cold (USED); passes 2-3 warm; outcome stubbed"}
json.dump(out, open(O / f"work/local_cold_{M}.json", "w"), indent=1); print(json.dumps(out))
