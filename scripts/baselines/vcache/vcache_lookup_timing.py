"""vCache-style baseline: our reimplementation of vCache's per-entry learned-threshold policy (Schroeder et al., ICLR 2026), not
the complete published system; it uses proxy correctness feedback and does not test the published error guarantee.

Local cache-overhead timing for vCache with the protocol of scripts/experiments/cost_and_latency/cold_lookup_timing.py: fresh process, engine load, ONE timed
pass over the frozen held-out stream (cold, USED), then two warm passes. That protocol stubs the drift scorer `mixed_age_replay.outcome`;
vCache cannot be stubbed wholesale because the same function supplies its explore-branch feedback (part of the policy). So every outcome() call is
timed and the time spent scoring EXPLOIT hits (pure drift bookkeeping, no decision effect) is subtracted; feedback-label time stays in the overhead."""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import collections, json, os, pathlib, sys, time; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
sys.dont_write_bytecode = True
HERE = ((OUTPUT_ROOT / "baselines/vcache")); sys.path.insert(0, str(REPO_ROOT / "scripts/baselines/vcache"))
import vcache_helpers as X
E = X.engine(); me, RT = E["me"], E["rounds"]; H = X.C.streams(E)["heldout"]; n = len(H)
sys.path.insert(1, str(REPO_ROOT / "scripts/baselines/vcache")); sys.path.insert(0, str(REPO_ROOT / "scripts/baselines/exact_ttl_and_l1_only")); sys.path.insert(0, str(REPO_ROOT / "scripts/baselines/vcache"))
import vcache_reference_replay as RF
assert "scripts/baselines/vcache" in RF.__file__ and RF.me is me
_o = me.outcome; passes = []
for k in range(3):
    acc = collections.defaultdict(float)
    def timed(uh, tc, tr, rounds, _f=_o):
        t0 = time.perf_counter(); r = _f(uh, tc, tr, rounds); acc[tr] += time.perf_counter() - t0; return r
    me.outcome = timed
    try: t0 = time.perf_counter(); m, o, ev = RF.reference_replay(H, rounds=RT, delta=0.02); tot = time.perf_counter() - t0
    finally: me.outcome = _o
    hit_t = {t for (t, r), e in zip(H, ev) if e["action"] == "hit"}
    hit_scoring = sum(v for tr, v in acc.items() if tr in hit_t); all_outcome = sum(acc.values())
    passes.append({"total_ms": tot * 1e3, "outcome_ms_all": all_outcome * 1e3, "outcome_ms_hit_scoring": hit_scoring * 1e3})
assert len({t for t, r in H}) == n, "request timestamps not unique; hit-scoring attribution would be ambiguous"
c0 = passes[0]
out = {"method": "vCache_delta0.02", "n": n, "passes": passes, "lookup_ms_per_request_used": round((c0["total_ms"] - c0["outcome_ms_hit_scoring"]) / n, 4),
       "cold_ms_per_request_including_hit_scoring": round(c0["total_ms"] / n, 4), "warm_ms_per_request_mean": round(sum((p["total_ms"] - p["outcome_ms_hit_scoring"]) for p in passes[1:]) / 2 / n, 4),
       "ops_during_timed_pass": {"search_calls": m["search_calls"], "fetches": m["fetches"], "generations": m["generations"], "hits": m["hits"]}, "loadavg_at_end": os.getloadavg(),
       "protocol": "cost-model cold-process protocol; drift scoring of exploit hits subtracted (cost-model stubs it); explore feedback labels kept (policy)"}
json.dump(out, open(X.O / "work/local_cold_vCache.json", "w"), indent=1); print(json.dumps({k: v for k, v in out.items() if k != "passes"}))
