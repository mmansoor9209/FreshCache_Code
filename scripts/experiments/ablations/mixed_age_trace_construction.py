#!/usr/bin/env python3
"""
scripts/experiments/ablations/mixed_age_trace_construction.py

Stage 1: run the reported heterogeneous-age replay once per method and cache
everything the downstream analyses need. The replays are the expensive step;
every later part reads these traces instead of re-running them.

Same workload, timestamps, schedule, semantic thresholds, freshness classes,
half-lives, epsilons, cache semantics and baseline configurations as the
recorded runs. The ONLY difference from the recorded mixed-age numbers is the
decision-precision fix already applied to tiered_cache_model.py p_stale(), which no
longer rounds the risk before comparing it against epsilon.

Methods whose temporal test never calls p_stale are expected to be bit-identical
to their recorded runs. That is not assumed: each is compared against the
recorded trace and any mismatch is reported.

  affected by the fix      FreshCache, L2Only, L3Only   (risk <= eps path)
  unaffected               EquivalentTTL (age <= TTL), FreshCache_AlwaysPassTemporal
                           (temporal always passes), SemanticTTL (fixed TTL),
                           TieredFixedTTL (half-life TTL), SCALM (no temporal test)

Per-arm age events for the age-bin analysis are captured by a monkey-patch:
mixed_age_replay.outcome() is wrapped so it records (call-site line, url_hash,
t_cache, t_request, label) and then delegates to the original. Engine behaviour
is unchanged; only observation is added.

Writes traces.pkl under outputs/replay/table2/<RECOMPUTE_ENGINE>/. Nothing else
is touched.

Run from the repo root:
    RECOMPUTE_ENGINE=reported python scripts/experiments/ablations/mixed_age_trace_construction.py
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage

import json; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
import os
import pathlib
import pickle
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")

ENG = os.environ["RECOMPUTE_ENGINE"]   # reported (the comparison copy is not released)
HERE = (OUTPUT_ROOT / "replay/table2") / ENG; HERE.mkdir(parents=True, exist_ok=True)   # outputs only here
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
for p in ("freshcache/method", "scripts/data", "", "scripts/baselines/vcache", "scripts/evaluation", "freshcache/simulation", "freshcache/selection_replay_engine"):
    sys.path.insert(0, str(ROOT / p) if p else str(ROOT))
sys.path.insert(0, str(ROOT / {"reported": "freshcache/replay_engine"}[ENG]))   # engine copy first
os.chdir(ROOT)

import numpy as np                    # noqa: E402
import tiered_cache_model as exp              # noqa: E402
import execution_semantics as ce         # noqa: E402
import method_replay as ea               # noqa: E402
import request_schedules as sc                # noqa: E402
import mixed_age_replay as me             # noqa: E402
from schedule_robustness import support_of  # noqa: E402

SCHEDULE, SEED = "zipf_uniform", 42
CHANGED, UNCHANGED = ce.CHANGED, ce.UNCHANGED

# L3Only is not in mixed_age_replay.VARIANTS; register it the same way the
# recorded L3Only pairwise analysis did, as the L3Only engine config under the
# ordinary risk-based temporal gate.
me.VARIANTS.setdefault("L3Only", ("L3Only", "risk"))
assert all(str(ROOT / {"reported": "freshcache/replay_engine"}[ENG]) in m.__file__ for m in (me, ce, ea)), (me.__file__, ce.__file__)

METHODS = ["FreshCache", "EquivalentTTL", "FreshCache_AlwaysPassTemporal",
           "L2Only", "L3Only", "SCALM", "SemanticTTL", "TieredFixedTTL"]

# Arms for which per-event ages are needed (the temporal-gate ablation).
AGE_EVENT_ARMS = {"FreshCache", "FreshCache_AlwaysPassTemporal"}

# Methods whose temporal predicate does not call p_stale, so the fix cannot
# move them. Verified, not assumed.
EXPECT_UNCHANGED = {"EquivalentTTL", "FreshCache_AlwaysPassTemporal",
                    "SCALM", "SemanticTTL", "TieredFixedTTL"}

_EV = []


def _patched(uh, tc, tr, rounds):
    lab = _patched._orig(uh, tc, tr, rounds)
    _EV.append((sys._getframe(1).f_lineno, uh, tc, tr, lab))
    return lab


def recorded():
    """Recorded per-variant rows for this schedule/seed, read-only."""
    pub = {}
    for fn in ("schedule_robustness_tasks.jsonl", "mixed_age_tasks.jsonl"):
        p = (OUTPUT_ROOT / "robustness_schedules") / fn
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("schedule") == SCHEDULE and r.get("seed") == SEED:
                    key = r.get("variant") or r.get("method")
                    if key:
                        pub.setdefault(key, r)
    return pub


def main():
    queries = exp.load_jsonl(exp.QUERIES_FILE)
    manifest = exp.load_jsonl(exp.MANIFEST_FILE)
    paras = (exp.load_jsonl(exp.PARAPHRASE_FILE)
             if exp.PARAPHRASE_FILE.exists() else [])
    records = exp.build_query_records(queries, manifest, paras)
    exp._QUERY_TO_IDX = {r["query"]: i for i, r in enumerate(records)}
    exp._SIM_MATRIX = np.load(str(exp.SIM_MATRIX_CACHE), mmap_mode="r")
    rich = ea._rich_feats()
    ea.set_cluster_base(records)
    rounds = {}
    with open((DATA_ROOT / "derived/round_table.jsonl"),
              encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            rounds[d["url_hash"]] = d["rounds"]

    stream = sc.build_stream(records, SCHEDULE, SEED)
    S = support_of(stream, rounds)
    t2q = {}
    for t, r in stream:
        assert t not in t2q, "arrival times not unique; cannot key age events"
        t2q[t] = r["query_id"]

    # cluster id per request, for the paired cluster bootstrap
    clusters = [(r.get("cluster_id") or r["query_id"]) for _, r in stream]
    qids = [r["query_id"] for _, r in stream]

    print(f"  stream {len(stream):,} requests   support |S| = {len(S):,}",
          flush=True)

    pub = recorded()
    traces = {"schedule": SCHEDULE, "seed": SEED, "n_requests": len(stream),
              "support": S, "qids": qids, "clusters": clusters,
              "arrival_times": [t for t, _ in stream],
              "methods": {}, "age_events": {}, "recorded": {}}

    cache_dir = HERE / "_method_cache"
    cache_dir.mkdir(exist_ok=True)

    for name in METHODS:
        t0 = time.time()
        # Resume: a completed replay is cached per method, so a failure later
        # in the list does not discard work already done.
        cf = cache_dir / f"{name}.pkl"
        if cf.exists():
            with open(cf, "rb") as fh:
                cached = pickle.load(fh)
            traces["methods"][name] = cached["m"]
            if cached.get("evs") is not None:
                traces["age_events"][name] = cached["evs"]
            if cached.get("pub") is not None:
                traces["recorded"][name] = cached["pub"]
            m = cached["m"]
            ds = (f"{100*m['_drift']:.4f}%" if m["_drift"] is not None else "N/A")
            print(f"  {name:<32} cached: saved {m['search_saved_pct']:.4f}%  "
                  f"drift {ds}  fetch {m['fetches']:,}", flush=True)
            continue
        capture = name in AGE_EVENT_ARMS
        if capture:
            _EV.clear()
            _patched._orig = me.outcome
            me.outcome = _patched
        m, per = me.replay(stream, rounds, name, rich=rich)
        if capture:
            me.outcome = _patched._orig
            evs = [e for e in _EV if t2q[e[3]] in S]
            traces["age_events"][name] = evs

        sig = [o for _, o in per]
        ins = [o for qid, o in per if o is not None and qid in S]
        ch = sum(1 for o in ins if o == CHANGED)
        un = sum(1 for o in ins if o == UNCHANGED)
        obs = ch + un
        m["_outcome_sig"] = sig
        m["_changed"] = ch
        m["_unchanged"] = un
        m["_determinate"] = obs
        m["_drift"] = (ch / obs) if obs else None
        m["_coverage"] = (obs / len(S)) if S else None
        m["_validation_errors"] = me.assert_run(m, stream)
        traces["methods"][name] = m

        # comparison against the recorded run, read-only
        p = pub.get(name)
        cmpr = None
        if p:
            fields = ("search_saved_pct", "search_calls", "fetches",
                      "l1_hits", "l2_hits", "l3_hits", "generations")
            cmpr = {f: {"recorded": p.get(f), "reported": m.get(f),
                        "same": p.get(f) == m.get(f)}
                    for f in fields if f in p}
            cmpr["changed_in_support"] = {
                "recorded": p.get("changed_in_support"), "reported": ch,
                "same": p.get("changed_in_support") == ch}
            cmpr["observable_hits_in_support"] = {
                "recorded": p.get("observable_hits_in_support"),
                "reported": obs,
                "same": p.get("observable_hits_in_support") == obs}
            traces["recorded"][name] = {
                k: p.get(k) for k in
                ("search_saved_pct", "drift_common_support",
                 "common_support_coverage", "search_calls", "fetches",
                 "generations", "l1_hits", "l2_hits", "l3_hits",
                 "conditional_gets", "observable_hits_in_support",
                 "changed_in_support", "support_size") if k in p}
        m["_vs_recorded"] = cmpr

        moved = (cmpr is not None
                 and not all(v["same"] for v in cmpr.values() if v["recorded"] is not None))
        tag = ""
        if cmpr is not None:
            if name in EXPECT_UNCHANGED:
                tag = "  UNCHANGED as expected" if not moved else \
                      "  *** MOVED BUT WAS EXPECTED UNCHANGED ***"
            else:
                tag = "  moved (expected: uses the risk path)" if moved else \
                      "  unchanged (did not expect that)"
        with open(cf, "wb") as fh:
            pickle.dump({"m": m, "evs": traces["age_events"].get(name),
                         "pub": traces["recorded"].get(name)}, fh, protocol=4)

        # L2Only performs no L3 content reuse, so it has no drift denominator.
        drift_s = f"{100*m['_drift']:7.4f}%" if m["_drift"] is not None else "    N/A"
        print(f"  {name:<32} saved {m['search_saved_pct']:8.4f}%  "
              f"drift {drift_s} ({ch:,}/{obs:,})  "
              f"fetch {m['fetches']:,}  {time.time()-t0:5.1f}s{tag}", flush=True)

    with open(HERE / "traces.pkl", "wb") as fh:
        pickle.dump(traces, fh, protocol=4)
    print(f"\n  wrote {HERE/'traces.pkl'}")


if __name__ == "__main__":
    main()
