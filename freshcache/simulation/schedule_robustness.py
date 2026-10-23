#!/usr/bin/env python3
"""
freshcache/simulation/schedule_robustness.py — mixed-age robustness across request
schedules and seeds.

3 schedules x 5 seeds x 5 methods = 75 runs. Within a (schedule, seed) cell every
method sees the IDENTICAL stream, so all differences are the method's.

Drift is always computed on the policy-independent common observable support for
that stream: the non-REAL_TIME requests whose own evidence has a substantive body
both at run_00 and at the round the request lands in. Unobservable outcomes are
excluded from both numerator and denominator and are never counted as unchanged.

Checkpointed per (schedule, seed, method); safe to interrupt and resume.

Run: C=12 python3 -u freshcache/simulation/schedule_robustness.py
"""
from __future__ import annotations

import sys as _sys; _sys.exit(print(__doc__ or "")) if __name__ == "__main__" and any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage
import json, math, multiprocessing as mp, os, pathlib, statistics, sys, time; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
HERE = ((OUTPUT_ROOT / "robustness_schedules"))
sys.path.insert(0, str(ROOT / "scripts/baselines/vcache")); sys.path.insert(0, str(ROOT / "")); sys.path.insert(0, str(ROOT / "scripts/data")); sys.path.insert(0, str(ROOT / "scripts/evaluation")); sys.path.insert(0, str(ROOT / "freshcache/method")); sys.path.insert(0, str(ROOT / "freshcache/selection_replay_engine")); sys.path.insert(0, str(ROOT / "freshcache/simulation"))
sys.path.insert(0, str(ROOT / "freshcache/selection_replay_engine")); sys.path.insert(0, str(ROOT / "freshcache/selection_replay_engine")); sys.path.insert(0, str(ROOT / "scripts/evaluation"))
sys.path.insert(0, str(ROOT / "freshcache/simulation"))
import numpy as np                    # noqa: E402
import tiered_cache_model as exp              # noqa: E402
import execution_semantics as ce         # noqa: E402
import method_replay as ea               # noqa: E402
import page_version_replay as ma             # noqa: E402
import request_schedules as sc                # noqa: E402
import mixed_age_replay as me             # noqa: E402

WORKERS = int(os.environ.get("C", "12"))
SEEDS = [42, 43, 44, 45, 46]
METHODS = ["FreshCache", "FreshCache_MLP", "SCALM", "SemanticTTL", "TieredFixedTTL"]
TASKS = HERE/"schedule_robustness_tasks.jsonl"
_REC = _ROUNDS = _RICH = None


def _init(rec, rounds, rich):
    global _REC, _ROUNDS, _RICH
    _REC, _ROUNDS, _RICH = rec, rounds, rich
    ea.set_cluster_base(rec)


def support_of(stream, rounds):
    S = set()
    for t, r in stream:
        if r["freshness_class"] == "REAL_TIME":
            continue
        v = ma.version_at(t)
        for u in r["urls"]:
            per = rounds.get(u["url_hash"])
            if per and per.get("run_00", {}).get("substantive") \
                    and per.get(v, {}).get("substantive"):
                S.add(r["query_id"]); break
    return S


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p, d = k/n, 1 + z*z/n
    c = p + z*z/(2*n)
    m = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return (max(0.0, (c-m)/d), min(1.0, (c+m)/d))


def _run(task):
    sched, seed, method = task
    t0 = time.time()
    stream = sc.build_stream(_REC, sched, seed)
    S = support_of(stream, _ROUNDS)
    m, per = me.replay(stream, _ROUNDS, method, rich=_RICH)
    errs = me.assert_run(m, stream)
    ins = [o for qid, o in per if o is not None and qid in S]
    ch = sum(1 for o in ins if o == ce.CHANGED)
    un = sum(1 for o in ins if o == ce.UNCHANGED)
    obs = ch + un
    lo, hi = wilson(ch, obs)
    m.update({"schedule": sched, "seed": seed, "method": method,
              "support_size": len(S),
              "observable_hits_in_support": obs, "changed_in_support": ch,
              "drift_common_support": (ch/obs) if obs else None,
              "drift_ci95": [round(lo, 6), round(hi, 6)] if obs else None,
              "common_support_coverage": round(obs/len(S), 6) if S else None,
              "validation_errors": errs, "secs": round(time.time()-t0, 1)})
    return m


def summarise(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    n = len(vals)
    mean = statistics.mean(vals)
    sd = statistics.stdev(vals) if n > 1 else 0.0
    half = 1.96*sd/math.sqrt(n) if n > 1 else 0.0
    return {"n": n, "mean": mean, "std": sd,
            "ci95": [mean-half, mean+half], "values": vals}


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
    for line in open((DATA_ROOT / "derived/round_table.jsonl"),
                     encoding="utf-8"):
        r = json.loads(line)
        rounds[r["url_hash"]] = r["rounds"]

    print(f"  records {len(records):,}  rounds {len(rounds):,}")
    print(f"  schedules {sc.SCHEDULES}  seeds {SEEDS}  methods {METHODS}")
    for s in sc.SCHEDULES:
        d = sc.describe(sc.build_stream(records, s, 42))
        print(f"    {s:<16}intra-cluster gap median {d['intra_cluster_gap_median_h']:6.2f}h  "
              f"p95 {d['intra_cluster_gap_p95_h']:6.2f}h", flush=True)

    cells = [(s, sd, m) for s in sc.SCHEDULES for sd in SEEDS for m in METHODS]
    done = {}
    if TASKS.exists():
        for line in open(TASKS):
            line = line.strip()
            if line:
                r = json.loads(line)
                done[(r["schedule"], r["seed"], r["method"])] = r
    todo = [c for c in cells if c not in done]
    print(f"\n  {len(cells)} runs, {len(done)} cached, {len(todo)} to run on "
          f"{WORKERS} workers", flush=True)
    if todo:
        with mp.Pool(WORKERS, initializer=_init,
                     initargs=(records, rounds, rich)) as pool:
            k = 0
            for m in pool.imap_unordered(_run, todo):
                done[(m["schedule"], m["seed"], m["method"])] = m
                with open(TASKS, "a") as f:
                    f.write(json.dumps(m)+"\n")
                k += 1
                flag = "" if not m["validation_errors"] else "  VALIDATION ERRORS"
                if k % 5 == 0 or m["validation_errors"]:
                    print(f"    [{k}/{len(todo)}] {m['schedule']:<15}s{m['seed']} "
                          f"{m['method']:<16}saved {m['search_saved_pct']:5.1f}%  "
                          f"drift {100*m['drift_common_support']:5.2f}%{flag}",
                          flush=True)

    bad = [k for k, v in done.items() if v["validation_errors"]]
    print(f"\n  runs with validation errors: {len(bad)}")
    for k in bad[:10]:
        print(f"    {k}: {done[k]['validation_errors']}")

    # ---------------- per-run table ----------------
    print(f"\n{'='*150}\n  EXPERIMENT 1 — ALL RUNS\n{'='*150}")
    print(f"  {'schedule':<15}{'seed':>5}{'method':<17}{'saved':>8}{'drift':>8}"
          f"{'cov':>7}{'L1':>8}{'L2':>8}{'L3':>8}{'search':>8}{'fetch':>8}"
          f"{'cGET':>6}{'gens':>8}{'medAge':>8}{'p95Age':>8}")
    rows = []
    for s in sc.SCHEDULES:
        for sd in SEEDS:
            for mth in METHODS:
                v = done[(s, sd, mth)]
                rows.append(v)
                ma_ = v["median_hit_age_all_h"]; pa = v["p95_hit_age_all_h"]
                print(f"  {s:<15}{sd:>5}{mth:<17}{v['search_saved_pct']:>7.1f}%"
                      f"{100*v['drift_common_support']:>7.2f}%"
                      f"{100*v['common_support_coverage']:>6.1f}%"
                      f"{v['l1_hits']:>8,}{v['l2_hits']:>8,}{v['l3_hits']:>8,}"
                      f"{v['search_calls']:>8,}{v['fetches']:>8,}"
                      f"{v['conditional_gets']:>6}{v['generations']:>8,}"
                      f"{(f'{ma_:.1f}h' if ma_ else '—'):>8}"
                      f"{(f'{pa:.1f}h' if pa else '—'):>8}")

    # ---------------- mean / std / CI ----------------
    METRICS = ["search_saved_pct", "drift_common_support", "common_support_coverage",
               "l1_hits", "l2_hits", "l3_hits", "search_calls", "fetches",
               "conditional_gets", "generations", "median_hit_age_all_h",
               "p95_hit_age_all_h"]
    agg = {}
    print(f"\n{'='*128}\n  EXPERIMENT 1 — MEAN +/- STD (95% CI) ACROSS "
          f"{len(SEEDS)} SEEDS\n{'='*128}")
    for s in sc.SCHEDULES:
        print(f"\n  schedule: {s}")
        print(f"    {'method':<17}{'saved %':>22}{'drift %':>22}{'coverage %':>22}"
              f"{'median age h':>20}")
        for mth in METHODS:
            vs = [done[(s, sd, mth)] for sd in SEEDS]
            a = {k: summarise([v[k] for v in vs]) for k in METRICS}
            agg[f"{s}|{mth}"] = a
            f = lambda k, sc_=1: (f"{sc_*a[k]['mean']:.2f}±{sc_*a[k]['std']:.2f} "
                                  f"[{sc_*a[k]['ci95'][0]:.2f},{sc_*a[k]['ci95'][1]:.2f}]"
                                  if a[k] else "—")
            print(f"    {mth:<17}{f('search_saved_pct'):>22}"
                  f"{f('drift_common_support',100):>22}"
                  f"{f('common_support_coverage',100):>22}"
                  f"{f('median_hit_age_all_h'):>20}")

    # ---------------- paired differences ----------------
    print(f"\n{'='*118}\n  PAIRED DIFFERENCE IN DRIFT, FreshCache MINUS BASELINE "
          f"(same schedule and seed)\n{'='*118}")
    print(f"  {'schedule':<16}{'baseline':<17}{'mean diff':>12}{'std':>9}"
          f"{'95% CI':>24}{'seeds FC lower':>16}")
    paired = {}
    for s in sc.SCHEDULES:
        for b in ["SCALM", "SemanticTTL", "TieredFixedTTL"]:
            d = [done[(s, sd, "FreshCache")]["drift_common_support"]
                 - done[(s, sd, b)]["drift_common_support"] for sd in SEEDS]
            a = summarise(d)
            lower = sum(1 for x in d if x < 0)
            paired[f"{s}|{b}"] = {**a, "seeds_freshcache_lower": lower}
            print(f"  {s:<16}{b:<17}{100*a['mean']:>11.2f}%{100*a['std']:>8.2f}%"
                  f"{'['+format(100*a['ci95'][0],'.2f')+'%, '+format(100*a['ci95'][1],'.2f')+'%]':>24}"
                  f"{str(lower)+'/'+str(len(SEEDS)):>16}")

    json.dump({"seeds": SEEDS, "schedules": sc.SCHEDULES, "methods": METHODS,
               "schedule_stats": {s: sc.describe(sc.build_stream(records, s, 42))
                                  for s in sc.SCHEDULES},
               "runs": rows, "aggregate": agg, "paired_drift": paired,
               "validation_error_runs": len(bad)},
              open(HERE/"schedule_robustness_results.json", "w"), indent=2)
    import csv
    with open(HERE/"schedule_robustness_runs.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["schedule", "seed", "method", "search_saved_pct",
                "drift_common_support", "common_support_coverage", "support_size",
                "observable_hits_in_support", "changed_in_support", "l1_hits",
                "l2_hits", "l3_hits", "search_calls", "fetches",
                "conditional_gets", "generations", "median_hit_age_all_h",
                "p95_hit_age_all_h"]
        wtr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wtr.writeheader()
        for r in rows:
            wtr.writerow(r)
    print(f"\n  wrote schedule_robustness_results.json, schedule_robustness_runs.csv")


if __name__ == "__main__":
    main()
