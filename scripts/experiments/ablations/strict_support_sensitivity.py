#!/usr/bin/env python3
"""
scripts/experiments/ablations/strict_support_sensitivity.py —
strict common-DETERMINATE-support sensitivity.

THE PROBLEM THIS TESTS.  In the recorded analysis every method's drift is
measured on the common observable support S (the policy-independent set of
non-REAL_TIME requests whose own evidence is substantive both at run_00 and at
the round the request lands in), but each method still contributes a DIFFERENT
subset of S to its own numerator and denominator, because a method only produces
a content-reuse outcome on requests where it actually reuses something. On
zipf_uniform/seed 42 FreshCache scores many more of the requests in S than
TieredFixedTTL does -- so the two drift figures are computed on
overlapping but unequal request sets, and part of any difference between them
could be composition rather than freshness behaviour.

THE TEST.  Restrict every method to the SAME requests: those on which ALL FOUR
primary methods -- FreshCache, SCALM, SemanticTTL, TieredFixedTTL -- return a
determinate outcome (CHANGED or UNCHANGED; not UNOBSERVABLE, and not "no reuse
happened"). On that single shared subset the denominators are identical by
construction and the only thing that can differ is whether the content each
method served had changed.

Two nested subsets are reported:
  STRICT          all four methods determinate.
  STRICT n S      the same, intersected with the common observable support S,
                  so it is also a subset of the recorded denominator.

No LLM judge and no search API is called: outcomes come from the immutable
data/derived/round_table.jsonl content hashes, exactly as in the
recorded analysis. The replays are re-executed only because per-request
outcome vectors were never persisted -- the aggregate metrics they reproduce
are asserted against outputs/robustness_schedules/schedule_robustness_tasks.jsonl, and the run aborts on mismatch.

3 schedules x 5 seeds x 4 methods = 60 runs, checkpointed.

Run: C=16 python3 -u scripts/experiments/ablations/strict_support_sensitivity.py
"""
from __future__ import annotations
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage

import csv, json, math, multiprocessing as mp, os, pathlib, statistics, sys, time; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve()
HERE = ((OUTPUT_ROOT / "class_predictor"))
for p in ("freshcache/method", "scripts/data", "", "scripts/baselines/vcache", "scripts/evaluation", "freshcache/simulation", "freshcache/selection_replay_engine"):
    sys.path.insert(0, str(ROOT/p) if p else str(ROOT))

import numpy as np                    # noqa: E402
import tiered_cache_model as exp              # noqa: E402
import execution_semantics as ce         # noqa: E402
import method_replay as ea               # noqa: E402
import request_schedules as sc                # noqa: E402
import mixed_age_replay as me             # noqa: E402
from schedule_robustness import support_of, wilson   # noqa: E402

WORKERS = int(os.environ.get("C", "16"))
SEEDS = [42, 43, 44, 45, 46]
METHODS = ["FreshCache", "SCALM", "SemanticTTL", "TieredFixedTTL"]
REF_CELL = ("zipf_uniform", 42)
TASKS = HERE/"class_code_tasks.jsonl"
CODE = {ce.CHANGED: "C", ce.UNCHANGED: "U", ce.UNOBS: "O", None: "-"}
_REC = _ROUNDS = _RICH = None


def _init(rec, rounds, rich):
    global _REC, _ROUNDS, _RICH
    _REC, _ROUNDS, _RICH = rec, rounds, rich
    ea.set_cluster_base(rec)


def _run(task):
    sched, seed, method = task
    t0 = time.time()
    stream = sc.build_stream(_REC, sched, seed)
    m, per = me.replay(stream, _ROUNDS, method, rich=_RICH)
    S = support_of(stream, _ROUNDS)
    ins = [o for qid, o in per if o is not None and qid in S]
    ch = sum(1 for o in ins if o == ce.CHANGED)
    obs = ch + sum(1 for o in ins if o == ce.UNCHANGED)
    m.update({"schedule": sched, "seed": seed, "method": method,
              "support_size": len(S), "observable_hits_in_support": obs,
              "changed_in_support": ch,
              "drift_common_support": (ch/obs) if obs else None,
              "validation_errors": me.assert_run(m, stream),
              "codes": "".join(CODE[o] for _, o in per),
              "qids": None, "secs": round(time.time()-t0, 1)})
    return m


def drift(codes, idxs):
    ch = sum(1 for i in idxs if codes[i] == "C")
    un = sum(1 for i in idxs if codes[i] == "U")
    n = ch+un
    lo, hi = wilson(ch, n)
    return {"n": n, "changed": ch, "unchanged": un,
            "drift": (ch/n) if n else None,
            "ci95": [round(lo, 6), round(hi, 6)] if n else None}


def summarise(v):
    v = [x for x in v if x is not None]
    if not v:
        return None
    n = len(v); m = statistics.mean(v)
    sd = statistics.stdev(v) if n > 1 else 0.0
    h = 1.96*sd/math.sqrt(n) if n > 1 else 0.0
    return {"n": n, "mean": m, "std": sd, "ci95": [m-h, m+h], "values": v}


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

    cells = [(s, sd, m) for s in sc.SCHEDULES for sd in SEEDS for m in METHODS]
    done = {}
    if TASKS.exists():
        for line in open(TASKS):
            line = line.strip()
            if line:
                r = json.loads(line)
                done[(r["schedule"], r["seed"], r["method"])] = r
    todo = [c for c in cells if c not in done]
    print(f"  {len(cells)} runs, {len(done)} cached, {len(todo)} to run on "
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
                print(f"    [{k}/{len(todo)}] {m['schedule']:<15}s{m['seed']} "
                      f"{m['method']:<16}drift {100*m['drift_common_support']:5.2f}%"
                      f"  {m['secs']:>7.1f}s"
                      f"{'  VALIDATION ERRORS' if m['validation_errors'] else ''}",
                      flush=True)

    # ---- reproduction check against the recorded robustness runs ----
    e1 = {}
    for line in open((OUTPUT_ROOT / "robustness_schedules/schedule_robustness_tasks.jsonl")):
        r = json.loads(line)
        e1[(r["schedule"], r["seed"], r["method"])] = r
    mism = []
    for k, v in done.items():
        if k not in e1:
            continue
        for f in ("search_saved_pct", "l1_hits", "l2_hits", "l3_hits",
                  "fetches", "observable_hits_in_support", "changed_in_support",
                  "support_size"):
            if v[f] != e1[k][f]:
                mism.append((k, f, v[f], e1[k][f]))
    print(f"\n  reproduction vs schedule_robustness_tasks.jsonl: "
          f"{len(done)} runs compared, {len(mism)} field mismatches")
    for x in mism[:10]:
        print(f"    MISMATCH {x}")
    if mism:
        raise SystemExit("  ABORT: replays do not reproduce the recorded runs.")
    bad = [k for k, v in done.items() if v["validation_errors"]]
    print(f"  runs with validation errors: {len(bad)}")

    # ---- strict subsets, per cell ----
    out_cells = {}
    for s in sc.SCHEDULES:
        for sd in SEEDS:
            stream = sc.build_stream(records, s, sd)
            S = support_of(stream, rounds)
            inS = [i for i, (t, r) in enumerate(stream) if r["query_id"] in S]
            codes = {m: done[(s, sd, m)]["codes"] for m in METHODS}
            n = len(stream)
            det = {m: [c in "CU" for c in codes[m]] for m in METHODS}
            strict = [i for i in range(n) if all(det[m][i] for m in METHODS)]
            strict_S = [i for i in strict if i in set(inS)]
            out_cells[f"{s}|{sd}"] = {
                "schedule": s, "seed": sd, "n_requests": n,
                "support_size": len(S),
                "determinate_per_method": {m: int(sum(det[m])) for m in METHODS},
                "strict_n": len(strict), "strict_in_support_n": len(strict_S),
                "strict": {m: drift(codes[m], strict) for m in METHODS},
                "strict_in_support": {m: drift(codes[m], strict_S) for m in METHODS},
                "recorded_support": {
                    m: {"n": done[(s, sd, m)]["observable_hits_in_support"],
                        "changed": done[(s, sd, m)]["changed_in_support"],
                        "drift": done[(s, sd, m)]["drift_common_support"]}
                    for m in METHODS}}

    # ---- reference cell table ----
    rc = out_cells[f"{REF_CELL[0]}|{REF_CELL[1]}"]
    print(f"\n{'='*120}\n  REFERENCE CELL {REF_CELL[0]} / seed {REF_CELL[1]} — "
          f"{rc['n_requests']:,} requests, common observable support "
          f"{rc['support_size']:,}\n{'='*120}")
    print(f"  determinate outcomes per method (of {rc['n_requests']:,} requests):")
    for m in METHODS:
        print(f"    {m:<18}{rc['determinate_per_method'][m]:>8,}")
    print(f"\n  STRICT shared subset (all four determinate): "
          f"{rc['strict_n']:,} requests")
    print(f"  STRICT n S (also in the common observable support): "
          f"{rc['strict_in_support_n']:,} requests\n")
    print(f"  {'method':<18}{'recorded n':>13}{'recorded drift':>17}"
           f"{'strict n':>10}{'strict drift':>15}{'strict 95% CI':>20}"
           f"{'delta':>9}")
    for m in METHODS:
        p, q = rc["recorded_support"][m], rc["strict"][m]
        print(f"  {m:<18}{p['n']:>13,}{100*p['drift']:>16.2f}%"
              f"{q['n']:>10,}{100*q['drift']:>14.2f}%"
              f"{('['+format(100*q['ci95'][0],'.2f')+', '+format(100*q['ci95'][1],'.2f')+']'):>20}"
              f"{100*(q['drift']-p['drift']):>8.2f}")

    # ---- all cells ----
    print(f"\n{'='*126}\n  ALL 15 CELLS — DRIFT ON THE STRICT SHARED SUBSET"
          f"\n{'='*126}")
    print(f"  {'schedule':<16}{'seed':>5}{'strict n':>10}"
          + "".join(f"{m:>19}" for m in METHODS))
    for s in sc.SCHEDULES:
        for sd in SEEDS:
            c = out_cells[f"{s}|{sd}"]
            cells_ = "".join(
                f"{(format(100*c['strict'][m]['drift'],'.2f')+'%'):>19}"
                for m in METHODS)
            print(f"  {s:<16}{sd:>5}{c['strict_n']:>10,}{cells_}")

    # ---- aggregates + paired differences on the strict subset ----
    agg, paired = {}, {}
    print(f"\n{'='*118}\n  MEAN ± STD ACROSS 5 SEEDS — STRICT SHARED SUBSET"
          f"\n{'='*118}")
    print(f"  {'schedule':<16}{'method':<18}{'strict drift':>26}"
          f"{'recorded drift':>26}")
    for s in sc.SCHEDULES:
        for m in METHODS:
            a = summarise([out_cells[f"{s}|{sd}"]["strict"][m]["drift"]
                           for sd in SEEDS])
            b = summarise([out_cells[f"{s}|{sd}"]["recorded_support"][m]["drift"]
                           for sd in SEEDS])
            agg[f"{s}|{m}"] = {"strict": a, "recorded": b}
            f_ = lambda x: (f"{100*x['mean']:.2f}±{100*x['std']:.2f} "
                            f"[{100*x['ci95'][0]:.2f},{100*x['ci95'][1]:.2f}]")
            print(f"  {s:<16}{m:<18}{f_(a):>26}{f_(b):>26}")
    print(f"\n{'='*112}\n  PAIRED DIFFERENCE ON THE STRICT SUBSET, "
          f"FreshCache MINUS BASELINE\n{'='*112}")
    print(f"  {'schedule':<16}{'baseline':<18}{'mean diff':>12}{'std':>9}"
          f"{'95% CI':>24}{'seeds FC lower':>16}{'significant':>13}")
    for s in sc.SCHEDULES:
        for b in METHODS[1:]:
            d = [out_cells[f"{s}|{sd}"]["strict"]["FreshCache"]["drift"]
                 - out_cells[f"{s}|{sd}"]["strict"][b]["drift"] for sd in SEEDS]
            a = summarise(d)
            sig = (a["ci95"][0] < 0) == (a["ci95"][1] < 0)
            paired[f"{s}|{b}"] = {**a, "seeds_freshcache_lower":
                                  sum(1 for x in d if x < 0), "significant": sig}
            print(f"  {s:<16}{b:<18}{100*a['mean']:>11.2f}%{100*a['std']:>8.2f}%"
                  f"{('['+format(100*a['ci95'][0],'.2f')+'%, '+format(100*a['ci95'][1],'.2f')+'%]'):>24}"
                  f"{(str(paired[f'{s}|{b}']['seeds_freshcache_lower'])+'/5'):>16}"
                  f"{('yes' if sig else 'no'):>13}")

    json.dump({"methods": METHODS, "seeds": SEEDS, "schedules": sc.SCHEDULES,
               "reference_cell": f"{REF_CELL[0]}|{REF_CELL[1]}",
               "cells": out_cells, "aggregate": agg, "paired_strict": paired,
               "reproduction_mismatches": len(mism),
               "validation_error_runs": len(bad)},
              open(HERE/"strict_support_sensitivity.json", "w"), indent=2)
    with open(HERE/"strict_support_sensitivity.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["schedule", "seed", "subset", "method", "n", "changed",
                    "unchanged", "drift_pct", "ci95_lo_pct", "ci95_hi_pct"])
        for k, c in out_cells.items():
            for sub in ("strict", "strict_in_support"):
                for m in METHODS:
                    d = c[sub][m]
                    w.writerow([c["schedule"], c["seed"], sub, m, d["n"],
                                d["changed"], d["unchanged"],
                                round(100*d["drift"], 4) if d["drift"] is not None else "",
                                round(100*d["ci95"][0], 4) if d["ci95"] else "",
                                round(100*d["ci95"][1], 4) if d["ci95"] else ""])
            for m in METHODS:
                p = c["recorded_support"][m]
                w.writerow([c["schedule"], c["seed"], "recorded_support", m,
                            p["n"], p["changed"], p["n"]-p["changed"],
                            round(100*p["drift"], 4), "", ""])
    print(f"\n  wrote strict_support_sensitivity.json, strict_support_sensitivity.csv")


if __name__ == "__main__":
    main()
