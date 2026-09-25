#!/usr/bin/env python3
"""
Section 8 -- summary of the REAL wall-clock latency measurement.

Adds to the raw per-request timings: p99, throughput, and 10,000-resample
bootstrap 95% confidence intervals for the mean and the median, per policy and
per stage. Nothing here mixes in a simulated latency constant: every number
comes from 08_latency/direct_per_request.csv, which holds one real timer per
real request.
"""
from __future__ import annotations
import csv, json, pathlib, random, statistics

HERE = pathlib.Path(__file__).resolve().parent
STAGES = ["embedding", "l1_lookup", "l2_lookup", "l3_processing",
          "search_api", "page_fetch", "llm_generation"]
B, SEED = 10_000, 42


def pct(v, p):
    s = sorted(v)
    return float(s[min(len(s) - 1, int(round(p * (len(s) - 1))))])


def boot(v, stat, b=B):
    rng = random.Random(SEED)
    n = len(v)
    if n < 2:
        return (None, None)
    out = []
    for _ in range(b):
        out.append(stat([v[rng.randrange(n)] for _ in range(n)]))
    out.sort()
    return (round(out[int(.025 * b)], 2), round(out[int(.975 * b)], 2))


def main():
    rows = list(csv.DictReader(open(HERE / "direct_per_request.csv",
                                    encoding="utf-8")))
    for r in rows:
        for k in STAGES + ["total_ms"]:
            r[k] = float(r[k])
    pols = []
    for r in rows:
        if r["policy"] not in pols:
            pols.append(r["policy"])
    out = {}
    print("  REAL wall-clock latency, held-out sample, one timer per request")
    print(f"  bootstrap: {B:,} resamples, seed {SEED}\n")
    print(f"  {'policy':<22}{'n':>5}{'mean ms':>10}{'mean 95% CI':>22}"
          f"{'p50':>9}{'p50 95% CI':>22}{'p95':>9}{'p99':>9}{'req/s':>9}")
    for p in pols:
        v = [r["total_ms"] for r in rows if r["policy"] == p]
        mean = statistics.mean(v)
        med = statistics.median(v)
        mci, dci = boot(v, statistics.mean), boot(v, statistics.median)
        tot_s = sum(v) / 1000.0
        out[p] = {"n": len(v), "mean_ms": round(mean, 2), "mean_ci95": mci,
                  "p50_ms": round(med, 2), "p50_ci95": dci,
                  "p95_ms": round(pct(v, .95), 2),
                  "p99_ms": round(pct(v, .99), 2),
                  "min_ms": round(min(v), 2), "max_ms": round(max(v), 2),
                  "throughput_req_per_s_sequential": round(len(v) / tot_s, 4),
                  "live_searches": sum(1 for r in rows
                                       if r["policy"] == p and r["search_api"] > 0),
                  "live_fetches": sum(int(r["n_fetched"]) for r in rows
                                      if r["policy"] == p),
                  "cache_reads": sum(int(r["n_from_cache"]) for r in rows
                                     if r["policy"] == p),
                  "generation_calls": sum(1 for r in rows
                                          if r["policy"] == p
                                          and r["llm_generation"] > 0),
                  "stages": {}}
        for s in STAGES:
            sv = [r[s] for r in rows if r["policy"] == p]
            out[p]["stages"][s] = {
                "mean_ms": round(statistics.mean(sv), 3),
                "mean_ci95": boot(sv, statistics.mean, 2000),
                "p50_ms": round(statistics.median(sv), 3),
                "p95_ms": round(pct(sv, .95), 3)}
        print(f"  {p:<22}{len(v):>5}{mean:>10.1f}{str(mci):>22}{med:>9.1f}"
              f"{str(dci):>22}{pct(v,.95):>9.1f}{pct(v,.99):>9.1f}"
              f"{len(v)/tot_s:>9.3f}")

    print(f"\n  per-stage mean ms")
    print(f"  {'policy':<22}" + "".join(f"{s[:11]:>13}" for s in STAGES))
    for p in pols:
        print(f"  {p:<22}" + "".join(
            f"{out[p]['stages'][s]['mean_ms']:>13.2f}" for s in STAGES))

    print(f"\n  live operations actually issued")
    print(f"  {'policy':<22}{'searches':>10}{'fetches':>10}"
          f"{'cache reads':>13}{'generations':>13}")
    ts = tf = 0
    for p in pols:
        r = out[p]
        ts += r["live_searches"]; tf += r["live_fetches"]
        print(f"  {p:<22}{r['live_searches']:>10,}{r['live_fetches']:>10,}"
              f"{r['cache_reads']:>13,}{r['generation_calls']:>13,}")
    print(f"  {'TOTAL':<22}{ts:>10,}{tf:>10,}")
    print(f"\n  API cost: the search API's per-query price is not recorded in "
          f"this repository, so no monetary figure is given -- only the "
          f"{ts:,} real search calls and {tf:,} real page fetches above.")
    print(f"  No simulated latency constant appears anywhere in this section.")

    json.dump({"measurement": "direct wall-clock, one process, per request",
               "bootstrap_resamples": B, "seed": SEED,
               "split": "held-out TEST clusters",
               "cost": "not reported: per-query price not recorded in the repo",
               "policies": out},
              open(HERE / "latency_summary.json", "w"), indent=2)
    with open(HERE / "latency_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["policy", "n", "mean_ms", "mean_ci_lo", "mean_ci_hi",
                    "p50_ms", "p50_ci_lo", "p50_ci_hi", "p95_ms", "p99_ms",
                    "throughput_req_per_s", "live_searches", "live_fetches",
                    "cache_reads", "generation_calls"])
        for p in pols:
            r = out[p]
            w.writerow([p, r["n"], r["mean_ms"], *r["mean_ci95"], r["p50_ms"],
                        *r["p50_ci95"], r["p95_ms"], r["p99_ms"],
                        r["throughput_req_per_s_sequential"],
                        r["live_searches"], r["live_fetches"],
                        r["cache_reads"], r["generation_calls"]])
    print("\n  wrote latency_summary.json/.csv")


if __name__ == "__main__":
    main()
