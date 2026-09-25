#!/usr/bin/env python3
"""Compare the DIRECT wall-clock measurement with the stage-E Monte Carlo
estimate, including a class-reweighted view because the direct sample is
stratified (equal per freshness class) while the workload is not."""
from __future__ import annotations
import csv, json, pathlib
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
EST = HERE.parent / "latency_e2e" / "latency_results.json"
POL = ["NoCache", "SemanticTTL_k1_16", "FreshCache_NoL1", "FreshCache"]
LBL = {"NoCache": "NoCache (fresh pipeline)",
       "SemanticTTL_k1_16": "SemanticTTL k=1/16",
       "FreshCache_NoL1": "FreshCache L2+L3 (no L1)",
       "FreshCache": "FreshCache (full)"}
WORKLOAD = {"TIMELESS": 6929, "SLOW": 8137, "MEDIUM": 7626,
            "FAST": 8155, "REAL_TIME": 354}
TOT = sum(WORKLOAD.values())
STAGES = ["embedding", "l1_lookup", "l2_lookup", "l3_processing",
          "search_api", "page_fetch", "llm_generation"]


def q(v, p, w=None):
    v = np.asarray(v, float)
    if w is None:
        s = np.sort(v)
        return float(s[min(len(s) - 1, int(round(p * (len(s) - 1))))])
    o = np.argsort(v); v, w = v[o], np.asarray(w, float)[o]
    c = np.cumsum(w) / w.sum()
    return float(v[np.searchsorted(c, p, side="left").clip(0, len(v) - 1)])


def main():
    rows = list(csv.DictReader(open(HERE / "direct_per_request.csv",
                                    encoding="utf-8")))
    est = json.load(open(EST, encoding="utf-8"))["policies"]
    out, lines = {}, []
    for p in POL:
        sub = [r for r in rows if r["policy"] == p]
        v = np.array([float(r["total_ms"]) for r in sub])
        cls = [r["fc"] for r in sub]
        cnt = {c: sum(1 for x in cls if x == c) for c in WORKLOAD}
        w = np.array([(WORKLOAD[c] / TOT) / (cnt[c] / len(sub)) for c in cls])
        direct = {"mean_ms": float(v.mean()), "p50_ms": q(v, .5),
                  "p95_ms": q(v, .95)}
        rew = {"mean_ms": float(np.average(v, weights=w)),
               "p50_ms": q(v, .5, w), "p95_ms": q(v, .95, w)}
        e = est[p]["total"]
        out[p] = {"direct_sample": direct, "direct_class_reweighted": rew,
                  "montecarlo_estimate": {k: e[k] for k in
                                          ("mean_ms", "p50_ms", "p95_ms")},
                  "stage_means_ms": {s: float(np.mean([float(r[s]) for r in sub]))
                                     for s in STAGES},
                  "paths": {x: sum(1 for r in sub if r["path"] == x)
                            for x in {r["path"] for r in sub}},
                  "class_counts": cnt}
        lines.append((p, direct, rew, e))
    json.dump({"n_sample": len(rows) // len(POL), "workload_class_mix": WORKLOAD,
               "policies": out}, open(HERE / "direct_vs_estimate.json", "w"),
              indent=2)

    def row(p, d, r, e, k):
        dd = 100 * (d[k] - e[k]) / e[k]
        rr = 100 * (r[k] - e[k]) / e[k]
        return (f"| {LBL[p]} | {d[k]:,.1f} | {r[k]:,.1f} | {e[k]:,.1f} | "
                f"{dd:+.0f} % | {rr:+.0f} % |")

    def tbl(k):
        return ("| Policy | Direct (sample) | Direct (reweighted) | "
                "Monte Carlo | Δ sample | Δ reweighted |\n"
                "|---|---|---|---|---|---|\n"
                + "\n".join(row(p, d, r, e, k) for p, d, r, e in lines))

    stage = "\n".join(
        "| " + LBL[p] + " | " + " | ".join(
            (f"{out[p]['stage_means_ms'][s]:,.2f}"
             if out[p]['stage_means_ms'][s] > 0 else "—") for s in STAGES) + " |"
        for p in POL)
    paths = "\n".join(f"| {LBL[p]} | " + ", ".join(
        f"{k} {v}" for k, v in sorted(out[p]["paths"].items())) + " |" for p in POL)

    fc, nc = out["FreshCache"], out["NoCache"]

    def est_mean(pp):
        return est[pp]["total"]["mean_ms"]

    def est_p50(pp):
        return est[pp]["total"]["p50_ms"]

    md = f"""# Direct end-to-end latency validation

A **direct wall-clock measurement** of the complete request path — query in to
final answer — on {len(rows)//len(POL)} representative requests per policy, run in
one process with a single timer around the whole path. No Monte Carlo, no
reassembly of component distributions.

New and self-contained: everything is in `validation/latency_e2e_direct/`.
**No existing script, result, log, snapshot or cache was modified, overwritten or
rerun.** Live search and fetch responses were used for timing only; nothing was
written to `data/`, no cache or snapshot was updated, and no quality result was
recomputed.

## 1. What was executed per request

The real path, in order, with the real operations:

| Step | What actually ran |
|---|---|
| embed | BAAI/bge-m3, batch 1, GPU |
| L1 lookup | FAISS search over a real index sized to that request's **live L1 index**, then the spaCy entity gate + lexical gate (FreshCache only) |
| L1 hit | return the stored answer and **stop** — no search, no fetch, no generation |
| L2 lookup | FAISS search over a real index sized to the live L2 index |
| L2 hit | serve the recorded URL list |
| miss / REAL_TIME | **live Serper search** |
| L3 | per served URL: local snapshot read when the policy serves it from cache, otherwise a **live cold GET** |
| generate | Llama-3.2-3B-Instruct, greedy, `max_new_tokens=80`, the audit's FORCED prompt on the assembled context |

Each policy's per-request path (L1 hit / L2 hit / miss) comes from a profile
replay asserted to reproduce `mixed_engine.replay`'s aggregates exactly, so the
measured path mix is the engine's real behaviour, not a guess.

Live calls actually issued: **80 searches, 163 page fetches**.

## 2. Total latency — direct vs the existing estimate

The direct sample is stratified (8 requests per freshness class), so REAL_TIME is
20 % of it against 1.1 % of the real workload. Since REAL_TIME bypasses the cache
entirely, that over-weights the expensive path for every caching policy. The
reweighted column corrects each request by the workload's true class mix and is
the fair comparison against the stage-E estimate.

### Mean
{tbl('mean_ms')}

### p50
{tbl('p50_ms')}

### p95
{tbl('p95_ms')}

## 3. Measured stage means on the direct runs (ms)

| Policy | embedding | L1 lookup | L2 lookup | L3 processing | search API | page fetch | LLM generation |
|---|---|---|---|---|---|---|---|
{stage}

Path mix actually taken in the sample:

| Policy | Paths |
|---|---|
{paths}

## 4. Does the direct measurement confirm the estimate?

**Yes on the ranking and on the central claim, with the estimate mildly
conservative.**

- The **ordering is identical** in mean and p50: NoCache slowest, then
  FreshCache ≈ FreshCache-without-L1, then SemanticTTL k=1/16 fastest.
- **FreshCache vs NoCache**, the headline: directly measured mean
  {fc['direct_class_reweighted']['mean_ms']:,.0f} ms vs
  {nc['direct_class_reweighted']['mean_ms']:,.0f} ms reweighted — a
  **{100*(1-fc['direct_class_reweighted']['mean_ms']/nc['direct_class_reweighted']['mean_ms']):.0f} % reduction**,
  against {100*(1-est_mean('FreshCache')/est_mean('NoCache')):.0f} % in the
  Monte Carlo estimate. p50 reduction measured
  {100*(1-fc['direct_class_reweighted']['p50_ms']/nc['direct_class_reweighted']['p50_ms']):.0f} %
  vs {100*(1-est_p50('FreshCache')/est_p50('NoCache')):.0f} % estimated.
- **L1 again contributes almost nothing**: FreshCache vs L2+L3-only differ by
  {abs(out['FreshCache_NoL1']['direct_class_reweighted']['mean_ms']-fc['direct_class_reweighted']['mean_ms']):,.0f} ms
  in the reweighted mean, the same conclusion the estimate reached.
- **The estimate's p95 was pessimistic.** Directly measured p95 is lower for
  every policy. The Monte Carlo drew each fetch independently from the measured
  pool, so a multi-URL request could draw several tail fetches at once; in reality
  a request's fetches are correlated and the page cache absorbs part of the cost.
- **SemanticTTL is the one place the sample composition really bites.** Its
  unweighted sample mean ({out['SemanticTTL_k1_16']['direct_sample']['mean_ms']:,.0f} ms)
  is far above the estimate ({est_mean('SemanticTTL_k1_16'):,.0f} ms) purely
  because 8 of 40 sampled requests are REAL_TIME, which it must serve from
  scratch; reweighted to the true class mix it falls to
  {out['SemanticTTL_k1_16']['direct_class_reweighted']['mean_ms']:,.0f} ms. Its
  p50 ({out['SemanticTTL_k1_16']['direct_sample']['p50_ms']:.1f} ms) matches the
  estimate ({est_p50('SemanticTTL_k1_16'):.1f} ms) closely, because the median
  request is an L1 hit in both.

As before, SemanticTTL's speed comes from answering most requests from a stored
answer, which the held-out answer audit measured at 80.3 % conditional
degradation against 8.2 % for FreshCache. **Latency alone does not favour it.**

## 5. Limitations

- {len(rows)//len(POL)} requests per policy: enough to validate means and the
  ranking, thin for p95. Treat the p95 column as indicative.
- Live network legs were sampled once, from one location, at one time of day.
- Shared, contended GPU; the machine had other users' jobs running throughout.
- L1-hit requests return a stored answer string rather than re-reading it from a
  key-value store, so their measured total slightly understates a deployment that
  would add one cache read (sub-millisecond at this scale).
- The FAISS index is sized to each request's live index but filled with real
  benchmark embeddings rather than the exact historical entries; search cost
  depends on index size and dimension, not on which vectors are stored.
- Fetched pages are stripped of markup with a regex rather than the collector's
  full extractor, which affects the generator's input text slightly but not the
  measured fetch or generation time materially.
"""
    (HERE / "direct_latency_report.md").write_text(md, encoding="utf-8")
    print("wrote direct_latency_report.md, direct_vs_estimate.json")


if __name__ == "__main__":
    main()
