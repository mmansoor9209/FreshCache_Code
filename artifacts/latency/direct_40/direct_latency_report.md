# Direct end-to-end latency validation

A **direct wall-clock measurement** of the complete request path — query in to
final answer — on 40 representative requests per policy, run in
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
| Policy | Direct (sample) | Direct (reweighted) | Monte Carlo | Δ sample | Δ reweighted |
|---|---|---|---|---|---|
| NoCache (fresh pipeline) | 3,255.3 | 3,271.4 | 3,940.4 | -17 % | -17 % |
| SemanticTTL k=1/16 | 703.3 | 142.7 | 328.8 | +114 % | -57 % |
| FreshCache L2+L3 (no L1) | 1,238.2 | 852.9 | 1,534.0 | -19 % | -44 % |
| FreshCache (full) | 1,214.4 | 885.9 | 1,514.9 | -20 % | -42 % |

### p50
| Policy | Direct (sample) | Direct (reweighted) | Monte Carlo | Δ sample | Δ reweighted |
|---|---|---|---|---|---|
| NoCache (fresh pipeline) | 3,014.4 | 3,014.4 | 3,417.4 | -12 % | -12 % |
| SemanticTTL k=1/16 | 13.5 | 12.9 | 11.4 | +19 % | +14 % |
| FreshCache L2+L3 (no L1) | 743.0 | 509.6 | 795.6 | -7 % | -36 % |
| FreshCache (full) | 563.8 | 521.5 | 759.7 | -26 % | -31 % |

### p95
| Policy | Direct (sample) | Direct (reweighted) | Monte Carlo | Δ sample | Δ reweighted |
|---|---|---|---|---|---|
| NoCache (fresh pipeline) | 5,050.3 | 5,050.3 | 7,860.9 | -36 % | -36 % |
| SemanticTTL k=1/16 | 3,832.5 | 24.5 | 3,056.2 | +25 % | -99 % |
| FreshCache L2+L3 (no L1) | 3,558.3 | 2,405.9 | 4,621.7 | -23 % | -48 % |
| FreshCache (full) | 3,107.3 | 2,942.3 | 4,624.1 | -33 % | -36 % |

## 3. Measured stage means on the direct runs (ms)

| Policy | embedding | L1 lookup | L2 lookup | L3 processing | search API | page fetch | LLM generation |
|---|---|---|---|---|---|---|---|
| NoCache (fresh pipeline) | — | — | — | — | 1,158.47 | 1,448.67 | 621.22 |
| SemanticTTL k=1/16 | 10.34 | 1.00 | — | — | 237.11 | 266.91 | 182.83 |
| FreshCache L2+L3 (no L1) | 9.17 | — | 17.22 | 0.26 | 394.30 | 239.22 | 571.12 |
| FreshCache (full) | 8.87 | 25.60 | 4.98 | 0.20 | 424.03 | 216.97 | 527.60 |

Path mix actually taken in the sample:

| Policy | Paths |
|---|---|
| NoCache (fresh pipeline) | nocache 40 |
| SemanticTTL k=1/16 | L1 31, miss 1, realtime_bypass 8 |
| FreshCache L2+L3 (no L1) | L2 25, miss 7, realtime_bypass 8 |
| FreshCache (full) | L1 3, L2 21, miss 8, realtime_bypass 8 |

## 4. Does the direct measurement confirm the estimate?

**Yes on the ranking and on the central claim, with the estimate mildly
conservative.**

- The **ordering is identical** in mean and p50: NoCache slowest, then
  FreshCache ≈ FreshCache-without-L1, then SemanticTTL k=1/16 fastest.
- **FreshCache vs NoCache**, the headline: directly measured mean
  886 ms vs
  3,271 ms reweighted — a
  **73 % reduction**,
  against 62 % in the
  Monte Carlo estimate. p50 reduction measured
  83 %
  vs 78 % estimated.
- **L1 again contributes almost nothing**: FreshCache vs L2+L3-only differ by
  33 ms
  in the reweighted mean, the same conclusion the estimate reached.
- **The estimate's p95 was pessimistic.** Directly measured p95 is lower for
  every policy. The Monte Carlo drew each fetch independently from the measured
  pool, so a multi-URL request could draw several tail fetches at once; in reality
  a request's fetches are correlated and the page cache absorbs part of the cost.
- **SemanticTTL is the one place the sample composition really bites.** Its
  unweighted sample mean (703 ms)
  is far above the estimate (329 ms) purely
  because 8 of 40 sampled requests are REAL_TIME, which it must serve from
  scratch; reweighted to the true class mix it falls to
  143 ms. Its
  p50 (13.5 ms) matches the
  estimate (11.4 ms) closely, because the median
  request is an L1 hit in both.

As before, SemanticTTL's speed comes from answering most requests from a stored
answer, which the held-out answer audit measured at 80.3 % conditional
degradation against 8.2 % for FreshCache. **Latency alone does not favour it.**

## 5. Limitations

- 40 requests per policy: enough to validate means and the
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
