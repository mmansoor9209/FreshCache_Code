# End-to-end latency experiment — FreshCache

New, self-contained experiment. **No existing script, result, log, snapshot or
cache was modified, overwritten or rerun**; every artifact produced here lives in
`validation/latency_e2e/`.

Live web measurements were used **only** to time the two network operations for
this benchmark. They were not used to recompute any accuracy, drift, freshness or
other saved FreshCache result, and nothing was written into `data/`.

## 1. What was measured, and how

Latency is composed per request from the operations the engine actually performs,
with every operation priced from a **measured** distribution rather than an
assumed constant.

| Component | How it was obtained | Sample |
|---|---|---|
| query embedding | BAAI/bge-m3, batch size 1 (serving path), GPU, wall-clock | n=200 |
| L1 / L2 lookup | FAISS `IndexFlatIP` over normalised vectors at the request's **live index size** | n=200 per index size |
| entity + lexical gate | spaCy NER span comparison + Jaccard/answer-type/subset guard (FreshCache's C18 L1 gate only) | n=200 |
| L3 processing | snapshot read + decode + exponential temporal gate, per page | n=200 |
| **search API** | **live Serper calls**, `collect.py`'s exact call shape | n=40 ok |
| **page fetching** | **live cold GETs**, `collect.py` headers and 20 s timeout | n=53 ok |
| LLM generation | Llama-3.2-3B-Instruct, batch 1, greedy, `max_new_tokens=80`, the audit's FORCED prompt on real stored contexts | n=120 |

Operation counts come from a replay whose aggregates and entire per-request
outcome sequence were **asserted identical to `mixed_engine.replay`** for every
policy, on the primary heterogeneous-age workload (31,201 requests,
`zipf_uniform`, seed 42). FreshCache reproduces its published profile exactly
(11,628 searches = 62.73 % savings, 1,175 L1, 18,398 L2).

Totals are formed by drawing each operation's cost from its measured sample pool
per request (seeded, deterministic), so the p95 of the total reflects real
component variance rather than an unrealistic sum of per-component p95s.

## 2. Measured component costs vs the constants the simulator assumes

| Operation | Measured | Simulator constant | Ratio |
|---|---|---|---|
| search API | p50 **1,546 ms** (mean 1,579, p95 1,969) | 500 ms | **3.1x slower** |
| page fetch | p50 **953 ms** (mean 1,254, p95 2,648) | 800 ms | 1.2x slower |
| LLM generation | p50 **341 ms** (mean 485, p95 1,136), 33 new tokens | 2000 ms | **5.9x faster** |
| L1/L2 lookup | p50 5.75 ms at 31k entries, plus 4.04 ms entity + 0.0091 ms lexical | 5 ms | ~2x slower at full index |
| L3 lookup | p50 0.023 ms | 5 ms | ~200x faster |
| query embedding | p50 **11.14 ms** | **absent from the model** | — |

The live fetch figure independently reproduces the project's June measurement
(`data/live_benchmark_summary.json`, cold p50 926.9 ms, n=203) at
953 ms, which is a useful cross-check that the
sample is not anomalous.

ANN lookup scales with cache size:

| Index entries | p50 (ms) | p95 (ms) |
|---|---|---|
| 1,000 | 0.1415 | 0.1702 |
| 5,000 | 0.6814 | 0.6867 |
| 10,000 | 1.3723 | 1.3883 |
| 20,000 | 3.2844 | 3.3952 |
| 31,000 | 5.7545 | 5.8741 |

## 3. End-to-end latency

| Policy | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|
| NoCache (fresh pipeline) | 3,940.4 | 3,417.4 | 7,860.9 | 11,896.6 |
| SemanticTTL k=1/16 | 328.8 | 11.4 | 3,056.2 | 6,371.6 |
| FreshCache L2+L3 (no L1) | 1,534.0 | 795.6 | 4,621.7 | 8,800.3 |
| FreshCache (full) | 1,514.9 | 759.7 | 4,624.1 | 8,320.3 |

## 4. Per-component breakdown

**NoCache (fresh pipeline)** — mean 3,940.4 ms, p50 3,417.4 ms, p95 7,860.9 ms

| Component | Mean (ms) | p50 (ms) | p95 (ms) | Mean share % |
|---|---|---|---|---|
| query embedding | — | — | — | — |
| L1 lookup | — | — | — | — |
| L2 lookup | — | — | — | — |
| L3 processing | — | — | — | — |
| search API | 1,580.56 | 1,540.12 | 2,172.95 | 40.1 |
| page fetching | 1,873.47 | 1,356.85 | 5,587.24 | 47.5 |
| LLM generation | 486.38 | 341.07 | 1,138.38 | 12.3 |

**SemanticTTL k=1/16** — mean 328.8 ms, p50 11.4 ms, p95 3,056.2 ms

| Component | Mean (ms) | p50 (ms) | p95 (ms) | Mean share % |
|---|---|---|---|---|
| query embedding | 11.06 | 11.14 | 11.46 | 3.4 |
| L1 lookup | 0.21 | 0.16 | 0.36 | 0.1 |
| L2 lookup | — | — | — | — |
| L3 processing | — | — | — | — |
| search API | 117.20 | 0.00 | 1,446.00 | 35.6 |
| page fetching | 164.60 | 0.00 | 993.95 | 50.1 |
| LLM generation | 35.73 | 0.00 | 295.13 | 10.9 |

**FreshCache L2+L3 (no L1)** — mean 1,534.0 ms, p50 795.6 ms, p95 4,621.7 ms

| Component | Mean (ms) | p50 (ms) | p95 (ms) | Mean share % |
|---|---|---|---|---|
| query embedding | 11.06 | 11.14 | 11.46 | 0.7 |
| L1 lookup | — | — | — | — |
| L2 lookup | 2.65 | 2.54 | 5.45 | 0.2 |
| L3 processing | 0.03 | 0.02 | 0.08 | 0.0 |
| search API | 587.72 | 0.00 | 1,842.34 | 38.3 |
| page fetching | 446.60 | 0.00 | 2,502.30 | 29.1 |
| LLM generation | 485.99 | 341.07 | 1,136.45 | 31.7 |

**FreshCache (full)** — mean 1,514.9 ms, p50 759.7 ms, p95 4,624.1 ms

| Component | Mean (ms) | p50 (ms) | p95 (ms) | Mean share % |
|---|---|---|---|---|
| query embedding | 11.06 | 11.14 | 11.46 | 0.7 |
| L1 lookup | 6.60 | 6.54 | 9.57 | 0.4 |
| L2 lookup | 2.43 | 2.31 | 5.22 | 0.2 |
| L3 processing | 0.03 | 0.02 | 0.07 | 0.0 |
| search API | 587.29 | 0.00 | 1,842.34 | 38.8 |
| page fetching | 438.98 | 0.00 | 2,453.99 | 29.0 |
| LLM generation | 468.47 | 336.65 | 1,136.45 | 30.9 |

## 5. Operation counts over the 31,201-request workload

| Policy | Embeddings | L1 lookups | L2 lookups | L1 hits | L2 hits | L3 lookups | Search calls | Page fetches | Generations |
|---|---|---|---|---|---|---|---|---|---|
| NoCache (fresh pipeline) | 0 | 0 | 0 | 0 | 0 | 0 | 31,201 | 46,703 | 31,201 |
| SemanticTTL k=1/16 | 30,847 | 30,847 | 0 | 28,879 | 0 | 0 | 2,322 | 4,113 | 2,322 |
| FreshCache L2+L3 (no L1) | 30,847 | 0 | 30,847 | 0 | 19,578 | 35,833 | 11,623 | 10,961 | 31,201 |
| FreshCache (full) | 30,847 | 30,847 | 29,672 | 1,175 | 18,398 | 33,907 | 11,628 | 10,959 | 30,026 |

## 6. What the numbers say

- **FreshCache cuts mean request latency by 62 %** against the fresh
  pipeline (1,515 ms vs
  3,940 ms) and p50 by
  78 %
  (760 ms vs
  3,417 ms). The saving comes from
  avoiding search and fetch on L2 hits, not from cheaper lookups.
- **Cache machinery is negligible.** Embedding + L1 + L2 + L3 together are
  20.1 ms
  of FreshCache's 1,515 ms mean
  (1.3 %).
  Latency is dominated by search (39 %),
  fetching (29 %) and
  generation (31 %).
- **L1 contributes almost nothing to latency.** Full FreshCache vs L2+L3-only
  differs by 19.2 ms
  in the mean (1.3 %)
  — consistent with the existing no-L1 ablation, where L1 added ~nothing to
  search savings while accounting for 1,175 avoided generations.
- **SemanticTTL k=1/16 is by far the fastest** (p50
  11.4 ms) because it
  answers 28,879 of 31,201
  requests from a stored answer, skipping search, fetch **and** generation
  entirely. Its p50 is essentially just an embedding plus an index probe.
  **This speed is not free**: the held-out answer audit measured 80.3 %
  conditional degradation for this exact configuration against 8.2 % for
  FreshCache, so the latency ranking and the answer-quality ranking point in
  opposite directions. Latency alone should not be used to prefer it.
- **The p95 tail is dominated by page fetching**, which is heavy-tailed
  (max 8,744 ms observed live). Policies that
  fetch fewer pages have shorter tails.

## 7. Limitations

- One hardware configuration: one A6000-class GPU on a **shared, contended**
  machine, local models, in-process index. Generation and embedding timings would
  differ on dedicated hardware or a hosted endpoint. During this run other users'
  jobs occupied most GPUs; measurements were pinned to a device that was free.
- Network legs were sampled at one time of day from one location:
  40 searches and 53 successful
  fetches, stratified by freshness class. Serper latency in particular is a
  single-provider, single-session figure.
- L3 processing was measured against a **warm OS page cache**, which models an
  in-memory/SSD content cache; cold storage would be slower.
- The entity/lexical gate is charged **once per L1 scan**. The engine evaluates it
  per candidate until one is accepted, so this is a lower bound on that component
  (it is 0.4 % of FreshCache's mean, so the effect on totals is small).
- Latency is composed from measured per-operation costs, not captured by
  instrumenting one continuous serving process; queueing, connection reuse and
  concurrency effects are not modelled.
- Totals assume operations are **serial**. A deployment that fetches pages
  concurrently would cut the fetch component substantially for multi-URL requests.
