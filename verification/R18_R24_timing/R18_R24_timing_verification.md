# R18 / R24: timing implementation and measurements

Audit date: 2026-09-24. Read-only: no manuscript, code, data or existing output was modified; no live Web or search call was made. Deliverables in this folder: `R18_R24_timing_verification.md`, `direct_timing_recompute.csv`, `class_reweighting.csv`, `stage_pool_inventory.csv`, `monte_carlo_recompute.json`, plus `timing_recompute.py`, `timing_results.json`, `code_excerpts.md`, `run.log`.

Tags: **[EXECUTED]** recomputed now from saved records; **[RECORD]** existing study file; **[UNVERIFIED]** not checkable from local files.

## 1. Timing artifacts

| what | where | date |
|---|---|---|
| 40-request direct wall-clock run (Table 17 source) | `validation/latency_e2e_direct/` : `prep_sample.py`, `run_direct.py`, `compare.py`, `direct_sample_spec.json`, `direct_per_request.csv` (160 rows), `direct_results.json`, `direct_vs_estimate.json`, `direct_latency_report.md` | 2026-09-18 01:30–01:43 |
| Monte-Carlo composition (3,940.4 / 1,514.9) and stage pools | `validation/latency_e2e/` : `stage_a_opcounts.py` (+ `opcounts_<policy>.csv`), `stage_b_gpu.py`, `stage_c1_ann.py`, `stage_c2_gates.py`, `stage_d_live.py`, `stage_e_compose.py`, `*_timings.json`, `stage_d_live_raw.jsonl`, `latency_results.json`, `per_request_latency.csv` | 2026-09-18 01:05–01:25 |
| 300-request direct run (the "withdrawn" 3,697.8 / 981.1) | `validation_V3/remaining_critical_issues/08_latency/` : `prep_latency.py`, `run_latency.py`, `summarize.py`, `direct_sample_spec.json`, `direct_per_request.csv` (1,800 rows), `direct_results.json`, `latency_summary.json/.csv`, log `../logs/08_latency.log` | 2026-09-21 12:12–13:21 |
| Manuscript-facing copy | `Reframe_Paper_Pro/artifacts/latency/`: `direct_results.json` and `direct_vs_estimate.json`, byte-identical to the 40-run files [EXECUTED sha256] | 2026-09-23 |

**The 40 request IDs** (`direct_sample_spec.json["sample_query_ids"]`, listed in `timing_results.json`) [EXECUTED]: p_q_000040_0, p_q_000142_1, p_q_000331_1, p_q_000331_3, p_q_0b7949bb_0, p_q_13071539_2, p_q_177880b1_0, p_q_1ef7404e_2, p_q_26309cb6_1, p_q_35a57f29_2, p_q_3704b8f2_3, p_q_43affc2b_0, p_q_4d556954_1, p_q_57755f05_2, p_q_5e007db1_0, p_q_6566e4ed_3, p_q_6e543930_3, p_q_75db3e8b_1, p_q_7b820346_0, p_q_813d0e0a_2, p_q_86cff9f1_0, p_q_89f7697b_3, p_q_8b7e079f_3, p_q_90168b95_3, p_q_91bc1860_3, p_q_99ec603a_3, p_q_a0cf70fb_0, p_q_a377fb63_0, p_q_a3aad92f_1, p_q_a7cec67e_0, p_q_b148b3f7_1, p_q_b9a75c82_3, p_q_bc882466_0, p_q_c17d761a_1, p_q_d79c71e4_2, p_q_d9e3bc12_2, p_q_dc84c82b_0, p_q_ef7d80f1_0, p_q_ef7d80f1_2, p_q_fea8dfeb_0.

- Same 40 IDs, same order, for all four policies (160 trace rows) [EXECUTED].
- Eight per class (TIMELESS, SLOW, MEDIUM, FAST, REAL_TIME) [EXECUTED].
- **Split: NOT all held-out.** 27 of the 40 fall in test clusters and 13 in validation clusters of `split.json` [EXECUTED]. `prep_sample.py:181-200` samples from the second half of the full 31,201-request stream without consulting the split. The manuscript's "40 held-out requests" (main text §Discussion and Limitations) is therefore wrong; the appendix's "eight per class from the seed-42 stream" is accurate. (The 300-request run did restrict to test clusters, `prep_latency.py:182-205`.)
- Profile-replay file: `direct_sample_spec.json["specs"][policy]` gives, per request, `path` (L1 / L2 / miss / realtime_bypass / nocache), `served` (per URL, `from_cache` flag), `l1_index`, `l2_index`, `matched_query`; produced by `prep_sample.py:profile`, which mirrors `mixed_engine.replay` and asserts equal aggregates [RECORD].

## 2. Direct wall-clock recompute [EXECUTED from `direct_per_request.csv`]

| policy | N | mean | p50 | p95 | manuscript |
|---|---|---|---|---|---|
| NoCache | 40 | 3,255.34 | 3,014.44 | 5,050.33 | 3,255.3 / 3,014.4 / 5,050.3 |
| FreshCache | 40 | 1,214.43 | 563.85 | 3,107.32 | 1,214.4 / 563.8 / 3,107.3 |
| FreshCache without L1 | 40 | 1,238.17 | 742.99 | 3,558.27 | 1,238.2 / 743.0 / 3,558.3 |
| SemanticTTL (θ=0.40, κ=1/16) | 40 | 703.26 | 13.51 | 3,832.47 | 703.3 / 13.5 / 3,832.5 |

All twelve values match. **Percentile convention** (`run_direct.py:70-72`, `compare.py:23-27`): sort ascending; take element at index round(p·(n−1)), so with n = 40 the "p50" is the 21st value (upper middle, not the mean of the two middle values; numpy's median differs, e.g. FreshCache without L1 655.7 vs 743.0) and the "p95" is the 38th value (numpy linear interpolation would give e.g. 3,148.6 for FreshCache). The manuscript should call the p50 column this rank statistic, or use the conventional median.

Trace integrity [EXECUTED]: every miss/nocache/REAL_TIME row has a positive search time (40 / 16 / 15 / 9 timed live searches), every fetched page a positive fetch time, and every generated answer has non-zero length. `run_direct.py:201-207, 223-230` swallows HTTP exceptions and keeps the elapsed time, and never retries, drops or replaces a request; whether any of the 163 fetches returned non-200 is not recorded [UNVERIFIED]. No run log exists for the 40-request run (only the 300-request run has `08_latency.log`) [EXECUTED search].

## 3. Timing implementation [EXECUTED from code; excerpts in `code_excerpts.md`]

| claim | evidence |
|---|---|
| FAISS `IndexFlatIP` | `run_direct.py:139` |
| L2-normalised BGE-M3 vectors | `:121-123` loads `data/query_embeddings_bgem3.npy`, `faiss.normalize_L2`; the query vector is encoded with `normalize_embeddings=True` (`:170-171`) |
| next power of two, minimum 1,024 | `:136-137`: `b = 1 << (n-1).bit_length(); b = max(1024, min(b, len(vecs)))` |
| populated with the first n vectors | `:140` `ix.add(vecs[:b])` |
| k = 5 | `:150, :180, :196` |
| does not decide admission; route imported | the search result is discarded; `hit = (r["path"] == "L1")` at `:186`, L2/miss branches keyed on `r["path"]` (`:193-199`); the C18 gates run on `r["matched_query"]` for timing only (`:181-184`) |

Timed downstream path per request (`:168-253`): BGE-M3 query embedding (skipped for NoCache and REAL_TIME, `:168`); L1 FAISS lookup plus entity and semantic-equivalence gates for FreshCache; stop with the stored answer on an L1 route; L2 FAISS lookup; live Serper search on replay-designated miss/nocache/REAL_TIME routes; per served URL either a local snapshot read plus temporal-gate call (from-cache) or a live cold GET; Llama-3.2-3B generation, greedy, 80 tokens, with `torch.cuda.synchronize()` before stopping the timer. The FAISS contents are the first n benchmark embeddings, not the request-specific replay cache; the study says so itself (`direct_latency_report.md` §5) and this audit makes no such claim.

## 4. Warm-up, hardware, execution

| item | evidence | status |
|---|---|---|
| one GPU, generator resident | `--gpu` pins `CUDA_VISIBLE_DEVICES` (`:33-38`); `device_map="cuda:0"` (`:117-118`) | [EXECUTED code] |
| that GPU is an A6000 | not recorded in any 40-run output; this host currently exposes eight RTX A6000 GPUs (`nvidia-smi -L` now); the Monte-Carlo stage recorded `gpu: 4, host: node1` | [UNVERIFIED] for the run itself |
| FAISS on CPU | `faiss.IndexFlatIP` built in the main process without any GPU resource (`:139-141`) | [EXECUTED code] |
| sequential, no concurrency | single process, nested `for pol … for r` loops (`:159-163`) | [EXECUTED code] |
| campus network | no record | [UNVERIFIED] |
| warm-up discarded | three untimed queries exercise the encoder, a 4,096-entry index and the generator before the timed loop (`:147-156`) | [EXECUTED code] |
| failures / retries / drops | none possible by construction (exceptions swallowed, no retry, no drop); non-200 responses not logged | partly [UNVERIFIED] |

## 5. Class-mix reweighting [EXECUTED; `class_reweighting.csv`]

Weights (`compare.py:16-17`, `:43`): W = {TIMELESS 6,929, SLOW 8,137, MEDIUM 7,626, FAST 8,155, REAL_TIME 354}, total 31,201 (replay request counts). Each request i of class c gets w_i = (W_c/31,201)/(n_c/40) with n_c = 8, so the weighted mean equals Σ_c (W_c/31,201)·(class mean_c). Results: NoCache **3,271.38 ms**, FreshCache **885.93 ms**, FreshCache without L1 852.89, SemanticTTL 142.67. The manuscript's 3,271 and 886 confirmed. The REAL_TIME weight is 0.0567 against a sample share of 0.20, which is why SemanticTTL drops from 703 to 143 ms.

## 6. Stage pools (`stage_pool_inventory.csv`) [EXECUTED counts]

| stage | file | n | units | sampled from |
|---|---|---|---|---|
| embedding | `stage_b_gpu_timings.json[embed]` | 200 | ms, BGE-M3 batch 1 GPU | separate microbenchmark (`stage_b_gpu.py`) |
| generation | `stage_b_gpu_timings.json[generate]` | 120 | ms, Llama-3.2-3B greedy 80 tokens (mean 33 new tokens) | separate microbenchmark |
| L1/L2 lookup | `stage_c1_ann_timings.json[ann][1000|5000|10000|20000|31000]` | 200 per size | ms, IndexFlatIP k=5 | separate microbenchmark (`stage_c1_ann.py`, 20 warm-ups) |
| entity gate, lexical gate | `stage_c2_gate_timings.json` | 200 each | ms, CPU | separate microbenchmark |
| L3 processing | `stage_c2_gate_timings.json[l3_lookup]` | 200 | ms per reused page | separate microbenchmark |
| search | `stage_d_live_raw.jsonl` kind=search ok | 40 (40 attempted) | ms per live Serper call | separate live microbenchmark (`stage_d_live.py`) |
| fetch | `stage_d_live_raw.jsonl` kind=fetch ok | 53 (60 attempted, 7 failures excluded) | ms per live cold GET | separate live microbenchmark |

**None of the pools comes from the 40-request direct run.** The manuscript's "draws stage times from the measured pools of the direct run" (Appendix F, "The estimate") is wrong: the pools are the stage B/C/D microbenchmarks of `validation/latency_e2e/`, timed on 2026-09-18 01:05–01:22, before the direct run.

## 7. Monte-Carlo composition [EXECUTED; `monte_carlo_recompute.json`]

Transcribing `stage_e_compose.py` with `numpy.random.default_rng(42)`, one draw per operation, a single pass (no replicates), over the 31,201 requests of each `opcounts_<policy>.csv` (route and operation counts fixed by the stage-A replay, asserted to reproduce `mixed_engine.replay`):

| policy | N | mean | p50 | p95 | recorded mean |
|---|---|---|---|---|---|
| NoCache | 31,201 | **3,940.41** | 3,417.35 | 7,860.86 | 3,940.4108 |
| FreshCache | 31,201 | **1,514.86** | 759.68 | 4,624.15 | 1,514.8600 |
| FreshCache without L1 | 31,201 | 1,534.05 | 795.58 | 4,621.73 | 1,534.0470 |
| SemanticTTL κ=1/16 | 31,201 | 328.80 | 11.36 | 3,056.17 | 328.7957 |

All means reproduce to 1e-6. Operation counts for FreshCache: 11,628 searches, 1,175 L1 hits, 18,398 L2 hits, 10,895 fetches. This is an operation-cost composition (each operation priced from a pool, summed per request), not a second wall-clock experiment; no network or GPU call occurs in `stage_e_compose.py`.

## 8. Assumed constants (500 / 800 / 150 / 2,000 ms) [EXECUTED grep]

Defined once: `experiment.py:79-88` (`LATENCY` dict, plus 5 ms lookups and `P95_FETCH_MULT`), used only by the legacy simulator's cost model (`experiment.py:592-597, 656-669, 731-745`). In the timing code they appear only as labels: `stage_d_live.py:179-180` records `assumed_search_ms/assumed_fetch_ms` for comparison and `stage_e_compose.py:182-185` writes them under `simulator_constants_for_reference`; neither value is read into any draw. They enter none of Table 17, the class-reweighted means, or the 3,940.4 / 1,514.9 composition. The manuscript's Table 21 statement is correct.

## 9. The "withdrawn" 300-request result

The change record (Table 18) and the Limitations text say the 300-request 3,697.8 → 981.1 ms figure "traced to no artifact and is withdrawn" and that the 40-request run "is the only direct measurement". **Both statements are false** [EXECUTED]:
- `validation_V3/remaining_critical_issues/08_latency/direct_results.json` records NoCache mean 3,697.79 / p50 2,858.5 / p95 7,095.8 and FreshCache 981.09 / 668.3 / 2,560.0 over 300 requests, with 1,800 per-request trace rows, a sample spec restricted to held-out test clusters, six policies, and a run log (`08_latency.log`, "warm-up done; starting timed runs").
- Dated 2026-09-21 13:20, three days after the 40-request run.
- Its files are the ones listed in the release manifest (`release/MANIFEST.json:942-946, 973, 1109-1111`) and present in `release/bundle/artifacts/validation_V3/remaining_critical_issues/08_latency/`.
The 300-request result is not used in any current manuscript calculation (Table 17, the reweighting and the composition all derive from the 40-run and `latency_e2e` files) [EXECUTED], but it has a complete artifact and the stated reason for withdrawal is not true. The two runs also disagree materially (FreshCache mean 981 vs 1,214 ms; NoCache 3,698 vs 3,255), which the manuscript does not discuss.

## 10. Release completeness [EXECUTED listings]

| required item | in `Reframe_Paper_Pro/artifacts/latency/` | in release bundle / MANIFEST |
|---|---|---|
| 40 request IDs (`direct_sample_spec.json`) | no | no |
| policy-route / profile records (`specs`) | no | no |
| per-request direct traces (`direct_per_request.csv`, 160 rows) | no | no (the bundle's file of that name is the 300-run, 1,800 rows) |
| measured stage pools (`stage_*_timings.json`, `stage_d_live_raw.jsonl`) | no | no |
| direct-summary script (`run_direct.py`) | no | no (bundle ships the 300-run scripts) |
| class-reweighting script (`compare.py`) | no | no |
| Monte-Carlo script (`stage_e_compose.py`) and opcounts | no | no |
| configuration / seed records | only inside `direct_results.json` (seed 42) | no |
| run log for the 40-run | none exists | — |
| manifest entries under `artifacts/latency/` | — | 0 |

Only `direct_results.json` and `direct_vs_estimate.json` of the 40-run are staged, and they are absent from the manifest. The appendix sentence "Per-request timing traces and the stage pools ship under `artifacts/latency/`" is not met.

## 11. Summary of manuscript corrections needed

1. "40 held-out requests" → 27 test and 13 validation clusters; say "40 requests from the seed-42 stream, eight per class".
2. "draws stage times from the measured pools of the direct run" → pools are separate microbenchmarks (stage B/C/D), not the direct run.
3. Change record and Limitations: the 300-request figure has a full artifact (in the release bundle); withdraw it for a stated methodological reason or reinstate it, but not "no artifact".
4. State the p50/p95 rank convention or report conventional medians.
5. Ship the 40-run spec, traces, pools and scripts under `artifacts/latency/` with manifest entries, or change the sentence.
6. A6000 and campus-network claims are not recorded for the 40-run; either add the record or soften.
