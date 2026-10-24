# FreshCache: Selective Intermediate-Artifact Reuse for Open-Web Retrieval-Augmented LLMs

This repository contains the implementation of FreshCache, the replay harness, every baseline and control, and the code of every
experiment, ablation and evaluation reported in the paper, together with the dataset-construction scripts and a small data sample.
It contains no results: all answers, judgments, metrics and tables are regenerated into `outputs/` (git-ignored).

* [1. Method](#1-method) · [2. Repository layout](#2-repository-layout) · [3. Setup](#3-setup) · [4. Data](#4-data)
* [5. Quick start on the sample](#5-quick-start-on-the-sample) · [6. Configuration](#6-configuration) · [7. Baselines and controls](#7-baselines-and-controls)
* [8. Reproducing the paper](#8-reproducing-the-paper) · [9. Paper-to-code map](#9-paper-to-code-map) · [10. Reproducibility notes](#10-reproducibility-notes)

---

## 1. Method

FreshCache stores the three intermediate artifacts of an open-web RAG request separately and reuses each one only when its own matching
rule and its own age rule both pass:

| Tier | Artifact | Matching | Age eligibility |
|---|---|---|---|
| **L1** | generated answer with its evidence manifest | cosine ≥ 0.80, content-word Jaccard ≥ 0.30, no strict subset, answer-type and entity agreement | ε = 0.10, m = 1.5, age from the oldest supporting page |
| **L2** | ranked URL list | query cosine ≥ θ_L2 (0.75 standalone; 0.80 in composition D) | ε = 0.20, m = 1.2, age from the original list discovery |
| **L3** | page content | exact URL | ε = 0.35, m = 1.0, age from the last successful fetch |

An artifact of age Δt is eligible while Δt ≤ T = −h·ln(1 − ε)/(m·ln 2), where h is the half-life of the freshness class
(TIMELESS, SLOW, MEDIUM, FAST, REAL_TIME). REAL_TIME requests bypass the cache. An L1 miss can still reuse an eligible URL list; each page of
that list is then checked independently in L3 and refetched when expired.

The evaluation replays 31,201 requests over seven simulated days (seed 42), split cluster-disjointly into 9,353 validation and 21,848
held-out requests. Compositions place a CategoryAware-style answer cache above FreshCache's lower tiers: **A** answer cache only,
**B** A + L3 pages, **C** A + L2 lists, **D** A + L2/L3; the write controls **D′**, **B⁻** and **B⁻ₑ** suppress answer writes derived from
reused lists (D′, B⁻ at the same request IDs; B⁻ₑ deployably, from its own registry).

---

## 2. Repository layout

```
freshcache/                       core library
  __init__.py, l1_cache.py, l2_cache.py, l3_cache.py, risk_model.py, ...   the three cache tiers as a library
  method/                         decision logic: tiered_cache_model.py (L1 checks, risk model, BGE-M3 embeddings, base replay),
                                  freshness_calibration.py (half-lives), l1_admission_gate.py, l1_structural_guards.py, l2_equivalence_fit.py
  replay_engine/                  replay engine of every reported number (mixed_age_replay, method_replay, execution_semantics, ...)
  selection_replay_engine/        replay engine of the frozen validation-selection stage and the audit samples
  keyset_replay_engine/           key-set engine, loaded as `keyset` only to rebuild the answer keys of the earlier audit samples
  simulation/                     request schedules, page-version replay, reference and step-wise online replays
  composition/                    CategoryAware-style answer cache composed with FreshCache L2/L3 and its write/admission controls
scripts/
  data/                           dataset construction: queries, paraphrases, BGE-M3 embeddings (query_embeddings.py), web panel,
                                  change detection, split, gold answers, statistics, page re-fetch (page_refetch.py)
  baselines/                      gptcache, scalm, vcache, matched_admission, exact_ttl_and_l1_only, semantic_ttl_category_aware,
                                  semantic_ttl_frontier, category_aware_frontier (runs, comparator grids, operating-point selection)
  experiments/                    main, ablations, operating_points, admission_controls, unseen_queries, live_streams,
                                  evaluation_sensitivity, cost_and_latency, workload_locality, calibration_and_measurement
  evaluation/                     generation and LLM judging, answer audits, reference validity, gold verification, generator
                                  replication, tier quality, external-judge agreement
  analysis/                       check_config_consistency.py
  prepare_output_dirs.py, run_sample_quickstart.sh
configs/                          freshcache/, baselines/, experiments/ (YAML), protocols/ (definitions only), integrity/, output_dirs.txt
data/                             README.md, LICENSE, sample/
run_all.sh                        every experiment and baseline, grouped by paper section
```

Every script prints its usage with `--help`.

---

## 3. Setup

```bash
conda env create -f environment.yml && conda activate freshcache     # or: python3.10 -m venv .venv && pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env                                                 # set API keys and, if needed, DATA_DIR / OUTPUT_DIR
```

All commands run from the repository root. Paths come from three variables: `FRESHCACHE_ROOT` (repository root, default the current
directory), `DATA_DIR` (dataset, default `data/`) and `OUTPUT_DIR` (results, default `outputs/`). Credentials are read from the
environment only: `SERPER_API_KEY` (web search; `SERPER_KEY_1`/`SERPER_KEY_2` for the QReCC collection) and `HF_TOKEN` (gated models).
Models are loaded from the Hugging Face Hub by name: `BAAI/bge-m3`, `meta-llama/Llama-3.2-3B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`,
`Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-14B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`, `microsoft/phi-4`,
`humarin/chatgpt_paraphraser_on_T5_base`. GPTCache is the official `gptcache==0.1.44` package from PyPI (not vendored).

---

## 4. Data

This repository includes a representative sample of FreshCache-Bench (data/sample/) for running the pipeline end to end. Third-party Web page content is referenced by URLs and content hashes rather than redistributed.

`data/sample/` holds 60 whole clusters with 60 requests per freshness class: questions, paraphrases, classes, splits, cluster IDs, arrival
times, references, and URL lists with content hashes and observation timestamps. The BGE-M3 embeddings and similarity matrix are generated
by `scripts/data/query_embeddings.py`. Stages that read page text use `scripts/data/page_refetch.py`, which re-fetches pages by URL and
checks them against the stored content hashes; results on re-fetched pages may differ from the paper because of web content drift.
`data/README.md` documents the schema and the source datasets, `data/LICENSE` the data terms. The dataset-construction block of
`run_all.sh` builds the same file layout from the source datasets and a new web collection.

A new web collection has five rounds: `run_00` and re-collections 1 h, 12 h, 24 h and 7 days later. `run_all.sh` runs only `run_00`;
start each later round separately at its delay after `run_00`, e.g. `LIVE=1 python scripts/data/web_collection.py --run-id rerun_1h`
(likewise `rerun_12h`, `rerun_24h`, `rerun_7d`). `run_all.sh` stops before `scripts/data/change_detection.py`, with a message naming the
missing rounds, until all five rounds are in `url_manifest.jsonl`.

---

## 5. Quick start on the sample

```bash
bash scripts/run_sample_quickstart.sh [--limit N]    # results in outputs/sample/
```

What the quickstart covers, in order:

1. Output directories and the configuration consistency check.
2. BGE-M3 query embeddings and similarity matrix of the sample (`scripts/data/query_embeddings.py`; the model is downloaded from the
   Hugging Face Hub on first use). CPU is enough.
3. Benchmark statistics, the validation/held-out split and the freshness-class predictor (CPU).
4. The standalone FreshCache replay with its internal controls and tier ablations (age gate off, EquivalentTTL, without L1, L2-only,
   SCALM-style, SemanticTTL, TieredFixedTTL), the supplementary replays (L3-only, L1-only, predicted classes, SemanticTTL points) and the
   ExactTTL and vCache-style replays (CPU, a few minutes).
5. Only with a CUDA GPU and page text (`REFETCH=1`, below): answer generation with Llama-3.2-3B-Instruct and the paper's two-judge
   correctness and served-evidence sufficiency judging with Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct, on `--limit` reference-bearing
   held-out requests (default 50; `scripts/evaluation/sample_answer_quality.py`, `scripts/evaluation/generation_judging_stage.py`).
   Results: `outputs/sample/sample_answer_quality/results.json` (fresh and FreshCache labels, WAI, sufficiency). The Llama models are
   gated on the Hugging Face Hub, so an authorised Hugging Face login is needed. Without a GPU, or without page text, this step is skipped
   with a message and the rest of the quickstart still runs.

Not covered by the quickstart: the dataset construction from the source datasets, the live-web studies, the audit samples and every
other GPU stage of `run_all.sh`. Sample numbers are not the paper's.

The sample carries URLs and content hashes, not page text. To run it on the pages as they are today:

```bash
REFETCH=1 bash scripts/run_sample_quickstart.sh   # web access, no API key; results in outputs/sample/
```

This first runs `scripts/data/page_refetch.py`, which downloads every page of the sample's baseline round, compares it with the stored
content hash and reports UNCHANGED, CHANGED, UNAVAILABLE or NO_STORED_HASH pages (`outputs/sample/page_refetch/refetch_report.csv`), then
runs all stages above on a data directory that holds the re-fetched pages (`outputs/sample/refetched_data`). The replays decide reuse from
the recorded content hashes, so their operation counts do not change; the page text is read by the generation and judging stages.

---

## 6. Configuration

`configs/freshcache/`, `configs/baselines/` and `configs/experiments/` record every parameter of the reported runs (thresholds, TTL budgets
and multipliers, half-lives, grids, schedule, seed and split, models, prompts, decoding and context budgets, gate and verifier settings).
Each entry names the code location that uses it; `python scripts/analysis/check_config_consistency.py` verifies that configuration and code
agree (first step of `run_all.sh`). `configs/protocols/` holds the protocols and pre-registrations frozen before the corresponding runs.
Values selected on validation data at run time are written to `outputs/` by the selection stages and read from there.

---

## 7. Baselines and controls

All baselines run under the same replay, operation accounting and cost model as FreshCache.

| Baseline / control | Implementation | Notes |
|---|---|---|
| NoCache | `freshcache/replay_engine/method_replay.py` (`NoCache`) | every request searches, fetches and generates |
| GPTCache | `scripts/baselines/gptcache/` | official `gptcache==0.1.44` package with BGE-M3 embeddings; native evaluator and FreshCache-matched admission; only the adapter (`gptcache_adapter.py`) is ours |
| SCALM-style | `method_replay.py` (`SCALM`), `scripts/baselines/scalm/`, `scripts/baselines/matched_admission/` | our reimplementation of SCALM's admission policy, not the complete system; omits SCALM's hierarchical cache structure and eviction |
| CategoryAware-style | `scripts/baselines/category_aware_frontier/`, `scripts/baselines/semantic_ttl_category_aware/`, `freshcache/composition/` | our reimplementation of the per-category threshold and TTL policy of category-aware semantic caching, not the complete system; omits quotas, the storage tier, indexing and load adaptation |
| vCache-style | `scripts/baselines/vcache/` | our reimplementation of vCache's per-entry learned-threshold policy; uses proxy correctness feedback and does not test the published error guarantee |
| ExactTTL, L1Only | `scripts/baselines/exact_ttl_and_l1_only/`, `scripts/experiments/main/exact_ttl_vcache_replay.py` | internal controls |
| SemanticTTL, TieredFixedTTL, EquivalentTTL | `method_replay.py`, `mixed_age_replay.py`; `scripts/experiments/main/freshcache_replay.py` | internal controls |
| Without L1, L2Only, Age gate off | `freshcache_replay.py` (`FreshCache_NoL1`, `L2Only`, `FreshCache_AlwaysPassTemporal`) | tier ablations (Table 10) |
| Compositions A, B, C, D, D′, B⁻, B⁻ₑ | `scripts/experiments/main/composition_ablation_replay.py`, `scripts/experiments/ablations/reuse_write_suppression.py`, `scripts/experiments/main/write_control_traces.py` | Tables 4, 7 |
| Page-support verifier γ, learned L2 gate τ / τ₂, answer verifier AV, L2 floor 0.80 | `scripts/experiments/admission_controls/`, `scripts/experiments/live_streams/answer_verifier.py`, `scripts/experiments/operating_points/` | admission variants (Tables 6, 12, 14) |

The comparator grid (36 SemanticTTL settings, 20 CategoryAware-style settings), the quality-, relaxed- and cost-matched selection on
validation and the θ_L2 selection are `scripts/experiments/operating_points/operating_point_validation_replay.py`,
`operating_point_validation_selection.py`, `scripts/baselines/category_aware_frontier/` and `scripts/experiments/operating_points/l2_threshold_sweep_*.py`.
Gate and verifier training: `scripts/experiments/admission_controls/l2_gate_training.py`, `verifier_threshold_selection.py` and
`scripts/experiments/live_streams/answer_verifier.py` (`train`).

---

## 8. Reproducing the paper

```bash
bash run_all.sh                # dataset in DATA_DIR; GPU=0 selects the GPU; LIVE=1 enables the live-web stages
```

`run_all.sh` runs every stage, grouped by paper section and ordered by data dependencies (sections are re-opened as
"(continued)" where a step needs results of a later section): dataset construction; calibration and measurement; the audit samples and reference
validity; standalone FreshCache and baselines (Tables 5, 10); operating points and admission controls (Table 6); compositions, write
controls and the common sample (Tables 4, 7, 9, 11); ablations, risk budgets and L1 diagnostics (Table 10, Appendices A-B); gold
verification, generator replication and tier diagnostics (Appendix B); unseen queries and deeper lists (Tables 12, 13); live streams and
complete policies (Tables 8, 9, 14); cost and latency, and workload locality (Section 5.1, Appendix D). Generation and judging stages
(`*_generation_judging.py`, `scripts/evaluation/generation_judging_stage.py`) process only items missing from their output pools, so a
stage can be resumed. Each script documents its modes (`--help`).

---

## 9. Paper-to-code map

| Paper item | Scripts |
|---|---|
| Method (Sec. 3, Fig. 1) | `freshcache/method/tiered_cache_model.py`, `freshcache/replay_engine/`, `freshcache/composition/`, `freshcache/` |
| Benchmark composition, half-lives (Table 2, Sec. 4.1, App. A) | `scripts/data/benchmark_statistics.py`, `freshcache/method/freshness_calibration.py`, `scripts/experiments/calibration_and_measurement/{half_life_refit,tier_calibration_targets}.py` |
| Settings (Table 3) | `scripts/evaluation/generation_judging_stage.py`, `scripts/experiments/unseen_queries/{generator_input_audit,deep_list_evaluation}.py`, `scripts/experiments/live_streams/{paraphrase_stream_scoring,qrecc_stream_scoring}.py` |
| Composition A-D, D′, B⁻, B⁻ₑ (Table 4, Sec. 5.2) | `scripts/experiments/main/composition_ablation_replay.py`, `scripts/experiments/ablations/reuse_write_suppression.py`, `scripts/experiments/main/write_control_traces.py`, `scripts/experiments/cost_and_latency/composition_cost_model.py` |
| Operation controls (Table 5) | `scripts/baselines/{gptcache,scalm,vcache,matched_admission}/`, `scripts/experiments/main/freshcache_replay.py` |
| Standalone operating points (Table 6) | `scripts/experiments/operating_points/`, `scripts/experiments/admission_controls/`, `scripts/baselines/{semantic_ttl_frontier,semantic_ttl_category_aware,category_aware_frontier}/` |
| Common sample (Table 7) | `scripts/experiments/main/{common_sample_construction,common_sample_scoring,loss_rate_denominators}.py` |
| Workloads, loss origins (Tables 8, 9) | `scripts/experiments/main/cross_workload_summary.py`, `scripts/experiments/live_streams/` |
| Standalone ablations (Table 10) | `scripts/experiments/main/{freshcache_replay,supplementary_replays}.py`, `scripts/baselines/exact_ttl_and_l1_only/` |
| Judging sensitivity (Table 11) | `scripts/experiments/evaluation_sensitivity/`, `scripts/evaluation/external_judge/`, `scripts/evaluation/reference_validation/model_adjudication_*.py` |
| Unseen queries, complete live policies (Table 12) | `scripts/experiments/unseen_queries/`, `scripts/evaluation/reference_validation/unseen_query_reference_inputs.py` |
| Prospective deeper lists (Table 13) | `scripts/experiments/unseen_queries/{deep_list_collection,deep_list_evaluation}.py` |
| Complete policies vs B (Table 14) | `scripts/experiments/live_streams/{admission_control_evaluation,answer_verifier,live_stream_admission_controls}.py`, `scripts/experiments/live_streams/recorded_stream_controls/` |
| Paraphrase and QReCC streams (Sec. 5.3, App. D) | `scripts/experiments/live_streams/` |
| Cost, break-even, live latency, monetary scenarios (Sec. 5.1, App. D) | `scripts/experiments/cost_and_latency/` |
| Locality, list provenance, MS MARCO, sub-streams, replications (Sec. 5.1, App. D) | `scripts/experiments/workload_locality/` (`strict_provenance_replay.py`: savings via lists sharing no URL with the own list), `scripts/experiments/ablations/composition_operation_replications.py` |
| Temporal eligibility, cross-round drift, density and sparsity (Sec. 5.1, App. D) | `scripts/experiments/ablations/{mixed_age_trace_construction,mixed_age_consistency_analysis,cross_round_drift_audit}.py`, `scripts/experiments/calibration_and_measurement/{observation_density_sensitivity,synthetic_sparsity_bias}.py` |
| L1 admission statistics, stricter L1 (App. A) | `scripts/experiments/calibration_and_measurement/l1_lexical_rule_audit.py`, `scripts/experiments/ablations/l1_precision_gate/`, `scripts/experiments/ablations/l1_validation/` |
| Parameter selection, class predictor (App. A) | `scripts/experiments/ablations/risk_budget_*.py`, `scripts/evaluation/answer_audits/risk_budget_*.py`, `scripts/experiments/calibration_and_measurement/freshness_class_prediction.py` |
| Observability and drift thresholds (App. B) | `scripts/experiments/calibration_and_measurement/{drift_observability_accounting,drift_length_threshold_sensitivity,observability_bounds,observability_recovery}.py` |
| Reference validity, screened subsets, 8B protocol, tier diagnostics (App. B) | `scripts/evaluation/{reference_validation,gold_verification,generator_replication,tier_quality}/`, `scripts/evaluation/answer_audits/answer_quality_evaluation.py` |
| Admission and training of γ, gate and AV (App. C) | `scripts/experiments/admission_controls/`, `scripts/experiments/live_streams/answer_verifier.py` |
| Holm families, WAI intervals | `scripts/experiments/operating_points/{holm_family_construction,holm_adjustment,two_test_holm_families,wai_confidence_intervals}.py`, `verifier_gamma_paired_tests.py`, `scripts/experiments/main/categoryaware_composition_paired_tests.py` |
| Paraphrase fidelity rates (App. D) | `scripts/experiments/live_streams/{paraphrase_equivalence_judging,paraphrase_equivalence_rates}.py` |

---

## 10. Reproducibility notes

* **Hardware.** The reported runs used NVIDIA A6000 GPUs (48 GB); the 14B models need one such GPU each.
* **Live components.** Web search, page fetching and the live streams depend on the state of the web at run time; a new collection
  reproduces the procedure, not the exact pages.
* **Page text and web drift.** Stages that read page text use `scripts/data/page_refetch.py`, which re-fetches pages by URL and checks
  them against the stored content hashes. Results on re-fetched pages may differ from the paper because of web content drift.
* **Integrity checks.** `configs/integrity/FILE_HASHES.csv` holds the hashes of the released code, configuration and sample data files only.
  Stages stop when released code or a protocol file differs from its recorded hash. Files in `OUTPUT_DIR` and `DATA_DIR` are not in the
  manifest; stages log them as "not in freeze" and continue, so the pipeline runs on `data/sample` and on re-fetched pages.
* **Reference values.** A value that one stage computes and later stages re-derive (for example FreshCache's held-out search savings, or the
  fresh-correct count of the held-out audit) is written by its single producing stage with `record()` to `OUTPUT_DIR/reference_values.json`;
  later stages only `check()` it (`scripts/analysis/reference_values.py`, which lists the producer of every value). A check whose value has
  not been recorded stops with "run <producing script> first". Other checks read the producing stage's own output file. No result numbers
  are written in the code.
* **Dataset sizes.** Size checks on loaded data (stream, split and audit-sample sizes) compare with `configs/data/dataset_sizes.yaml`
  (`scripts/analysis/dataset_sizes.py`). When `DATA_DIR` is the bundled sample (or a re-fetched copy of it) they are skipped with a
  message instead of failing.
* **Protocols and results.** `configs/protocols/` holds protocol definitions only (test lists, thresholds, criteria, tie-breaks). Every
  measured value is regenerated into `outputs/`: the protocol scripts write their instantiated pre-registrations there
  (`outputs/quality_selection/quality_selection_prereg.json`, `outputs/l1_precision/l1_precision_prereg.json`), and
  `holm_family_construction.py` builds the 24-test Holm family (`outputs/operating_point_summaries/holm_family_24.json`) that
  `holm_adjustment.py` checks. `l2_audit_excluded_requests.csv` lists the requests of an independent L2 retrieval audit that are excluded
  from verifier labelling.
* **External judge.** The blinded external-judge check exports batches (`external_judge_batch.py`); the replies are pasted by hand into
  `outputs/external_judge/chatgpt_batch/reply_XX.txt` before `external_judge_agreement.py`.
* **Driver details.** A few stages use fixed GPU ids or loop over GPU passes as in the original runs (the answer-verifier completion
  spawns jobs on its own GPU list; the shared-floor control uses GPU 0); `live_stream_admission_controls.py replay` may fetch pages
  that are missing from the recorded streams. The model-adjudication addendum adds the selected operating points with
  `model_adjudication_extra_points.py` after the operating-point section.
* **Engines.** A single engine, `freshcache/replay_engine/` (label `reported`), produces every reported number. The key-set engine
  (`freshcache/keyset_replay_engine/`, label `keyset`) only rebuilds the answer keys of the earlier audit samples so that their judged
  answers can be reused; it produces no reported number.
* **Calibration.** `freshcache/method/freshness_calibration.py` writes its report to `outputs/calibration/` and does not modify any
  source file. With `--patch-library` it also rewrites the calibrated constants of the library tier model `freshcache/risk_model.py`.
  The operational half-lives of the replay experiments are the constants recorded in `configs/freshcache/standalone.yaml`.
