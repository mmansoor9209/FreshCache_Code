# Run order

Stages are listed in dependency order. Stages 1–3 require network access;
everything after stage 4 runs offline from stored artifacts.

| # | Stage | Entry point | Network | GPU | Notes |
|---|---|---|---|---|---|
| 1 | Query construction | `build_queries.py` | yes | no | builds the query set and freshness classes |
| 2 | Web collection (`run_00`) | `collect.py` | yes | no | `MAX_URLS_PER_QUERY = 2`; writes `data/snapshots/run_00/` |
| 3 | Re-collection rounds | `collect.py` | yes | no | `rerun_1h`, `rerun_12h`, `rerun_24h`, `rerun_7d` |
| 4 | Round table | `v13_corrected/a1b_round_table.py` | no | no | `substantive` = ≥400 chars ∧ HTTP 200 ∧ no block-page pattern |
| 5 | Embeddings + similarity | `experiment.precompute_similarity_matrix` | no | yes | BGE-M3; `data/sim_matrix_bgem3.npy` |
| 6 | Half-life calibration | `calibrate.py` | no | no | per-class MLE, asymmetric temporal holdout |
| 7 | Policy replays | `v16_exp12/mixed_engine.py`, `v14_baselines/engine_all.py` | no | no | all main and baseline results |
| 8 | Answer generation | `remaining_critical_issues/06_stronger_generator/gen_judge.py gen` | no | yes | Llama-3.2-3B, FORCED prompt, greedy, 80 new tokens |
| 9 | Correctness judging | same module, `judge` | no | yes | Llama-3.1-8B + Qwen2.5-7B, greedy, 6 new tokens |
| 10 | Analysis and audits | `validation_V3/*/` | no | no | each workstream has its own runner and `FINAL_REPORT.md` |
| 11 | Verification | `validation_V3/group1_resolution/p1_f08_f10_f26.py` | no | no | 75 row-level checks |

**Split discipline.** `validation/heldout_baseline_tuning/split.json` defines
1,877 validation clusters (9,353 requests) and 4,381 held-out test clusters
(21,848 requests); the two are cluster-disjoint. Any parameter selection runs
on validation only and must be frozen and hashed before the held-out replay.
