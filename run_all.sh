#!/usr/bin/env bash
# Reproduces every experiment and baseline of the paper, grouped by paper section and run in dependency order, on the dataset in DATA_DIR (default data/).
# Requirements: a dataset in DATA_DIR in the layout of data/README.md (built by the first block below), NVIDIA GPUs for the generation and judging stages, and
# SEARCH/SERPER keys for the live-web stages (enabled with LIVE=1). Results are written to OUTPUT_DIR (default outputs/).
# Usage: [GPU=0] [LIVE=1] [PYTHON=python] bash run_all.sh            # quick CPU run on the bundled sample: scripts/run_sample_quickstart.sh
set -euo pipefail
case "${1:-}" in -h|--help) sed -n "2,/^set -euo/p" "$0" | sed "/^set -euo/d; s/^# \{0,1\}//"; exit 0;; esac
cd "$(dirname "$0")"
[ -f .env ] && set -a && . ./.env && set +a
export FRESHCACHE_ROOT="${FRESHCACHE_ROOT:-$(pwd)}"
OUT="${OUTPUT_DIR:-$FRESHCACHE_ROOT/outputs}"   # results directory (scripts read OUTPUT_DIR, default outputs/)
PY=${PYTHON:-python}; GPU=${GPU:-0}; LIVE=${LIVE:-0}
py()    { echo "+ $*"; "$PY" "$@"; }
modes() { local s=$1; shift; for m in "$@"; do py "$s" $m; done; }                       # run the listed modes one after another
round() { local s=$1; shift; py "$s" gen --gpu "$GPU" "$@"                                # generation, two-judge correctness, sufficiency
          for j in llama8b qwen7b; do py "$s" judge2 --judge $j --gpu "$GPU" "$@"; done
          for j in llama8b qwen7b; do py "$s" suffv --judge $j --gpu "$GPU" "$@"; done; }
round2(){ local s=$1; shift; py "$s" gen --gpu "$GPU" "$@"; for j in llama8b qwen7b; do py "$s" judge2 --judge $j --gpu "$GPU" "$@"; done; }
audit() { local s=$1; shift; py "$s" gen --gpu "$GPU" "$@"; for j in llama8b qwen7b; do py "$s" judge --judge $j --gpu "$GPU" "$@"; done; }
live()  { if [ "$LIVE" = 1 ]; then "$@"; else echo "skip (LIVE=0): $*"; fi; }
# A step never needs the outputs of a step listed after it; where that differs from paper order the section is re-opened as "(continued)".

py scripts/prepare_output_dirs.py
py scripts/analysis/check_config_consistency.py

# ---- Section 4.1 and Table 2: benchmark construction --------------------------------------------------------------------
py scripts/data/query_construction.py
py scripts/data/korean_query_generation.py
live py scripts/data/web_collection.py --run-id run_00   # baseline round; writes url_manifest.jsonl, snapshots/, change_log.jsonl
# The later rounds are NOT run here: start each one separately, at its delay after run_00 (1 h, 12 h, 24 h, 7 days), e.g.
#   LIVE=1 python scripts/data/web_collection.py --run-id rerun_1h      (likewise rerun_12h, rerun_24h, rerun_7d)
# change_detection.py and every later step need all five rounds in url_manifest.jsonl; the check below stops if one is missing.
py scripts/data/paraphrase_generation.py   # reads url_manifest.jsonl (paraphrases only questions with run_00 URLs)
py scripts/data/query_embeddings.py   # BGE-M3 query embeddings and similarity matrix (read by every replay)
"$PY" -B -c 'import json, pathlib, sys
p = pathlib.Path(sys.argv[1]) / "url_manifest.jsonl"; need = ("run_00", "rerun_1h", "rerun_12h", "rerun_24h", "rerun_7d")
have = {json.loads(l).get("run_id") for l in open(p, encoding="utf-8") if l.strip()} if p.exists() else set()
miss = ", ".join(r for r in need if r not in have)
sys.exit(f"collection rounds missing from {p}: {miss}. Run each with LIVE=1 python scripts/data/web_collection.py --run-id <round> "
         "at its delay after run_00, then re-run run_all.sh." if miss else None)' "${DATA_DIR:-$FRESHCACHE_ROOT/data}"
py scripts/data/change_detection.py
py scripts/data/query_rich_features.py
py freshcache/method/freshness_calibration.py
py freshcache/method/tiered_cache_model.py
py scripts/data/snapshot_audit.py
py scripts/data/round_content_table.py
py scripts/data/version_timeline.py
py scripts/data/validation_heldout_split.py
py scripts/data/benchmark_statistics.py
CUDA_VISIBLE_DEVICES=$GPU py scripts/data/paraphrase_fidelity_judging.py
py scripts/data/gold_answer_construction.py
py scripts/experiments/admission_controls/page_support_embeddings.py

# ---- Appendix A-B: calibration and measurement --------------------------------------------------------------------------
py scripts/experiments/calibration_and_measurement/half_life_refit.py
py scripts/experiments/calibration_and_measurement/tier_calibration_targets.py
py scripts/evaluation/freshness_classifier_evaluation.py
py scripts/experiments/calibration_and_measurement/freshness_class_prediction.py

# ---- Audit samples, reference validity and quality selection (shared by Tables 5-7) --------------------------------------
py scripts/experiments/main/freshcache_replay.py reported main   # reported-engine primary / held-out replays; records the reference values the audit replay checks
py freshcache/selection_replay_engine/audit_replay.py
audit scripts/evaluation/answer_audits/mixed_age_audit_generation_judging.py
py scripts/evaluation/answer_audits/mixed_age_audit_analysis.py
py scripts/evaluation/answer_audits/semanticttl_audit_sample.py
audit scripts/evaluation/answer_audits/semanticttl_audit_generation_judging.py
py scripts/evaluation/answer_audits/semanticttl_audit_analysis.py
py scripts/baselines/semantic_ttl_category_aware/validation_split_tuning.py
py scripts/baselines/semantic_ttl_category_aware/heldout_split_evaluation.py
# (moved here: reads outputs of heldout_split_evaluation.py)
py scripts/evaluation/answer_audits/general_audit_sample.py
audit scripts/evaluation/answer_audits/general_audit_generation_judging.py
py scripts/evaluation/answer_audits/general_audit_analysis.py
py scripts/baselines/semantic_ttl_category_aware/semanticttl_joint_grid_sweep.py
py scripts/evaluation/answer_audits/quality_selection_protocol.py
py scripts/evaluation/answer_audits/quality_selection_sample.py
py scripts/evaluation/answer_audits/quality_selection_generation.py --gpu "$GPU"
for j in llama8b qwen7b; do py scripts/evaluation/answer_audits/quality_selection_judging.py --judge $j --gpu "$GPU"; done
py scripts/evaluation/answer_audits/quality_selection_gold_verification.py build
for m in fidelity validity; do for j in llama8b qwen7b; do py scripts/evaluation/answer_audits/quality_selection_gold_verification.py $m --judge $j --gpu "$GPU"; done; done
py scripts/evaluation/answer_audits/quality_selection_gold_verification.py tiebreak
py scripts/evaluation/answer_audits/quality_selection_gold_verification.py validity --judge mistral7b --gpu "$GPU" --only-disagreements   # third judge on the pair's disagreements
py scripts/evaluation/answer_audits/quality_selection_gold_verification.py combine
py scripts/evaluation/answer_audits/quality_selection.py
py scripts/evaluation/answer_audits/heldout_sensitivity_protocol.py
py scripts/evaluation/answer_audits/heldout_sensitivity_sample.py
py scripts/evaluation/answer_audits/heldout_sensitivity_generation_judging.py seed
py scripts/evaluation/answer_audits/heldout_sensitivity_generation_judging.py gen --gpu "$GPU"
for j in llama8b qwen7b; do py scripts/evaluation/answer_audits/heldout_sensitivity_generation_judging.py judge --judge $j --gpu "$GPU"; done
# (moved here: reads outputs of heldout_sensitivity_sample.py)
py scripts/evaluation/reference_validation/reference_validation_inputs.py
py scripts/evaluation/reference_validation/reference_validation_protocol.py
JUDGE_GPUS="${JUDGE_GPUS:-$GPU $GPU $GPU}" bash scripts/evaluation/reference_validation/run_reference_judges.sh
py scripts/evaluation/reference_validation/reference_validation_reparse.py
py scripts/evaluation/reference_validation/reference_validation_analysis.py
py scripts/evaluation/answer_audits/predicted_class_audit_sample.py

# ---- Tables 5 and 10: standalone FreshCache, scripts/baselines and tier ablations ------------------------------------------------
py scripts/experiments/main/freshcache_replay.py reported mlp
py scripts/experiments/main/supplementary_replays.py reported
py scripts/experiments/main/exact_ttl_vcache_replay.py reported
py scripts/baselines/exact_ttl_and_l1_only/exact_ttl_l1_only_baselines.py
py scripts/evaluation/answer_audits/audit_answer_keys.py reported
py scripts/evaluation/answer_audits/audit_answer_keys.py keyset
py scripts/baselines/scalm/scalm_lookup_timing.py SCALM
py scripts/baselines/scalm/scalm_lookup_timing.py FreshCache_control
py scripts/baselines/vcache/vcache_reference_replay.py

# ---- Table 6: comparator grid, operating points and admission controls ---------------------------------------------------
py scripts/experiments/admission_controls/verifier_selection_rule.py
py scripts/experiments/admission_controls/verifier_sufficiency_labels.py build
for j in llama8b qwen7b; do py scripts/experiments/admission_controls/verifier_sufficiency_labels.py judge --judge $j --gpu "$GPU"; done
py scripts/experiments/admission_controls/verifier_sufficiency_labels.py combine
py scripts/experiments/admission_controls/verifier_threshold_selection.py
py scripts/experiments/admission_controls/verifier_heldout_replay_selection_engine.py
py scripts/experiments/admission_controls/verifier_answer_audit.py build
GPU=$GPU bash scripts/experiments/admission_controls/run_verifier_answer_audit.sh
py scripts/experiments/admission_controls/verifier_heldout_replay.py
py scripts/experiments/admission_controls/verifier_audit_keys.py

# ---- Tables 4, 7, 9 and 11: composition A-D, write controls (D', B-, Be-) and the common sample ----------------------------
py scripts/experiments/main/mixed_age_common_support.py

# ---- Table 10 and Appendices A-B: temporal eligibility, observability, risk budgets and L1 diagnostics -----------------------
RECOMPUTE_ENGINE=reported py scripts/experiments/ablations/mixed_age_trace_construction.py
py scripts/experiments/ablations/cross_round_drift_audit.py
py scripts/experiments/ablations/strict_support_sensitivity.py
# (moved here: reads outputs of strict_support_sensitivity.py)
RECOMPUTE_ENGINE=reported py scripts/experiments/ablations/mixed_age_consistency_analysis.py
py scripts/experiments/calibration_and_measurement/drift_length_threshold_sensitivity.py
py scripts/experiments/calibration_and_measurement/observability_bounds.py
py scripts/experiments/calibration_and_measurement/observability_recovery.py

# ---- (continued) Appendix A-B: calibration and measurement --------------------------------------------------------------------------
# (moved here: reads outputs of observability_recovery.py)
py scripts/experiments/calibration_and_measurement/drift_observability_accounting.py

# ---- (continued) Table 10 and Appendices A-B: temporal eligibility, observability, risk budgets and L1 diagnostics -----------------------
py scripts/experiments/ablations/risk_budget_validation_sweep.py
py scripts/experiments/ablations/risk_budget_heldout_fit.py
py scripts/experiments/ablations/risk_budget_selection.py reported
py scripts/experiments/ablations/l1_precision_gate/l1_precision_protocol.py
py scripts/experiments/ablations/l1_precision_gate/l1_precision_validation_grid.py replay
for j in llama3b llama8b qwen7b mistral7b; do py scripts/experiments/ablations/l1_precision_gate/l1_precision_generation_judging.py equiv --set validation --judge $j --gpu "$GPU"; done
py scripts/experiments/ablations/l1_precision_gate/l1_precision_validation_grid.py score
py scripts/experiments/ablations/l1_precision_gate/l1_precision_selection.py stage1
py scripts/experiments/ablations/l1_precision_gate/l1_precision_generation_judging.py gen --set validation --gpu "$GPU"
for j in llama8b qwen7b; do py scripts/experiments/ablations/l1_precision_gate/l1_precision_generation_judging.py judge --set validation --judge $j --gpu "$GPU"; done
py scripts/experiments/ablations/l1_precision_gate/l1_precision_selection.py stage2
py scripts/experiments/ablations/l1_precision_gate/l1_hit_audit_tasks.py
py scripts/experiments/ablations/l1_precision_gate/l1_hit_generation_judging_protocol.py gen --gpu "$GPU"
for j in llama3b llama8b qwen7b mistral7b; do py scripts/experiments/ablations/l1_precision_gate/l1_hit_generation_judging_protocol.py judge --judge $j --gpu "$GPU"; py scripts/experiments/ablations/l1_precision_gate/l1_hit_generation_judging_protocol.py equiv --judge $j --gpu "$GPU"; done
py scripts/experiments/ablations/l1_precision_gate/stricter_l1_validation_sweep.py
py scripts/experiments/ablations/l1_precision_gate/strict_l1_gate_tasks.py
py scripts/experiments/ablations/l1_precision_gate/strict_l1_gate_generation_judging.py gen --gpu "$GPU"
for j in llama3b llama8b qwen7b mistral7b; do py scripts/experiments/ablations/l1_precision_gate/strict_l1_gate_generation_judging.py equiv --judge $j --gpu "$GPU"; py scripts/experiments/ablations/l1_precision_gate/strict_l1_gate_generation_judging.py judge --judge $j --gpu "$GPU"; done
py scripts/experiments/ablations/l1_validation/l1_validation_protocol.py
py scripts/experiments/ablations/l1_validation/l1_validation_replay.py
for j in llama3b llama8b qwen7b mistral7b; do py scripts/experiments/ablations/l1_validation/l1_validation_generation_judging.py --stage equiv --judge $j --gpu "$GPU"; done
py scripts/experiments/ablations/l1_validation/l1_validation_generation_judging.py --stage gen --gpu "$GPU"
py scripts/experiments/ablations/l1_validation/l1_validation_judging_tasks.py
for j in llama8b qwen7b; do py scripts/experiments/ablations/l1_validation/l1_validation_generation_judging.py --stage judge --judge $j --gpu "$GPU"; done
py scripts/experiments/ablations/l1_validation/l1_validation_selection.py
py scripts/experiments/ablations/tier_diagnostic_replay.py
py scripts/experiments/calibration_and_measurement/request_list_export.py
py freshcache/method/l2_equivalence_fit.py

# ---- Appendix B: gold verification, generator replication, tier diagnostics and the external judge ------------------------
py scripts/evaluation/gold_verification/audit_reproduction_gate.py
py scripts/evaluation/gold_verification/gold_verification_inputs.py
py scripts/evaluation/gold_verification/gold_temporal_volatility.py
for j in llama8b qwen7b; do py scripts/evaluation/gold_verification/paraphrase_fidelity_screen.py judge --judge $j --gpu "$GPU"; done
py scripts/evaluation/gold_verification/paraphrase_fidelity_screen.py analyze
py scripts/evaluation/gold_verification/reference_validity_screen.py build
for j in llama8b qwen7b; do py scripts/evaluation/gold_verification/reference_validity_screen.py judge --judge $j --gpu "$GPU"; done
py scripts/evaluation/gold_verification/reference_validity_screen.py analyze
py scripts/evaluation/gold_verification/gold_verification_analysis.py
py scripts/evaluation/gold_verification/timestamp_validity_tasks.py
for j in llama8b qwen7b; do py scripts/evaluation/answer_audits/timestamp_validity_judging.py --judge $j --gpu "$GPU"; done
py scripts/evaluation/gold_verification/timestamp_validity_tiebreaks.py
py scripts/evaluation/answer_audits/timestamp_validity_judging.py --judge mistral7b --gpu "$GPU" --only-disagreements
py scripts/evaluation/gold_verification/timestamp_validity_analysis.py

# ---- (continued) Audit samples, reference validity and quality selection (shared by Tables 5-7) --------------------------------------
# (moved here: reads outputs of timestamp_validity_analysis.py)
py scripts/evaluation/answer_audits/heldout_sensitivity_analysis.py

# ---- (continued) Appendix B: gold verification, generator replication, tier diagnostics and the external judge ------------------------
py scripts/evaluation/gold_verification/reference_verification.py
py scripts/evaluation/gold_verification/screened_subset_crosswalk.py
py scripts/evaluation/answer_audits/quality_selection_keys.py reported
py scripts/evaluation/answer_audits/quality_selection_keys.py keyset
py scripts/evaluation/generator_replication/stronger_generator_audit_tasks.py
for g in llama3b llama8b_forced llama8b_abstain; do py scripts/experiments/ablations/l1_precision_gate/audit_generation_judging_protocol.py gen --gen $g --gpu "$GPU"; done
for g in llama3b llama8b_forced llama8b_abstain; do for j in llama3b llama8b qwen7b mistral7b; do py scripts/experiments/ablations/l1_precision_gate/audit_generation_judging_protocol.py judge --judge $j --gen $g --gpu "$GPU"; done; done

# ---- (continued) Table 10 and Appendices A-B: temporal eligibility, observability, risk budgets and L1 diagnostics -----------------------
# (moved here: reads outputs of audit_generation_judging_protocol.py)
py scripts/experiments/ablations/l1_precision_gate/l1_precision_heldout_replay.py
py scripts/experiments/ablations/l1_precision_gate/l1_precision_generation_judging.py gen --set heldout --gpu "$GPU"
for j in llama3b llama8b qwen7b mistral7b; do py scripts/experiments/ablations/l1_precision_gate/l1_precision_generation_judging.py equiv --set heldout --judge $j --gpu "$GPU"; done
for j in llama8b qwen7b; do py scripts/experiments/ablations/l1_precision_gate/l1_precision_generation_judging.py judge --set heldout --judge $j --gpu "$GPU"; done
py scripts/experiments/ablations/l1_precision_gate/l1_precision_heldout_analysis.py
# (moved here: reads outputs of l1_precision_heldout_replay.py)
py scripts/evaluation/answer_audits/risk_budget_audit_records.py
py scripts/evaluation/answer_audits/risk_budget_audit_sample.py
audit scripts/evaluation/answer_audits/risk_budget_audit_generation_judging.py
py scripts/evaluation/answer_audits/risk_budget_answer_keys.py reported
py scripts/experiments/ablations/stricter_l1_replay.py reported
py scripts/evaluation/answer_audits/stricter_l1_validation_keys.py reported

# ---- (continued) Appendix B: gold verification, generator replication, tier diagnostics and the external judge ------------------------
py scripts/evaluation/generator_replication/stronger_generator_audit_analysis.py
py scripts/evaluation/tier_quality/tier_quality_inputs.py
py scripts/evaluation/tier_quality/tier_quality_protocol.py
py scripts/evaluation/tier_quality/tier_quality_generation.py --gpu "$GPU"
py scripts/evaluation/tier_quality/tier_quality_judging_tasks.py
JUDGE_GPUS="${JUDGE_GPUS:-$GPU $GPU $GPU}" bash scripts/evaluation/tier_quality/run_tier_quality_judges.sh
py scripts/evaluation/tier_quality/tier_quality_analysis.py

# ---- (continued) Tables 5 and 10: standalone FreshCache, scripts/baselines and tier ablations ------------------------------------------------
# (moved here: reads outputs of tier_quality_analysis.py)
py scripts/evaluation/answer_audits/audit_generation_tasks.py gen
py scripts/evaluation/generation_judging_stage.py gen --gpu "$GPU"
py scripts/evaluation/answer_audits/audit_generation_tasks.py judge   # 2-judge items, 4-juror pairs, 3-judge tasks, sufficiency tasks (needs the answers)
for j in llama8b qwen7b; do py scripts/evaluation/generation_judging_stage.py judge2 --judge $j --gpu "$GPU"; done
for j in llama8b qwen7b mistral7b; do py scripts/evaluation/generation_judging_stage.py judge3 --judge $j --gpu "$GPU"; py scripts/evaluation/generation_judging_stage.py suff3 --judge $j --gpu "$GPU"; done
for j in llama8b qwen7b; do py scripts/evaluation/generation_judging_stage.py suffv --judge $j --gpu "$GPU"; done
for j in llama3b llama8b qwen7b mistral7b; do py scripts/evaluation/generation_judging_stage.py equiv --judge $j --gpu "$GPU"; done

# ---- (continued) Table 6: comparator grid, operating points and admission controls ---------------------------------------------------
py scripts/experiments/admission_controls/verifier_evaluation.py
# (moved here: reads outputs of verifier_evaluation.py)
py scripts/experiments/operating_points/l2_threshold_sweep_replay.py
py scripts/experiments/operating_points/l2_threshold_sweep_generation_judging.py gen --gpu "$GPU"
py scripts/experiments/operating_points/l2_threshold_sweep_scoring.py items
for j in llama8b qwen7b; do py scripts/experiments/operating_points/l2_threshold_sweep_generation_judging.py judge2 --judge $j --gpu "$GPU"; py scripts/experiments/operating_points/l2_threshold_sweep_generation_judging.py suffv --judge $j --gpu "$GPU"; done
py scripts/experiments/operating_points/l2_threshold_sweep_scoring.py score
py scripts/experiments/admission_controls/l2_gate_candidate_tracing.py
for j in llama8b qwen7b; do py scripts/experiments/admission_controls/l2_gate_generation_judging.py suffv --judge $j --gpu "$GPU"; done   # validation labels only (no gen / judge2 in this stage)
py scripts/experiments/admission_controls/l2_gate_training.py

# ---- (continued) Table 10 and Appendices A-B: temporal eligibility, observability, risk budgets and L1 diagnostics -----------------------
py scripts/experiments/calibration_and_measurement/l1_lexical_rule_audit.py

# ---- (continued) Appendix B: gold verification, generator replication, tier diagnostics and the external judge ------------------------
py scripts/evaluation/answer_audits/answer_quality_evaluation.py
py scripts/evaluation/answer_audits/replication_generation_tasks.py gen
py scripts/evaluation/generation_judging_stage.py gen --gpu "$GPU" --indir in3 --outdir out3
for m in forced abstain; do py scripts/evaluation/generation_judging_stage.py gen8 --mode $m --gpu "$GPU" --indir in3 --outdir out3; done
py scripts/evaluation/answer_audits/replication_generation_tasks.py judge   # judge items once the round-3 answers exist
for j in llama8b qwen7b; do py scripts/evaluation/generation_judging_stage.py judge2 --judge $j --gpu "$GPU" --indir in3 --outdir out3; done
for m in forced abstain; do for j in qwen7b mistral7b; do py scripts/evaluation/generation_judging_stage.py judge8 --mode $m --judge $j --gpu "$GPU" --indir in3 --outdir out3; done; done
py scripts/evaluation/answer_audits/quality_selection_and_8b_replication.py

# ---- (continued) Table 6: comparator grid, operating points and admission controls ---------------------------------------------------
py scripts/baselines/semantic_ttl_frontier/semanticttl_frontier_validation_replay.py
py scripts/baselines/semantic_ttl_frontier/semanticttl_frontier_generation_judging.py gen --gpu "$GPU" --tag val
py scripts/baselines/semantic_ttl_frontier/semanticttl_frontier_judging_tasks.py val
for j in llama8b qwen7b; do py scripts/baselines/semantic_ttl_frontier/semanticttl_frontier_generation_judging.py judge2 --judge $j --gpu "$GPU" --tag val; done
py scripts/baselines/semantic_ttl_frontier/semanticttl_frontier_validation_scoring.py
py scripts/baselines/semantic_ttl_frontier/semanticttl_frontier_heldout_replay.py
py scripts/baselines/semantic_ttl_frontier/semanticttl_frontier_generation_judging.py gen --gpu "$GPU" --tag ho
py scripts/baselines/semantic_ttl_frontier/semanticttl_frontier_judging_tasks.py ho
for j in llama8b qwen7b; do py scripts/baselines/semantic_ttl_frontier/semanticttl_frontier_generation_judging.py judge2 --judge $j --gpu "$GPU" --tag ho; done
py scripts/baselines/semantic_ttl_frontier/semanticttl_frontier_heldout_scoring.py
py scripts/baselines/semantic_ttl_category_aware/baseline_validation_sweep.py
cp "$OUT/baseline_selection/gpu/in/val_gen_tasks.jsonl" "$OUT/baseline_selection/gpu/in/gen_tasks.jsonl"   # the GPU stage reads gen_tasks.jsonl
py scripts/baselines/semantic_ttl_category_aware/baseline_generation_judging.py gen --gpu "$GPU"
py scripts/baselines/semantic_ttl_category_aware/baseline_validation_selection.py items
for j in llama8b qwen7b; do py scripts/baselines/semantic_ttl_category_aware/baseline_generation_judging.py judge2 --judge $j --gpu "$GPU"; done
py scripts/baselines/semantic_ttl_category_aware/baseline_validation_selection.py select
py scripts/baselines/semantic_ttl_category_aware/baseline_heldout_replay.py
py scripts/baselines/semantic_ttl_category_aware/baseline_generation_judging.py gen --gpu "$GPU"
py scripts/baselines/semantic_ttl_category_aware/baseline_heldout_scoring.py items
for j in llama8b qwen7b; do py scripts/baselines/semantic_ttl_category_aware/baseline_generation_judging.py judge2 --judge $j --gpu "$GPU"; done
py scripts/baselines/semantic_ttl_category_aware/baseline_heldout_scoring.py score

# ---- (continued) Appendix B: gold verification, generator replication, tier diagnostics and the external judge ------------------------
py scripts/evaluation/generator_replication/generator_replication_inputs.py
for m in forced abstain; do py scripts/evaluation/generator_replication/generator_replication_generation_judging.py gen --mode $m --gpu "$GPU"; done
py scripts/evaluation/generator_replication/generator_replication_judging_tasks.py
for m in forced abstain; do for j in qwen7b mistral7b; do py scripts/evaluation/generator_replication/generator_replication_generation_judging.py judge --mode $m --judge $j --gpu "$GPU"; done; done
for m in forced abstain; do for j in llama8b qwen7b mistral7b; do py scripts/evaluation/generator_replication/generator_replication_generation_judging.py judge3 --mode $m --judge $j --gpu "$GPU"; done; done
py scripts/evaluation/generator_replication/generator_replication_analysis.py
py scripts/evaluation/answer_audits/reference_and_tier_recomputation.py
py scripts/evaluation/tier_quality/tier_diagnostic_analysis.py
py scripts/experiments/operating_points/two_test_holm_families.py   # App. B-C: Holm within the theta_L2 0.80 and L2 inherited-cached families

# ---- Tables 12 and 13: unseen queries and prospective deeper lists ---------------------------------------------------------
py scripts/experiments/unseen_queries/unseen_query_embeddings.py
py scripts/experiments/unseen_queries/unseen_query_candidate_matching.py
py scripts/experiments/unseen_queries/unseen_query_cache_state_matching.py
live py scripts/experiments/unseen_queries/unseen_query_live_search.py
live py scripts/experiments/unseen_queries/unseen_query_page_refetch.py
py scripts/experiments/unseen_queries/unseen_query_evidence_paths.py
py scripts/evaluation/reference_validation/unseen_query_reference_inputs.py
for j in llama8b qwen7b mistral7b; do py scripts/evaluation/reference_validation/reference_validation_judging.py --study unseen_queries --stage reference --judge $j --gpu "$GPU"; done
py scripts/experiments/unseen_queries/generator_input_audit.py
live py scripts/experiments/unseen_queries/deep_list_collection.py t0
live py scripts/experiments/unseen_queries/deep_list_collection.py t1
py scripts/experiments/unseen_queries/deep_list_evaluation.py build
py scripts/experiments/unseen_queries/deep_list_evaluation.py gen14 --gpu "$GPU"
py scripts/evaluation/composition_study_generation_judging.py gen --gpu "$GPU" --workspace common_sample   # 3B cell (gpu/in/gen_tasks.jsonl)
for j in llama8b qwen7b; do py scripts/evaluation/composition_study_generation_judging.py suffv --judge $j --gpu "$GPU" --workspace common_sample; done
py scripts/experiments/unseen_queries/deep_list_evaluation.py items
for j in llama8b qwen7b; do py scripts/evaluation/composition_study_generation_judging.py judge2 --judge $j --gpu "$GPU" --workspace common_sample; done
py scripts/experiments/unseen_queries/deep_list_evaluation.py score

# ---- Tables 8, 9 and 14: live paraphrase and QReCC streams, complete admission policies ------------------------------------
py scripts/experiments/live_streams/live_query_block_design.py
py scripts/experiments/live_streams/paraphrase_stream_design.py
for j in llama8b qwen7b; do py scripts/experiments/live_streams/paraphrase_equivalence_judging.py judge --judge $j --gpu "$GPU"; done
py scripts/experiments/live_streams/paraphrase_equivalence_judging.py score
py scripts/experiments/live_streams/paraphrase_equivalence_rates.py   # App. D: per-judge paraphrase-fidelity rates
py scripts/experiments/live_streams/qrecc_stream_design.py

# ---- Section 5.1 and Appendix D: cost and latency ----------------------------------------------------------------------------
py scripts/experiments/cost_and_latency/component_timing_cpu.py
py scripts/experiments/cost_and_latency/component_timing_repeated.py
py scripts/experiments/cost_and_latency/component_timing_gpu.py "$GPU"
py scripts/experiments/cost_and_latency/gpu_timing_stability.py "$GPU"
py scripts/experiments/cost_and_latency/cpu_timing_stability.py
py scripts/experiments/cost_and_latency/latency_gpu_measurement.py --gpu "$GPU"
live py scripts/experiments/cost_and_latency/latency_network_measurement.py
# (moved here: reads outputs of latency_network_measurement.py)
py scripts/experiments/cost_and_latency/component_cost_break_even.py
py scripts/experiments/cost_and_latency/concurrency_benchmark.py "$GPU"
for m in FreshCache "CategoryAware-style" "SemanticTTL (.80,1/4)" "SemanticTTL (.80,2)" "SemanticTTL (.60,1/2)"; do py scripts/experiments/cost_and_latency/cold_lookup_timing.py "$m"; done
py scripts/experiments/cost_and_latency/serving_cost_summary.py

# ---- (continued) Audit samples, reference validity and quality selection (shared by Tables 5-7) --------------------------------------
# (moved here: reads outputs of baseline_heldout_scoring.py)
py scripts/evaluation/reference_validation/model_adjudication_inputs.py
for j in llama8b qwen7b mistral7b; do py scripts/evaluation/reference_validation/reference_validation_judging.py --study model_adjudication --stage answer --judge $j --gpu "$GPU"; done
# the selected standalone operating points are added after Table 6 (model_adjudication_extra_points.py, below savings_matched_operating_point.py)
py scripts/evaluation/reference_validation/model_adjudication_analysis.py

# ---- (continued) Tables 5 and 10: standalone FreshCache, scripts/baselines and tier ablations ------------------------------------------------
py scripts/baselines/matched_admission/matched_admission_baselines.py replay
py scripts/baselines/matched_admission/matched_admission_generation_judging.py gen --gpu "$GPU"
py scripts/baselines/matched_admission/matched_admission_baselines.py items
for j in llama8b qwen7b; do py scripts/baselines/matched_admission/matched_admission_generation_judging.py judge2 --judge $j --gpu "$GPU"; done
py scripts/baselines/matched_admission/matched_admission_baselines.py score
py scripts/baselines/gptcache/gptcache_topk_bound.py
# (moved here: reads outputs of gptcache_topk_bound.py)
for v in GPTCache_native GPTCache_matchedFC; do for s in primary heldout; do py scripts/baselines/gptcache/gptcache_baseline.py replay_one $v $s; done; done
modes scripts/baselines/gptcache/gptcache_baseline.py emulate collect
py scripts/baselines/gptcache/gptcache_generation_judging.py gen --gpu "$GPU"
py scripts/baselines/gptcache/gptcache_baseline.py items
for j in llama8b qwen7b; do py scripts/baselines/gptcache/gptcache_generation_judging.py judge2 --judge $j --gpu "$GPU"; done
py scripts/baselines/gptcache/gptcache_baseline.py score
py scripts/baselines/scalm/scalm_heldout_metrics.py
py scripts/baselines/vcache/vcache_lookup_timing.py   # writes work/local_cold_vCache.json, read by vcache_baseline.py score
py scripts/baselines/vcache/vcache_baseline.py replay
py scripts/baselines/vcache/vcache_generation_judging.py gen --gpu "$GPU"
py scripts/baselines/vcache/vcache_baseline.py items
for j in llama8b qwen7b; do py scripts/baselines/vcache/vcache_generation_judging.py judge2 --judge $j --gpu "$GPU"; done
py scripts/baselines/vcache/vcache_baseline.py score

# ---- (continued) Table 6: comparator grid, operating points and admission controls ---------------------------------------------------
py scripts/baselines/category_aware_frontier/frontier_validation_sweep.py
py scripts/baselines/category_aware_frontier/frontier_generation_judging.py gen --gpu "$GPU"
py scripts/baselines/category_aware_frontier/frontier_validation_selection.py items
for j in llama8b qwen7b; do py scripts/baselines/category_aware_frontier/frontier_generation_judging.py judge2 --judge $j --gpu "$GPU"; done
py scripts/baselines/category_aware_frontier/frontier_validation_selection.py select
py scripts/baselines/category_aware_frontier/frontier_heldout_replay.py
py scripts/baselines/category_aware_frontier/frontier_generation_judging.py gen --gpu "$GPU"
for j in llama8b qwen7b; do py scripts/baselines/category_aware_frontier/frontier_generation_judging.py suffv --judge $j --gpu "$GPU"; done
py scripts/baselines/category_aware_frontier/frontier_heldout_scoring.py   # takes no mode
py scripts/experiments/operating_points/operating_point_validation_replay.py
py scripts/experiments/operating_points/operating_point_generation_judging.py gen --gpu "$GPU"
py scripts/experiments/operating_points/operating_point_validation_selection.py items
for j in llama8b qwen7b; do py scripts/experiments/operating_points/operating_point_generation_judging.py judge2 --judge $j --gpu "$GPU"; done
py scripts/experiments/operating_points/operating_point_validation_selection.py select
py scripts/experiments/operating_points/operating_point_heldout_evaluation.py replay
py scripts/experiments/operating_points/operating_point_generation_judging.py gen --gpu "$GPU"
py scripts/experiments/operating_points/operating_point_heldout_evaluation.py items
for j in llama8b qwen7b; do py scripts/experiments/operating_points/operating_point_generation_judging.py judge2 --judge $j --gpu "$GPU"; py scripts/experiments/operating_points/operating_point_generation_judging.py suffv --judge $j --gpu "$GPU"; done
py scripts/experiments/operating_points/operating_point_heldout_evaluation.py score
py scripts/experiments/operating_points/savings_matched_operating_point.py replay
py scripts/experiments/operating_points/operating_point_generation_judging.py gen --gpu "$GPU"
py scripts/experiments/operating_points/savings_matched_operating_point.py items
for j in llama8b qwen7b; do py scripts/experiments/operating_points/operating_point_generation_judging.py judge2 --judge $j --gpu "$GPU"; py scripts/experiments/operating_points/operating_point_generation_judging.py suffv --judge $j --gpu "$GPU"; done
py scripts/experiments/operating_points/savings_matched_operating_point.py score
# Table 11: add the selected operating points to the three-model adjudication, judge only the new pairs, re-run the analysis
for m in points addendum; do
  py scripts/evaluation/reference_validation/model_adjudication_extra_points.py $m
  for j in llama8b qwen7b mistral7b; do py scripts/evaluation/reference_validation/reference_validation_judging.py --study model_adjudication --stage answer --judge $j --gpu "$GPU"; done
done
py scripts/evaluation/reference_validation/model_adjudication_analysis.py
py scripts/experiments/operating_points/nocache_sufficiency_tasks.py
for j in llama8b qwen7b; do py scripts/experiments/operating_points/nocache_sufficiency_judging.py suffv --judge $j --gpu "$GPU"; done
py scripts/experiments/operating_points/nocache_sufficiency_scoring.py
py scripts/experiments/admission_controls/l2_gate_heldout_evaluation.py replay
py scripts/experiments/admission_controls/l2_gate_heldout_evaluation.py items
py scripts/experiments/admission_controls/l2_gate_generation_judging.py gen --gpu "$GPU"
py scripts/experiments/admission_controls/l2_gate_heldout_evaluation.py items   # again: judge items are written only for answers that exist
for j in llama8b qwen7b; do py scripts/experiments/admission_controls/l2_gate_generation_judging.py judge2 --judge $j --gpu "$GPU"; done
# items also writes gpu/in/ho_suffv_tasks.jsonl, which l2_gate_generation_judging.py suffv does not read (it reads suffv_tasks.jsonl); in the
# reported run that file was empty (all held-out evidence labels reused), so no held-out sufficiency pass is run here
py scripts/experiments/admission_controls/l2_gate_heldout_evaluation.py score
py scripts/experiments/admission_controls/l2_gate_savings_matched_threshold.py
GATE_POINT=_tau2 py scripts/experiments/admission_controls/l2_gate_heldout_evaluation.py replay
GATE_POINT=_tau2 py scripts/experiments/admission_controls/l2_gate_heldout_evaluation.py items
# items writes gpu/in/gen_tasks_tau2.jsonl / judge2_items_tau2.jsonl / ho_suffv_tasks_tau2.jsonl, which the GPU stage does not read; they were
# empty in the reported run (everything reused), so no GPU pass is run for the savings-matched point
GATE_POINT=_tau2 py scripts/experiments/admission_controls/l2_gate_heldout_evaluation.py score

# ---- (continued) Tables 4, 7, 9 and 11: composition A-D, write controls (D', B-, Be-) and the common sample ----------------------------
py scripts/experiments/main/composition_engine_gate.py
for v in H75 H80 H55; do py scripts/experiments/cost_and_latency/categoryaware_composition_timing.py $v validation; done   # read by categoryaware_composition.py select
for v in H75 H80 H55; do py scripts/experiments/cost_and_latency/categoryaware_composition_timing.py $v heldout; done   # read by categoryaware_composition.py score
# (moved here: reads outputs of categoryaware_composition_timing.py)
py scripts/experiments/main/categoryaware_composition.py val
py scripts/experiments/main/categoryaware_composition_generation_judging.py gen --gpu "$GPU"
py scripts/experiments/main/categoryaware_composition.py val_items
for j in llama8b qwen7b; do py scripts/experiments/main/categoryaware_composition_generation_judging.py judge2 --judge $j --gpu "$GPU"; py scripts/experiments/main/categoryaware_composition_generation_judging.py suffv --judge $j --gpu "$GPU"; done
py scripts/experiments/main/categoryaware_composition.py select
py scripts/experiments/main/categoryaware_composition.py heldout
py scripts/experiments/main/categoryaware_composition_generation_judging.py gen --gpu "$GPU"
py scripts/experiments/main/categoryaware_composition.py items
for j in llama8b qwen7b; do py scripts/experiments/main/categoryaware_composition_generation_judging.py judge2 --judge $j --gpu "$GPU"; py scripts/experiments/main/categoryaware_composition_generation_judging.py suffv --judge $j --gpu "$GPU"; done
py scripts/experiments/main/categoryaware_composition.py score
py scripts/experiments/main/categoryaware_composition_paired_tests.py
py scripts/experiments/operating_points/verifier_gamma_paired_tests.py
py scripts/experiments/cost_and_latency/generation_cost_break_even.py   # reads the gate, verifier and composition outputs above
py scripts/experiments/operating_points/holm_family_construction.py   # Table 6: 24-test Holm family -> outputs/operating_point_summaries/holm_family_24.json
py scripts/experiments/operating_points/holm_adjustment.py
py scripts/experiments/main/composition_ablation_replay.py replay
py scripts/experiments/main/composition_ablation_replay.py items   # generation tasks
py scripts/experiments/main/composition_generation_judging.py gen --gpu "$GPU"
py scripts/experiments/main/composition_ablation_replay.py items   # judge2 items and SUFFV tasks once no answer is missing
for j in llama8b qwen7b; do py scripts/experiments/main/composition_generation_judging.py judge2 --judge $j --gpu "$GPU"; py scripts/experiments/main/composition_generation_judging.py suffv --judge $j --gpu "$GPU"; done
py scripts/experiments/main/composition_ablation_replay.py score
py scripts/experiments/main/composition_paired_quality.py
py scripts/experiments/main/changed_request_audit.py freeze
py scripts/experiments/main/changed_request_audit.py items
py scripts/evaluation/composition_study_generation_judging.py gen --gpu "$GPU" --workspace composition_checks
py scripts/experiments/main/changed_request_audit.py items
for j in llama8b qwen7b; do py scripts/evaluation/composition_study_generation_judging.py judge2 --judge $j --gpu "$GPU" --workspace composition_checks; py scripts/evaluation/composition_study_generation_judging.py suffv --judge $j --gpu "$GPU" --workspace composition_checks; done
py scripts/experiments/main/changed_request_audit.py score
py scripts/experiments/main/changed_population.py
py scripts/experiments/main/changed_audit_by_list_depth.py
py scripts/experiments/ablations/reuse_write_suppression.py replay
py scripts/experiments/ablations/reuse_write_suppression.py items
py scripts/evaluation/composition_study_generation_judging.py gen --gpu "$GPU" --workspace write_controls
py scripts/experiments/ablations/reuse_write_suppression.py items
for j in llama8b qwen7b; do py scripts/evaluation/composition_study_generation_judging.py judge2 --judge $j --gpu "$GPU" --workspace write_controls; py scripts/evaluation/composition_study_generation_judging.py suffv --judge $j --gpu "$GPU" --workspace write_controls; done
py scripts/experiments/ablations/reuse_write_suppression.py score
py scripts/experiments/main/write_control_traces.py
py scripts/experiments/main/common_sample_construction.py build
py scripts/experiments/main/common_sample_construction.py items
py scripts/evaluation/composition_study_generation_judging.py gen --gpu "$GPU" --workspace common_sample
py scripts/experiments/main/common_sample_construction.py items
for j in llama8b qwen7b; do py scripts/evaluation/composition_study_generation_judging.py judge2 --judge $j --gpu "$GPU" --workspace common_sample; py scripts/evaluation/composition_study_generation_judging.py suffv --judge $j --gpu "$GPU" --workspace common_sample; done
py scripts/experiments/main/common_sample_scoring.py
for s in pre diff; do py scripts/experiments/main/root_clone_diagnostic.py replay $s; py scripts/experiments/main/root_clone_diagnostic.py items $s; done
cat "$OUT"/common_sample/gpu/in/gen_tasks_2x2_{pre,diff}.jsonl > "$OUT/common_sample/gpu/in/gen_tasks.jsonl"   # the shared GPU stage reads gen_tasks.jsonl / judge2_items.jsonl
py scripts/evaluation/composition_study_generation_judging.py gen --gpu "$GPU" --workspace common_sample
for s in pre diff; do py scripts/experiments/main/root_clone_diagnostic.py items $s; done
cat "$OUT"/common_sample/gpu/in/judge2_items_2x2_{pre,diff}.jsonl > "$OUT/common_sample/gpu/in/judge2_items.jsonl"
for j in llama8b qwen7b; do py scripts/evaluation/composition_study_generation_judging.py judge2 --judge $j --gpu "$GPU" --workspace common_sample; done
for s in pre diff; do py scripts/experiments/main/root_clone_diagnostic.py score $s; done
py scripts/experiments/main/loss_rate_denominators.py
py scripts/experiments/evaluation_sensitivity/judging_sensitivity.py items
py scripts/evaluation/composition_study_generation_judging.py judge2 --judge mistral7b --gpu "$GPU" --workspace common_sample   # third judge family
py scripts/experiments/evaluation_sensitivity/judging_sensitivity.py phi4 --gpu "$GPU"
for j in llama8b qwen7b mistral7b; do py scripts/experiments/evaluation_sensitivity/judging_sensitivity.py refval --judge $j --gpu "$GPU"; done
py scripts/experiments/evaluation_sensitivity/judging_sensitivity.py score
py scripts/experiments/evaluation_sensitivity/common_sample_larger_context.py build
py scripts/experiments/evaluation_sensitivity/common_sample_larger_context.py gen --gpu "$GPU"
py scripts/experiments/evaluation_sensitivity/common_sample_larger_context.py items
for j in llama8b qwen7b; do py scripts/experiments/live_streams/live_stream_generation_judging.py judge2 --judge $j --gpu "$GPU"; done
py scripts/experiments/evaluation_sensitivity/common_sample_larger_context.py score

# ---- (continued) Table 10 and Appendices A-B: temporal eligibility, observability, risk budgets and L1 diagnostics -----------------------
py scripts/experiments/ablations/composition_operation_replications.py

# ---- (continued) Appendix B: gold verification, generator replication, tier diagnostics and the external judge ------------------------
py scripts/evaluation/external_judge/external_judge_batch.py      # the external-judge replies are pasted by hand (see README)
# paste each outputs/external_judge/chatgpt_batch/batch_XX.txt into the external judge and save the reply as chatgpt_batch/reply_XX.txt
if ls "$OUT"/external_judge/chatgpt_batch/reply_*.txt >/dev/null 2>&1; then py scripts/evaluation/external_judge/external_judge_agreement.py; else echo "skip (no pasted replies): scripts/evaluation/external_judge/external_judge_agreement.py"; fi

# ---- (continued) Tables 12 and 13: unseen queries and prospective deeper lists ---------------------------------------------------------
py scripts/experiments/unseen_queries/unseen_query_generation_judging.py gen --gpu "$GPU"
for j in llama8b qwen7b; do py scripts/experiments/unseen_queries/unseen_query_generation_judging.py suffv --judge $j --gpu "$GPU"; done
py scripts/experiments/unseen_queries/unseen_query_scoring.py items
for j in llama8b qwen7b; do py scripts/experiments/unseen_queries/unseen_query_generation_judging.py judge2 --judge $j --gpu "$GPU"; done
py scripts/experiments/unseen_queries/unseen_query_scoring.py score

# ---- (continued) Table 6: comparator grid, operating points and admission controls ---------------------------------------------------
# (moved here: reads outputs of unseen_query_scoring.py)
py scripts/experiments/admission_controls/l2_gate_heldout_evaluation.py live   # needs the unseen-query live cases (Tables 12-13 section)

# ---- (continued) Tables 12 and 13: unseen queries and prospective deeper lists ---------------------------------------------------------
py scripts/experiments/unseen_queries/unseen_query_similarity_bands.py
py scripts/experiments/unseen_queries/unseen_query_reference_validity.py
py scripts/experiments/unseen_queries/live_gate_scores.py

# ---- (continued) Tables 4, 7, 9 and 11: composition A-D, write controls (D', B-, Be-) and the common sample ----------------------------
# (moved here: reads outputs of live_gate_scores.py)
py scripts/experiments/evaluation_sensitivity/stronger_generator_answers.py tasks
py scripts/experiments/evaluation_sensitivity/stronger_generator_answers.py gen --gpu "$GPU" --shard 0/1
py scripts/experiments/evaluation_sensitivity/stronger_generator_answers.py merge
py scripts/experiments/evaluation_sensitivity/stronger_generator_scoring.py items
for j in llama8b qwen7b; do py scripts/evaluation/composition_study_generation_judging.py judge2 --judge $j --gpu "$GPU" --workspace composition_checks; done
py scripts/experiments/evaluation_sensitivity/stronger_generator_scoring.py score
py scripts/experiments/operating_points/wai_confidence_intervals.py   # reads the audit outputs of Tables 2 and 4 (moved after their producers)

# ---- (continued) Tables 12 and 13: unseen queries and prospective deeper lists ---------------------------------------------------------
py scripts/experiments/unseen_queries/live_admission_policies.py
py scripts/experiments/unseen_queries/live_full_state_policies.py build
py scripts/experiments/main/composition_generation_judging.py gen --gpu "$GPU"
for j in llama8b qwen7b; do py scripts/experiments/main/composition_generation_judging.py suffv --judge $j --gpu "$GPU"; done
py scripts/experiments/unseen_queries/live_full_state_judging_items.py
for j in llama8b qwen7b; do py scripts/experiments/main/composition_generation_judging.py judge2 --judge $j --gpu "$GPU"; done
py scripts/experiments/unseen_queries/live_full_state_policies.py score

# ---- (continued) Tables 8, 9 and 14: live paraphrase and QReCC streams, complete admission policies ------------------------------------
live py scripts/experiments/live_streams/paraphrase_stream_collection.py
py scripts/experiments/live_streams/paraphrase_stream_scoring.py build
py scripts/experiments/live_streams/paraphrase_stream_scoring.py gen14 --gpu "$GPU"
py scripts/evaluation/composition_study_generation_judging.py gen --gpu "$GPU" --workspace common_sample   # configuration P (3B)
for j in llama8b qwen7b; do py scripts/evaluation/composition_study_generation_judging.py suffv --judge $j --gpu "$GPU" --workspace common_sample; done
py scripts/experiments/live_streams/paraphrase_stream_scoring.py items
for j in llama8b qwen7b; do py scripts/evaluation/composition_study_generation_judging.py judge2 --judge $j --gpu "$GPU" --workspace common_sample; done
py scripts/experiments/live_streams/paraphrase_stream_scoring.py score
py scripts/experiments/live_streams/paraphrase_stream_strata.py
live py scripts/experiments/live_streams/qrecc_stream_collection.py
py scripts/experiments/live_streams/qrecc_stream_scoring.py build
GEN_TASKS=gen_tasks_qr.jsonl GEN_TAG=_qr py scripts/experiments/live_streams/live_stream_generation_judging.py gen --gpu "$GPU"   # configuration P (3B)
py scripts/experiments/live_streams/qrecc_stream_scoring.py gen14 --gpu "$GPU"
for j in llama8b qwen7b; do py scripts/experiments/live_streams/live_stream_generation_judging.py suffv --judge $j --gpu "$GPU"; done
py scripts/experiments/live_streams/qrecc_stream_scoring.py items
for j in llama8b qwen7b; do py scripts/experiments/live_streams/live_stream_generation_judging.py judge2 --judge $j --gpu "$GPU"; done
modes scripts/experiments/live_streams/qrecc_stream_scoring.py score measured predict
py scripts/experiments/live_streams/answer_verifier.py valset
py scripts/experiments/live_streams/live_stream_generation_judging.py gen --gpu "$GPU"
py scripts/experiments/live_streams/answer_verifier.py valitems
for j in llama8b qwen7b; do py scripts/experiments/live_streams/live_stream_generation_judging.py judge2 --judge $j --gpu "$GPU"; done
py scripts/experiments/live_streams/answer_verifier.py train
for pass in $(seq 1 40); do   # held-out D+AV replay, rerun until no answer is missing (residual misses fail open)
  py scripts/experiments/live_streams/answer_verifier.py heldout
  n=$( { awk 'NF' "$OUT/natural_workload/gpu/in/gen_tasks.jsonl" 2>/dev/null || true; } | wc -l ); [ "$n" -eq 0 ] && break
  py scripts/experiments/live_streams/live_stream_generation_judging.py gen --gpu "$GPU"
done
py scripts/experiments/live_streams/list_control_replays.py
py scripts/experiments/live_streams/admission_control_evaluation.py build
py scripts/experiments/live_streams/admission_control_evaluation.py items
py scripts/experiments/live_streams/live_stream_generation_judging.py gen --gpu "$GPU"
py scripts/experiments/live_streams/admission_control_evaluation.py items   # judge items for the new answers
for j in llama8b qwen7b; do py scripts/experiments/live_streams/live_stream_generation_judging.py judge2 --judge $j --gpu "$GPU"; done
py scripts/experiments/live_streams/admission_control_evaluation.py score
for pass in $(seq 1 8); do   # configuration P: D+AV replay, rerun until no AV answer is missing
  py scripts/experiments/live_streams/live_stream_admission_controls.py replay P
  n=$( { awk 'NF' "$OUT/natural_workload/gpu/in/gen_tasks_cctl.jsonl" 2>/dev/null || true; } | wc -l ); [ "$n" -eq 0 ] && break
  GEN_TASKS=gen_tasks_cctl.jsonl GEN_TAG=_cctl py scripts/experiments/live_streams/live_stream_generation_judging.py gen --gpu "$GPU"
done
for pass in $(seq 1 8); do   # configuration S (14B): same loop with the script's own gen14 stage
  py scripts/experiments/live_streams/live_stream_admission_controls.py replay S
  n=$( { awk 'NF' "$OUT/natural_workload/gpu/in/cctl_tasks14.jsonl" 2>/dev/null || true; } | wc -l ); [ "$n" -eq 0 ] && break
  py scripts/experiments/live_streams/live_stream_admission_controls.py gen14 --gpu "$GPU"
done
py scripts/experiments/live_streams/live_stream_admission_controls.py items P
GEN_TASKS=gen_tasks_cctl.jsonl GEN_TAG=_cctl py scripts/experiments/live_streams/live_stream_generation_judging.py gen --gpu "$GPU"
py scripts/experiments/live_streams/live_stream_admission_controls.py items P
for j in llama8b qwen7b; do py scripts/experiments/live_streams/live_stream_generation_judging.py judge2 --judge $j --gpu "$GPU"; done
py scripts/experiments/live_streams/live_stream_admission_controls.py items S
py scripts/experiments/live_streams/live_stream_admission_controls.py gen14 --gpu "$GPU"
py scripts/experiments/live_streams/live_stream_admission_controls.py items S
for j in llama8b qwen7b; do py scripts/experiments/live_streams/live_stream_generation_judging.py judge2 --judge $j --gpu "$GPU"; done
py scripts/experiments/live_streams/live_stream_admission_controls.py score   # scores both configurations
py scripts/experiments/live_streams/paraphrase_stream_root_diagnostic.py replay
live py scripts/experiments/live_streams/paraphrase_stream_root_diagnostic.py fetchmissing   # pages no policy fetched (web access); needed before items builds contexts
py scripts/experiments/live_streams/paraphrase_stream_root_diagnostic.py items
py scripts/experiments/live_streams/live_stream_generation_judging.py gen --gpu "$GPU"
py scripts/experiments/live_streams/paraphrase_stream_root_diagnostic.py items
for j in llama8b qwen7b; do py scripts/experiments/live_streams/live_stream_generation_judging.py judge2 --judge $j --gpu "$GPU"; done
py scripts/experiments/live_streams/paraphrase_stream_root_diagnostic.py score

# ---- (continued) Tables 4, 7, 9 and 11: composition A-D, write controls (D', B-, Be-) and the common sample ----------------------------
# (moved here: reads outputs of paraphrase_stream_root_diagnostic.py)
py scripts/experiments/main/root_contrasts_leave_one_out.py   # also needs the confirmation-stream roots (paraphrase_stream_root_diagnostic.py score)

# ---- (continued) Tables 8, 9 and 14: live paraphrase and QReCC streams, complete admission policies ------------------------------------
py scripts/experiments/live_streams/paraphrase_stream_search_latency.py
# answer_verifier_completion.py starts its own GPU subprocesses (cross_workload_generation_judging.py) on fixed GPU ids 1, 3, 4, 7, 9; edit CODE there if needed
py scripts/experiments/live_streams/recorded_stream_controls/answer_verifier_completion.py loop heldout
py scripts/experiments/live_streams/recorded_stream_controls/answer_verifier_completion.py loop S
py scripts/experiments/live_streams/recorded_stream_controls/answer_verifier_completion.py label   # runs hoitems itself, with its generation and judging passes
py scripts/experiments/live_streams/recorded_stream_controls/answer_verifier_completion.py audit
for c in P S; do
  py scripts/experiments/live_streams/recorded_stream_controls/qrecc_admission_controls.py replay $c
  for it in 0 1 2 3 4 5; do   # score -> generate -> judge -> sufficiency until nothing is pending (as run_shared_floor_control.sh)
    py scripts/experiments/live_streams/recorded_stream_controls/qrecc_admission_controls.py score $c
    I="$OUT/cross_workload/gpu/in"
    g=$( { awk 'NF' "$I/qrecc_gen_$c.jsonl" 2>/dev/null || true; } | wc -l ); jn=$( { awk 'NF' "$I/qrecc_judge_$c.jsonl" 2>/dev/null || true; } | wc -l ); v=$( { awk 'NF' "$I/qrecc_suffv_$c.jsonl" 2>/dev/null || true; } | wc -l )
    [ "$g$jn$v" = "000" ] && break
    if [ "$g" != 0 ]; then if [ $c = S ]; then st=gen14; else st=gen3; fi; py scripts/experiments/live_streams/recorded_stream_controls/cross_workload_generation_judging.py $st --gpu "$GPU" --tasks qrecc_gen_$c.jsonl --tag qc$c$it; fi
    if [ "$jn" != 0 ]; then for j in llama8b qwen7b; do py scripts/experiments/live_streams/recorded_stream_controls/cross_workload_generation_judging.py judge2 --judge $j --gpu "$GPU" --tasks qrecc_judge_$c.jsonl; done; fi
    if [ "$v" != 0 ]; then for j in llama8b qwen7b; do py scripts/experiments/live_streams/recorded_stream_controls/cross_workload_generation_judging.py suffv --judge $j --gpu "$GPU" --tasks qrecc_suffv_$c.jsonl --tag qc; done; fi
  done
done
py scripts/experiments/live_streams/recorded_stream_controls/qrecc_shared_floor_control.py replay
py scripts/experiments/live_streams/recorded_stream_controls/qrecc_shared_floor_control.py fixedstate
bash scripts/experiments/live_streams/recorded_stream_controls/run_shared_floor_control.sh   # score loop with its GPU stages (GPU 0)
py scripts/experiments/main/cross_workload_summary.py

# ---- (continued) Section 5.1 and Appendix D: cost and latency ----------------------------------------------------------------------------
for v in A B C D; do py scripts/experiments/cost_and_latency/decision_path_timing.py $v; done
pids=(); for v in A B C D; do py scripts/experiments/cost_and_latency/decision_path_timing.py $v parallel & pids+=($!); done; for p in "${pids[@]}"; do wait "$p"; done   # concurrent run (conservative bound)
py scripts/experiments/cost_and_latency/storage_payload.py
py scripts/experiments/cost_and_latency/local_runtime_check.py
py scripts/experiments/cost_and_latency/admission_inspection_cost.py
py scripts/experiments/cost_and_latency/composition_cost_model.py
py scripts/experiments/cost_and_latency/vector_index_timing.py
py scripts/experiments/cost_and_latency/vector_index_parity.py
py scripts/experiments/cost_and_latency/monetary_cost.py
py scripts/experiments/cost_and_latency/monetary_cost_table.py
live py scripts/experiments/cost_and_latency/live_network_latency.py
py scripts/experiments/cost_and_latency/live_network_latency_windows.py plan
live py scripts/experiments/cost_and_latency/live_network_latency_windows.py run 7000 16000 20000
py scripts/experiments/cost_and_latency/live_network_latency_summary.py
# (moved here: reads outputs of live_network_latency_summary.py)
py scripts/experiments/cost_and_latency/break_even_analysis.py

# ---- Section 5.1 and Appendix D: workload locality and temporal checks ------------------------------------------------------
py scripts/experiments/workload_locality/cluster_locality_decomposition.py
py scripts/experiments/workload_locality/paraphrase_share_sensitivity.py
py scripts/experiments/workload_locality/paraphrase_share_report.py
py scripts/experiments/workload_locality/strict_provenance_replay.py   # Sec. 5.1: held-out savings via lists sharing no URL with the own list
py scripts/experiments/workload_locality/neighbour_density.py
py scripts/experiments/workload_locality/substream_replay.py
py scripts/experiments/calibration_and_measurement/observation_density_sensitivity.py
py scripts/experiments/calibration_and_measurement/synthetic_sparsity_bias.py
echo "all stages finished; results are under ${OUTPUT_DIR:-outputs}"
