#!/usr/bin/env bash
# End-to-end run on the bundled sample (data/sample, 300 requests in 60 clusters; the replays need only a CPU and a few minutes). The first run downloads the
# BGE-M3 model from the Hugging Face Hub to build the query embeddings; the rest needs no web access unless REFETCH=1. Stages:
# benchmark statistics, validation/held-out split, freshness-class predictor, standalone FreshCache with its internal controls and
# tier ablations (age gate off, EquivalentTTL, without L1, L2-only, SCALM-style, SemanticTTL, TieredFixedTTL), the supplementary
# replays (L3-only, L1-only, predicted classes, SemanticTTL points) and the ExactTTL / vCache-style replays; then, when a CUDA GPU is
# available and DATA_DIR holds page text (REFETCH=1), answer generation with Llama-3.2-3B-Instruct and the two-judge correctness and
# evidence-sufficiency judging with Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct on --limit held-out requests (default 50).
# Usage: bash scripts/run_sample_quickstart.sh [--limit N]   (results in outputs/sample/; numbers are sample-sized, not the paper's)
#        REFETCH=1 bash scripts/run_sample_quickstart.sh   re-fetches the sample's pages first (web access, no API key; a few minutes),
#        reports changed/unavailable pages, and runs every stage below on a data directory that holds the re-fetched pages
#        (outputs/sample/refetched_data). The replays use the recorded content hashes, so their operation counts do not change;
#        the re-fetched page text is what snapshot_audit.py checks here and what the generation and judging stages read.
set -euo pipefail
cd "$(dirname "$0")/.."
export FRESHCACHE_ROOT="$(pwd)"
export DATA_DIR="${DATA_DIR:-$(pwd)/data/sample}" OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/outputs/sample}"
PY=${PYTHON:-python}; LIMIT=50; GPU=${GPU:-0}
while [ $# -gt 0 ]; do case "$1" in --limit) LIMIT="$2"; shift 2;; -h|--help) sed -n "2,/^set -euo/p" "$0" | sed "/^set -euo/d; s/^# \{0,1\}//"; exit 0;; *) echo "unknown argument: $1" >&2; exit 2;; esac; done
run() { echo "+ $*"; "$PY" "$@"; }
run scripts/prepare_output_dirs.py
run scripts/analysis/check_config_consistency.py
run scripts/data/query_embeddings.py              # BGE-M3 embeddings and similarity matrix of DATA_DIR (kept if present)
if [ "${REFETCH:-0}" = 1 ]; then
  run scripts/data/page_refetch.py --out "$OUTPUT_DIR/page_refetch" --build-data "$OUTPUT_DIR/refetched_data"
  export DATA_DIR="$OUTPUT_DIR/refetched_data"
  run scripts/data/snapshot_audit.py          # observability of the re-fetched pages (writes only DATA_DIR/derived)
fi
run scripts/data/benchmark_statistics.py
run scripts/data/validation_heldout_split.py
run scripts/evaluation/freshness_classifier_evaluation.py
run scripts/experiments/calibration_and_measurement/freshness_class_prediction.py
run scripts/experiments/main/freshcache_replay.py reported main
run scripts/experiments/main/supplementary_replays.py reported
run scripts/experiments/main/exact_ttl_vcache_replay.py reported
# ---- answer generation, two-judge correctness and evidence sufficiency on a small subset (GPU, the paper's models) ----
if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "skip answer generation and judging: no CUDA GPU available (they need a GPU and the Hugging Face models meta-llama/Llama-3.2-3B-Instruct,"
  echo "  meta-llama/Llama-3.1-8B-Instruct and Qwen/Qwen2.5-7B-Instruct)"
elif ! ls "$DATA_DIR"/snapshots/*/*.json >/dev/null 2>&1; then
  echo "skip answer generation and judging: DATA_DIR has no page text; rerun with REFETCH=1 to re-fetch the sample's pages"
else
  GJ=(--gpu "$GPU" --indir sample_in --outdir sample_out)
  run scripts/evaluation/sample_answer_quality.py tasks --limit "$LIMIT"
  run scripts/evaluation/generation_judging_stage.py gen "${GJ[@]}"
  run scripts/evaluation/sample_answer_quality.py items
  for j in llama8b qwen7b; do run scripts/evaluation/generation_judging_stage.py judge2 --judge "$j" "${GJ[@]}"; done
  for j in llama8b qwen7b; do run scripts/evaluation/generation_judging_stage.py suffv --judge "$j" "${GJ[@]}"; done
  run scripts/evaluation/sample_answer_quality.py score
fi
echo "sample run finished; results in $OUTPUT_DIR (replay metrics: $OUTPUT_DIR/replay/results/replays_reported.json; answer quality, if run: $OUTPUT_DIR/sample_answer_quality/results.json)"
