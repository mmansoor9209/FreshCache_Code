#!/bin/bash
# Runs all judge passes of the reference-validity study (reference stage, then answer stage), one judge per GPU.
# Run from the repository root:  JUDGE_GPUS="0 1 2" bash scripts/evaluation/reference_validation/run_reference_judges.sh
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGS="${OUTPUT_DIR:-${FRESHCACHE_ROOT:-.}/outputs}/reference_validation/logs"; mkdir -p "$LOGS"
read -r G1 G2 G3 <<< "${JUDGE_GPUS:-0 1 2}"
for spec in "llama8b $G1" "qwen7b $G2" "mistral7b $G3"; do set -- $spec
  ( ${PYTHON:-python} "$HERE/reference_validation_judging.py" --stage reference --judge $1 --gpu $2 && ${PYTHON:-python} "$HERE/reference_validation_judging.py" --stage answer --judge $1 --gpu $2 ) > "$LOGS/judge_$1.log" 2>&1 &
done
wait; echo ALL_JUDGES_DONE >> "$LOGS/judges_finished.flag"
