#!/bin/bash
# Runs all judge passes of the tier-specific quality study (sufficiency, then answer grading), one judge per GPU.
# Run from the repository root:  JUDGE_GPUS="0 1 2" bash scripts/evaluation/tier_quality/run_tier_quality_judges.sh
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGS="${OUTPUT_DIR:-${FRESHCACHE_ROOT:-.}/outputs}/tier_quality/logs"; mkdir -p "$LOGS"
read -r G1 G2 G3 <<< "${JUDGE_GPUS:-0 1 2}"
for spec in "llama8b $G1" "qwen7b $G2" "mistral7b $G3"; do set -- $spec
  ( ${PYTHON:-python} "$HERE/tier_quality_judging.py" --kind sufficiency --judge $1 --gpu $2 && ${PYTHON:-python} "$HERE/tier_quality_judging.py" --kind answer --judge $1 --gpu $2 ) > "$LOGS/judge_$1.log" 2>&1 &
done
wait; echo done > "$LOGS/judges_finished.flag"
