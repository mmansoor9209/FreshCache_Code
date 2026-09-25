#!/bin/bash
# Revised Stage 3b pipeline. Read-only on all pre-existing artifacts.
set -u
P=<HOME>/miniconda3/envs/graphrag/bin/python
D=<PROJECT_ROOT>/validation/gold_verification_400/revised_3b
GPU=${GPU:-3}
cd <PROJECT_ROOT>
echo "=== revised Stage 3b started $(date -Is)  GPU=$GPU ==="
echo "--- r1 build (already run; re-running is idempotent) ---"; $P $D/r1_build.py || exit 1
echo "--- r2 judge llama8b ---";  $P $D/r2_judge.py --judge llama8b  --gpu $GPU || exit 1
echo "--- r2 judge qwen7b ---";   $P $D/r2_judge.py --judge qwen7b   --gpu $GPU || exit 1
echo "--- r3 tie-break set ---";  $P $D/r3_tiebreak_set.py || exit 1
echo "--- r2 judge mistral7b (disagreements only) ---"
$P $D/r2_judge.py --judge mistral7b --gpu $GPU --only-disagreements || exit 1
echo "--- r4 analyze ---";        $P $D/r4_analyze.py || exit 1
echo "=== revised Stage 3b finished $(date -Is) ==="
