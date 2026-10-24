#!/bin/bash
# GPU stages of the page-support verifier answer audit, in order (run verifier_answer_audit.py build first).
# Run from the repository root:  GPU=0 bash scripts/experiments/admission_controls/run_verifier_answer_audit.sh
set -u
P=${PYTHON:-python}
D="$(cd "$(dirname "$0")" && pwd)"
G=${GPU:-0}
echo "=== verifier answer audit started $(date -Is) GPU=$G ==="
$P $D/verifier_answer_audit.py gen --gpu $G || exit 1
$P $D/verifier_answer_audit.py judge --judge llama8b --gpu $G || exit 1
$P $D/verifier_answer_audit.py judge --judge qwen7b  --gpu $G || exit 1
$P $D/verifier_answer_audit.py suff --judge llama8b --gpu $G || exit 1
$P $D/verifier_answer_audit.py suff --judge qwen7b  --gpu $G || exit 1
$P $D/verifier_answer_audit.py analyze || exit 1
echo "=== finished $(date -Is) ==="
