#!/bin/bash
set -u
P=<HOME>/miniconda3/envs/graphrag/bin/python
D=<PROJECT_ROOT>/validation_V2/l2_evidence_verification
G=${GPU:-3}
cd <PROJECT_ROOT>
echo "=== s6 audit started $(date -Is) GPU=$G ==="
$P $D/s6_answer_audit.py gen --gpu $G || exit 1
$P $D/s6_answer_audit.py judge --judge llama8b --gpu $G || exit 1
$P $D/s6_answer_audit.py judge --judge qwen7b  --gpu $G || exit 1
$P $D/s6_answer_audit.py suff --judge llama8b --gpu $G || exit 1
$P $D/s6_answer_audit.py suff --judge qwen7b  --gpu $G || exit 1
$P $D/s6_answer_audit.py analyze || exit 1
echo "=== finished $(date -Is) ==="
