#!/bin/bash
# Shared-floor QReCC control: iterate score -> generate (3B) -> judge (two judges) -> sufficiency until nothing is pending (at most 6 passes). GPU 0.
# Run from the repository root:  bash scripts/experiments/live_streams/recorded_stream_controls/run_shared_floor_control.sh
ROOT=${FRESHCACHE_ROOT:-$(pwd)}; R=${OUTPUT_DIR:-$ROOT/outputs}/cross_workload; PY=${PYTHON:-python}
D="$(cd "$(dirname "$0")" && pwd)"; mkdir -p $R/logs
export OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
cnt () { if [ -f $R/gpu/in/$1 ]; then grep -c . $R/gpu/in/$1 || true; else echo 0; fi; }   # grep -c prints 0 itself on no match
for it in 0 1 2 3 4 5; do
  $PY $D/qrecc_shared_floor_control.py score >> $R/logs/shared_floor_score.out 2>&1
  g=$(cnt shared_floor_gen_P.jsonl); j=$(cnt shared_floor_judge_P.jsonl); v=$(cnt shared_floor_suffv_P.jsonl); echo "$(date +%T) pass $it gen $g judge $j suffv $v" >> $R/logs/shared_floor.log
  [ "$g$j$v" = "000" ] && { echo "$(date +%T) DONE" >> $R/logs/shared_floor.log; exit 0; }
  [ "$g" != 0 ] && $PY $D/cross_workload_generation_judging.py gen3 --gpu 0 --tasks shared_floor_gen_P.jsonl --tag sfP$it >> $R/logs/shared_floor_gen.out 2>&1
  if [ "$j" != 0 ]; then $PY $D/cross_workload_generation_judging.py judge2 --judge llama8b --gpu 0 --tasks shared_floor_judge_P.jsonl >> $R/logs/shared_floor_judge.out 2>&1
                          $PY $D/cross_workload_generation_judging.py judge2 --judge qwen7b --gpu 0 --tasks shared_floor_judge_P.jsonl >> $R/logs/shared_floor_judge.out 2>&1; fi
  if [ "$v" != 0 ]; then $PY $D/cross_workload_generation_judging.py suffv --judge llama8b --gpu 0 --tasks shared_floor_suffv_P.jsonl --tag sf >> $R/logs/shared_floor_suffv.out 2>&1
                          $PY $D/cross_workload_generation_judging.py suffv --judge qwen7b --gpu 0 --tasks shared_floor_suffv_P.jsonl --tag sf >> $R/logs/shared_floor_suffv.out 2>&1; fi
done
echo "$(date +%T) STOPPED after 6 passes" >> $R/logs/shared_floor.log
