#!/bin/bash
# 3RScan ours (ours_290e110 maps through eval/chain_fill) rescored after chain_fill takes the run's own moves
# (10-29: no 50 % surface re-check, as eval/timeline_complete.py). Official evaluator, online slice, 0.5 m, objects only.
# The earlier 0.5 m metrics stay as METHOD_DIR/khronos_metrics_m050_before_A. Evaluation only, nothing remapped.
FT=/home/jixian/Desktop/FT; R=$FT/wt_takeover_runs; PY=/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python
LOW=(ionice -c3 nice -n 19 taskset -c 0-9)
step() { echo "$(date +%H:%M) $*"; }
for ref in $(grep -v '^#' $FT/datasets/3rscan/CHAINS_P3.txt | cut -c1-8); do
  room=p3_$ref; d=$R/3rscan/$room/ours_290e110
  [[ -d $d/khronos_metrics && ! -e $d/khronos_metrics_m050_before_A ]] && mv $d/khronos_metrics $d/khronos_metrics_m050_before_A
  rm -rf $d/khronos_metrics
  THREERSCAN_OBJECTS_ONLY=1 THREERSCAN_JOBS=2 "${LOW[@]}" $PY $FT/eval/threerscan_khronos_eval.py $room ours $d > $R/3rscan/$room/khronos_metrics_ours_chainfill_A.log 2>&1 \
    && step "$(tail -1 $R/3rscan/$room/khronos_metrics_ours_chainfill_A.log | cut -c1-160)" || step "$room FAILED ($(tail -1 $R/3rscan/$room/khronos_metrics_ours_chainfill_A.log | cut -c1-160))"
done
step "3RScan ours rescored with chain_fill A"
