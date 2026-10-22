#!/usr/bin/env bash
# Hindsight intervals (CPU) for B, then the B close-ups in hindsight (GPU, under runs/GPU.lock), then the clips (CPU).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)                 # this folder of update_layer/extras (the scripts)
R=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006; V=$R/analysis/paper_figs_20261022/video4d; H=$V/hindsight; D=$V/d1_4d
PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 $PY -W ignore $HERE/hindsight_intervals.py $R/syn_snap_c298ed5 b $D/out $H/intervals_b.json > $H/intervals_b.log 2>&1
grep -E "unit +(7|155|163|164|171|173) |track" $H/intervals_b.log
for e in "b_164 29.6 34.0" "b_72 40.0 47.5"; do
  set -- $e
  OMP_NUM_THREADS=2 $HERE/../video4d/gpu_run.sh $PY -W ignore $HERE/hindsight_render.py $HERE/../d1_4d/cam_$1.json $R/syn_snap_c298ed5 b $2:$3:0.0833333 $H/intervals_b.json $D/out $H/hind_$1 --lamps $V/v2/isaac_done_a.json > $H/hind_$1.log 2>&1
  echo "$1: $(grep -c RENDERED $H/hind_$1.log) rendered, $(grep 'card peak\|stopped' $H/hind_$1.log)"
done
CUDA_VISIBLE_DEVICES= $PY -W ignore $HERE/compose_hindsight_clips.py $D $H $H/clips b_164:B:table_0003 b_72:B:chair_0004 > $H/compose.log 2>&1; grep VIDEO $H/compose.log
