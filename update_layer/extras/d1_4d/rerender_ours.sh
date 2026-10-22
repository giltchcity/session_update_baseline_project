#!/usr/bin/env bash
# Ours with the tracked furniture again, the place each model left cleared of unlabelled rows (render_d1_4d.py, 10-23),
# for the four watched moves (each under runs/GPU.lock); then the clips composed again.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)                 # this folder of update_layer/extras (the scripts)
R=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006; V=$R/analysis/paper_figs_20261022/video4d; D=$V/d1_4d
PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python; export OMP_NUM_THREADS=2
for e in "b 72 40.0 47.5" "b 164 29.6 34.0" "b 109 110.5 118.5" "a 164 115.0 122.5"; do
  set -- $e; n=$1_$2; rm -rf $D/ours_$n
  $HERE/../video4d/gpu_run.sh $PY -W ignore $HERE/render_d1_4d.py $HERE/cam_$n.json $R/syn_snap_c298ed5 $1 $3:$4:0.0833333 $D/ours_$n --lamps $V/v2/isaac_done_a.json --tracks $(ls $D/out/$1_*.json | tr '\n' ' ') > $D/ours_$n.log 2>&1
  echo "$(date +%T) ours $n: $(grep -c RENDERED $D/ours_$n.log) $(grep 'card peak' $D/ours_$n.log)"
done
rm -rf $D/clips; CUDA_VISIBLE_DEVICES= $PY -W ignore $HERE/compose_d1_clips.py $D $D/clips > $D/compose.log 2>&1; grep VIDEO $D/compose.log
