#!/usr/bin/env bash
# GaME rows (only long GPU job). syn_game is done. Row 3 real first (no decision needed), then
# row 4 with the static-frame decision ($R/STATIC_DECISION, "off" if absent after 180 min), then rows 2 and 1.
set -u
R=/home/jixian/Desktop/FT/runs
G=$R/game_final_20260925
GPY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python
cd /home/jixian/Desktop/FT/wt_layer_43c
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
score() { local ds=$1 run=$2 out=$3
  if [[ $ds == synthetic ]]; then bash /home/jixian/Desktop/FT/eval/scene/eval_synthetic_scene.sh $run $out 0.02 0.01 > $out.log 2>&1
  else nice -n 10 bash /home/jixian/Desktop/FT/eval/scene/eval_real_scene.sh $run $out 0.02 0.01 > $out.log 2>&1; fi; }
game() { local ds=$1 mode=$2 short=syn; shift 2; [[ $ds == real ]] && short=real
  if $GPY -m update_layer.run_game $ds $G/${short}_$mode --mode $mode "$@" > $G/${short}_$mode.log 2>&1; then
    score $ds $G/${short}_$mode $G/eval_${short}_$mode; echo "$(date +%T) done game ${short}_$mode"
  else echo "$(date +%T) FAILED game ${short}_$mode"; fi; }
echo "$(date +%T) GaME queue 2 started"
game real game
for k in $(seq 1 180); do [[ -s $R/STATIC_DECISION ]] && break; sleep 60; done
d=$(cat $R/STATIC_DECISION 2>/dev/null || echo off); flag=""; [[ $d == on ]] && flag="--static-frames"
echo "$(date +%T) static decision for GaME + layer: $d"
game synthetic layer $flag
game real layer $flag
bash /home/jixian/Desktop/FT/eval/scene/post_retention.sh synthetic $G > /dev/null 2>&1
bash /home/jixian/Desktop/FT/eval/scene/post_retention.sh real $G > /dev/null 2>&1
echo "$(date +%T) GAME_34_DONE"
game synthetic naive
game synthetic scratch
game real naive
game real scratch
bash /home/jixian/Desktop/FT/eval/scene/post_retention.sh synthetic $G > /dev/null 2>&1
bash /home/jixian/Desktop/FT/eval/scene/post_retention.sh real $G > /dev/null 2>&1
echo "$(date +%T) GAME_DONE"
