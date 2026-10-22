#!/usr/bin/env bash
# Isaac ground truth of the whole 4D video, every 0.25 s of scene time (A 0-139.5 s, B 0-118.5 s), each image its time's
# alone (--history cut --subframes 24: no ghost of the previous frame). One GPU job per chunk under runs/GPU.lock.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)                 # this folder of update_layer/extras (the scripts)
V=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/analysis/paper_figs_20261022/video4d; mkdir -p $V/v2/full2
for c in "a 0 34.75" "a 35 69.75" "a 70 104.75" "a 105 139.5" "b 0 39.75" "b 40 79.75" "b 80 118.5"; do
  set -- $c
  $HERE/gpu_run.sh bash $HERE/isaac_run.sh $1 $2 $3 0.25 video_v2_full2 --history cut --subframes 24 > $V/v2/full2/isaac_$1_$2.log 2>&1
  echo "$(date +%T) Isaac $1 $2-$3: $(grep -c RENDERED $V/v2/full2/isaac_$1_$2.log) rendered"
done
echo ISAAC_DONE
