#!/usr/bin/env bash
# D1 as 4D Gaussians, the three watched moves in close-up, every 1/12 s: Isaac's ground truth, our 4D map with the
# tracked furniture (track_d1.py), and the plain 4D map (rounds only). Each GPU job under runs/GPU.lock (gpu_run.sh).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)                 # this folder of update_layer/extras (the scripts)
R=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006; V=$R/analysis/paper_figs_20261022/video4d; D=$V/d1_4d
PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python; IW='D:\3Study\ETH\FT\Synthetic_ABC\final_state_renders_20261022'
I=/mnt/d/3Study/ETH/FT/Synthetic_ABC/final_state_renders_20261022
export OMP_NUM_THREADS=2
isaac() {   # SESSION CAMERA_NAME T0 T1 OUT_NAME
  cd $I && timeout 3600 cmd.exe /c "D:\\3Study\\Isaac\\python.bat $IW\\render_video_isaac.py --session $1 --camera $IW\\$2.json --t0 $3 --t1 $4 --step 0.0833333 --output $IW\\$5 --history cut --subframes 24"
}
export -f isaac; export I IW
for e in "b 72 40.0 47.5" "b 164 29.6 34.0" "a 164 115.0 122.5"; do
  set -- $e; n=$1_$2
  $HERE/../video4d/gpu_run.sh bash -c "isaac $1 cam_$n $3 $4 d1_$n" > $D/isaac_$n.log 2>&1
  echo "$(date +%T) Isaac $n: $(grep -c RENDERED $D/isaac_$n.log)"
  tr=$(ls $D/out/$1_*.json | tr '\n' ' ')
  $HERE/../video4d/gpu_run.sh $PY -W ignore $HERE/render_d1_4d.py $HERE/cam_$n.json $R/syn_snap_c298ed5 $1 $3:$4:0.0833333 $D/ours_$n --lamps $V/v2/isaac_done_a.json --tracks $tr > $D/ours_$n.log 2>&1
  $HERE/../video4d/gpu_run.sh $PY -W ignore $HERE/render_d1_4d.py $HERE/cam_$n.json $R/syn_snap_c298ed5 $1 $3:$4:0.0833333 $D/rounds_$n --lamps $V/v2/isaac_done_a.json --rounds-only > $D/rounds_$n.log 2>&1
  echo "$(date +%T) ours $n: $(grep -c RENDERED $D/ours_$n.log), rounds only: $(grep -c RENDERED $D/rounds_$n.log)"
done
echo RENDER_DONE
