#!/usr/bin/env bash
# The plant moved at the end of B (flower_0003, unit 109), after render_all.sh: Isaac, ours with the tracked furniture,
# rounds only (each under runs/GPU.lock); then the four clips composed (CPU).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)                 # this folder of update_layer/extras (the scripts)
R=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006; V=$R/analysis/paper_figs_20261022/video4d; D=$V/d1_4d
PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python; IW='D:\3Study\ETH\FT\Synthetic_ABC\final_state_renders_20261022'
I=/mnt/d/3Study/ETH/FT/Synthetic_ABC/final_state_renders_20261022
export OMP_NUM_THREADS=2
until grep -q RENDER_DONE $D/render_all.log; do sleep 10; done
isaac() { cd $I && timeout 3600 cmd.exe /c "D:\\3Study\\Isaac\\python.bat $IW\\render_video_isaac.py --session $1 --camera $IW\\$2.json --t0 $3 --t1 $4 --step 0.0833333 --output $IW\\$5 --history cut --subframes 24"; }
export -f isaac; export I IW
$HERE/../video4d/gpu_run.sh bash -c "isaac b cam_b_109 110.5 118.5 d1_b_109" > $D/isaac_b_109.log 2>&1
$HERE/../video4d/gpu_run.sh $PY -W ignore $HERE/render_d1_4d.py $HERE/cam_b_109.json $R/syn_snap_c298ed5 b 110.5:118.5:0.0833333 $D/ours_b_109 --lamps $V/v2/isaac_done_a.json --tracks $(ls $D/out/b_*.json | tr '\n' ' ') > $D/ours_b_109.log 2>&1
$HERE/../video4d/gpu_run.sh $PY -W ignore $HERE/render_d1_4d.py $HERE/cam_b_109.json $R/syn_snap_c298ed5 b 110.5:118.5:0.0833333 $D/rounds_b_109 --lamps $V/v2/isaac_done_a.json --rounds-only > $D/rounds_b_109.log 2>&1
echo "$(date +%T) plant: Isaac $(grep -c RENDERED $D/isaac_b_109.log), ours $(grep -c RENDERED $D/ours_b_109.log), rounds only $(grep -c RENDERED $D/rounds_b_109.log)"
CUDA_VISIBLE_DEVICES= $PY -W ignore $HERE/compose_d1_clips.py $D $D/clips > $D/compose.log 2>&1
tail -2 $D/compose.log
