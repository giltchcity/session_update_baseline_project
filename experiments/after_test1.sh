#!/usr/bin/env bash
# After quick mapping test 1: the bathroom probe (CPU, the 10-13 ruler) and the raw-image renders of the bathroom cameras (GPU, own lock).
set -u
X=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$X"); R=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006
S=/tmp/claude-1000/-home-jixian-Desktop-FT/633eaaed-1a6e-4213-936b-f03c710b0681/scratchpad; PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python
T=$REPO/runs_quick3k/map70_final0638_b9400c9
until [ -s $T/status.txt ]; do sleep 30; done
D=$REPO/runs_quick3k/probe_bath_test1; mkdir -p $D; cd $D
$X/run_logged.sh . env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 $PY -W ignore $X/probe_bathroom_rows.py $T
D=$REPO/runs_quick3k/render_bath_test1; mkdir -p $D; cd $D
flock /home/jixian/Desktop/FT/runs/GPU.lock $X/run_logged.sh . env OMP_NUM_THREADS=2 $PY -W ignore $X/render_views.py $R/worktrees/final0638_b9400c9 $D/img a2700,a2850,a2985 A_input=$R/syn_row4s0/checkpoint_a.pt test1_B70=$T/checkpoint_b.pt R11p_fullB=$R/syn_row4f3e5_r11p/checkpoint_b.pt
