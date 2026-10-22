#!/usr/bin/env bash
# The 4D video v7: equal panels cut to the same window at their rendered size, 1 px outlines (CPU only).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)                 # this folder of update_layer/extras (the scripts)
R=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006; V=$R/analysis/paper_figs_20261022/video4d
I=/mnt/d/3Study/ETH/FT/Synthetic_ABC/final_state_renders_20261022; O=$I/video_v2_out7; mkdir -p $O $V/v2/full7
compose() { CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 /home/jixian/Desktop/FT/envs/gs-cu128/bin/python -W ignore $HERE/compose_video.py $HERE/camera_video_v2.json $I/video_v2_full2 $V/v2/ours2 $V/v2/changes5.json $O "$@"; }
compose --sessions a > $V/v2/full7/compose_a.log 2>&1 &
pa=$!
compose --sessions b > $V/v2/full7/compose_b.log 2>&1
wait $pa
compose --sessions x --encode --fps 24 --name syn_4d_v7 > $V/v2/full7/encode.log 2>&1
cp $O/syn_4d_v7.mp4 $V/syn_4d_v7.mp4 && echo "VIDEO $V/syn_4d_v7.mp4 $(grep -c FRAME $V/v2/full7/compose_a.log) + $(grep -c FRAME $V/v2/full7/compose_b.log) frames"
