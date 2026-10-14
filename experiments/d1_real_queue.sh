#!/usr/bin/env bash
# D1 voxel verdicts on the real chain (real_row4f3e5 A/B/C, the accepted F3e5 chain), GPU, one job at a time behind the lock
for s in c a b; do S=$(echo $s | tr a-z A-Z); D=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/code_mine_20261014/runs_quick3k/d1_full_real${s}_65ee401; mkdir -p $D; cd $D
  flock /home/jixian/Desktop/FT/runs/GPU.lock /home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/code_mine_20261014/experiments/run_logged.sh . env OMP_NUM_THREADS=4 /home/jixian/Desktop/FT/envs/gs-cu128/bin/python -W ignore /home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/code_mine_20261014/experiments/d1_voxels.py /home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/real_row4f3e5/checkpoint_$s.pt real $s /home/jixian/Desktop/FT/results/abc_eval_v2/geometry/${S}_reference_1cm.ply 10
  echo "^done real $s $(date '+%T') $(cat status.txt)" >> /home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/code_mine_20261014/runs_quick3k/d1_real_queue_65ee401.log
done
