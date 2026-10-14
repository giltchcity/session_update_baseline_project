#!/usr/bin/env bash
# D1 voxel verdicts on the real chain (real_row4f3e5 A/B/C) on the CPU, one heavy process at a time: waits for the synthetic B run,
# then C, A, B in turn; a run starts only with >= 12 GB of free host memory.
set -u
X=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$X"); R=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006
G=/home/jixian/Desktop/FT/results/abc_eval_v2/geometry; PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python; Q=$REPO/runs_quick3k
until [ -s $Q/d1_full_syn_65ee401/status.txt ]; do sleep 20; done
for s in c a b; do S=$(echo $s | tr a-z A-Z); D=$Q/d1_full_real${s}_$(git -C $REPO rev-parse --short HEAD); mkdir -p $D; cd $D
  until [ $(free -g | awk '/Mem:/{print $7}') -ge 12 ]; do sleep 30; done
  $X/run_logged.sh . env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 nice -n 5 $PY -W ignore $X/d1_voxels.py $R/real_row4f3e5/checkpoint_$s.pt real $s $G/${S}_reference_1cm.ply 10
  echo "^done real $s $(date '+%T') $(cat status.txt)" >> $Q/d1_real_cpu_queue.log
done
