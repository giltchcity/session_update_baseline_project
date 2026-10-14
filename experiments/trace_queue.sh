#!/usr/bin/env bash
# CPU, one heavy process at a time: D1 run as in the chain on synthetic B and real C/A/B, every processed frame's tracks recorded
# (d1_trace.py) -- the evidence for the object-level move rule (observed-centroid displacement >= min_dynamic_displacement).
set -u
X=$(cd "$(dirname "$0")" && pwd); R=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006; PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python
Q=$R/code_mine_20261014/runs_quick3k; L=$Q/trace_queue.log; W=$R/worktrees/static_kf_e37823e
for job in "synthetic b $R/syn_row4f3e5_r11p/checkpoint_b.pt" "real c $R/real_row4f3e5/checkpoint_c.pt" "real a $R/real_row4f3e5/checkpoint_a.pt" "real b $R/real_row4f3e5/checkpoint_b.pt"; do
  set -- $job; D=$Q/trace_$1_$2; mkdir -p $D; cd $D
  until [ $(free -g | awk '/Mem:/{print $7}') -ge 12 ]; do sleep 30; done
  echo "^start trace $1 $2 $(date '+%T')" >> $L
  $X/run_logged.sh . env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 nice -n 5 $PY -W ignore $X/d1_trace.py $3 $1 $2 0 $D/trace.jsonl
  echo "^done trace $1 $2 $(date '+%T') $(cat status.txt)" >> $L
done
echo "^trace queue end $(date '+%T')" >> $L
