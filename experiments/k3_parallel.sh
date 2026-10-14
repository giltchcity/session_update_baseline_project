#!/usr/bin/env bash
# CPU: the K3 offline counts (k3_offline.py, the chain's own K3 code from static_kf3) on synthetic B, real B and real C in
# parallel (3 threads each; the D1 trace queue holds 6 of the 16 cores). Markers in runs_quick3k/k3_queue.log.
set -u
X=$(cd "$(dirname "$0")" && pwd); PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python
Q=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/code_mine_20261014/runs_quick3k; L=$Q/k3_queue.log
CODE=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/worktrees/static_kf3; K=$(git -C $CODE rev-parse --short HEAD)
git -C $CODE status --porcelain | grep -q . && { echo "^refused dirty static_kf3" >> $L; exit 3; }
one() { local D=$Q/k3_$1_$2_$K; rm -rf $D; mkdir -p $D; echo "^start k3 $1 $2 $(date '+%T')" >> $L
  $X/run_logged.sh $D env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=3 nice -n 5 $PY -W ignore $X/k3_offline.py $CODE $1 $2 $D/k3.jsonl
  echo "^done k3 $1 $2 $(date '+%T') $(cat $D/status.txt) $(grep K3_OFFLINE $D/run.log)" >> $L; }
one synthetic b & one real b & one real c &
wait
echo "^k3 queue end $(date '+%T')" >> $L
