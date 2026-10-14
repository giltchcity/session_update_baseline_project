#!/usr/bin/env bash
# restore_cf3.py on the synthetic chain's B (the arm 'every A row back to its A values'; supervisor 13:44), under GPU.lock (~1-2 min).
set -u
X=$(cd "$(dirname "$0")" && pwd); RR=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006; Q=$RR/code_mine_20261014/runs_quick3k
PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python; L=$Q/cf2_queue.log; LOCK=/home/jixian/Desktop/FT/runs/GPU.lock; S=$RR/syn_chain_4ca6976/map
D=$Q/cf3_syn_chain_4ca6976; mkdir -p $D
echo "^start cf3 $(date '+%T')" >> $L
flock $LOCK $X/run_logged.sh $D env OMP_NUM_THREADS=2 $PY -W ignore $X/restore_cf3.py $RR/worktrees/r21c_base $D/renders $RR/syn_row4s0/checkpoint_a.pt $S/checkpoint_b.pt a2700,a2850,a2985,b2625,b3447
echo "^done cf3 $(date '+%T') $(cat $D/status.txt)" >> $L
