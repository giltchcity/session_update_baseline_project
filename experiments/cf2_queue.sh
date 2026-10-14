#!/usr/bin/env bash
# The bathroom counterfactual (restore_cf2.py) on the full synthetic chain's B (syn_chain_4ca6976, the final mapping) as soon as its
# checkpoint is tidied: A = the accepted A (syn_row4s0). GPU renders only, under GPU.lock (~3 min). Markers in runs_quick3k/cf2_queue.log.
set -u
X=$(cd "$(dirname "$0")" && pwd); RR=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006; Q=$RR/code_mine_20261014/runs_quick3k
PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python; L=$Q/cf2_queue.log; LOCK=/home/jixian/Desktop/FT/runs/GPU.lock; S=$RR/syn_chain_4ca6976/map
until grep -q "^\^tidied $S " $Q/chain_queue.log 2>/dev/null; do sleep 30; done
D=$Q/cf2_syn_chain_4ca6976; mkdir -p $D
echo "^start cf2 $(date '+%T')" >> $L
flock $LOCK $X/run_logged.sh $D env OMP_NUM_THREADS=2 $PY -W ignore $X/restore_cf2.py $RR/worktrees/r21c_base $D/renders $RR/syn_row4s0/checkpoint_a.pt $S/checkpoint_b.pt a2700,a2850,a2985,b2625,b3447
echo "^done cf2 $(date '+%T') $(cat $D/status.txt)" >> $L
