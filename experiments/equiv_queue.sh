#!/usr/bin/env bash
# Equivalence of the cleaned tree (chain_clean) with the tree that produces the chain's results (r21c_base 4ca6976; its mapping code is
# final0638's): the same inputs as existing runs of the old trees, so nothing old is rerun (supervisor 10:08, option b):
#   synthetic B 150 frames (quick_map)  vs speed150_final0638_b9400c9 (same mapping code; refinement off in both)
#   real B -> C 30 frames each           vs smoke30_real_bc_4ca6976 (the chain queue's own smoke)
#   4000-step refinement (quick_screen)  vs speed3k_r21c_base_4ca6976 (same input R11p B, same sampling restore)
# Each GPU step takes GPU.lock on its own, so the steps fall between the chain's steps. Markers in runs_quick3k/equiv_queue.log.
set -u
X=$(cd "$(dirname "$0")" && pwd); RR=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006; CODE=$RR/worktrees/chain_clean
PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python; Q=$RR/code_mine_20261014/runs_quick3k; L=$Q/equiv_queue.log
LOCK=/home/jixian/Desktop/FT/runs/GPU.lock; K=$(git -C $CODE rev-parse --short HEAD)
git -C $CODE status --porcelain | grep -q . && { echo "^refused dirty $CODE" >> $L; exit 3; }
run() { local name=$1; shift; local D=$Q/$name; mkdir -p $D; cd $D; git -C $CODE rev-parse HEAD > code_commit.txt
  echo "^start $name $(date '+%T')" >> $L; local t0=$(date +%s)
  flock $LOCK $X/run_logged.sh . env OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$@"
  echo "^done $name $(date '+%T') seconds=$(( $(date +%s) - t0 )) $(cat status.txt 2>/dev/null)" >> $L; }
run equiv150_clean_$K $PY -W ignore $X/quick_map.py $CODE $Q/equiv150_clean_$K/map 150; $X/clean_speed_ckpts.sh equiv150_clean_$K
until grep -q '^\^done code_mine_20261014/runs_quick3k/smoke30_real_bc_' $Q/chain_queue.log 2>/dev/null; do sleep 30; done
run equiv30_real_bc_clean_$K $PY -W ignore $X/quick_map.py $CODE $Q/equiv30_real_bc_clean_$K/map 30 real bc
for f in $(ls $Q/equiv30_real_bc_clean_$K/map/checkpoint_*.pt $Q/equiv30_real_bc_clean_$K/map/session_*/timeline*.pkl 2>/dev/null); do
  [ -L "$f" ] || { echo "$(date '+%F %H:%M') equivalence smoke (real B->C 30 frames, chain_clean): $(stat -c %s $f) $f" >> $RR/analysis/DISK_RECLAIM_20261013.txt; rm -- "$f"; }; done
run equiv4k_clean_$K $PY -W ignore $X/quick_screen.py short4k clean synthetic $Q/equiv4k_clean_$K $CODE
echo "^equiv queue end $(date '+%T')" >> $L
