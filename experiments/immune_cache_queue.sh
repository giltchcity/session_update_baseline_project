#!/usr/bin/env bash
# Measurement of the immune() cache (branch immune_cache, measurement only; supervisor 13:20): the first 150 frames of real B from
# the accepted real A, with every immune() call computing the uncached mask too (timed, synchronised) and asserting it equals the
# cached one. Under GPU.lock (~4-5 min). The run's checkpoints are deleted afterwards (ledger); the IMMUNE_CACHE lines stay in run.log.
set -u
X=$(cd "$(dirname "$0")" && pwd); RR=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006; CODE=$RR/worktrees/immune_cache
PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python; Q=$RR/code_mine_20261014/runs_quick3k; L=$Q/immune_cache_queue.log
LOCK=/home/jixian/Desktop/FT/runs/GPU.lock; K=$(git -C $CODE rev-parse --short HEAD); D=$Q/immune150_real_b_$K
git -C $CODE status --porcelain | grep -q . && { echo "^refused dirty $CODE" >> $L; exit 3; }
until grep -q '^\^done real_chain_4ca6976' $Q/chain_queue.log 2>/dev/null; do sleep 60; done      # after the timed real chain
mkdir -p $D; cd $D; git -C $CODE rev-parse HEAD > code_commit.txt
echo "^start immune150 $K $(date '+%T')" >> $L
flock $LOCK $X/run_logged.sh . env OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -W ignore $X/quick_map.py $CODE $D/map 150 real b
echo "^done immune150 $K $(date '+%T') $(cat status.txt 2>/dev/null)" >> $L
for f in $(ls $D/map/checkpoint_*.pt $D/map/session_*/timeline*.pkl 2>/dev/null); do
  [ -L "$f" ] || { echo "$(date '+%F %H:%M') immune-cache measurement (real B 150 frames; only run.log is read): $(stat -c %s $f) $f" >> $RR/analysis/DISK_RECLAIM_20261013.txt; rm -- "$f"; }; done
grep "IMMUNE_CACHE" $D/run.log | tail -1 >> $L
echo "^immune cache queue end $(date '+%T')" >> $L
