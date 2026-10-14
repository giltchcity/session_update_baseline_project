#!/usr/bin/env bash
# Measurement of the per-iteration parts (branch parts_timing, measurement only; supervisor 13:50): the first 150 frames of real B from
# the accepted real A, every part the chain adds per iteration timed with synchronised timers. Under GPU.lock (~4-5 min). The run's checkpoints are deleted afterwards (ledger); the PARTS_TIMING lines stay in run.log.
set -u
X=$(cd "$(dirname "$0")" && pwd); RR=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006; CODE=$RR/worktrees/parts_timing
PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python; Q=$RR/code_mine_20261014/runs_quick3k; L=$Q/parts_timing_queue.log
LOCK=/home/jixian/Desktop/FT/runs/GPU.lock; K=$(git -C $CODE rev-parse --short HEAD); D=$Q/parts150_real_b_$K
git -C $CODE status --porcelain | grep -q . && { echo "^refused dirty $CODE" >> $L; exit 3; }
until grep -q '^\^done refine synthetic b' $Q/post_chain_queue.log 2>/dev/null; do sleep 60; done      # after the synthetic refinement (supervisor 13:50)
mkdir -p $D; cd $D; git -C $CODE rev-parse HEAD > code_commit.txt
echo "^start parts150 $K $(date '+%T')" >> $L
flock $LOCK $X/run_logged.sh . env OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -W ignore $X/quick_map.py $CODE $D/map 150 real b
echo "^done parts150 $K $(date '+%T') $(cat status.txt 2>/dev/null)" >> $L
for f in $(ls $D/map/checkpoint_*.pt $D/map/session_*/timeline*.pkl 2>/dev/null); do
  [ -L "$f" ] || { echo "$(date '+%F %H:%M') immune-cache measurement (real B 150 frames; only run.log is read): $(stat -c %s $f) $f" >> $RR/analysis/DISK_RECLAIM_20261013.txt; rm -- "$f"; }; done
grep "PARTS_TIMING" $D/run.log | tail -1 >> $L
echo "^immune cache queue end $(date '+%T')" >> $L
