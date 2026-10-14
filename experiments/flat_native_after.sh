#!/usr/bin/env bash
# Flat native (GaME's official run, unmodified code; version-independent) in the GPU gap after R11e's real B scoring, while the
# data-driven agreement test is built and checked offline on the CPU. run_flat.sh takes GPU.lock itself (no outer flock: the same
# lock twice would deadlock). Command of post_chain_queue.sh.
set -u
Q=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/code_mine_20261014/runs_quick3k; L=$Q/flat_native_after.log
D=/home/jixian/Desktop/FT/runs/game_flat_official_20261007
until grep -q '^\^real B G1\|^\^real B below\|^\^refused\|^\^stopped by the stop file' $Q/evstep_chain.log 2>/dev/null; do sleep 30; done
echo "^start flat native $(date '+%T')" >> $L
( cd $D && [ -f run.log ] && mv -f run.log run_before_$(date +%H%M).log; rm -rf $D/out; mkdir -p $D/out; bash $D/run_flat.sh ) > $Q/flat_native_stdout.log 2>&1
echo "^done flat native $(date '+%T') $(grep -m1 '^exit' $D/run.log)" >> $L
