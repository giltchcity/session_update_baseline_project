#!/usr/bin/env bash
# Hold Flat ours (supervisor 15:30: Flat ours depends on the mapping version, which the decisive experiment decides; Flat native is
# version-independent and runs after the experiment, from the ordered sequence).
# The post-chain queue's next step after the real scoring is Flat ours. It is stopped only when the queue has logged '^start flat ours'
# and its flock is still WAITING: the decisive experiment holds the lock by then, and the flock has no child yet. A queue whose Flat ours
# python already runs is left alone and reported. The real scoring and its acceptance step are never touched. PIDs are taken from the
# process tree of the given queue PID; kill by PID only.
# usage: hold_flat_ours.sh POST_CHAIN_QUEUE_PID
set -u
QP=$1; Q=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/code_mine_20261014/runs_quick3k; L=$Q/post_chain_queue.log
until grep -q '^\^start flat ours' $L 2>/dev/null; do kill -0 $QP 2>/dev/null || { echo "^hold: queue $QP gone before Flat ours $(date '+%T')" >> $L; exit 0; }; sleep 5; done
sleep 3
desc() { local c; for c in $(ps -o pid= --ppid $1); do echo $c; desc $c; done; }
D=$(desc $QP | tr '\n' ' ')
RUN=$( [ -n "$D" ] && ps -o pid=,args= -p ${D// /,} 2>/dev/null | grep "update_layer.run" | grep -v " flock " )
if [ -n "$RUN" ]; then echo "^hold NOT applied $(date '+%T'): Flat ours already runs: $RUN" >> $L; exit 5; fi
echo "^hold $(date '+%T') by Claude: Flat ours stopped while its flock waited (no child); queue $QP and its waiting descendants $D (supervisor 15:30)" >> $L
kill $QP
for p in $D; do kill $p 2>/dev/null; done
echo "^post chain queue stopped before Flat ours $(date '+%T'); Flat native runs from the ordered sequence after the decisive experiment" >> $L
