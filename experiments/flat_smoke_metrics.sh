#!/usr/bin/env bash
# Flat smoke (ours): the two render-metric steps the stopped flat_smoke_queue.sh still had, with its commands (lines 30-35 of
# harness_4cca3e6). The supervisor asked at 15:36 to run them in a GPU gap now. The smoke map and its post_ref view were made at 15:23
# by r21c, and the same tree computes the metrics. One hold of GPU.lock for both steps; afterwards the smoke checkpoints are deleted
# as that queue did.
set -u
RR=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006; RT=$RR/worktrees/r21c; PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python
O=$RR/flat_smoke_20261014/ours; L=$RR/code_mine_20261014/runs_quick3k/flat_smoke_queue.log; LOCK=/home/jixian/Desktop/FT/runs/GPU.lock
[ "$(git -C $RT rev-parse HEAD)" = "$(cat $O/code_commit.txt)" ] || { echo "^refused: r21c moved since the smoke map" >> $L; exit 3; }
echo "^queued flat smoke ours metrics $(date '+%T')" >> $L
exec 9>>$LOCK
flock 9
echo "^start flat smoke ours metrics (lock held) $(date '+%T')" >> $L
for v in pre_ref post_ref; do src=$O; [ $v = post_ref ] && src=$O/post_ref
  ( cd $RT && env OMP_NUM_THREADS=2 $PY -W ignore -m update_layer.eval.game_render_metrics $src b flat $O/render_metrics_$v.json ) > $O/render_metrics_$v.log 2>&1
  rc=$?; echo "^done flat smoke ours metrics $v $(date '+%T') rc=$rc $(head -c 300 $O/render_metrics_$v.json 2>/dev/null)" >> $L
done
flock -u 9; exec 9>&-
for f in $(ls $O/checkpoint_*.pt $O/session_*/timeline*.pkl 2>/dev/null); do [ -L "$f" ] || { echo "$(date '+%F %H:%M') Flat smoke (ours, 40 frames per run; logs and metrics kept): $(stat -c %s $f) $f" >> $RR/analysis/DISK_RECLAIM_20261013.txt; rm -- "$f"; }; done
echo "^flat smoke ours metrics end $(date '+%T')" >> $L
