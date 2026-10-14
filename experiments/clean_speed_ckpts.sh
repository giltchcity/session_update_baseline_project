#!/usr/bin/env bash
# Disk guard for the 150-frame speed tests: each writes checkpoint_b.pt and a 0-step post_ref copy (~18 GB each, the whole A map is
# inside); only session_b/run.json is read by the speed gate. After each speed run has finished (its status.txt), record and delete them.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd); Q=$REPO/runs_quick3k; M=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/analysis/DISK_RECLAIM_20261013.txt
for name in speed150_final0638_b9400c9 speed150_static_kf_e37823e; do
  until [ -s $Q/$name/status.txt ]; do sleep 30; done
  for f in $Q/$name/map/checkpoint_b.pt $Q/$name/map/checkpoint_b_post_ref.pt; do
    [ -f "$f" ] && [ ! -L "$f" ] && { echo "$(date '+%F %H:%M') speed-test checkpoint (only run.json is used): $(stat -c %s "$f") $f" >> $M; rm -- "$f"; }
  done
done
