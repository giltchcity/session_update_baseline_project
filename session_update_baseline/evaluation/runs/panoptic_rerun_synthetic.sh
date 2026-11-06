#!/usr/bin/env bash
# Panoptic Mapping rerun, synthetic A -> B (B loads A's map), for the results table's online object F1: the settings of
# the earlier synthetic run (datasets/synthetic_ab/evaluation_complete_20260917/run_panoptic_ab.sh: projective
# tracking, merged deactivated submaps, 60/60/60, the synthetic camera, A stamped from 1 s and B from 141 s), on the
# input and launcher analysis/panoptic_synthetic_prepare.py rebuilt with the 09-17 scripts, and, as panoptic_rerun.sh, a
# map saved every 2 s of dataset time after the mapper's ACK. 5 cm objects and background (user 10-31, every method at
# the same resolution; the earlier run's 10 cm background: runs/panoptic_mapping/rerun_snap2s_20261029/synthetic).
# Waits for WAIT_PID to exit, then each visit under runs/HEAVY.lock.
# usage: panoptic_rerun_synthetic.sh OUT_DIR [WAIT_PID]     (OUT_DIR as given to panoptic_synthetic_prepare.py)
set -Eeuo pipefail
OUT=$(realpath "$1"); WAIT_PID=${2:-}
FT=/home/jixian/Desktop/FT
RUNNER=$OUT/synthetic_launcher/run_panoptic_session.sh
step() { echo "[$(date +%F' '%T)] $*"; }
mkdir -p "$OUT/synthetic"
if [[ -n $WAIT_PID ]]; then
  step "waiting for pid $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
COMMON=(PANOPTIC_IMAGE=panoptic_mapping:synthetic-ack
        FLAT_DATASET_ROOT="$OUT/synthetic_input" MAX_FRAMES=0 PLAY_RATE=1
        VISUALIZE=false USE_MONOTONIC_PLAYER=true FLAT_PLAYER_REQUIRE_ACK=1 PANOPTIC_SNAPSHOT_PERIOD_S=2
        POST_WAIT=2 PLAYBACK_TIMEOUT=0 PANOPTIC_MEMORY=12g PANOPTIC_MEMORY_SWAP=16g
        ID_TRACKER_TYPE=projective MERGE_DEACTIVATED_SUBMAPS=true UPDATE_ONLY_TRACKED_SUBMAPS=true
        SMALL_INSTANCE_VOXEL_SIZE=0.05 MEDIUM_INSTANCE_VOXEL_SIZE=0.05 LARGE_INSTANCE_VOXEL_SIZE=0.05
        BACKGROUND_VOXEL_SIZE=0.05 UNKNOWN_VOXEL_SIZE=0.05
        REQUIRED_REOBSERVATIONS=60 DEACTIVATE_AFTER_MISSED_DETECTIONS=60 CHANGE_DETECTION_FREQUENCY=60
        CAMERA_FX=588.897289890142 CAMERA_FY=588.897289890142 CAMERA_CX=340 CAMERA_CY=240
        CAMERA_WIDTH=680 CAMERA_HEIGHT=480)
visit() {  # visit RUN_NAME ENV...
  local name=$1 dir=$OUT/synthetic; shift
  if [[ -s $dir/$name.panmap ]]; then step "$name: already saved"; return; fi
  step "$name: start"
  flock "$FT/runs/HEAVY.lock" env "${COMMON[@]}" "$@" bash "$RUNNER" "$name" "$dir" > "$dir/$name.log" 2>&1
  [[ -s $dir/$name.panmap ]] || { step "$name: no map saved, see $dir/$name.log"; exit 3; }
  step "$name: saved $(du -h "$dir/$name.panmap" | cut -f1), $(ls "$dir/${name}_snapshots" | wc -l) maps during the visit"
}
visit session_a FLAT_PLAYER_STAMP_OFFSET=1
visit session_b FLAT_PLAYER_STAMP_OFFSET=141 LOAD_MAP=true LOAD_FILE="$OUT/synthetic/session_a.panmap"
step "synthetic done: $(du -sh "$OUT/synthetic" | cut -f1)"
