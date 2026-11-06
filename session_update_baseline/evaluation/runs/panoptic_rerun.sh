#!/usr/bin/env bash
# Panoptic Mapping (ethz-asl/panoptic_mapping) rerun for the results table's online object F1: Flat run1 -> run2 and
# real A -> B -> C, each visit loading the previous visit's map (official loadMap), with the settings of the runs the
# table used before (Flat: the default flat_groundtruth config, as runs/panoptic_mapping/flat_ab_20261023; real:
# scripts/panoptic_mapping/run_live_chain_abc.sh's COMMON_ENV), two changes:
#   every frame is processed before the next is played (FLAT_PLAYER_REQUIRE_ACK=1; the old real run played at 0.25x
#   without it and its mapper skipped 1 % of A's frames, 7 % of B's and 34 % of C's);
#   a map is saved every 2 s of dataset time (PANOPTIC_SNAPSHOT_PERIOD_S, flat_data_player_monotonic.py), as the
#   official series evaluation reads them (panoptic_mapping_utils evaluate_series.launch / evaluation_manager.py).
# The image panoptic_mapping:synthetic-ack has the processed-input ACK (same mapper binary as eval-observer).
# Waits for WAIT_PID (the running GaME mapping) to exit, then each visit under runs/HEAVY.lock.
# usage: panoptic_rerun.sh OUT_DIR [WAIT_PID]
set -Eeuo pipefail
OUT=$(realpath -m "$1"); WAIT_PID=${2:-}
FT=/home/jixian/Desktop/FT
RUNNER=$FT/scripts/panoptic_mapping/run_flat_save_monotonic.sh
LOCK=$FT/runs/HEAVY.lock
step() { echo "[$(date +%F' '%T)] $*"; }
mkdir -p "$OUT/flat" "$OUT/real"
if [[ -n $WAIT_PID ]]; then
  step "waiting for pid $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
COMMON=(PANOPTIC_IMAGE=panoptic_mapping:synthetic-ack FLAT_PLAYER_REQUIRE_ACK=1 PANOPTIC_SNAPSHOT_PERIOD_S=2
        USE_MONOTONIC_PLAYER=true VISUALIZE=false PLAY_RATE=20 MAX_FRAMES=0 PLAYBACK_TIMEOUT=3600)
visit() {  # visit DIR RUN_NAME ENV...
  local dir=$1 name=$2; shift 2
  if [[ -s $dir/$name.panmap ]]; then step "$name: already saved"; return; fi
  step "$name: start"
  flock "$LOCK" env "${COMMON[@]}" "$@" bash "$RUNNER" "$name" "$dir" > "$dir/$name.log" 2>&1
  [[ -s $dir/$name.panmap ]] || { step "$name: no map saved, see $dir/$name.log"; exit 3; }
  step "$name: saved $(du -h "$dir/$name.panmap" | cut -f1), $(ls "$dir/${name}_snapshots" | wc -l) maps during the visit"
}
F=$FT/datasets/panoptic_mapping/flat_dataset
visit "$OUT/flat" run1 FLAT_DATASET_ROOT="$F"
visit "$OUT/flat" run2 FLAT_DATASET_ROOT="$F" LOAD_MAP=true LOAD_FILE="$OUT/flat/run1.panmap"
DS=$F/local_ab_official_contract_640x480_v1
REAL=(FLAT_DATASET_ROOT="$DS"
      ID_TRACKER_TYPE=projective MERGE_DEACTIVATED_SUBMAPS=true UPDATE_ONLY_TRACKED_SUBMAPS=true
      SMALL_INSTANCE_VOXEL_SIZE=0.05 MEDIUM_INSTANCE_VOXEL_SIZE=0.05 LARGE_INSTANCE_VOXEL_SIZE=0.05
      BACKGROUND_VOXEL_SIZE=0.05 UNKNOWN_VOXEL_SIZE=0.05
      REQUIRED_REOBSERVATIONS=60 DEACTIVATE_AFTER_MISSED_DETECTIONS=60 CHANGE_DETECTION_FREQUENCY=60
      CAMERA_FX=319.513224284 CAMERA_FY=425.777832031 CAMERA_CX=318.840154546 CAMERA_CY=243.558051891
      CAMERA_WIDTH=640 CAMERA_HEIGHT=480)
visit "$OUT/real" session_a "${REAL[@]}"
visit "$OUT/real" session_b "${REAL[@]}" LOAD_MAP=true LOAD_FILE="$OUT/real/session_a.panmap" \
  WORLD_TRANSFORM=$FT/datasets/local_ab/alignment/session_b_to_session_a.txt
visit "$OUT/real" session_c "${REAL[@]}" LOAD_MAP=true LOAD_FILE="$OUT/real/session_b.panmap" \
  WORLD_TRANSFORM=$FT/datasets/local_ab/alignment/session_c_to_session_a.txt
step "Flat and real done: $(du -sh "$OUT" | cut -f1)"
