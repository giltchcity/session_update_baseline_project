#!/usr/bin/env bash
# Panoptic Mapping on the 3RScan P3 rooms, for the results table: each room's visits in order
# (datasets/3rscan/mapping_configs/p3_ROOM/chain.json), the first from an empty map, each later one loading the one
# before (official loadMap) with its rescan-to-reference transform; the input of analysis/panoptic_3rscan_input.py
# (480x270, the visit's own intrinsics). Panoptic's own 3RScan (RIO) configuration with ground-truth segmentation
# (panoptic_mapping_ros/config/mapper/rio_groundtruth.yaml), at 5 cm as every method on 3RScan; every frame processed
# before the next is played, and a map saved every 2 s of dataset time, as panoptic_rerun.sh.
# Each visit under runs/HEAVY.lock.
# usage: panoptic_rerun_3rscan.sh RERUN_DIR [ROOM ...]      (RERUN_DIR/3rscan_input as made; maps into RERUN_DIR/3rscan)
set -Eeuo pipefail
RUN=$(realpath "$1"); shift
FT=/home/jixian/Desktop/FT; D3=$FT/datasets/3rscan
RUNNER=$FT/scripts/panoptic_mapping/run_flat_save_monotonic.sh
step() { echo "[$(date +%F' '%T)] $*"; }
ROOMS=("$@"); [[ ${#ROOMS[@]} -gt 0 ]] || mapfile -t ROOMS < <(grep -v '^#' $D3/CHAINS_P3.txt | cut -c1-8)
COMMON=(PANOPTIC_IMAGE=panoptic_mapping:synthetic-ack CONFIG=rio_groundtruth FLAT_PLAYER_REQUIRE_ACK=1 PANOPTIC_SNAPSHOT_PERIOD_S=2
        USE_MONOTONIC_PLAYER=true VISUALIZE=false PLAY_RATE=20 MAX_FRAMES=0 PLAYBACK_TIMEOUT=3600
        SMALL_INSTANCE_VOXEL_SIZE=0.05 MEDIUM_INSTANCE_VOXEL_SIZE=0.05 LARGE_INSTANCE_VOXEL_SIZE=0.05
        BACKGROUND_VOXEL_SIZE=0.05 UNKNOWN_VOXEL_SIZE=0.05)
for room in "${ROOMS[@]}"; do
  cfg=$D3/mapping_configs/p3_$room; out=$RUN/3rscan/p3_$room; mkdir -p "$out"
  mapfile -t visits < <(python3 -c "import json,sys; print('\n'.join(json.load(open(sys.argv[1]))['visits']))" $cfg/chain.json)
  prev=""
  for k in "${!visits[@]}"; do
    scan=${visits[$k]}
    if [[ -s $out/$scan.panmap ]]; then step "$room visit $k: already saved"; prev=$out/$scan.panmap; continue; fi
    read -r fx fy cx cy w h < "$RUN/3rscan_input/p3_$room/$scan/camera.txt"
    more=(CAMERA_FX=$fx CAMERA_FY=$fy CAMERA_CX=$cx CAMERA_CY=$cy CAMERA_WIDTH=$w CAMERA_HEIGHT=$h)
    [[ -n $prev ]] && more+=(LOAD_MAP=true LOAD_FILE="$prev" WORLD_TRANSFORM="$cfg/world_transform_${scan}_to_${visits[0]}.txt")
    rm -rf "$out/${scan}_snapshots"
    step "$room visit $k ($scan): start"
    flock "$FT/runs/HEAVY.lock" env "${COMMON[@]}" FLAT_DATASET_ROOT="$RUN/3rscan_input/p3_$room" "${more[@]}" \
      bash "$RUNNER" "$scan" "$out" > "$out/$scan.log" 2>&1 || true
    [[ -s $out/$scan.panmap ]] || { step "$room visit $k: no map saved, see $out/$scan.log"; exit 3; }
    step "$room visit $k: saved $(du -h "$out/$scan.panmap" | cut -f1), $(ls "$out/${scan}_snapshots" | wc -l) maps during the visit"
    prev=$out/$scan.panmap
  done
done
step "3RScan done: $(du -sh "$RUN/3rscan" | cut -f1)"
