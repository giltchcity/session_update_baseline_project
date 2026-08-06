#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jixian/Desktop/FT"
ROUND="${ROOT}/session_update_baseline/rounds/khronos_office_reversed_ab_30s_20260730"

case "${1:-}" in
  A|a)
    MAP="${ROUND}/A_khronos_v3/final.4dmap"
    ;;
  B|b)
    MAP="${ROUND}/B_khronos_from_scratch/final.4dmap"
    ;;
  A-improved|a-improved)
    MAP="${ROUND}/A_ours/improved_final.4dmap"
    ;;
  B-memory|b-memory)
    MAP="${ROUND}/B_ours_with_A_memory/improved_final.4dmap"
    ;;
  *)
    echo "Usage: $0 A|B|A-improved|B-memory" >&2
    exit 2
    ;;
esac

# shellcheck disable=SC1091
source /home/jixian/ros2_ws/install/setup.bash
exec ros2 launch khronos_ros object_mesh_with_global.launch.yaml map_file:="${MAP}"
