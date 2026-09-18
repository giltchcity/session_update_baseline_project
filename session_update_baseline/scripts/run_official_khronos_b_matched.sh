#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OFFICIAL_BUILD="${OFFICIAL_KHRONOS_BUILD:-/tmp/session_update_official_matched_build_20260820}"
OUTPUT="${1:-}"
FRAME_LIMIT="${2:-0}"

if [[ -z "${OUTPUT}" ]]; then
  echo "Usage: $0 OUTPUT_DIR [FRAME_LIMIT]" >&2
  exit 2
fi
[[ "${FRAME_LIMIT}" =~ ^[0-9]+$ ]] || { echo "invalid frame limit: ${FRAME_LIMIT}" >&2; exit 2; }
[[ ! -e "${OUTPUT}" ]] || { echo "output already exists: ${OUTPUT}" >&2; exit 3; }

ADAPTER="${OFFICIAL_BUILD}/adapter/official_khronos_session_node"
OFFICIAL_PREFIX="${OFFICIAL_BUILD}/install"
OFFICIAL_SOURCE="${ROOT}/vendor/khronos_official_63faadde"
RUN_DIR="/home/jixian/Desktop/FT/datasets/local_ab/rgbd/session_b_20260810_030502_620_flat_rgbd_30hz_1080p"
SEMANTIC_DIR="/home/jixian/Desktop/FT/datasets/local_ab/semantics/session_b"
WORLD_TRANSFORM="/home/jixian/Desktop/FT/datasets/local_ab/alignment/session_b_to_session_a.txt"
MAPPER_CONFIG="${OFFICIAL_SOURCE}/khronos_ros/config/mapper/uHumans2.yaml"
INPUT_CONFIG="${ROOT}/configs/nss_flat_input.yaml"
LABELSPACE_CONFIG="${ROOT}/configs/nss_ade20k_room_label_space.yaml"
FASTRTPS_PROFILE="${ROOT}/configs/fastdds_session_update_ack.xml"

for path in "${ADAPTER}" "${OFFICIAL_PREFIX}/local_setup.bash" "${MAPPER_CONFIG}" \
            "${INPUT_CONFIG}" "${LABELSPACE_CONFIG}" "${WORLD_TRANSFORM}" \
            "${FASTRTPS_PROFILE}"; do
  [[ -e "${path}" ]] || { echo "missing required input: ${path}" >&2; exit 3; }
done

STAGING="${OUTPUT}.incomplete.$$"
CONTROL="${STAGING}/control"
STATE="${STAGING}/state"
mkdir -p "${CONTROL}/logs"

set +u
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
source /home/jixian/ros2_ws/install/setup.bash
# shellcheck disable=SC1091
source "${OFFICIAL_PREFIX}/local_setup.bash"
set -u

export ROS_DOMAIN_ID="${OFFICIAL_ROS_DOMAIN_ID:-48}"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export RMW_FASTRTPS_USE_QOS_FROM_XML=1
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_PROFILE}"
unset FASTDDS_DEFAULT_PROFILES_FILE || true

COMMON_THRESHOLDS="{active_window: {min_output_separation: 0.4, volumetric_map: {voxel_size: 0.05, truncation_distance: 0.15}, projective_integrator: {interpolation_method: {max_depth_difference_m: 0.05}}, tracking_integrator: {temporal_buffer: 2.0}, motion_detector: {min_cluster_size: 3000}, object_detector: {grid_size: 0.05}, object_extractor: {object_reconstruction_resolution: 0.02, projective_integrator: {num_threads: 16}}, frame_data_buffer: {max_buffer_size: 100, store_every_n_frames: 3}}, frontend: {pgmo: {mesh_resolution: 0.005, time_horizon: 15.0, d_graph_resolution: 2.5}}, backend: {type: Backend, fix_input_poses: true, change_detection: {run_every_n_frames: 5, ray_verificator: {radial_tolerance: 0.08, depth_tolerance: 0.3, active_window_duration: 3.0}, ray_change_detector: {temporal_resolution: 5.0, window_size: 5, use_relative_confidence: true, absence_confidence: 0.6, presence_confidence: 0.5}, objects: {time_filtering_threshold: 3.0}}}}"

"${ADAPTER}" \
  --config-utilities-file "${INPUT_CONFIG}@input" \
  --config-utilities-file "${MAPPER_CONFIG}" \
  --config-utilities-file "${LABELSPACE_CONFIG}" \
  --config-utilities-yaml "{robot_id: 0, odom_frame: odom, robot_frame: robot_0, map_frame: map}" \
  --config-utilities-yaml "{sensor_frame: left_cam, store_visualization_details: true, experiment: {output_dir: '${STATE}', overwrite: true, save_every_n_frames: 0, save_full_state: false, log_timing: false, log_timing_details: false, exit_after_clock: false}}" \
  --config-utilities-yaml "{input: {inputs: {left_cam: {sensor: {min_range: 0.1, max_range: 5.0, extrinsics: {sensor_frame: left_cam}}}}}}" \
  --config-utilities-yaml "${COMMON_THRESHOLDS}" \
  --ros-args \
  -r "~/input/left_cam/depth_registered/image_rect:=/nss/depth/image_raw" \
  -r "~/input/left_cam/rgb/image_raw:=/nss/rgb/image_raw" \
  -r "~/input/left_cam/rgb/camera_info:=/nss/rgb/camera_info" \
  -r "~/input/left_cam/semantic/image_raw:=/nss/semantic/image_raw" \
  >"${CONTROL}/logs/khronos.log" 2>&1 &
MAPPER_PID=$!
printf '%s\n' "${MAPPER_PID}" >"${CONTROL}/khronos.pid"

MAPPER_REAPED=false
cleanup() {
  local status=$?
  if [[ "${MAPPER_REAPED}" != true ]] && kill -0 "${MAPPER_PID}" 2>/dev/null; then
    kill -TERM "${MAPPER_PID}" 2>/dev/null || true
    wait "${MAPPER_PID}" 2>/dev/null || true
  fi
  if [[ ${status} -ne 0 ]]; then
    echo "OFFICIAL_MATCHED_FAILED staging=${STAGING}" >&2
  fi
  return "${status}"
}
trap cleanup EXIT INT TERM

/usr/bin/python3 -u "${ROOT}/scripts/nss_flat_ros2_player.py" \
  --run-dir "${RUN_DIR}" \
  --label-dir "${SEMANTIC_DIR}" \
  --world-transform "${WORLD_TRANSFORM}" \
  --session-start-ns 1786302302620000000 \
  --tf-settle-s 0.02 \
  --play-rate 100 \
  --image-scale 0.5 \
  --flow-control ack \
  --ack-timeout-s 180 \
  --finish-timeout-s 1800 \
  --discovery-timeout-s 180 \
  --post-wait-s 0 \
  --frame-limit "${FRAME_LIMIT}" \
  --manifest "${CONTROL}/playback_manifest.json" \
  >"${CONTROL}/logs/player.log" 2>&1

wait "${MAPPER_PID}"
MAPPER_REAPED=true
printf '0\n' >"${CONTROL}/khronos.exit_code"
[[ -s "${STATE}/final.4dmap" ]] || { echo "missing final.4dmap" >&2; exit 5; }

/usr/bin/python3 - "${CONTROL}/provenance.json" "${OFFICIAL_SOURCE}" \
  "${MAPPER_CONFIG}" "${COMMON_THRESHOLDS}" "${FRAME_LIMIT}" <<'PY'
import hashlib
import json
import pathlib
import sys

output, source, config, thresholds, frame_limit = sys.argv[1:]
source = pathlib.Path(source)
backend = source / "khronos/src/backend/backend.cpp"
payload = {
    "schema": "official_khronos_matched/v1",
    "upstream_snapshot": "MIT-SPARK/Khronos@63faadde",
    "upstream_source": str(source.resolve()),
    "upstream_backend_sha256": hashlib.sha256(backend.read_bytes()).hexdigest(),
    "mapper_config": str(pathlib.Path(config).resolve()),
    "common_threshold_overrides": thresholds,
    "frame_limit": int(frame_limit),
    "input_protocol": "semantic_only",
    "prior_state": None,
    "controlled_difference": "upstream ConnectedSemantics+MaxIouTracker+Backend; no A memory",
}
pathlib.Path(output).write_text(json.dumps(payload, indent=2) + "\n")
PY

mv "${STAGING}" "${OUTPUT}"
trap - EXIT INT TERM
echo "OFFICIAL_MATCHED_COMPLETE output=${OUTPUT} map=${OUTPUT}/state/final.4dmap"
