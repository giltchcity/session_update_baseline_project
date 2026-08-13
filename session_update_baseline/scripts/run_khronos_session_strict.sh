#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jixian/Desktop/FT/session_update_baseline"
# shellcheck disable=SC1091
source "${ROOT}/scripts/khronos_env.sh"

RUN_DIR=""
LABEL_DIR=""
WORLD_TRANSFORM=""
PRIOR_MAP=""
PRIOR_SEED_MAP=""
OUTPUT_DIR=""
TF_SETTLE_S="0.02"
PLAY_RATE="1.0"
IMAGE_SCALE="1.0"
FLOW_CONTROL="ack"
ACK_TIMEOUT_S="120"
DISCOVERY_TIMEOUT_S="30.0"
FRAME_LIMIT="0"
CHANGE_DETECTION_EVERY_N_FRAMES="-1"
SAVE_EVERY_N_FRAMES="0"
STORE_VISUALIZATION_DETAILS="true"
SAVE_FULL_STATE="false"
SENSOR_MAX_RANGE="5.0"
VOXEL_SIZE=""
TRUNCATION_DISTANCE=""
MOTION_MIN_CLUSTER_SIZE=""
MASK_OBJECTS_FROM_BACKGROUND=""
KHRONOS_EXTRA_YAML=""
MIN_GLOG_LEVEL="1"
GLOG_VERBOSITY="0"
MAPPER_CONFIG="${ROOT}/../configs/khronos/uHumans2_map_update_5_change_no_obj_bg_remove.yaml"
INPUT_CONFIG="${ROOT}/configs/nss_flat_input.yaml"
LABELSPACE_CONFIG="${ROOT}/configs/nss_ade20k_room_label_space.yaml"

usage() {
  cat <<EOF
Usage: $0 --run-dir DIR --output-dir DIR [options]

Required:
  --run-dir DIR
  --output-dir DIR

Session input:
  --label-dir DIR
  --world-transform FILE
  --prior-map FILE
  --prior-seed-map FILE

Playback:
  --flow-control ack|realtime
  --play-rate RATE
  --image-scale SCALE
  --frame-limit N
  --tf-settle-s SECONDS
  --ack-timeout-s SECONDS
  --discovery-timeout-s SECONDS

Khronos:
  --mapper-config FILE
  --input-config FILE
  --labelspace-config FILE
  --change-detection-every-n-frames N
  --save-every-n-frames N
  --save-full-state true|false
  --store-visualization-details true|false
  --sensor-max-range METERS
  --voxel-size METERS
  --truncation-distance METERS
  --motion-min-cluster-size PIXELS
  --min-glog-level N
  --verbosity N
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

require_bool() {
  local name=$1
  local value=$2
  [[ "${value}" == "true" || "${value}" == "false" ]] || \
    die "${name} must be true or false"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --label-dir) LABEL_DIR="$2"; shift 2 ;;
    --world-transform) WORLD_TRANSFORM="$2"; shift 2 ;;
    --prior-map) PRIOR_MAP="$2"; shift 2 ;;
    --prior-seed-map) PRIOR_SEED_MAP="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --tf-settle-s) TF_SETTLE_S="$2"; shift 2 ;;
    --play-rate) PLAY_RATE="$2"; shift 2 ;;
    --image-scale) IMAGE_SCALE="$2"; shift 2 ;;
    --flow-control) FLOW_CONTROL="$2"; shift 2 ;;
    --ack-timeout-s) ACK_TIMEOUT_S="$2"; shift 2 ;;
    --discovery-timeout-s) DISCOVERY_TIMEOUT_S="$2"; shift 2 ;;
    --frame-limit) FRAME_LIMIT="$2"; shift 2 ;;
    --change-detection-every-n-frames) CHANGE_DETECTION_EVERY_N_FRAMES="$2"; shift 2 ;;
    --save-every-n-frames) SAVE_EVERY_N_FRAMES="$2"; shift 2 ;;
    --store-visualization-details) STORE_VISUALIZATION_DETAILS="$2"; shift 2 ;;
    --save-full-state) SAVE_FULL_STATE="$2"; shift 2 ;;
    --sensor-max-range) SENSOR_MAX_RANGE="$2"; shift 2 ;;
    --voxel-size) VOXEL_SIZE="$2"; shift 2 ;;
    --truncation-distance) TRUNCATION_DISTANCE="$2"; shift 2 ;;
    --motion-min-cluster-size) MOTION_MIN_CLUSTER_SIZE="$2"; shift 2 ;;
    --mask-objects-from-background) MASK_OBJECTS_FROM_BACKGROUND="$2"; shift 2 ;;
    --min-glog-level) MIN_GLOG_LEVEL="$2"; shift 2 ;;
    --verbosity) GLOG_VERBOSITY="$2"; shift 2 ;;
    --mapper-config) MAPPER_CONFIG="$2"; shift 2 ;;
    --input-config) INPUT_CONFIG="$2"; shift 2 ;;
    --labelspace-config) LABELSPACE_CONFIG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "${RUN_DIR}" ]] || die "--run-dir is required"
[[ -n "${OUTPUT_DIR}" ]] || die "--output-dir is required"
[[ "${FLOW_CONTROL}" == "ack" || "${FLOW_CONTROL}" == "realtime" ]] || \
  die "--flow-control must be ack or realtime"
[[ "${FRAME_LIMIT}" =~ ^[0-9]+$ ]] || die "--frame-limit must be a non-negative integer"
[[ "${SAVE_EVERY_N_FRAMES}" =~ ^[0-9]+$ ]] || \
  die "--save-every-n-frames must be a non-negative integer"
require_bool "--store-visualization-details" "${STORE_VISUALIZATION_DETAILS}"
require_bool "--save-full-state" "${SAVE_FULL_STATE}"

[[ -n "${LABEL_DIR}" ]] || LABEL_DIR="${RUN_DIR}"
for path in "${RUN_DIR}" "${LABEL_DIR}" "${MAPPER_CONFIG}" "${INPUT_CONFIG}" \
            "${LABELSPACE_CONFIG}"; do
  [[ -e "${path}" ]] || die "missing input: ${path}"
done
[[ -z "${WORLD_TRANSFORM}" || -f "${WORLD_TRANSFORM}" ]] || \
  die "missing world transform: ${WORLD_TRANSFORM}"
[[ -z "${PRIOR_MAP}" || -f "${PRIOR_MAP}" ]] || die "missing prior map: ${PRIOR_MAP}"
[[ -z "${PRIOR_SEED_MAP}" || -f "${PRIOR_SEED_MAP}" ]] || \
  die "missing prior seed map: ${PRIOR_SEED_MAP}"
[[ -n "${PRIOR_MAP}" || -z "${PRIOR_SEED_MAP}" ]] || \
  die "--prior-seed-map requires --prior-map"

RUN_DIR="$(realpath "${RUN_DIR}")"
LABEL_DIR="$(realpath "${LABEL_DIR}")"
MAPPER_CONFIG="$(realpath "${MAPPER_CONFIG}")"
INPUT_CONFIG="$(realpath "${INPUT_CONFIG}")"
LABELSPACE_CONFIG="$(realpath "${LABELSPACE_CONFIG}")"
[[ -z "${WORLD_TRANSFORM}" ]] || WORLD_TRANSFORM="$(realpath "${WORLD_TRANSFORM}")"
[[ -z "${PRIOR_MAP}" ]] || PRIOR_MAP="$(realpath "${PRIOR_MAP}")"
[[ -z "${PRIOR_SEED_MAP}" ]] || PRIOR_SEED_MAP="$(realpath "${PRIOR_SEED_MAP}")"
OUTPUT_DIR="$(realpath -m "${OUTPUT_DIR}")"
CONTROL_DIR="${OUTPUT_DIR}_control"

# Never merge a retry into an old or partially written experiment.
[[ ! -e "${OUTPUT_DIR}" ]] || die "output already exists: ${OUTPUT_DIR}"
[[ ! -e "${CONTROL_DIR}" ]] || die "control output already exists: ${CONTROL_DIR}"

# Only the parent is created here. Khronos allocates OUTPUT_DIR itself, and it
# unconditionally remove_all()s that path first, so pre-creating it is pointless.
mkdir -p "$(dirname "${OUTPUT_DIR}")" "${CONTROL_DIR}/logs"
export ROS_HOME="${CONTROL_DIR}/ros_home"
export ROS_LOG_DIR="${CONTROL_DIR}/logs/ros"
export ROS2CLI_NO_DAEMON=1
mkdir -p "${ROS_HOME}" "${ROS_LOG_DIR}"

cat >"${CONTROL_DIR}/command.txt" <<EOF
run_dir=${RUN_DIR}
label_dir=${LABEL_DIR}
world_transform=${WORLD_TRANSFORM:-IDENTITY}
prior_map=${PRIOR_MAP:-NONE}
prior_seed_map=${PRIOR_SEED_MAP:-NONE}
output_dir=${OUTPUT_DIR}
tf_settle_s=${TF_SETTLE_S}
play_rate=${PLAY_RATE}
image_scale=${IMAGE_SCALE}
flow_control=${FLOW_CONTROL}
ack_timeout_s=${ACK_TIMEOUT_S}
discovery_timeout_s=${DISCOVERY_TIMEOUT_S}
frame_limit=${FRAME_LIMIT}
change_detection_every_n_frames=${CHANGE_DETECTION_EVERY_N_FRAMES}
save_every_n_frames=${SAVE_EVERY_N_FRAMES}
store_visualization_details=${STORE_VISUALIZATION_DETAILS}
save_full_state=${SAVE_FULL_STATE}
sensor_max_range=${SENSOR_MAX_RANGE}
voxel_size=${VOXEL_SIZE:-CONFIG}
truncation_distance=${TRUNCATION_DISTANCE:-CONFIG}
motion_min_cluster_size=${MOTION_MIN_CLUSTER_SIZE:-CONFIG}
min_glog_level=${MIN_GLOG_LEVEL}
verbosity=${GLOG_VERBOSITY}
mapper_config=${MAPPER_CONFIG}
input_config=${INPUT_CONFIG}
labelspace_config=${LABELSPACE_CONFIG}
EOF

if [[ -z "${PRIOR_MAP}" ]]; then
  BACKEND_YAML="{type: Backend, change_detection: {run_every_n_frames: ${CHANGE_DETECTION_EVERY_N_FRAMES}}, fix_input_poses: true}"
else
  BACKEND_YAML="{type: SessionBackend, prior_map: '${PRIOR_MAP}', prior_seed_map: '${PRIOR_SEED_MAP}', change_detection: {run_every_n_frames: ${CHANGE_DETECTION_EVERY_N_FRAMES}}, fix_input_poses: true}"
fi

# experiment.overwrite must be true. Upstream builds the DataDirectory config
# with swapped positional args -- ExperimentManager passes
# `DataDirectory::Config{true, config.overwrite}` while the struct declares
# `{overwrite, allocate, ...}`. So `overwrite` is pinned true upstream and this
# flag actually lands on `allocate`; with false the output directory is never
# created and Khronos aborts opening experiment_log.txt / mesh.ply. The
# pre-flight check above already guarantees the directory does not exist, so
# the upstream-forced remove_all has nothing to delete.
SESSION_YAML="{semantic_colormap_file: '', store_visualization_details: ${STORE_VISUALIZATION_DETAILS}, backend: ${BACKEND_YAML}, sensor_frame: left_cam, experiment: {output_dir: '${OUTPUT_DIR}', overwrite: true, save_every_n_frames: ${SAVE_EVERY_N_FRAMES}, save_full_state: ${SAVE_FULL_STATE}, log_timing: false, log_timing_details: false, exit_after_clock: false}}"

# Optional ablation: keep object-labelled surfaces out of the global/background
# TSDF so every movable thing lives only in the object layer.
[[ -z "${MASK_OBJECTS_FROM_BACKGROUND}" ]] || KHRONOS_EXTRA_YAML="{active_window: {mask_object_labels_from_background: ${MASK_OBJECTS_FROM_BACKGROUND}}}"

KHRONOS_ARGS=(
  --config-utilities-file "${INPUT_CONFIG}@input"
  --config-utilities-file "${MAPPER_CONFIG}"
  --config-utilities-file "${LABELSPACE_CONFIG}"
  --config-utilities-yaml "{robot_id: 0, odom_frame: odom, robot_frame: robot_0, map_frame: map}"
  --config-utilities-yaml "{glog_level: ${MIN_GLOG_LEVEL}, glog_verbosity: ${GLOG_VERBOSITY}}"
  --config-utilities-yaml "${SESSION_YAML}"
  --config-utilities-yaml "{input: {inputs: {left_cam: {sensor: {min_range: 0.1, max_range: ${SENSOR_MAX_RANGE}, extrinsics: {sensor_frame: left_cam}}}}}}"
)

# Override only values explicitly supplied by the caller. Never invent companion defaults.
[[ -z "${VOXEL_SIZE}" ]] || KHRONOS_ARGS+=(
  --config-utilities-yaml "{active_window: {volumetric_map: {voxel_size: ${VOXEL_SIZE}}}}"
)
[[ -z "${TRUNCATION_DISTANCE}" ]] || KHRONOS_ARGS+=(
  --config-utilities-yaml "{active_window: {volumetric_map: {truncation_distance: ${TRUNCATION_DISTANCE}}}}"
)
[[ -z "${MOTION_MIN_CLUSTER_SIZE}" ]] || KHRONOS_ARGS+=(
  --config-utilities-yaml "{active_window: {motion_detector: {min_cluster_size: ${MOTION_MIN_CLUSTER_SIZE}}}}"
)
[[ -z "${KHRONOS_EXTRA_YAML}" ]] || KHRONOS_ARGS+=(
  --config-utilities-yaml "${KHRONOS_EXTRA_YAML}"
)

"${BASE1_BUILD_DIR}/session_khronos_node" \
  "${KHRONOS_ARGS[@]}" \
  --ros-args \
  -r "~/input/left_cam/depth_registered/image_rect:=/nss/depth/image_raw" \
  -r "~/input/left_cam/rgb/image_raw:=/nss/rgb/image_raw" \
  -r "~/input/left_cam/rgb/camera_info:=/nss/rgb/camera_info" \
  -r "~/input/left_cam/semantic/image_raw:=/nss/semantic/image_raw" \
  >"${CONTROL_DIR}/logs/khronos.log" 2>&1 &
KHRONOS_PID=$!
printf '%s\n' "${KHRONOS_PID}" >"${CONTROL_DIR}/khronos.pid"

PLAYER_ARGS=(
  --run-dir "${RUN_DIR}"
  --label-dir "${LABEL_DIR}"
  --tf-settle-s "${TF_SETTLE_S}"
  --play-rate "${PLAY_RATE}"
  --image-scale "${IMAGE_SCALE}"
  --flow-control "${FLOW_CONTROL}"
  --ack-timeout-s "${ACK_TIMEOUT_S}"
  --discovery-timeout-s "${DISCOVERY_TIMEOUT_S}"
  --post-wait-s 0
  --frame-limit "${FRAME_LIMIT}"
  --manifest "${CONTROL_DIR}/playback_manifest.json"
)
[[ -z "${WORLD_TRANSFORM}" ]] || PLAYER_ARGS+=(--world-transform "${WORLD_TRANSFORM}")

set +e
"${BASE1_PYTHON}" "${ROOT}/scripts/nss_flat_ros2_player.py" \
  "${PLAYER_ARGS[@]}" >"${CONTROL_DIR}/logs/player.log" 2>&1
PLAYER_RC=$?
set -e
printf '%s\n' "${PLAYER_RC}" >"${CONTROL_DIR}/player.exit_code"
if [[ ${PLAYER_RC} -ne 0 ]]; then
  echo "PLAYBACK_FAILED exit_code=${PLAYER_RC} khronos_pid=${KHRONOS_PID}" >&2
  echo "Khronos was not signalled or killed by this runner." >&2
  exit 4
fi

if [[ ! -d "/proc/${KHRONOS_PID}" ]]; then
  set +e
  wait "${KHRONOS_PID}"
  KHRONOS_RC=$?
  set -e
  echo "KHRONOS_EXITED_BEFORE_SAVE exit_code=${KHRONOS_RC}" >&2
  exit 5
fi

# The player already published the finish signal on its last step. Khronos saves
# synchronously and exits on its own; this runner never signals or kills it.
set +e
wait "${KHRONOS_PID}"
KHRONOS_RC=$?
set -e
printf '%s\n' "${KHRONOS_RC}" >"${CONTROL_DIR}/khronos.exit_code"

[[ -s "${OUTPUT_DIR}/final.4dmap" ]] || \
  die "save service returned but final.4dmap is missing: ${OUTPUT_DIR}/final.4dmap"
grep -q "Experiment Finished Cleanly" "${OUTPUT_DIR}/experiment_log.txt" || \
  die "save service returned but the experiment lacks the clean-finish flag"

"${BASE1_PYTHON}" - "${CONTROL_DIR}/playback_manifest.json" \
  "${CONTROL_DIR}/processing_manifest.json" "${KHRONOS_RC}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    playback = json.load(stream)

published = int(playback["frames_published"])
flow_control = playback["flow_control"]
result = {
    "frames_published": published,
    "frames_processed_by_active_window": published if flow_control == "ack" else None,
    "processing_certainty": "DIRECT_PER_FRAME_ACK" if flow_control == "ack" else "REALTIME_UNVERIFIED",
    "khronos_exit_code_after_clean_save": int(sys.argv[3]),
}
with open(sys.argv[2], "w", encoding="utf-8") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
PY

echo "SESSION_COMPLETE map=${OUTPUT_DIR}/final.4dmap frames=$("${BASE1_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["frames_published"])' "${CONTROL_DIR}/playback_manifest.json")"
ls -lh "${OUTPUT_DIR}/final.4dmap" "${OUTPUT_DIR}/experiment_log.txt"
