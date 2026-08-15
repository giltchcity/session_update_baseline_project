#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${SESSION_UPDATE_INTERNAL_STRICT:-0}" != "1" ]]; then
  echo "INTERNAL_ENTRY_ONLY: use ${ROOT}/scripts/run_session.sh" >&2
  exit 64
fi
# shellcheck disable=SC1091
source "${ROOT}/scripts/khronos_env.sh"
"${ROOT}/scripts/check_canonical_runtime.sh" --require-built >/dev/null

RUN_DIR=""
LABEL_DIR=""
INSTANCE_DIR=""
WORLD_TRANSFORM=""
INPUT_STATE=""
OUTPUT_DIR=""
TF_SETTLE_S="0.02"
PLAY_RATE="1.0"
IMAGE_SCALE="1.0"
FLOW_CONTROL="ack"
ACK_TIMEOUT_S="120"
FINALIZATION_TIMEOUT_S="1800"
DISCOVERY_TIMEOUT_S="30.0"
FRAME_LIMIT="0"
SESSION_START_NS=""
CHANGE_DETECTION_EVERY_N_FRAMES="-1"
SAVE_EVERY_N_FRAMES="0"
STORE_VISUALIZATION_DETAILS="true"
SAVE_FULL_STATE="false"
SENSOR_MAX_RANGE="5.0"
VOXEL_SIZE=""
TRUNCATION_DISTANCE=""
MOTION_MIN_CLUSTER_SIZE=""
MIN_GLOG_LEVEL="1"
GLOG_VERBOSITY="0"
MAPPER_CONFIG="${ROOT}/configs/room18_instance_5cm.yaml"
INPUT_CONFIG="${ROOT}/configs/nss_flat_input.yaml"
LABELSPACE_CONFIG="${ROOT}/configs/nss_ade20k_room_label_space.yaml"
PHYSICAL_CATALOG="${ROOT}/configs/room18_physical_catalog.json"
FASTRTPS_ACK_PROFILE="${ROOT}/configs/fastdds_session_update_ack.xml"

usage() {
  cat <<EOF
Usage: $0 --run-dir DIR --output-dir DIR [options]

Required:
  --run-dir DIR
  --output-dir DIR

Session input:
  --label-dir DIR
  --instance-dir DIR
  --world-transform FILE
  --input-state FILE

Playback:
  --flow-control ack|realtime
  --play-rate RATE
  --image-scale SCALE
  --frame-limit N
  --session-start-ns NS
  --tf-settle-s SECONDS
  --ack-timeout-s SECONDS
  --finalization-timeout-s SECONDS
  --discovery-timeout-s SECONDS

Khronos:
  --mapper-config FILE
  --input-config FILE
  --labelspace-config FILE
  --physical-catalog FILE
  --change-detection-every-n-backend-updates N
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
    --instance-dir) INSTANCE_DIR="$2"; shift 2 ;;
    --world-transform) WORLD_TRANSFORM="$2"; shift 2 ;;
    --input-state) INPUT_STATE="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --tf-settle-s) TF_SETTLE_S="$2"; shift 2 ;;
    --play-rate) PLAY_RATE="$2"; shift 2 ;;
    --image-scale) IMAGE_SCALE="$2"; shift 2 ;;
    --flow-control) FLOW_CONTROL="$2"; shift 2 ;;
    --ack-timeout-s) ACK_TIMEOUT_S="$2"; shift 2 ;;
    --finalization-timeout-s) FINALIZATION_TIMEOUT_S="$2"; shift 2 ;;
    --discovery-timeout-s) DISCOVERY_TIMEOUT_S="$2"; shift 2 ;;
    --frame-limit) FRAME_LIMIT="$2"; shift 2 ;;
    --session-start-ns) SESSION_START_NS="$2"; shift 2 ;;
    --change-detection-every-n-backend-updates) CHANGE_DETECTION_EVERY_N_FRAMES="$2"; shift 2 ;;
    # Compatibility spelling. The value has always been counted after the
    # frontend output gate, so it never meant 30 FPS input camera frames.
    --change-detection-every-n-frames) CHANGE_DETECTION_EVERY_N_FRAMES="$2"; shift 2 ;;
    --save-every-n-frames) SAVE_EVERY_N_FRAMES="$2"; shift 2 ;;
    --store-visualization-details) STORE_VISUALIZATION_DETAILS="$2"; shift 2 ;;
    --save-full-state) SAVE_FULL_STATE="$2"; shift 2 ;;
    --sensor-max-range) SENSOR_MAX_RANGE="$2"; shift 2 ;;
    --voxel-size) VOXEL_SIZE="$2"; shift 2 ;;
    --truncation-distance) TRUNCATION_DISTANCE="$2"; shift 2 ;;
    --motion-min-cluster-size) MOTION_MIN_CLUSTER_SIZE="$2"; shift 2 ;;
    --min-glog-level) MIN_GLOG_LEVEL="$2"; shift 2 ;;
    --verbosity) GLOG_VERBOSITY="$2"; shift 2 ;;
    --mapper-config) MAPPER_CONFIG="$2"; shift 2 ;;
    --input-config) INPUT_CONFIG="$2"; shift 2 ;;
    --labelspace-config) LABELSPACE_CONFIG="$2"; shift 2 ;;
    --physical-catalog) PHYSICAL_CATALOG="$2"; shift 2 ;;
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
[[ "${FINALIZATION_TIMEOUT_S}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || \
  die "--finalization-timeout-s must be a finite positive number"
"${BASE1_PYTHON:-/usr/bin/python3}" - "${FINALIZATION_TIMEOUT_S}" <<'PY' || \
  die "--finalization-timeout-s must be a finite positive number"
import math
import sys
value = float(sys.argv[1])
raise SystemExit(0 if math.isfinite(value) and value > 0.0 else 1)
PY

[[ -n "${LABEL_DIR}" ]] || LABEL_DIR="${RUN_DIR}"
for path in "${RUN_DIR}" "${LABEL_DIR}" "${MAPPER_CONFIG}" "${INPUT_CONFIG}" \
            "${LABELSPACE_CONFIG}" "${PHYSICAL_CATALOG}" \
            "${FASTRTPS_ACK_PROFILE}"; do
  [[ -e "${path}" ]] || die "missing input: ${path}"
done
[[ -z "${INSTANCE_DIR}" || -d "${INSTANCE_DIR}" ]] || \
  die "missing instance dir: ${INSTANCE_DIR}"
[[ -z "${WORLD_TRANSFORM}" || -f "${WORLD_TRANSFORM}" ]] || \
  die "missing world transform: ${WORLD_TRANSFORM}"
[[ -z "${INPUT_STATE}" || -f "${INPUT_STATE}" ]] || \
  die "missing input state: ${INPUT_STATE}"

RUN_DIR="$(realpath "${RUN_DIR}")"
LABEL_DIR="$(realpath "${LABEL_DIR}")"
[[ -z "${INSTANCE_DIR}" ]] || INSTANCE_DIR="$(realpath "${INSTANCE_DIR}")"
MAPPER_CONFIG="$(realpath "${MAPPER_CONFIG}")"
INPUT_CONFIG="$(realpath "${INPUT_CONFIG}")"
LABELSPACE_CONFIG="$(realpath "${LABELSPACE_CONFIG}")"
PHYSICAL_CATALOG="$(realpath "${PHYSICAL_CATALOG}")"
FASTRTPS_ACK_PROFILE="$(realpath "${FASTRTPS_ACK_PROFILE}")"
[[ -z "${WORLD_TRANSFORM}" ]] || WORLD_TRANSFORM="$(realpath "${WORLD_TRANSFORM}")"
[[ -z "${INPUT_STATE}" ]] || INPUT_STATE="$(realpath "${INPUT_STATE}")"
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
export GLOG_log_dir="${CONTROL_DIR}/logs/glog"
export ROS2CLI_NO_DAEMON=1
# Fast DDS 2.14 on this runtime loads profile files from the legacy-named
# FASTRTPS variable. The newer FASTDDS spelling is accepted by newer releases
# but is ignored here, so erase it to keep the effective source unambiguous.
unset FASTDDS_DEFAULT_PROFILES_FILE
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_ACK_PROFILE}"
export RMW_FASTRTPS_USE_QOS_FROM_XML=1
mkdir -p "${ROS_HOME}" "${ROS_LOG_DIR}" "${GLOG_log_dir}"

"${BASE1_PYTHON}" "${ROOT}/scripts/verify_fastdds_ack_profile.py" >/dev/null
FAST_DDS_NATIVE_CONTRACT_LOG="${CONTROL_DIR}/logs/fastdds_writer_contract.log"
if ! "${BASE1_BUILD_DIR}/test_fastdds_ack_profile" \
    >"${FAST_DDS_NATIVE_CONTRACT_LOG}" 2>&1; then
  cat "${FAST_DDS_NATIVE_CONTRACT_LOG}" >&2
  die "native Fast DDS transaction-writer contract failed"
fi
for topic in /session_update/frame_processed /nss/rgb/image_raw \
             /nss/depth/image_raw /nss/semantic/image_raw; do
  grep -Fq "FAST_DDS_WRITER_CONTRACT_OK topic=${topic} " \
    "${FAST_DDS_NATIVE_CONTRACT_LOG}" || \
    die "native Fast DDS writer proof missing for ${topic}"
done
grep -Fq "FAST_DDS_TRANSACTION_PROFILE_OK writers=4" \
  "${FAST_DDS_NATIVE_CONTRACT_LOG}" || \
  die "native Fast DDS transaction proof is incomplete"
FASTRTPS_ACK_PROFILE_SHA256="$(sha256sum "${FASTRTPS_ACK_PROFILE}" | cut -d' ' -f1)"
"${BASE1_PYTHON}" - "${CONTROL_DIR}/transport_provenance.json" \
  "${FASTRTPS_ACK_PROFILE}" "${FASTRTPS_ACK_PROFILE_SHA256}" <<'PY'
import json
import sys

output, profile, digest = sys.argv[1:]
payload = {
    "schema": "session_update_transport/v2",
    "rmw_implementation": "rmw_fastrtps_cpp",
    "profile_env_variable": "FASTRTPS_DEFAULT_PROFILES_FILE",
    "profile_path": profile,
    "profile_sha256": digest,
    "rmw_fastrtps_use_qos_from_xml": "1",
    "transaction_writers": [
        {
            "role": role,
            "ros_topic": topic,
            "native_dds_topic": f"rt{topic}",
            "reliability": "RELIABLE",
            "history": "KEEP_LAST",
            "depth": 10,
            "initial_heartbeat_ns": 1_000_000,
            "heartbeat_period_ns": 10_000_000,
            "nack_response_delay_ns": 1_000_000,
        }
        for role, topic in (
            ("frame_processed_ack", "/session_update/frame_processed"),
            ("rgb_input", "/nss/rgb/image_raw"),
            ("depth_input", "/nss/depth/image_raw"),
            ("packed_semantic_instance_input", "/nss/semantic/image_raw"),
        )
    ],
    "native_writer_validation": {
        "pre_session_all_transaction_writers": "FAIL_CLOSED",
        "in_mapper_frame_processed_writer": "FAIL_CLOSED",
        "pre_session_proof_log": "logs/fastdds_writer_contract.log",
    },
}
with open(output, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2)
    stream.write("\n")
PY

cat >"${CONTROL_DIR}/command.txt" <<EOF
run_dir=${RUN_DIR}
label_dir=${LABEL_DIR}
instance_dir=${INSTANCE_DIR:-NONE}
world_transform=${WORLD_TRANSFORM:-IDENTITY}
input_state=${INPUT_STATE:-NONE}
output_dir=${OUTPUT_DIR}
tf_settle_s=${TF_SETTLE_S}
play_rate=${PLAY_RATE}
image_scale=${IMAGE_SCALE}
flow_control=${FLOW_CONTROL}
ack_timeout_s=${ACK_TIMEOUT_S}
finalization_timeout_s=${FINALIZATION_TIMEOUT_S}
discovery_timeout_s=${DISCOVERY_TIMEOUT_S}
frame_limit=${FRAME_LIMIT}
session_start_ns=${SESSION_START_NS:-AUTO_FROM_ACQUISITION_NAME}
change_detection_every_n_backend_updates=${CHANGE_DETECTION_EVERY_N_FRAMES}
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
physical_catalog=${PHYSICAL_CATALOG}
fastdds_ack_profile=${FASTRTPS_ACK_PROFILE}
fastdds_ack_profile_sha256=${FASTRTPS_ACK_PROFILE_SHA256}
fastdds_native_transaction_contract=FAIL_CLOSED_FOUR_WRITERS
fastdds_profile_env_variable=FASTRTPS_DEFAULT_PROFILES_FILE
rmw_fastrtps_use_qos_from_xml=${RMW_FASTRTPS_USE_QOS_FROM_XML}
rmw_implementation=${RMW_IMPLEMENTATION}
EOF

if [[ -z "${INPUT_STATE}" ]]; then
  BACKEND_YAML="{type: Backend, change_detection: {run_every_n_frames: ${CHANGE_DETECTION_EVERY_N_FRAMES}}, fix_input_poses: true}"
else
  BACKEND_YAML="{type: SessionBackend, input_state: '${INPUT_STATE}', change_detection: {run_every_n_frames: ${CHANGE_DETECTION_EVERY_N_FRAMES}}, fix_input_poses: true}"
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
mapper_reaped=false
cleanup_mapper() {
  local status=$?
  if [[ "${mapper_reaped}" != true ]] && kill -0 "${KHRONOS_PID}" 2>/dev/null; then
    kill -TERM "${KHRONOS_PID}" 2>/dev/null || true
    wait "${KHRONOS_PID}" 2>/dev/null || true
  fi
  return "${status}"
}
trap cleanup_mapper EXIT

PLAYER_ARGS=(
  --run-dir "${RUN_DIR}"
  --label-dir "${LABEL_DIR}"
  --tf-settle-s "${TF_SETTLE_S}"
  --play-rate "${PLAY_RATE}"
  --image-scale "${IMAGE_SCALE}"
  --flow-control "${FLOW_CONTROL}"
  --ack-timeout-s "${ACK_TIMEOUT_S}"
  --finish-timeout-s "${FINALIZATION_TIMEOUT_S}"
  --discovery-timeout-s "${DISCOVERY_TIMEOUT_S}"
  --post-wait-s 0
  --frame-limit "${FRAME_LIMIT}"
  --manifest "${CONTROL_DIR}/playback_manifest.json"
)
[[ -z "${SESSION_START_NS}" ]] || PLAYER_ARGS+=(--session-start-ns "${SESSION_START_NS}")
[[ -z "${WORLD_TRANSFORM}" ]] || PLAYER_ARGS+=(--world-transform "${WORLD_TRANSFORM}")
[[ -z "${INSTANCE_DIR}" ]] || PLAYER_ARGS+=(--instance-dir "${INSTANCE_DIR}")
[[ -z "${INSTANCE_DIR}" ]] || PLAYER_ARGS+=(--physical-catalog "${PHYSICAL_CATALOG}")

set +e
"${BASE1_PYTHON}" "${ROOT}/scripts/nss_flat_ros2_player.py" \
  "${PLAYER_ARGS[@]}" >"${CONTROL_DIR}/logs/player.log" 2>&1
PLAYER_RC=$?
set -e
printf '%s\n' "${PLAYER_RC}" >"${CONTROL_DIR}/player.exit_code"
if [[ ${PLAYER_RC} -ne 0 ]]; then
  echo "PLAYBACK_FAILED exit_code=${PLAYER_RC} khronos_pid=${KHRONOS_PID}" >&2
  exit 4
fi

# The player only returns after receiving the mapper's terminal-save ACK. Give
# the in-process shutdown service a bounded grace period; a DDS/runtime teardown
# bug must fail transactionally instead of hanging the production runner. The
# expensive terminal change detection and serialization already completed
# before that ACK and are bounded separately by FINALIZATION_TIMEOUT_S.
FINAL_TIMEOUT_MARKER="${CONTROL_DIR}/finalization.timeout"
(
  sleep 30
  if kill -0 "${KHRONOS_PID}" 2>/dev/null; then
    printf 'mapper_pid=%s\n' "${KHRONOS_PID}" >"${FINAL_TIMEOUT_MARKER}"
    kill -TERM "${KHRONOS_PID}" 2>/dev/null || true
    sleep 5
    kill -KILL "${KHRONOS_PID}" 2>/dev/null || true
  fi
) &
WATCHDOG_PID=$!
set +e
wait "${KHRONOS_PID}"
KHRONOS_RC=$?
set -e
mapper_reaped=true
kill "${WATCHDOG_PID}" 2>/dev/null || true
wait "${WATCHDOG_PID}" 2>/dev/null || true
if [[ -e "${FINAL_TIMEOUT_MARKER}" ]]; then
  echo "KHRONOS_EXIT_TIMEOUT pid=${KHRONOS_PID} after terminal save ACK" >&2
  exit 5
fi
printf '%s\n' "${KHRONOS_RC}" >"${CONTROL_DIR}/khronos.exit_code"
if [[ ${KHRONOS_RC} -ne 0 ]]; then
  echo "KHRONOS_EXIT_NONZERO exit_code=${KHRONOS_RC}" >&2
  exit 5
fi

[[ -s "${OUTPUT_DIR}/final.4dmap" ]] || \
  die "khronos exited cleanly but final.4dmap is missing: ${OUTPUT_DIR}/final.4dmap"
grep -q "Experiment Finished Cleanly" "${OUTPUT_DIR}/experiment_log.txt" || \
  die "save service returned but the experiment lacks the clean-finish flag"
grep -q "FAST_DDS_ACK_CONTRACT_OK topic=/session_update/frame_processed" \
  "${CONTROL_DIR}/logs/khronos.log" || \
  die "mapper exited without native Fast DDS ACK timing proof"

"${BASE1_PYTHON}" - "${CONTROL_DIR}/playback_manifest.json" \
  "${CONTROL_DIR}/processing_manifest.json" "${KHRONOS_RC}" \
  "${CONTROL_DIR}/transport_provenance.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    playback = json.load(stream)
with open(sys.argv[4], encoding="utf-8") as stream:
    transport = json.load(stream)

published = int(playback["frames_published"])
flow_control = playback["flow_control"]
result = {
    "frames_published": published,
    "frames_processed_by_active_window": published if flow_control == "ack" else None,
    "processing_certainty": "DIRECT_PER_FRAME_ACK" if flow_control == "ack" else "REALTIME_UNVERIFIED",
    "khronos_exit_code_after_clean_save": int(sys.argv[3]),
    "transport_provenance": transport,
}
with open(sys.argv[2], "w", encoding="utf-8") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
PY

echo "SESSION_COMPLETE map=${OUTPUT_DIR}/final.4dmap frames=$("${BASE1_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["frames_published"])' "${CONTROL_DIR}/playback_manifest.json")"
ls -lh "${OUTPUT_DIR}/final.4dmap" "${OUTPUT_DIR}/experiment_log.txt"
