#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/khronos_env.sh"

RUN_DIR=""
OUTPUT_DIR=""
PLAY_RATE="5"
TF_SETTLE_S="0.02"
FRAME_LIMIT="0"
SAVE_WAIT_S="300"
SAVE_CALL_TIMEOUT_S="1800"
START_VISUALIZER="false"
CHANGE_DETECTION_EVERY_N_FRAMES="-1"
MIN_GLOG_LEVEL="1"
GLOG_VERBOSITY="0"
MAPPER_CONFIG="${ROOT}/../configs/khronos/uHumans2_map_update_5_change_no_obj_bg_remove.yaml"
INPUT_CONFIG="${ROOT}/configs/nss_flat_input.yaml"
LABELSPACE_CONFIG="${ROOT}/configs/nss_surface_label_space.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --play-rate) PLAY_RATE="$2"; shift 2 ;;
    --tf-settle-s) TF_SETTLE_S="$2"; shift 2 ;;
    --frame-limit) FRAME_LIMIT="$2"; shift 2 ;;
    --save-wait-s) SAVE_WAIT_S="$2"; shift 2 ;;
    --save-call-timeout-s) SAVE_CALL_TIMEOUT_S="$2"; shift 2 ;;
    --start-visualizer) START_VISUALIZER="$2"; shift 2 ;;
    --change-detection-every-n-frames) CHANGE_DETECTION_EVERY_N_FRAMES="$2"; shift 2 ;;
    --min-glog-level) MIN_GLOG_LEVEL="$2"; shift 2 ;;
    --verbosity) GLOG_VERBOSITY="$2"; shift 2 ;;
    --mapper-config) MAPPER_CONFIG="$2"; shift 2 ;;
    --input-config) INPUT_CONFIG="$2"; shift 2 ;;
    --labelspace-config) LABELSPACE_CONFIG="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 --run-dir DIR --output-dir DIR [--play-rate N] [--frame-limit N] [--min-glog-level N] [--verbosity N]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${RUN_DIR}" || -z "${OUTPUT_DIR}" ]]; then
  echo "--run-dir and --output-dir are required" >&2
  exit 2
fi
for path in "${RUN_DIR}" "${MAPPER_CONFIG}" "${INPUT_CONFIG}" "${LABELSPACE_CONFIG}"; do
  if [[ ! -e "${path}" ]]; then
    echo "MISSING_INPUT ${path}" >&2
    exit 2
  fi
done

mkdir -p "$(dirname "${OUTPUT_DIR}")"
RUN_DIR="$(realpath "${RUN_DIR}")"
OUTPUT_DIR="$(realpath -m "${OUTPUT_DIR}")"
MAPPER_CONFIG="$(realpath "${MAPPER_CONFIG}")"
INPUT_CONFIG="$(realpath "${INPUT_CONFIG}")"
LABELSPACE_CONFIG="$(realpath "${LABELSPACE_CONFIG}")"
CONTROL_DIR="${OUTPUT_DIR}_control"
mkdir -p "${CONTROL_DIR}/logs"
export ROS_HOME="${CONTROL_DIR}/ros_home"
export ROS_LOG_DIR="${CONTROL_DIR}/logs/ros"
mkdir -p "${ROS_HOME}" "${ROS_LOG_DIR}"

cat >"${CONTROL_DIR}/command.txt" <<EOF
$0 --run-dir ${RUN_DIR} --output-dir ${OUTPUT_DIR} --play-rate ${PLAY_RATE} --tf-settle-s ${TF_SETTLE_S} --frame-limit ${FRAME_LIMIT} --save-wait-s ${SAVE_WAIT_S} --save-call-timeout-s ${SAVE_CALL_TIMEOUT_S} --change-detection-every-n-frames ${CHANGE_DETECTION_EVERY_N_FRAMES} --min-glog-level ${MIN_GLOG_LEVEL} --verbosity ${GLOG_VERBOSITY}
MAPPER_CONFIG=${MAPPER_CONFIG}
INPUT_CONFIG=${INPUT_CONFIG}
LABELSPACE_CONFIG=${LABELSPACE_CONFIG}
EOF

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill -INT -- "-${pid}" 2>/dev/null || kill -INT "${pid}" 2>/dev/null || true
  done
  for _ in $(seq 1 20); do
    any_alive=false
    for pid in "${pids[@]:-}"; do
      if kill -0 -- "-${pid}" 2>/dev/null || kill -0 "${pid}" 2>/dev/null; then
        any_alive=true
      fi
    done
    [[ "${any_alive}" == "false" ]] && break
    sleep 1
  done
  for pid in "${pids[@]:-}"; do
    if kill -0 -- "-${pid}" 2>/dev/null || kill -0 "${pid}" 2>/dev/null; then
      kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
      sleep 1
    fi
    if kill -0 -- "-${pid}" 2>/dev/null || kill -0 "${pid}" 2>/dev/null; then
      kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
    fi
    wait "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

EXTRA_YAML="{semantic_colormap_file: '', experiment: {save_every_n_frames: 0, save_full_state: true, log_timing_details: false, exit_after_clock: false}, store_visualization_details: false, backend: {change_detection: {run_every_n_frames: ${CHANGE_DETECTION_EVERY_N_FRAMES}}}}"

setsid ros2 launch khronos_ros khronos.launch.yaml \
  input_config:="${INPUT_CONFIG}" \
  mapper_config:="${MAPPER_CONFIG}" \
  labelspace_config:="${LABELSPACE_CONFIG}" \
  extra_yaml:="${EXTRA_YAML}" \
  output_dir:="${OUTPUT_DIR}" \
  evaluate:=false \
  start_visualizer:="${START_VISUALIZER}" \
  use_gt_frame:=true \
  sensor_min_range:=0.1 \
  sensor_max_range:=20.0 \
  sensor_frame:=left_cam \
  robot_frame:=robot_0 \
  odom_frame:=odom \
  map_frame:=map \
  world_frame:=world \
  rgb_topic:=/nss/rgb/image_raw \
  rgb_info_topic:=/nss/rgb/camera_info \
  depth_topic:=/nss/depth/image_raw \
  label_topic:=/nss/semantic/image_raw \
  min_glog_level:="${MIN_GLOG_LEVEL}" \
  verbosity:="${GLOG_VERBOSITY}" \
  >"${CONTROL_DIR}/logs/khronos.log" 2>&1 &
KHRONOS_PID=$!
pids+=("${KHRONOS_PID}")

sleep 3
"${BASE1_PYTHON}" "${ROOT}/scripts/nss_flat_ros2_player.py" \
  --run-dir "${RUN_DIR}" \
  --play-rate "${PLAY_RATE}" \
  --tf-settle-s "${TF_SETTLE_S}" \
  --post-wait-s 5 \
  --frame-limit "${FRAME_LIMIT}" \
  --manifest "${CONTROL_DIR}/playback_manifest.json" \
  >"${CONTROL_DIR}/logs/player.log" 2>&1

SAVE_SERVICE=""
for _ in $(seq 1 30); do
  services="$(ros2 service list 2>>"${CONTROL_DIR}/logs/service_wait.log" || true)"
  if grep -qx "/khronos_node/finish_mapping_and_save" <<<"${services}"; then
    SAVE_SERVICE="/khronos_node/finish_mapping_and_save"
    break
  fi
  if grep -qx "/finish_mapping_and_save" <<<"${services}"; then
    SAVE_SERVICE="/finish_mapping_and_save"
    break
  fi
  sleep 1
done
if [[ -z "${SAVE_SERVICE}" ]]; then
  echo "FAILED_SERVICE_NOT_READY" >&2
  tail -100 "${CONTROL_DIR}/logs/khronos.log" >&2 || true
  exit 4
fi
printf '%s\n' "${SAVE_SERVICE}" >"${CONTROL_DIR}/save_service_name.txt"
if ! timeout "${SAVE_CALL_TIMEOUT_S}" ros2 service call \
    "${SAVE_SERVICE}" std_srvs/srv/Empty \
    >"${CONTROL_DIR}/logs/save_service.log" 2>&1; then
  echo "FAILED_SAVE_SERVICE_CALL service=${SAVE_SERVICE} timeout_s=${SAVE_CALL_TIMEOUT_S}" >&2
  tail -120 "${CONTROL_DIR}/logs/save_service.log" >&2 || true
  tail -120 "${CONTROL_DIR}/logs/khronos.log" >&2 || true
  exit 5
fi

for _ in $(seq 1 "${SAVE_WAIT_S}"); do
  if [[ -s "${OUTPUT_DIR}/final.4dmap" ]] && \
      grep -q "Experiment Finished Cleanly" "${OUTPUT_DIR}/experiment_log.txt" 2>/dev/null; then
    break
  fi
  sleep 1
done

if [[ ! -s "${OUTPUT_DIR}/final.4dmap" ]]; then
  echo "FAILED_MISSING_FINAL_4DMAP ${OUTPUT_DIR}/final.4dmap" >&2
  tail -120 "${CONTROL_DIR}/logs/khronos.log" >&2 || true
  exit 6
fi
if ! grep -q "Experiment Finished Cleanly" "${OUTPUT_DIR}/experiment_log.txt" 2>/dev/null; then
  echo "FAILED_EXPERIMENT_NOT_CLEAN ${OUTPUT_DIR}" >&2
  tail -120 "${OUTPUT_DIR}/experiment_log.txt" >&2 || true
  exit 7
fi

echo "SUCCESS_NSS_KHRONOS_FINAL_4DMAP ${OUTPUT_DIR}/final.4dmap"
ls -lh "${OUTPUT_DIR}/final.4dmap" "${OUTPUT_DIR}/experiment_log.txt"
