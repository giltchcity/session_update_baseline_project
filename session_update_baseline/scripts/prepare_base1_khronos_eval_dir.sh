#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 BASE1_RUN_DIR EVAL_EXPERIMENT_DIR [MAP_NAME]" >&2
  echo "Example:" >&2
  echo "  $0 session_update_baseline/runs/loop08_full /tmp/base1_eval_loop08_full improved_final.4dmap" >&2
  exit 2
fi

BASE1_RUN_DIR="$1"
EVAL_EXPERIMENT_DIR="$2"
MAP_NAME="${3:-improved_final.4dmap}"

SOURCE_MAP="${BASE1_RUN_DIR}/${MAP_NAME}"
if [[ ! -f "${SOURCE_MAP}" ]]; then
  echo "Missing map: ${SOURCE_MAP}" >&2
  exit 1
fi

mkdir -p "${EVAL_EXPERIMENT_DIR}"
rm -f "${EVAL_EXPERIMENT_DIR}/final.4dmap"
cp "${SOURCE_MAP}" "${EVAL_EXPERIMENT_DIR}/final.4dmap"

for name in \
  command.txt \
  config.yaml \
  evidence_summary.json \
  mesh_update_summary.csv \
  mesh_vertex_update_summary.csv \
  object_audit.csv \
  object_memory.json \
  object_update_summary.csv; do
  if [[ -f "${BASE1_RUN_DIR}/${name}" ]]; then
    cp "${BASE1_RUN_DIR}/${name}" "${EVAL_EXPERIMENT_DIR}/${name}"
  fi
done

cat > "${EVAL_EXPERIMENT_DIR}/base1_eval_source.txt" <<EOF
base1_run_dir=${BASE1_RUN_DIR}
source_map=${SOURCE_MAP}
prepared_final_map=${EVAL_EXPERIMENT_DIR}/final.4dmap
prepared_at=$(date '+%Y-%m-%d %H:%M:%S %Z')

Khronos final-map-only eval command:
  source /home/jixian/ros2_ws/install/setup.bash
  /home/jixian/ros2_ws/install/khronos_eval/lib/khronos_eval/exp_pipeline \\
    /home/jixian/ros2_ws/install/khronos_eval/share/khronos_eval/config/pipeline/office.yaml \\
    ${EVAL_EXPERIMENT_DIR} true true true

Notes:
  - This only prepares the directory and does not run the memory-heavy evaluator.
  - The evaluator expects a file named final.4dmap in the experiment directory.
EOF

cat > "${EVAL_EXPERIMENT_DIR}/experiment_log.txt" <<EOF
[FLAG] [Experiment Finished Cleanly] Base1 prepared eval directory
[INFO] This directory was prepared by session_update_baseline/scripts/prepare_base1_khronos_eval_dir.sh.
[INFO] Source map: ${SOURCE_MAP}
EOF

echo "PREPARED ${EVAL_EXPERIMENT_DIR}"
ls -lh "${EVAL_EXPERIMENT_DIR}/final.4dmap"
