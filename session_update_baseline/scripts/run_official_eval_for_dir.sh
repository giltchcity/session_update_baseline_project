#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 EVAL_DIR [EVAL_CONFIG]" >&2
  exit 2
fi

EVAL_DIR="$1"
EVAL_CONFIG="${2:-/home/jixian/Desktop/FT/configs/khronos/eval_office_local.yaml}"

if [[ ! -f "$EVAL_DIR/final.4dmap" ]]; then
  echo "Missing $EVAL_DIR/final.4dmap" >&2
  exit 1
fi

if [[ ! -f "$EVAL_DIR/experiment_log.txt" ]]; then
  cat > "$EVAL_DIR/experiment_log.txt" <<EOF
[FLAG] [Experiment Finished Cleanly] Base1 prepared eval directory
[INFO] Created by run_official_eval_for_dir.sh
EOF
fi

set +u
source /home/jixian/ros2_ws/install/setup.bash
set -u

mkdir -p "$EVAL_DIR"
/usr/bin/time -v \
  /home/jixian/ros2_ws/install/khronos_eval/lib/khronos_eval/exp_pipeline \
  "$EVAL_CONFIG" \
  "$EVAL_DIR" \
  true \
  true \
  true \
  > "$EVAL_DIR/eval_final_only_local.log" 2>&1

test -f "$EVAL_DIR/results/background_mesh.csv"
test -f "$EVAL_DIR/results/static_objects.csv"
echo "EVAL_DONE $EVAL_DIR"
