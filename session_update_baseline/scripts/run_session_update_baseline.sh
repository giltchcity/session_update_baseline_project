#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT}/build"

if [[ -f /home/jixian/ros2_ws/install/setup.bash ]]; then
  set +u
  # shellcheck disable=SC1091
  source /home/jixian/ros2_ws/install/setup.bash
  set -u
fi

cmake -S "${ROOT}" -B "${BUILD_DIR}" >/dev/null
cmake --build "${BUILD_DIR}" -j"$(nproc)" >/dev/null

exec "${BUILD_DIR}/run_session_update_baseline" "$@"
