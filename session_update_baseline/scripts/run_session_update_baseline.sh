#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT}/build"

# shellcheck disable=SC1091
source "${ROOT}/scripts/khronos_env.sh"

cmake -S "${ROOT}" -B "${BUILD_DIR}" >/dev/null
cmake --build "${BUILD_DIR}" -j"$(nproc)" >/dev/null

exec "${BUILD_DIR}/run_session_update_baseline" "$@"
