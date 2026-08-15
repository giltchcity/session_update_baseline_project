#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(realpath "${ROOT}/..")"
MAPPING_SOURCE="${ROOT}/ports/mapping_core"
CANONICAL_ROOT="${SESSION_UPDATE_CANONICAL_ROOT:-${ROOT}/.canonical_mapping}"
CANONICAL_PREFIX="${SESSION_UPDATE_CANONICAL_PREFIX:-${CANONICAL_ROOT}/install}"
BASELINE_BUILD="${SESSION_UPDATE_CANONICAL_BUILD:-${ROOT}/build_canonical}"
ROS_SETUP="${SESSION_UPDATE_ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
DEPENDENCY_SETUP="${SESSION_UPDATE_DEPENDENCY_SETUP:-/home/jixian/ros2_ws/install/setup.bash}"
JOBS="${SESSION_UPDATE_BUILD_JOBS:-$(nproc)}"
RUN_TESTS=true
ALLOW_INCREMENTAL=false
FINGERPRINT_TOOL="${ROOT}/scripts/fingerprint_sources.py"
FINGERPRINT_PYTHON="${BASE1_PYTHON:-/usr/bin/python3}"
MAPPING_FINGERPRINT_BEFORE=""
BASELINE_FINGERPRINT_BEFORE=""

usage() {
  cat <<'EOF'
Usage: build_canonical.sh [--jobs N] [--no-tests] [--incremental]

Builds ports/mapping_core into .canonical_mapping/install and then builds the
baseline against that exact prefix. It never resolves Khronos or khronos_ros
from an external workspace. External Hydra/PGMO/etc. packages are still used as
dependency packages until they are moved into this repository.

By default the script requires empty output paths. It never deletes an existing
build or install directory. Set SESSION_UPDATE_CANONICAL_ROOT and
SESSION_UPDATE_CANONICAL_BUILD to unique paths for an isolated clean build, or
use --incremental to continue a build that you own.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs) JOBS=$2; shift 2 ;;
    --no-tests) RUN_TESTS=false; shift ;;
    --incremental) ALLOW_INCREMENTAL=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "CANONICAL_BUILD_ERROR unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "${JOBS}" =~ ^[1-9][0-9]*$ ]] || { echo "CANONICAL_BUILD_ERROR invalid jobs: ${JOBS}" >&2; exit 2; }
[[ -f "${ROS_SETUP}" ]] || { echo "CANONICAL_BUILD_ERROR missing ROS setup: ${ROS_SETUP}" >&2; exit 3; }
[[ -f "${DEPENDENCY_SETUP}" ]] || { echo "CANONICAL_BUILD_ERROR missing dependency setup: ${DEPENDENCY_SETUP}" >&2; exit 3; }
command -v colcon >/dev/null || { echo "CANONICAL_BUILD_ERROR colcon is not installed" >&2; exit 3; }
command -v flock >/dev/null || { echo "CANONICAL_BUILD_ERROR flock is not installed" >&2; exit 3; }
[[ -f "${FINGERPRINT_TOOL}" ]] || { echo "CANONICAL_BUILD_ERROR missing source fingerprint tool" >&2; exit 3; }
[[ -x "${FINGERPRINT_PYTHON}" ]] || { echo "CANONICAL_BUILD_ERROR missing Python: ${FINGERPRINT_PYTHON}" >&2; exit 3; }

"${ROOT}/scripts/check_canonical_runtime.sh" --source-only

MAPPING_FINGERPRINT_BEFORE="$(
  "${FINGERPRINT_PYTHON}" "${FINGERPRINT_TOOL}" --root "${ROOT}" ports/mapping_core
)"
BASELINE_FINGERPRINT_BEFORE="$(
  "${FINGERPRINT_PYTHON}" "${FINGERPRINT_TOOL}" --root "${ROOT}" \
    CMakeLists.txt app include src ports/panoptic_core
)"

# Serialize callers that selected the same output root. The lock file is kept
# deliberately: unlike deleting a stale build directory, opening and locking it
# is non-destructive and the kernel releases the lock when this process exits.
LOCK_PATH="${CANONICAL_ROOT}.build.lock"
mkdir -p "$(dirname "${LOCK_PATH}")"
exec 9>"${LOCK_PATH}"
flock -n 9 || {
  echo "CANONICAL_BUILD_ERROR another build owns ${CANONICAL_ROOT}" >&2
  exit 4
}

if [[ "${ALLOW_INCREMENTAL}" != true ]]; then
  for output in "${CANONICAL_ROOT}/build" "${CANONICAL_PREFIX}" \
                "${CANONICAL_ROOT}/log" "${BASELINE_BUILD}"; do
    if [[ -e "${output}" ]]; then
      echo "CANONICAL_BUILD_ERROR output already exists: ${output}" >&2
      echo "Use unique SESSION_UPDATE_CANONICAL_ROOT/BUILD paths; this script never clears shared builds." >&2
      exit 4
    fi
  done
fi

case "$-" in *u*) RESTORE_NOUNSET=true ;; *) RESTORE_NOUNSET=false ;; esac
set +u
# shellcheck disable=SC1090
source "${ROS_SETUP}"
# shellcheck disable=SC1090
source "${DEPENDENCY_SETUP}"
[[ "${RESTORE_NOUNSET}" == true ]] && set -u

mkdir -p "${CANONICAL_ROOT}" "${BASELINE_BUILD}"
export CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}"
colcon --log-base "${CANONICAL_ROOT}/log" build \
  --base-paths "${MAPPING_SOURCE}" \
  --packages-select khronos_msgs khronos khronos_ros \
  --allow-overriding khronos_msgs khronos khronos_ros \
  --build-base "${CANONICAL_ROOT}/build" \
  --install-base "${CANONICAL_PREFIX}" \
  --merge-install \
  --cmake-clean-cache \
  --cmake-force-configure \
  --cmake-args \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE=/usr/bin/python3 \
    -DPYTHON_EXECUTABLE=/usr/bin/python3

printf '%s\n' "$(realpath "${MAPPING_SOURCE}")" > \
  "${CANONICAL_PREFIX}/.session_update_mapping_source"
printf '%s\n' "$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo UNKNOWN)" > \
  "${CANONICAL_PREFIX}/.session_update_repository_commit"

set +u
# Use local_setup: dependencies were sourced explicitly above, so the generated
# absolute prefix chain cannot silently choose a different Khronos installation.
# shellcheck disable=SC1090
source "${CANONICAL_PREFIX}/local_setup.bash"
[[ "${RESTORE_NOUNSET}" == true ]] && set -u

cmake -S "${ROOT}" -B "${BASELINE_BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DSESSION_UPDATE_CANONICAL_MAPPING_PREFIX="${CANONICAL_PREFIX}"
cmake --build "${BASELINE_BUILD}" -j"${JOBS}"

MAPPING_FINGERPRINT_AFTER="$(
  "${FINGERPRINT_PYTHON}" "${FINGERPRINT_TOOL}" --root "${ROOT}" ports/mapping_core
)"
BASELINE_FINGERPRINT_AFTER="$(
  "${FINGERPRINT_PYTHON}" "${FINGERPRINT_TOOL}" --root "${ROOT}" \
    CMakeLists.txt app include src ports/panoptic_core
)"
[[ "${MAPPING_FINGERPRINT_BEFORE}" == "${MAPPING_FINGERPRINT_AFTER}" ]] || {
  echo "CANONICAL_BUILD_ERROR mapping source changed during build" >&2
  exit 5
}
[[ "${BASELINE_FINGERPRINT_BEFORE}" == "${BASELINE_FINGERPRINT_AFTER}" ]] || {
  echo "CANONICAL_BUILD_ERROR baseline source changed during build" >&2
  exit 5
}
printf '%s\n' "${MAPPING_FINGERPRINT_AFTER}" > \
  "${CANONICAL_PREFIX}/.session_update_mapping_fingerprint"
printf '%s\n' "${BASELINE_FINGERPRINT_AFTER}" > \
  "${BASELINE_BUILD}/.session_update_baseline_fingerprint"

if [[ "${RUN_TESTS}" == true ]]; then
  ctest --test-dir "${CANONICAL_ROOT}/build/khronos" --output-on-failure
  ctest --test-dir "${BASELINE_BUILD}" --output-on-failure
fi
"${ROOT}/scripts/check_canonical_runtime.sh" --require-built
