#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAPPING_SOURCE="${ROOT}/ports/mapping_core"
CANONICAL_ROOT="${SESSION_UPDATE_CANONICAL_ROOT:-${ROOT}/.canonical_mapping}"
CANONICAL_PREFIX="${SESSION_UPDATE_CANONICAL_PREFIX:-${BASE1_CANONICAL_PREFIX:-${CANONICAL_ROOT}/install}}"
BASELINE_BUILD="${SESSION_UPDATE_CANONICAL_BUILD:-${BASE1_BUILD_DIR:-${ROOT}/build_canonical}}"
CONFIG="${SESSION_UPDATE_CANONICAL_CONFIG:-${ROOT}/configs/room18_instance_5cm.yaml}"
FINGERPRINT_TOOL="${ROOT}/scripts/fingerprint_sources.py"
FINGERPRINT_PYTHON="${BASE1_PYTHON:-/usr/bin/python3}"
REQUIRE_BUILT=false

usage() {
  cat <<'EOF'
Usage: check_canonical_runtime.sh [--source-only|--require-built]

Checks that the canonical mapper source is ports/mapping_core, that the 5 cm
configuration has the frozen values, and (with --require-built) that the strict
runtime binary resolves Khronos only from the canonical install prefix.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-only) REQUIRE_BUILT=false; shift ;;
    --require-built) REQUIRE_BUILT=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "CANONICAL_CHECK_ERROR unknown argument: $1" >&2; exit 2 ;;
  esac
done

fail() {
  echo "CANONICAL_CHECK_ERROR $*" >&2
  exit 3
}

[[ -f "${MAPPING_SOURCE}/khronos/CMakeLists.txt" ]] || \
  fail "missing canonical Khronos source: ${MAPPING_SOURCE}/khronos"
[[ -f "${MAPPING_SOURCE}/khronos_ros/CMakeLists.txt" ]] || \
  fail "missing canonical Khronos ROS source: ${MAPPING_SOURCE}/khronos_ros"
[[ -f "${ROOT}/scripts/run_session.sh" ]] || fail "missing canonical session runner"
[[ -f "${FINGERPRINT_TOOL}" ]] || fail "missing source fingerprint tool"
[[ -x "${FINGERPRINT_PYTHON}" ]] || fail "missing Python: ${FINGERPRINT_PYTHON}"
[[ -f "${CONFIG}" ]] || fail "missing canonical configuration: ${CONFIG}"

grep -Fq 'SESSION_UPDATE_CANONICAL_MAPPING_PREFIX' "${ROOT}/CMakeLists.txt" || \
  fail "top-level CMake does not enforce a canonical mapping prefix"
grep -Fq 'LEGACY_AB_PIPELINE_DISABLED' "${ROOT}/scripts/run_final_ab_pipeline.sh" || \
  fail "legacy A/B pipeline is not explicitly disabled"

"${BASE1_PYTHON:-/usr/bin/python3}" - "${CONFIG}" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
values = {}
stack = []
for number, original in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    line = original.split("#", 1)[0].rstrip()
    if not line.strip() or ":" not in line:
        continue
    indent = len(line) - len(line.lstrip(" "))
    key, value = line.strip().split(":", 1)
    while stack and stack[-1][0] >= indent:
        stack.pop()
    current = tuple(item[1] for item in stack) + (key,)
    value = value.strip()
    if value:
        values[current] = (value, number)
    else:
        stack.append((indent, key))

expected = {
    ("shared_parameters", "change_detection_every_n_backend_updates"):
        "&change_detection_every_n_backend_updates 5",
    ("active_window", "min_output_separation"): "0.4",
    ("active_window", "frame_data_buffer", "max_buffer_size"): "100",
    ("active_window", "frame_data_buffer", "store_every_n_frames"): "3",
    ("active_window", "volumetric_map", "voxel_size"): "0.05",
    ("active_window", "volumetric_map", "truncation_distance"): "0.15",
    ("frontend", "pgmo", "mesh_resolution"): "0.005",
    ("backend", "change_detection", "run_every_n_frames"):
        "*change_detection_every_n_backend_updates",
}
for key, wanted in expected.items():
    actual = values.get(key)
    if actual is None or actual[0] != wanted:
        dotted = ".".join(key)
        got = "missing" if actual is None else repr(actual[0])
        raise SystemExit(f"CANONICAL_CHECK_ERROR {dotted}: expected {wanted!r}, got {got}")
PY

if [[ "${REQUIRE_BUILT}" != true ]]; then
  echo "CANONICAL_SOURCE_OK source=${MAPPING_SOURCE} config=${CONFIG}"
  exit 0
fi

SOURCE_MARKER="${CANONICAL_PREFIX}/.session_update_mapping_source"
MAPPING_FINGERPRINT_MARKER="${CANONICAL_PREFIX}/.session_update_mapping_fingerprint"
BASELINE_FINGERPRINT_MARKER="${BASELINE_BUILD}/.session_update_baseline_fingerprint"
[[ -f "${SOURCE_MARKER}" ]] || fail "missing source marker; run scripts/build_canonical.sh"
[[ "$(cat "${SOURCE_MARKER}")" == "$(realpath "${MAPPING_SOURCE}")" ]] || \
  fail "canonical install was not built from ${MAPPING_SOURCE}"
[[ -f "${MAPPING_FINGERPRINT_MARKER}" ]] || \
  fail "canonical mapping build predates source fingerprinting; rebuild it"
[[ -f "${BASELINE_FINGERPRINT_MARKER}" ]] || \
  fail "baseline build predates source fingerprinting; rebuild it"
CURRENT_MAPPING_FINGERPRINT="$(
  "${FINGERPRINT_PYTHON}" "${FINGERPRINT_TOOL}" --root "${ROOT}" ports/mapping_core
)"
CURRENT_BASELINE_FINGERPRINT="$(
  "${FINGERPRINT_PYTHON}" "${FINGERPRINT_TOOL}" --root "${ROOT}" \
    CMakeLists.txt app include src ports/panoptic_core
)"
[[ "$(cat "${MAPPING_FINGERPRINT_MARKER}")" == "${CURRENT_MAPPING_FINGERPRINT}" ]] || \
  fail "canonical mapper binary is stale relative to ports/mapping_core"
[[ "$(cat "${BASELINE_FINGERPRINT_MARKER}")" == "${CURRENT_BASELINE_FINGERPRINT}" ]] || \
  fail "baseline binary is stale relative to current C++ source"

cache_value() {
  local cache=$1
  local key=$2
  awk -F= -v key="${key}" '$1 ~ ("^" key ":[^=]+$") {print $2; exit}' "${cache}"
}

MAPPING_CACHE="${CANONICAL_ROOT}/build/khronos/CMakeCache.txt"
MAPPING_ROS_CACHE="${CANONICAL_ROOT}/build/khronos_ros/CMakeCache.txt"
BASELINE_CACHE="${BASELINE_BUILD}/CMakeCache.txt"
for cache in "${MAPPING_CACHE}" "${MAPPING_ROS_CACHE}" "${BASELINE_CACHE}"; do
  [[ -f "${cache}" ]] || fail "missing build cache: ${cache}"
done

[[ "$(realpath "$(cache_value "${MAPPING_CACHE}" CMAKE_HOME_DIRECTORY)")" == \
   "$(realpath "${MAPPING_SOURCE}/khronos")" ]] || \
  fail "khronos build cache points at another source tree"
[[ "$(realpath "$(cache_value "${MAPPING_ROS_CACHE}" CMAKE_HOME_DIRECTORY)")" == \
   "$(realpath "${MAPPING_SOURCE}/khronos_ros")" ]] || \
  fail "khronos_ros build cache points at another source tree"
[[ "$(realpath "$(cache_value "${BASELINE_CACHE}" CMAKE_HOME_DIRECTORY)")" == \
   "$(realpath "${ROOT}")" ]] || fail "baseline build cache points at another source tree"
[[ "$(realpath "$(cache_value "${BASELINE_CACHE}" khronos_DIR)")" == \
   "$(realpath "${CANONICAL_PREFIX}/lib/cmake/khronos")" ]] || \
  fail "baseline resolved khronos outside the canonical prefix"
[[ "$(realpath "$(cache_value "${BASELINE_CACHE}" khronos_ros_DIR)")" == \
   "$(realpath "${CANONICAL_PREFIX}/share/khronos_ros/cmake")" ]] || \
  fail "baseline resolved khronos_ros outside the canonical prefix"

BINARY="${BASELINE_BUILD}/session_khronos_node"
[[ -x "${BINARY}" ]] || fail "missing canonical runtime binary: ${BINARY}"
[[ -x "${BASELINE_BUILD}/inspect_session_state" ]] || \
  fail "missing required state inspector: ${BASELINE_BUILD}/inspect_session_state"
LDD_OUTPUT="$(LD_LIBRARY_PATH="${CANONICAL_PREFIX}/lib:${LD_LIBRARY_PATH:-}" ldd "${BINARY}")"
KHRONOS_LINE="$(grep -E '^[[:space:]]*libkhronos\.so' <<<"${LDD_OUTPUT}" || true)"
KHRONOS_ROS_LINE="$(grep -E '^[[:space:]]*libkhronos_ros\.so' <<<"${LDD_OUTPUT}" || true)"
[[ "${KHRONOS_LINE}" == *"${CANONICAL_PREFIX}/lib/libkhronos.so"* ]] || \
  fail "runtime libkhronos is not canonical: ${KHRONOS_LINE:-missing}"
[[ "${KHRONOS_ROS_LINE}" == *"${CANONICAL_PREFIX}/lib/libkhronos_ros.so"* ]] || \
  fail "runtime libkhronos_ros is not canonical: ${KHRONOS_ROS_LINE:-missing}"

echo "CANONICAL_RUNTIME_OK binary=${BINARY} source=${MAPPING_SOURCE} config=${CONFIG}"
