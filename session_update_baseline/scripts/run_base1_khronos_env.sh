#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/khronos_env.sh"

usage() {
  cat <<'EOF'
Usage: run_base1_khronos_env.sh COMMAND [arguments]

Commands:
  check                  verify the unified Khronos/Base1 environment
  build                  build Base1 and its Panoptic-derived portable core
  test                   build and run the portable-core tests
  khronos-map ARGS...    run ObjectGuidedMapReconciler on a Khronos .4dmap
  export-mesh ARGS...    export the latest global mesh from a Khronos .4dmap
  flat-build ARGS...     build one final map from a virtual-flat session
  flat-memory ARGS...    reconcile Session A memory with Session B
  state-audit ARGS...    audit A/B object-memory state transitions
  nss-render ARGS...     render an NSS mesh into virtual-flat input

All commands run in ROS2 Jazzy + /home/jixian/ros2_ws/install and use
/usr/bin/python3 unless BASE1_PYTHON is explicitly overridden.
EOF
}

build_base1() {
  cmake -S "${ROOT}" -B "${ROOT}/build"
  cmake --build "${ROOT}/build" -j"$(nproc)"
}

check_environment() {
  "${BASE1_PYTHON}" - <<'PY'
import sys
import cv2
import numpy
import scipy
import yaml

print(f"BASE1_PYTHON={sys.executable}")
print(f"PYTHON_VERSION={sys.version.split()[0]}")
print(f"NUMPY_VERSION={numpy.__version__}")
print(f"SCIPY_VERSION={scipy.__version__}")
print(f"OPENCV_VERSION={cv2.__version__}")
print(f"PYYAML_VERSION={yaml.__version__}")
PY
  command -v ros2
  test -f /home/jixian/ros2_ws/install/khronos/lib/cmake/khronos/khronosConfig.cmake
  build_base1 >/dev/null
  if ldd "${ROOT}/build/run_session_update_baseline" | grep -q "not found"; then
    ldd "${ROOT}/build/run_session_update_baseline" | grep "not found" >&2
    echo "BASE1_ENV_ERROR unresolved shared libraries" >&2
    return 3
  fi
  echo "BASE1_ENV_OK Khronos, Base1 Python, and portable Panoptic core are ready"
}

command_name="${1:-}"
if [[ -z "${command_name}" ]]; then
  usage
  exit 2
fi
shift

case "${command_name}" in
  check)
    check_environment
    ;;
  build)
    build_base1
    ;;
  test)
    build_base1
    ctest --test-dir "${ROOT}/build" --output-on-failure
    ;;
  khronos-map)
    build_base1 >/dev/null
    exec "${ROOT}/build/run_session_update_baseline" "$@"
    ;;
  export-mesh)
    build_base1 >/dev/null
    exec "${ROOT}/build/export_4dmap_mesh_ply" "$@"
    ;;
  flat-build)
    exec "${BASE1_PYTHON}" "${ROOT}/src/base1/flat_clean_map_builder.py" "$@"
    ;;
  flat-memory)
    exec "${BASE1_PYTHON}" "${ROOT}/src/base1/flat_memory_update_builder.py" "$@"
    ;;
  state-audit)
    exec "${BASE1_PYTHON}" "${ROOT}/src/base1/base12_state_audit.py" "$@"
    ;;
  nss-render)
    exec "${BASE1_PYTHON}" "${ROOT}/../scripts/nss/render_virtual_flat_dataset.py" "$@"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown command: ${command_name}" >&2
    usage >&2
    exit 2
    ;;
esac
