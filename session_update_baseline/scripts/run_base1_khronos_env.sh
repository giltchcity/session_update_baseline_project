#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/khronos_env.sh"

usage() {
  cat <<'EOF'
Usage: run_base1_khronos_env.sh COMMAND [arguments]

Commands:
  check                  verify the source-pinned diagnostic environment
  build                  run the canonical mapping + baseline build
  test                   build and run the portable-core tests
  khronos-map ARGS...    run ObjectGuidedMapReconciler on a Khronos .4dmap
  export-mesh ARGS...    export the latest global mesh from a Khronos .4dmap
  flat-build ARGS...     build one final map from a virtual-flat session
  flat-memory ARGS...    reconcile Session A memory with Session B
  state-audit ARGS...    audit A/B object-memory state transitions
  nss-render ARGS...     render an NSS mesh into virtual-flat input

These are diagnostic/export commands, not the production session runner.
Use run_session.sh for mapping. Khronos resolves from ports/mapping_core;
installed dependency packages are supplied by BASE1_DEPENDENCY_SETUP.
EOF
}

build_base1() {
  "${ROOT}/scripts/build_canonical.sh"
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
  "${ROOT}/scripts/check_canonical_runtime.sh" --require-built
  if ldd "${BASE1_BUILD_DIR}/run_session_update_baseline" | grep -q "not found"; then
    ldd "${BASE1_BUILD_DIR}/run_session_update_baseline" | grep "not found" >&2
    echo "BASE1_ENV_ERROR unresolved shared libraries" >&2
    return 3
  fi
  echo "BASE1_ENV_OK canonical mapping core and diagnostic tools are ready"
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
    ;;
  khronos-map)
    "${ROOT}/scripts/check_canonical_runtime.sh" --require-built >/dev/null
    exec "${BASE1_BUILD_DIR}/run_session_update_baseline" "$@"
    ;;
  export-mesh)
    "${ROOT}/scripts/check_canonical_runtime.sh" --require-built >/dev/null
    exec "${BASE1_BUILD_DIR}/export_4dmap_mesh_ply" "$@"
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
