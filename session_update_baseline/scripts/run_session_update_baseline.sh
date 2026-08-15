#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT}/build_canonical"

if [[ "${SESSION_UPDATE_ENABLE_LEGACY_POSTPROCESS:-0}" != "1" ]]; then
  echo "LEGACY_POSTPROCESS_DISABLED" >&2
  echo "This tool edits a completed map and is not the recurrent session mapper." >&2
  echo "Use ${ROOT}/scripts/run_session.sh, or explicitly set" >&2
  echo "SESSION_UPDATE_ENABLE_LEGACY_POSTPROCESS=1 for a diagnostic post-process run." >&2
  exit 64
fi

# shellcheck disable=SC1091
source "${ROOT}/scripts/khronos_env.sh"

"${ROOT}/scripts/check_canonical_runtime.sh" --require-built >/dev/null

exec "${BUILD_DIR}/run_session_update_baseline" "$@"
