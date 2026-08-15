#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "LEGACY_SESSION_ENTRY_DISABLED" >&2
echo "This launcher resolves ros2 launch packages from whichever overlay was sourced." >&2
echo "Use the source-pinned entry: ${ROOT}/scripts/run_session.sh" >&2
exit 64
