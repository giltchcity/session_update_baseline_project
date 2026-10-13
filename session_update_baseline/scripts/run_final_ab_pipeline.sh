#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "LEGACY_AB_PIPELINE_DISABLED" >&2
echo "This entry mixed B-from-scratch, A-seeded B, and a second offline A+B reconciliation." >&2
echo "Use ${ROOT}/scripts/run_session.sh once per session and pass the previous output with --input-state." >&2
exit 64
