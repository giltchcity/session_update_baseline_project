#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 MAP_FILE OUTPUT_ROOT [PRIOR_MAP] [PRIOR_OBJECT_MEMORY]" >&2
  exit 2
fi

MAP_FILE="$1"
OUTPUT_ROOT="$2"
PRIOR_MAP="${3:-}"
PRIOR_OBJECT_MEMORY="${4:-}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="${ROOT}/scripts/run_session_update_baseline.sh"

mkdir -p "${OUTPUT_ROOT}"

run_case() {
  local name="$1"
  shift
  local out_dir="${OUTPUT_ROOT}/${name}"
  rm -rf "${out_dir}"
  "${RUNNER}" --map_file "${MAP_FILE}" --output_dir "${out_dir}" "$@"
}

run_case "00_noop" --mode no_op --dynamic_mode within_session
run_case "01_audit" --mode audit --dynamic_mode within_session
run_case "02_cleanup_dryrun_005" --mode cleanup --dynamic_mode within_session \
  --object_distance_m 0.05 --dry_run true
run_case "03_cleanup_002" --mode cleanup --dynamic_mode within_session \
  --object_distance_m 0.02
run_case "04_cleanup_003" --mode cleanup --dynamic_mode within_session \
  --object_distance_m 0.03
run_case "05_cleanup_005" --mode cleanup --dynamic_mode within_session \
  --object_distance_m 0.05
run_case "06_cleanup_008" --mode cleanup --dynamic_mode within_session \
  --object_distance_m 0.08
run_case "07_cleanup_005_same_label" --mode cleanup --dynamic_mode within_session \
  --object_distance_m 0.05 --require_same_label true

if [[ -n "${PRIOR_MAP}" ]]; then
  run_case "08_cross_cleanup_005" --mode cleanup --dynamic_mode cross_session \
    --object_distance_m 0.05 --prior_map "${PRIOR_MAP}" \
    --prior_object_memory "${PRIOR_OBJECT_MEMORY}"
fi

python3 - "${OUTPUT_ROOT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/evidence_summary.json")):
    data = json.loads(path.read_text())
    row = {"case": path.parent.name}
    for key in [
        "mode",
        "dynamic_mode",
        "prior_map_loaded",
        "output_scope",
        "object_distance_m",
        "bbox_margin_m",
        "min_object_mesh_vertices",
        "require_same_label",
        "require_bbox_containment",
        "dry_run",
        "initial_vertices",
        "initial_faces",
        "final_vertices",
        "final_faces",
        "candidate_vertices",
        "removed_vertices",
        "objects_total",
        "objects_with_private_mesh",
        "objects_used_for_cleanup",
        "cleanup_source_points",
    ]:
        row[key] = data.get(key, "")
    rows.append(row)

fieldnames = [
    "case",
    "mode",
    "dynamic_mode",
    "prior_map_loaded",
    "output_scope",
    "object_distance_m",
    "bbox_margin_m",
    "min_object_mesh_vertices",
    "require_same_label",
    "require_bbox_containment",
    "dry_run",
    "initial_vertices",
    "initial_faces",
    "final_vertices",
    "final_faces",
    "candidate_vertices",
    "removed_vertices",
    "objects_total",
    "objects_with_private_mesh",
    "objects_used_for_cleanup",
    "cleanup_source_points",
]
with (root / "sweep_summary.csv").open("w", newline="") as fout:
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(root / "sweep_summary.csv")
PY

