#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STRICT_RUNNER="${ROOT}/scripts/run_khronos_session_strict.sh"

RUN_DIR=""
SEMANTIC_DIR=""
INSTANCE_DIR=""
INPUT_STATE=""
OUTPUT_STATE=""
WORLD_TRANSFORM=""
SESSION_ID=""
MAPPER_CONFIG="${ROOT}/configs/room18_instance_5cm.yaml"
INPUT_CONFIG="${ROOT}/configs/nss_flat_input.yaml"
LABELSPACE_CONFIG="${ROOT}/configs/nss_ade20k_room_label_space.yaml"
PHYSICAL_CATALOG="${ROOT}/configs/room18_physical_catalog.json"
IMAGE_SCALE="0.5"
PLAY_RATE="100"
FRAME_LIMIT="0"
SESSION_START_NS=""
TF_SETTLE_S="0.02"
ACK_TIMEOUT_S="180"
FINALIZATION_TIMEOUT_S="1800"
DISCOVERY_TIMEOUT_S="180"
SENSOR_MAX_RANGE="5.0"

usage() {
  cat <<'EOF'
Usage:
  run_session.sh \
    --run-dir RGBD_DIR \
    --semantic-dir SEMANTIC_DIR \
    --instance-dir INSTANCE_DIR \
    --output-state STATE_DIR \
    [--input-state PREVIOUS_ACCEPTED_STATE_DIR] \
    [--world-transform FILE] [options]

This is the only production session transition entry point:

  empty + observations_1 -> state_1
  state_1 + observations_2 -> state_2
  state_2 + observations_3 -> state_3

It invokes the mapper exactly once. It never starts a from-scratch control and
never runs the offline Base1 reconciler. STATE_DIR must not already exist and
the resulting map is STATE_DIR/final.4dmap.

Options:
  --session-id NAME
  --mapper-config FILE       default: configs/room18_instance_5cm.yaml
  --input-config FILE        default: configs/nss_flat_input.yaml
  --labelspace-config FILE   default: configs/nss_ade20k_room_label_space.yaml
  --physical-catalog FILE     default: configs/room18_physical_catalog.json
  --image-scale FLOAT        default: 0.5
  --play-rate FLOAT          default: 100
  --frame-limit N            default: 0 (all)
  --session-start-ns NS      default: parse acquisition name in Asia/Shanghai
  --tf-settle-s FLOAT        default: 0.02
  --ack-timeout-s FLOAT      default: 180
  --finalization-timeout-s N default: 1800
  --discovery-timeout-s N    default: 180
  --sensor-max-range FLOAT   default: 5.0
EOF
}

die() {
  echo "SESSION_TRANSITION_ERROR $*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR=$2; shift 2 ;;
    --semantic-dir) SEMANTIC_DIR=$2; shift 2 ;;
    --instance-dir) INSTANCE_DIR=$2; shift 2 ;;
    --input-state) INPUT_STATE=$2; shift 2 ;;
    --output-state) OUTPUT_STATE=$2; shift 2 ;;
    --world-transform) WORLD_TRANSFORM=$2; shift 2 ;;
    --session-id) SESSION_ID=$2; shift 2 ;;
    --mapper-config) MAPPER_CONFIG=$2; shift 2 ;;
    --input-config) INPUT_CONFIG=$2; shift 2 ;;
    --labelspace-config) LABELSPACE_CONFIG=$2; shift 2 ;;
    --physical-catalog) PHYSICAL_CATALOG=$2; shift 2 ;;
    --image-scale) IMAGE_SCALE=$2; shift 2 ;;
    --play-rate) PLAY_RATE=$2; shift 2 ;;
    --frame-limit) FRAME_LIMIT=$2; shift 2 ;;
    --session-start-ns) SESSION_START_NS=$2; shift 2 ;;
    --tf-settle-s) TF_SETTLE_S=$2; shift 2 ;;
    --ack-timeout-s) ACK_TIMEOUT_S=$2; shift 2 ;;
    --finalization-timeout-s) FINALIZATION_TIMEOUT_S=$2; shift 2 ;;
    --discovery-timeout-s) DISCOVERY_TIMEOUT_S=$2; shift 2 ;;
    --sensor-max-range) SENSOR_MAX_RANGE=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "${RUN_DIR}" ]] || die "--run-dir is required"
[[ -n "${SEMANTIC_DIR}" ]] || die "--semantic-dir is required"
[[ -n "${INSTANCE_DIR}" ]] || die "--instance-dir is required"
[[ -n "${OUTPUT_STATE}" ]] || die "--output-state is required"

for directory in "${RUN_DIR}" "${SEMANTIC_DIR}" "${INSTANCE_DIR}"; do
  [[ -d "${directory}" ]] || die "missing input directory: ${directory}"
done

# Load the one canonical runtime environment before importing the ROS player
# for preflight. A direct invocation must not depend on the caller having
# already sourced ROS or a personal workspace.
# shellcheck disable=SC1091
source "${ROOT}/scripts/khronos_env.sh"
"${ROOT}/scripts/check_canonical_runtime.sh" --require-built >/dev/null
[[ -x "${BASE1_BUILD_DIR}/inspect_session_state" ]] || \
  die "canonical build lacks inspect_session_state: ${BASE1_BUILD_DIR}"
[[ "${FINALIZATION_TIMEOUT_S}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || \
  die "--finalization-timeout-s must be a finite positive number"
"${BASE1_PYTHON:-/usr/bin/python3}" - "${FINALIZATION_TIMEOUT_S}" <<'PY' || \
  die "--finalization-timeout-s must be a finite positive number"
import math
import sys

value = float(sys.argv[1])
raise SystemExit(0 if math.isfinite(value) and value > 0.0 else 1)
PY

# Validate the complete dual-label contract before creating an output directory or
# starting ROS. The player uses the same resolver, so a canonical
# <id>_segmentation.png instance directory and the reviewed legacy
# <id>_instances.png directory are both accepted without copying/staging.
INPUT_PREFLIGHT_JSON="$("${BASE1_PYTHON:-/usr/bin/python3}" - \
  "${ROOT}/scripts/nss_flat_ros2_player.py" \
  "${RUN_DIR}" "${SEMANTIC_DIR}" "${INSTANCE_DIR}" \
  "${PHYSICAL_CATALOG}" "${WORLD_TRANSFORM}" <<'PY'
import importlib.util
import json
import pathlib
import sys

module_path, run_dir, semantic_dir, instance_dir, catalog = map(
    pathlib.Path, sys.argv[1:6]
)
world_transform = pathlib.Path(sys.argv[6]) if sys.argv[6] else None
spec = importlib.util.spec_from_file_location("session_update_nss_player", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
frames = module.read_frames(run_dir)
module.load_intrinsics(run_dir)
module.load_world_transform(world_transform)
summary = module.preflight_session_inputs(
    frames, run_dir, semantic_dir, instance_dir, catalog
)
spacing_s = [
    (current[1] - previous[1]) * 1.0e-9
    for previous, current in zip(frames, frames[1:])
]
summary["min_spacing_s"] = min(spacing_s) if spacing_s else None
print(json.dumps(summary, sort_keys=True))
PY
 )" || die "complete input preflight failed"
echo "SESSION_INPUT_OK ${INPUT_PREFLIGHT_JSON}"
for file in "${MAPPER_CONFIG}" "${INPUT_CONFIG}" "${LABELSPACE_CONFIG}" \
            "${PHYSICAL_CATALOG}"; do
  [[ -f "${file}" ]] || die "missing configuration: ${file}"
done

# Per-frame ACK is only sound when the receiver accepts every dataset frame.
# Parse the small input config without adding a Python YAML dependency, and
# reject a rate gate that would silently discard adjacent timestamps.
"${BASE1_PYTHON:-/usr/bin/python3}" - "${RUN_DIR}" "${INPUT_CONFIG}" <<'PY'
import csv
import pathlib
import re
import sys

run_dir, config_path = map(pathlib.Path, sys.argv[1:])
with (run_dir / "timestamps.csv").open(newline="") as stream:
    rows = list(csv.DictReader(stream))
stamps = [int(row.get("TimeStamp", row.get("timestamp", ""))) for row in rows]
spacings = [(current - previous) * 1.0e-9 for previous, current in zip(stamps, stamps[1:])]
config_text = config_path.read_text(encoding="utf-8")
match = re.search(
    r"(?m)^\s*input_separation_s\s*:\s*([0-9.eE+-]+)\s*(?:#.*)?$",
    config_text,
)
if match is None:
    raise SystemExit("SESSION_INPUT_ERROR input config lacks input_separation_s")
gate = float(match.group(1))
if spacings and gate >= min(spacings) - 1.0e-12:
    raise SystemExit(
        "SESSION_INPUT_ERROR receiver would drop dataset frames under per-frame ACK: "
        f"input_separation_s={gate} min_dataset_spacing_s={min(spacings)}"
    )
minimum = min(spacings) if spacings else None
print(f"SESSION_RATE_OK input_separation_s={gate} min_dataset_spacing_s={minimum}")
PY
[[ -z "${WORLD_TRANSFORM}" || -f "${WORLD_TRANSFORM}" ]] || \
  die "missing world transform: ${WORLD_TRANSFORM}"

if [[ -n "${INPUT_STATE}" ]]; then
  [[ -d "${INPUT_STATE}" ]] || \
    die "production --input-state must be an accepted state directory, not a bare map"
  INPUT_STATE_DIR="$(realpath "${INPUT_STATE}")"
  for state_file in final.4dmap transition_manifest.json state_summary.json; do
    [[ -s "${INPUT_STATE_DIR}/${state_file}" ]] || \
      die "input state is not an accepted transition: missing ${state_file}"
  done
  INPUT_STATE="${INPUT_STATE_DIR}/final.4dmap"
  "${BASE1_PYTHON:-/usr/bin/python3}" - \
    "${INPUT_STATE_DIR}/transition_manifest.json" "${INPUT_STATE}" <<'PY'
import hashlib
import json
import pathlib
import sys

manifest_path, map_path = map(pathlib.Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("schema") != "session_update_transition/v1":
    raise SystemExit("SESSION_INPUT_STATE_ERROR unsupported transition manifest")
digest = hashlib.sha256()
with map_path.open("rb") as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(block)
if digest.hexdigest() != manifest.get("output_state_sha256"):
    raise SystemExit("SESSION_INPUT_STATE_ERROR final.4dmap checksum differs from manifest")
PY
fi

RUN_DIR="$(realpath "${RUN_DIR}")"
SEMANTIC_DIR="$(realpath "${SEMANTIC_DIR}")"
INSTANCE_DIR="$(realpath "${INSTANCE_DIR}")"
MAPPER_CONFIG="$(realpath "${MAPPER_CONFIG}")"
INPUT_CONFIG="$(realpath "${INPUT_CONFIG}")"
LABELSPACE_CONFIG="$(realpath "${LABELSPACE_CONFIG}")"
PHYSICAL_CATALOG="$(realpath "${PHYSICAL_CATALOG}")"
[[ -z "${WORLD_TRANSFORM}" ]] || WORLD_TRANSFORM="$(realpath "${WORLD_TRANSFORM}")"
OUTPUT_STATE="$(realpath -m "${OUTPUT_STATE}")"
[[ ! -e "${OUTPUT_STATE}" ]] || die "output state already exists: ${OUTPUT_STATE}"
[[ ! -e "${OUTPUT_STATE}_control" ]] || die "output control already exists: ${OUTPUT_STATE}_control"
[[ -n "${SESSION_ID}" ]] || SESSION_ID="$(basename "${RUN_DIR}")"

# Resolve the complete observation interval deterministically before creating
# any output. For recurrent transitions the first new observation must be
# strictly newer than the latest serialized state; execution wall time never
# participates in this decision.
TIME_PREFLIGHT_JSON="$("${BASE1_PYTHON:-/usr/bin/python3}" - \
  "${ROOT}/scripts/nss_flat_ros2_player.py" "${RUN_DIR}" \
  "${FRAME_LIMIT}" "${SESSION_START_NS}" "${INPUT_STATE}" \
  "${BASE1_BUILD_DIR}/inspect_session_state" <<'PY'
import importlib.util
import json
import pathlib
import subprocess
import sys

module_path = pathlib.Path(sys.argv[1])
run_dir = pathlib.Path(sys.argv[2])
frame_limit = int(sys.argv[3])
explicit_text = sys.argv[4]
input_state = sys.argv[5]
inspector = sys.argv[6]
spec = importlib.util.spec_from_file_location("session_update_nss_player", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
frames = module.read_frames(run_dir)
if frame_limit > 0:
    frames = frames[:frame_limit]
explicit = int(explicit_text) if explicit_text else None
contract = module.resolve_session_time_contract(run_dir, frames, explicit)
if input_state:
    prior = json.loads(subprocess.check_output([inspector, input_state], text=True))
    prior_latest = int(prior["latest_stamp_ns"])
    try:
        module.validate_recurrent_session_start(contract, prior_latest)
    except ValueError as error:
        raise SystemExit(f"SESSION_TIME_ERROR {error}") from error
    contract["input_latest_stamp_ns"] = prior_latest
else:
    contract["input_latest_stamp_ns"] = None
print(json.dumps(contract, sort_keys=True))
PY
)" || die "timestamp preflight failed"
SESSION_START_NS="$("${BASE1_PYTHON:-/usr/bin/python3}" -c \
  'import json,sys; print(json.loads(sys.argv[1])["session_start_ns"])' \
  "${TIME_PREFLIGHT_JSON}")"
echo "SESSION_TIME_OK ${TIME_PREFLIGHT_JSON}"

# A production state becomes visible at OUTPUT_STATE only after every mapper,
# timestamp, identity, and recursive-state gate has passed.  All intermediate
# files live under a uniquely named sibling directory.  On failure we remove
# only that directory, which this process created itself; a partial map can
# therefore never be mistaken for an accepted P_N.
OUTPUT_PARENT="$(dirname "${OUTPUT_STATE}")"
OUTPUT_NAME="$(basename "${OUTPUT_STATE}")"
mkdir -p "${OUTPUT_PARENT}"
OUTPUT_LOCK="${OUTPUT_STATE}.lock"
mkdir "${OUTPUT_LOCK}" 2>/dev/null || \
  die "another transition owns output state: ${OUTPUT_STATE}"
STAGING_ROOT="$(mktemp -d "${OUTPUT_PARENT}/.${OUTPUT_NAME}.incomplete.XXXXXX")"
STAGING_STATE="${STAGING_ROOT}/state"
CONTROL_SOURCE="${STAGING_STATE}_control"
REJECTED_ROOT="${SESSION_UPDATE_REJECTED_ROOT:-${TMPDIR:-/tmp}/session_update_rejected}"
KEEP_REJECTED_MAP="${SESSION_UPDATE_KEEP_REJECTED_MAP:-0}"
transition_committed=false

preserve_rejected_diagnostics() {
  local status=$1
  local rejected_dir=""
  mkdir -p -- "${REJECTED_ROOT}" 2>/dev/null || return 0
  rejected_dir="$(mktemp -d "${REJECTED_ROOT}/${OUTPUT_NAME}.rejected.XXXXXX")" || return 0
  mkdir -p -- "${rejected_dir}/state" "${rejected_dir}/control"

  local filename
  for filename in state_summary.json transition_manifest.json \
                  object_changes.csv background_changes.csv \
                  config.txt experiment_log.txt; do
    if [[ -f "${STAGING_STATE}/${filename}" ]]; then
      cp -a -- "${STAGING_STATE}/${filename}" "${rejected_dir}/state/${filename}" || true
    fi
  done

  local control_source=""
  if [[ -d "${STAGING_STATE}/control" ]]; then
    control_source="${STAGING_STATE}/control"
  elif [[ -d "${CONTROL_SOURCE}" ]]; then
    control_source="${CONTROL_SOURCE}"
  fi
  if [[ -n "${control_source}" ]]; then
    if [[ -d "${control_source}/logs" ]]; then
      cp -a -- "${control_source}/logs" "${rejected_dir}/control/logs" || true
    fi
    for filename in command.txt playback_manifest.json input_state_summary.json \
                    transport_provenance.json processing_manifest.json \
                    player.exit_code khronos.exit_code finalization.timeout; do
      if [[ -f "${control_source}/${filename}" ]]; then
        cp -a -- "${control_source}/${filename}" \
          "${rejected_dir}/control/${filename}" || true
      fi
    done
  fi

  local map_retained=false
  if [[ "${KEEP_REJECTED_MAP}" == "1" && -s "${STAGING_STATE}/final.4dmap" ]]; then
    cp -a -- "${STAGING_STATE}/final.4dmap" "${rejected_dir}/state/final.4dmap" || true
    [[ -s "${rejected_dir}/state/final.4dmap" ]] && map_retained=true
  fi

  "${BASE1_PYTHON:-/usr/bin/python3}" - \
    "${rejected_dir}/rejection_manifest.json" "${status}" \
    "${OUTPUT_STATE}" "${map_retained}" <<'PY' || true
import datetime
import json
import pathlib
import sys

path, status, output_state, map_retained = sys.argv[1:]
root = pathlib.Path(path).parent
state_summary_path = root / "state" / "state_summary.json"
retained = map_retained == "true"
if state_summary_path.is_file():
    try:
        summary = json.loads(state_summary_path.read_text(encoding="utf-8"))
        summary["map"] = str(root / "state" / "final.4dmap") if retained else None
        summary["rejected_map_retained"] = retained
        state_summary_path.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, TypeError):
        pass
payload = {
    "schema": "session_update_rejection/v1",
    "exit_code": int(status),
    "formal_output_state": output_state,
    "formal_state_published": False,
    "final_4dmap_retained": retained,
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "state_files": sorted(p.name for p in (root / "state").iterdir()),
    "control_files": sorted(
        str(p.relative_to(root / "control"))
        for p in (root / "control").rglob("*")
        if p.is_file()
    ),
}
pathlib.Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
  echo "SESSION_TRANSITION_REJECTED diagnostics=${rejected_dir} exit_code=${status} map_retained=${map_retained}" >&2
}

cleanup_transition() {
  local status=$?
  if [[ "${transition_committed}" != true ]]; then
    preserve_rejected_diagnostics "${status}"
    rm -rf -- "${STAGING_ROOT}"
  fi
  rmdir "${OUTPUT_LOCK}" 2>/dev/null || true
  return "${status}"
}
trap cleanup_transition EXIT

ARGS=(
  --run-dir "${RUN_DIR}"
  --label-dir "${SEMANTIC_DIR}"
  --instance-dir "${INSTANCE_DIR}"
  --output-dir "${STAGING_STATE}"
  --mapper-config "${MAPPER_CONFIG}"
  --input-config "${INPUT_CONFIG}"
  --labelspace-config "${LABELSPACE_CONFIG}"
  --physical-catalog "${PHYSICAL_CATALOG}"
  --image-scale "${IMAGE_SCALE}"
  --play-rate "${PLAY_RATE}"
  --frame-limit "${FRAME_LIMIT}"
  --session-start-ns "${SESSION_START_NS}"
  --tf-settle-s "${TF_SETTLE_S}"
  --flow-control ack
  --ack-timeout-s "${ACK_TIMEOUT_S}"
  --finalization-timeout-s "${FINALIZATION_TIMEOUT_S}"
  --discovery-timeout-s "${DISCOVERY_TIMEOUT_S}"
  --change-detection-every-n-backend-updates 5
  --save-every-n-frames 0
  --store-visualization-details true
  --save-full-state false
  --sensor-max-range "${SENSOR_MAX_RANGE}"
)
[[ -z "${WORLD_TRANSFORM}" ]] || ARGS+=(--world-transform "${WORLD_TRANSFORM}")
if [[ -n "${INPUT_STATE}" ]]; then
  ARGS+=(--input-state "${INPUT_STATE}")
fi

SESSION_UPDATE_INTERNAL_STRICT=1 "${STRICT_RUNNER}" "${ARGS[@]}"
[[ -s "${STAGING_STATE}/final.4dmap" ]] || \
  die "canonical mapper returned without final.4dmap"
[[ -d "${CONTROL_SOURCE}" ]] || die "canonical mapper returned without control evidence"
[[ ! -e "${STAGING_STATE}/control" ]] || die "output state already contains control evidence"
mv "${CONTROL_SOURCE}" "${STAGING_STATE}/control"

STATE_SUMMARY="${STAGING_STATE}/state_summary.json"
"${BASE1_BUILD_DIR}/inspect_session_state" \
  "${STAGING_STATE}/final.4dmap" >"${STATE_SUMMARY}"
INPUT_STATE_SUMMARY=""
if [[ -n "${INPUT_STATE}" ]]; then
  INPUT_STATE_SUMMARY="${STAGING_STATE}/control/input_state_summary.json"
  "${BASE1_BUILD_DIR}/inspect_session_state" \
    "${INPUT_STATE}" >"${INPUT_STATE_SUMMARY}"
fi

"${BASE1_PYTHON:-/usr/bin/python3}" - \
  "${STAGING_STATE}/transition_manifest.json" \
  "${SESSION_ID}" "${INPUT_STATE}" "${STAGING_STATE}/final.4dmap" \
  "${OUTPUT_STATE}/final.4dmap" \
  "${RUN_DIR}" "${SEMANTIC_DIR}" "${INSTANCE_DIR}" "${WORLD_TRANSFORM}" \
  "${MAPPER_CONFIG}" "${INPUT_CONFIG}" "${LABELSPACE_CONFIG}" \
  "${PHYSICAL_CATALOG}" "${WORLD_TRANSFORM}" "${ROOT}/ports/mapping_core" \
  "${STAGING_STATE}/control/playback_manifest.json" "${STATE_SUMMARY}" \
  "${INPUT_STATE_SUMMARY}" \
  "${STAGING_STATE}/control/transport_provenance.json" <<'PY'
import datetime
import hashlib
import json
import pathlib
import sys

(manifest_path, session_id, input_state, output_map_source, output_map_record,
 run_dir, semantic_dir,
 instance_dir, world_transform, mapper_config, input_config,
 labelspace_config, physical_catalog, transform_file, mapping_source,
 playback_path, state_summary_path, input_summary_path,
 transport_provenance_path) = sys.argv[1:]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

with open(playback_path, encoding="utf-8") as stream:
    playback = json.load(stream)
with open(state_summary_path, encoding="utf-8") as stream:
    state_summary = json.load(stream)
with open(physical_catalog, encoding="utf-8") as stream:
    catalog = json.load(stream)
with open(transport_provenance_path, encoding="utf-8") as stream:
    transport_provenance = json.load(stream)

# The inspector necessarily sees the private staging path. Normalize the
# persisted summary (and the copy embedded below) to the path that becomes
# valid after the atomic commit; accepted state metadata must never point back
# into a removed .incomplete directory.
inspected_map = pathlib.Path(state_summary.get("map", ""))
if inspected_map.resolve() != pathlib.Path(output_map_source).resolve():
    raise SystemExit(
        "SESSION_OUTPUT_ERROR state summary does not describe staging final.4dmap"
    )
state_summary["map"] = output_map_record
with open(state_summary_path, "w", encoding="utf-8") as stream:
    json.dump(state_summary, stream, indent=2)
    stream.write("\n")

published_bounds = playback.get("published_bounds_ns")
if not published_bounds or published_bounds[1] is None:
    raise SystemExit("SESSION_OUTPUT_ERROR no frame was published")
if int(playback["frames_published"]) != int(playback["frames_encountered"]):
    raise SystemExit(
        "SESSION_OUTPUT_ERROR not every selected observation was published: "
        f"{playback['frames_published']} != {playback['frames_encountered']}"
    )
if int(playback.get("frames_skipped_empty_depth", 0)) != 0:
    raise SystemExit("SESSION_OUTPUT_ERROR empty-depth observations were skipped")
if int(state_summary["latest_stamp_ns"]) != int(published_bounds[1]):
    raise SystemExit(
        "SESSION_OUTPUT_ERROR final state timestamp does not equal the last ACKed frame: "
        f"state={state_summary['latest_stamp_ns']} ack={published_bounds[1]}"
    )
if not state_summary["strictly_increasing_stamps"]:
    raise SystemExit("SESSION_OUTPUT_ERROR state timestamps are not strictly increasing")
if int(state_summary["current"]["global_mesh_vertices"]) == 0:
    raise SystemExit("SESSION_OUTPUT_ERROR final current state has an empty global mesh")
duplicates = state_summary["current"]["duplicate_current_physical_ids"]
if duplicates:
    raise SystemExit(
        "SESSION_OUTPUT_ERROR current state contains duplicate physical objects: "
        + ",".join(map(str, duplicates))
    )

# Dataset-specific review gates. These are judgements about a finished map,
# not data contracts: the map is built and committed first, then reviewed
# (and possibly acted on). Resolution or instance-rate differences must never
# hard-reject a finished map, so every mismatch is recorded in the manifest's
# "review_gates" field and echoed to stderr instead of aborting the
# transition. Data-contract checks above (frame accounting, checksums,
# seed equivalence) still hard-fail.
review_gates = {}
session_key = next(
    (key for key in ("session_a", "session_b") if pathlib.Path(run_dir).name.startswith(key)),
    None,
)
is_complete_observation_run = int(playback["frames_encountered"]) == int(
    playback["frames_available"]
)
expectation = catalog.get("known_complete_sessions", {}).get(session_key or "", {})
if is_complete_observation_run and expectation:
    violations = []
    expected_ids = sorted(map(int, expectation["active_physical_instance_ids"]))
    actual_ids = sorted(map(int, state_summary["current"]["current_physical_ids"]))
    if actual_ids != expected_ids:
        violations.append(
            f"{session_key} current physical IDs differ: {actual_ids} != {expected_ids}"
        )
    if int(state_summary["time_steps"]) < int(expectation["minimum_time_steps"]):
        violations.append(
            f"{session_key} timeline is too short: "
            f"{state_summary['time_steps']} < {expectation['minimum_time_steps']}"
        )
    trajectories = int(state_summary["current"]["current_trajectory_objects"])
    if trajectories < int(expectation["minimum_trajectory_objects"]):
        violations.append(
            f"{session_key} D1 trajectory gate failed: "
            f"{trajectories} < {expectation['minimum_trajectory_objects']}"
        )
    entity_semantics = {
        int(item["physical_instance_id"]): int(item["semantic_id"])
        for item in catalog["entities"]
    }
    actual_semantics = state_summary["current"]["current_physical_id_semantic_labels"]
    actual_meshes = state_summary["current"]["current_physical_id_private_mesh_vertices"]
    for physical_id in expected_ids:
        labels = list(map(int, actual_semantics.get(str(physical_id), [])))
        wanted = entity_semantics[physical_id]
        if labels != [wanted]:
            violations.append(
                f"I{physical_id} semantic labels {labels} != [{wanted}]"
            )
        if expectation.get("require_private_mesh_for_every_physical_instance") and int(
            actual_meshes.get(str(physical_id), 0)
        ) == 0:
            violations.append(f"I{physical_id} has no current private mesh")
    if violations:
        review_gates["dataset_contract"] = violations
        print(
            "SESSION_REVIEW_GATES " + "; ".join(violations),
            file=sys.stderr,
        )
if input_state and int(state_summary["time_steps"]) < 2:
    raise SystemExit("SESSION_OUTPUT_ERROR recurrent state lacks its initial seed")
if input_state:
    with open(input_summary_path, encoding="utf-8") as stream:
        input_summary = json.load(stream)
    if int(state_summary["first_stamp_ns"]) != int(input_summary["latest_stamp_ns"]):
        raise SystemExit("SESSION_OUTPUT_ERROR output did not begin at input latest stamp")
    seed_initial = state_summary["initial"]
    seed_input = input_summary["current"]
    canonical_fields = (
        "canonical_current_scene_schema",
        "canonical_current_scene_bytes",
        "canonical_current_scene_objects",
        "canonical_current_scene_fingerprint_fnv1a64",
    )
    missing = [
        field
        for field in canonical_fields
        if field not in seed_initial or field not in seed_input
    ]
    if missing:
        raise SystemExit(
            "SESSION_OUTPUT_ERROR inspector lacks canonical seed fields: "
            + ",".join(missing)
        )
    if any(seed_initial[field] != seed_input[field] for field in canonical_fields):
        output_fingerprint = {
            field: seed_initial[field] for field in canonical_fields
        }
        input_fingerprint = {
            field: seed_input[field] for field in canonical_fields
        }
        raise SystemExit(
            "SESSION_OUTPUT_ERROR output seed canonical current scene differs from input: "
            f"output={output_fingerprint} input={input_fingerprint}"
        )
    structural_fields = (
        "global_mesh_vertices",
        "global_mesh_faces",
        "current_object_nodes",
        "current_private_mesh_vertices",
        "current_private_mesh_faces",
        "current_physical_ids",
        "current_physical_id_node_counts",
        "current_physical_id_semantic_labels",
    )
    if any(seed_initial[field] != seed_input[field] for field in structural_fields):
        raise SystemExit(
            "SESSION_OUTPUT_ERROR output seed structure differs from input current state"
        )
    if int(state_summary["latest_stamp_ns"]) <= int(input_summary["latest_stamp_ns"]):
        raise SystemExit("SESSION_OUTPUT_ERROR current session did not advance state time")
else:
    input_summary = None

payload = {
    "schema": "session_update_transition/v1",
    "session_id": session_id,
    "transition": "bootstrap" if not input_state else "recurrent",
    "input_state": input_state or None,
    "input_state_sha256": sha256(input_state) if input_state else None,
    "input_state_summary": input_summary,
    "output_state": output_map_record,
    "output_state_bytes": pathlib.Path(output_map_source).stat().st_size,
    "output_state_sha256": sha256(output_map_source),
    "state_summary": state_summary,
    "last_acked_frame_stamp_ns": published_bounds[1],
    "timestamp_provenance": playback.get("timestamp_provenance"),
    "transport_provenance": transport_provenance,
    "observations": {
        "rgbd": run_dir,
        "semantic": semantic_dir,
        "instance": instance_dir,
        "world_transform": world_transform or None,
    },
    "mapper_config": mapper_config,
    "mapper_config_sha256": sha256(mapper_config),
    "input_config": input_config,
    "input_config_sha256": sha256(input_config),
    "labelspace_config": labelspace_config,
    "labelspace_config_sha256": sha256(labelspace_config),
    "physical_catalog": physical_catalog,
    "physical_catalog_sha256": sha256(physical_catalog),
    "world_transform_sha256": sha256(transform_file) if transform_file else None,
    "input_preflight": playback.get("input_preflight"),
    "mapping_source": str(pathlib.Path(mapping_source).resolve()),
    "review_gates": review_gates,
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
with open(manifest_path, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2)
    stream.write("\n")
PY

mv "${STAGING_STATE}" "${OUTPUT_STATE}"
rmdir "${STAGING_ROOT}"
transition_committed=true
echo "SESSION_TRANSITION_COMPLETE state=${OUTPUT_STATE} map=${OUTPUT_STATE}/final.4dmap"
