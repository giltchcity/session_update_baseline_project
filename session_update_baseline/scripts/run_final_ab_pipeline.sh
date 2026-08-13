#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jixian/Desktop/FT/session_update_baseline
# The Khronos runner sources this in a child shell. Source it here as well so
# the baseline executables launched by this parent inherit the ROS libraries.
source "${ROOT}/scripts/khronos_env.sh"
OUT_ROOT=${OUT_ROOT:-/home/jixian/Desktop/FT/runs/azure_room_multisession_final}
KHRONOS_RUNNER=${ROOT}/scripts/run_khronos_session_strict.sh
BASELINE=${BASE1_BUILD_DIR}/run_session_update_baseline

A_RUN=/mnt/d/3Study/ETH/FT/datasets/azure_kinect/session_a_20260809_204010_201_flat_rgbd_30hz_1080p
A_LABELS=/mnt/d/3Study/ETH/FT/runs/final_outputs_20260812/semantics/session_a/semantic_masks_ade_ids
B_RUN=/mnt/d/3Study/ETH/FT/datasets/azure_kinect/session_b_20260810_030502_620_flat_rgbd_30hz_1080p
B_LABELS=/mnt/d/3Study/ETH/FT/runs/final_outputs_20260812/semantics/session_b/semantic_masks_ade_ids
B_TO_A=/mnt/d/3Study/ETH/FT/runs/final_outputs_20260812/alignment/session_b_to_session_a.txt
MAPPER=${ROOT}/configs/room18_official_2cm.yaml
INPUT=/mnt/d/3Study/ETH/FT/configs/khronos/room18_final_30fps_input.yaml
LABELSPACE=${ROOT}/configs/nss_ade20k_room_label_space.yaml

A_KHRONOS=${OUT_ROOT}/session_a/khronos
A_OURS=${OUT_ROOT}/session_a/memory_noop_v2
B_EVIDENCE=${OUT_ROOT}/session_b/khronos_evidence
B_FROM_SCRATCH=${OUT_ROOT}/session_b/khronos_from_scratch
B_OURS=${OUT_ROOT}/session_b/conservative_candidate_v3

run_khronos() {
  local run_dir=$1
  local labels=$2
  local output=$3
  local transform=${4:-}
  local prior_map=${5:-}
  local prior_seed_map=${6:-}
  local control="${output}_control"
  if [[ -s ${output}/final.4dmap && \
        -s ${control}/playback_manifest.json && \
        -s ${control}/processing_manifest.json && \
        -s ${output}/experiment_log.txt ]] && \
      grep -q "Experiment Finished Cleanly" "${output}/experiment_log.txt" && \
      { [[ -z ${prior_map} ]] || grep -Fq -- "prior_map=${prior_map}" "${control}/command.txt"; }; then
    echo "SKIP_COMPLETE ${output}/final.4dmap"
    return
  fi

  local args=(
    --run-dir "${run_dir}"
    --label-dir "${labels}"
    --output-dir "${output}"
    --tf-settle-s 0.02
    --play-rate 0.1
    --flow-control ack
    --ack-timeout-s 180
    --frame-limit 0
    --change-detection-every-n-frames 50
    --save-every-n-frames 0
    --store-visualization-details true
    --save-full-state false
    --sensor-max-range 5.0
    --mapper-config "${MAPPER}"
    --input-config "${INPUT}"
    --labelspace-config "${LABELSPACE}"
  )
  if [[ -n ${transform} ]]; then
    args+=(--world-transform "${transform}")
  fi
  if [[ -n ${prior_map} ]]; then
    args+=(
      --prior-map "${prior_map}"
      --prior-seed-map "${prior_seed_map}"
      --discovery-timeout-s 180
    )
  fi
  "${KHRONOS_RUNNER}" "${args[@]}"
}

baseline_common=(
  --map_time latest
  --object_move_decision hard
  --object_distance_m 0.05
  --bbox_margin_m 0.05
  --min_object_mesh_vertices 20
  --require_same_label false
  --require_bbox_containment true
)

run_baseline_a() {
  if [[ -s ${A_OURS}/improved_final.4dmap && -s ${A_OURS}/object_memory.json ]]; then
    echo "SKIP_COMPLETE ${A_OURS}/improved_final.4dmap"
    return
  fi
  local change_args=()
  if [[ -s ${A_KHRONOS}/object_changes.csv ]]; then
    change_args=(--object_changes_csv "${A_KHRONOS}/object_changes.csv")
  fi
  "${BASELINE}" \
    --map_file "${A_KHRONOS}/final.4dmap" \
    --output_dir "${A_OURS}" \
    --dynamic_mode within_session \
    --mode no_op \
    --save_original_copy false \
    "${change_args[@]}" \
    "${baseline_common[@]}"
}

run_baseline_b() {
  if [[ -s ${B_OURS}/improved_final.4dmap && \
        -s ${B_OURS}/config.yaml ]] && \
      grep -q '^dynamic_mode: "cross_session"$' "${B_OURS}/config.yaml" && \
      grep -q "^prior_map: \"${A_KHRONOS}/final.4dmap\"$" \
        "${B_OURS}/config.yaml"; then
    echo "SKIP_COMPLETE ${B_OURS}/improved_final.4dmap"
    return
  fi
  local change_args=()
  if [[ -s ${B_EVIDENCE}/object_changes.csv ]]; then
    change_args=(--object_changes_csv "${B_EVIDENCE}/object_changes.csv")
  fi
  "${BASELINE}" \
    --map_file "${B_EVIDENCE}/final.4dmap" \
    --output_dir "${B_OURS}" \
    --dynamic_mode cross_session \
    --prior_map "${A_KHRONOS}/final.4dmap" \
    --prior_object_memory "${A_OURS}/object_memory.json" \
    --mode cleanup \
    --save_original_copy false \
    --object_cleanup_reobservation_gate true \
    --cross_session_remove_absent_prior true \
    "${change_args[@]}" \
    "${baseline_common[@]}"
}

mkdir -p "${OUT_ROOT}"
run_khronos "${A_RUN}" "${A_LABELS}" "${A_KHRONOS}"
run_baseline_a
# Untouched clean-backend control for Session B. This is required to separate
# ordinary Khronos mapping from the cross-session extension.
run_khronos "${B_RUN}" "${B_LABELS}" "${B_FROM_SCRATCH}" "${B_TO_A}"
run_khronos "${B_RUN}" "${B_LABELS}" "${B_EVIDENCE}" "${B_TO_A}" \
  "${A_KHRONOS}/final.4dmap" "${A_KHRONOS}/final.4dmap"
run_baseline_b

# The cross-session Khronos map is the accepted output until a candidate passes
# a registered-GT accuracy gate. Never silently replace the control with an
# unevaluated post-processing result.
ln -sfn "${B_EVIDENCE}/final.4dmap" "${OUT_ROOT}/accepted_final.4dmap"
ln -sfn "${B_OURS}/improved_final.4dmap" "${OUT_ROOT}/candidate_final.4dmap"
ln -sfn "${B_EVIDENCE}/final.4dmap" "${OUT_ROOT}/final_ab.4dmap"

VIS_DIR=${OUT_ROOT}/process_visualization_ab_rgb
"${BASE1_PYTHON}" "${ROOT}/scripts/prepare_flat_ab_process_visualization.py" \
  --a-map "${A_KHRONOS}/final.4dmap" \
  --b-map "${B_EVIDENCE}/final.4dmap" \
  --a-dsg "${A_KHRONOS}/backend/dsg_with_mesh.sparkdsg" \
  --b-dsg "${B_EVIDENCE}/backend/dsg_with_mesh.sparkdsg" \
  --improved-map "${B_OURS}/improved_final.4dmap" \
  --evidence-summary "${B_OURS}/evidence_summary.json" \
  --a-flat "${A_RUN}" \
  --b-flat "${B_RUN}" \
  --a-playback "${A_KHRONOS}_control/playback_manifest.json" \
  --b-playback "${B_EVIDENCE}_control/playback_manifest.json" \
  --b-transform "${B_TO_A}" \
  --output-dir "${VIS_DIR}"

ln -sfn "${OUT_ROOT}" /home/jixian/Desktop/FT/runs/azure_room_multisession_current
cat >"${OUT_ROOT}/RUN_COMPLETE.json" <<EOF
{
  "status": "complete",
  "official_khronos_commit": "63faadde6ed92220e78fb2f6ca86dcc54bb5cf9e",
  "session_a_map": "${A_KHRONOS}/final.4dmap",
  "session_b_from_scratch_map": "${B_FROM_SCRATCH}/final.4dmap",
  "session_b_map": "${B_EVIDENCE}/final.4dmap",
  "accepted_map": "${B_EVIDENCE}/final.4dmap",
  "candidate_map": "${B_OURS}/improved_final.4dmap",
  "accuracy_certified": false,
  "accuracy_status": "registered GT is unavailable; candidate must not replace accepted map",
  "visualization_manifest": "${VIS_DIR}/sequence_manifest.csv"
}
EOF
echo "AB_PIPELINE_COMPLETE ${OUT_ROOT}/accepted_final.4dmap"
