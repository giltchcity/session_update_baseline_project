#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jixian/Desktop/FT"
PYTHON="/usr/bin/python3"
EDITOR="${ROOT}/session_update_baseline/scripts/semantic_mask_editor.py"
export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"

case "${1:-B}" in
  A|a)
    RECORDING="recording_20260809_204010_201"
    TITLE="Session A - 20260809_204010"
    ;;
  B|b)
    RECORDING="recording_20260810_030502_620"
    TITLE="Session B - 20260810_030502"
    ;;
  *)
    echo "Usage: $0 A|B" >&2
    exit 2
    ;;
esac

BASE="${ROOT}/datasets/azure_kinect/${RECORDING}"
exec "${PYTHON}" "${EDITOR}" \
  --input-dir "${BASE}/semantic_keyframes_1hz_ade20k_auto" \
  --output-dir "${BASE}/semantic_keyframes_1hz_manual_masks" \
  --labelspace "${ROOT}/session_update_baseline/configs/nss_ade20k_room_label_space.yaml" \
  --title "${TITLE}"
