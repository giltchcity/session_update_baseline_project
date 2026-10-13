#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jixian/Desktop/FT"
PAIR="${1:-12}"
AUTO_DIR="$ROOT/runs/nss/bldg1_auto_alignment_20260723"
OUTPUT_DIR="$ROOT/runs/nss/bldg1_manual_pair_alignment_20260723"

case "$PAIR" in
  12) AUTO_INITIAL="$AUTO_DIR/stage2_to_stage1.txt"; MANUAL_INITIAL="$OUTPUT_DIR/stage2_to_stage1.txt" ;;
  13) AUTO_INITIAL="$AUTO_DIR/stage3_to_stage1.txt"; MANUAL_INITIAL="$OUTPUT_DIR/stage3_to_stage1.txt" ;;
  23) AUTO_INITIAL="$AUTO_DIR/stage3_to_stage2.txt"; MANUAL_INITIAL="$OUTPUT_DIR/stage3_to_stage2.txt" ;;
  *) echo "Usage: $0 {12|13|23}" >&2; exit 2 ;;
esac

if [[ -f "$MANUAL_INITIAL" ]]; then
  INITIAL="$MANUAL_INITIAL"
  echo "LOADING_MANUAL_TRANSFORM=$INITIAL"
else
  INITIAL="$AUTO_INITIAL"
  echo "LOADING_AUTO_TRANSFORM=$INITIAL"
fi

cd "$ROOT"
exec /home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python \
  scripts/nss/manual_align_stage_meshes.py \
  --raw-root datasets/nss/raw_data \
  --building 1 \
  --pair "$PAIR" \
  --output-dir "$OUTPUT_DIR" \
  --initial-transform "$INITIAL" \
  --rotation-step-deg 0.5
