#!/usr/bin/env bash
# Synthetic D1 metric of the TSDF version (live dynamics: time P/R/F1 of visible motion, trajectory coverage, centroid
# RMSE, identity switches; VISIBLE_MOTION_TRACKING) on a layer run with the D1 front end:
#   RUN_DIR/session_{a,b}/live_tracks.jsonl -> OUT/live/ours_live/{a,b}/live_tracks.jsonl (links)
#   -> the unchanged Synthetic_ABC/scripts/evaluate_live_dynamics.py through eval_tools_full_20260922/run_original.py
#      (ours only, outputs only below OUT) -> OUT/dynamics/{LIVE_DYNAMICS_SUMMARY,VISIBLE_MOTION_TRACKING_SUMMARY}.csv
#   d1_synthetic.sh RUN_DIR OUT_DIR
set -Eeuo pipefail
RUN=$(realpath "$1"); OUT=$(realpath -m "$2")
FT=/home/jixian/Desktop/FT
PY=/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python
SYN=/mnt/d/3Study/ETH/FT/Synthetic_ABC/scripts
TOOLS=$FT/datasets/synthetic_ab/eval_tools_full_20260922
BANK=$FT/datasets/synthetic_ab/evaluation_complete_20260917/causal_reference
for s in a b; do
  [ -s "$RUN/session_$s/live_tracks.jsonl" ] || { echo "missing $RUN/session_$s/live_tracks.jsonl"; exit 2; }
  mkdir -p "$OUT/live/ours_live/$s"; ln -sf "$RUN/session_$s/live_tracks.jsonl" "$OUT/live/ours_live/$s/live_tracks.jsonl"
done
cd "$SYN"
env PYTHONPATH="$SYN" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 FULL_SYN_SCRIPTS="$SYN" FULL_NATIVE_ROOT="$OUT/live" \
    FULL_LIVE_ROOT="$OUT/live" FULL_CAUSAL_BANK="$BANK" FULL_OUT_ROOT="$OUT" nice -n 10 "$PY" "$TOOLS/run_original.py" dynamics
ls "$OUT/dynamics"
