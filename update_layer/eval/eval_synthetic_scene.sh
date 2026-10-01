#!/usr/bin/env bash
# Synthetic A/B metrics for a map given as EvaluationScene timelines (any representation).
#
#   eval_synthetic_scene.sh RUN_DIR OUT_DIR [BG_SPACING OBJ_SPACING]
#
# RUN_DIR/session_{a,b}/timeline.pkl (any timeline with stamps()/scene(t)) are written as the
# native export (update_layer/eval/emit_synthetic.py, surfels), then the validated harness scripts run
# unchanged on it: Mesh F1 (evaluate_cross_session_geometry.py) and D2/D3 State
# (score_complete_states.py -> group_state_and_report.py), same as eval/synthetic_base_v38.
set -Eeuo pipefail
RUN=$(realpath "$1"); OUT=$(realpath -m "$2"); BG=${3:-0.02}; OBJ=${4:-0.01}
PY=${PY:-/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python}
PROJECT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)     # session_update_baseline_project
SYN=/mnt/d/3Study/ETH/FT/Synthetic_ABC/scripts
ORIG=/home/jixian/Desktop/FT/datasets/synthetic_ab/evaluation_complete_20260917
H=/home/jixian/Desktop/FT/eval/synthetic_base_v38/harness
mkdir -p "$OUT/eval_logs"
for s in a b; do
  if [[ ! -s $OUT/ours/native/$s/meshes.jsonl ]]; then
    echo "[$(date +%T)] emit native $s"
    OMP_NUM_THREADS=4 "$PY" - "$RUN/session_$s/timeline.pkl" "$OUT/ours/native/$s" "$BG" "$OBJ" "$PROJECT" <<'EOF'
import sys
sys.path.insert(0, sys.argv[5])
from update_layer.eval.scenelist import load_timeline
from update_layer.eval.emit_synthetic import write_native
tl = load_timeline(sys.argv[1])
write_native(tl, sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
EOF
  fi
done
run() { env PYTHONPATH="$SYN" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
        HARNESS_IN_ROOT="$OUT" HARNESS_OUT_ROOT="$OUT" HARNESS_CAUSAL_BANK="$ORIG/causal_reference" \
        HARNESS_SYN_SCRIPTS="$SYN" HARNESS_SKIP_RUN_CHECKS=1 nice -n 10 "$PY" "$@"; }
echo "[$(date +%T)] geometry"; run "$H/evaluate_cross_session_geometry.py" --methods ours > "$OUT/eval_logs/geometry.log" 2>&1 || { tail -20 "$OUT/eval_logs/geometry.log"; exit 1; }
echo "[$(date +%T)] state";    run "$H/score_complete_states.py" --ours-only > "$OUT/eval_logs/state.log" 2>&1 || { tail -20 "$OUT/eval_logs/state.log"; exit 1; }
# the native export is only an input of the two scorers above (tens of GB per chain); the scores
# stay. KEEP_INTERMEDIATES=1 keeps it for debugging; it can be rebuilt from the timelines.
[[ ${KEEP_INTERMEDIATES:-0} == 1 ]] || rm -rf "$OUT/ours/native"
echo "[$(date +%T)] done"; ls "$OUT"
