#!/usr/bin/env bash
# Synthetic A/B metrics for a map given as EvaluationScene timelines (any representation).
#
#   eval_synthetic_scene.sh RUN_DIR OUT_DIR [BG_SPACING OBJ_SPACING]
#
# RUN_DIR/session_{a,b}/timeline.pkl (any timeline with stamps()/scene(t)) are written as the
# native export (update_layer/eval/emit_synthetic.py, surfels), then the validated harness scripts run
# unchanged: Mesh F1 (evaluate_cross_session_geometry.py) and D2/D3 State
# (score_complete_states.py -> group_state_and_report.py), same as eval/synthetic_base_v38.
# Mesh F1 scores the map's surface. A Gaussian map's (session_<s>/run.json backend "game") is its median-depth TSDF mesh
# (eval/tsdf_export.py, the surface the real G1 scores; made here from checkpoint_<s>.pt when missing), written as a
# one-snapshot native export of its own; other representations are scored on the surfel export. D2/D3 State read the
# surfel export of every representation (each object's geometry per snapshot).
set -Eeuo pipefail
RUN=$(realpath "$1"); OUT=$(realpath -m "$2"); BG=${3:-0.02}; OBJ=${4:-0.01}
PY=${PY:-/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python}
GPY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python          # the Gaussian map's TSDF export (GPU)
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
backend() { "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("backend", ""))' "$RUN/session_$1/run.json" 2>/dev/null || true; }
surface() {   # session -> the median-depth TSDF mesh of the Gaussian map's checkpoint (exported from it whenever it is here)
  local m=$RUN/session_$1/tsdf_final.ply ck=$RUN/checkpoint_$1.pt
  if [[ -e $ck ]]; then
    ( cd "$PROJECT" && nice -n 10 "$GPY" -W ignore -m update_layer.eval.tsdf_export "$ck" "$m.tmp.ply" synthetic ) \
        >> "$RUN/tsdf_export.log" 2>&1 \
        || { echo "session $1: TSDF export failed ($RUN/tsdf_export.log)" >&2; return 1; }
    mv -f "$m.tmp.ply" "$m"                 # replaces a symlinked mesh instead of writing through it
  elif [[ ! -s $m ]]; then
    echo "session $1: no checkpoint_$1.pt and no tsdf_final.ply" >&2; return 1
  fi
  echo "$m"
}
# The evaluation units are the mapping's physical identities: the dataset's 29-unit scope, the one the label images
# number (user 10-21: input and evaluation must match; the large sofa and its throw pillows are two units, B parts them).
UNITS=${UNITS:-/home/jixian/Desktop/FT/datasets/synthetic_ab/results/cvpr_surface_final_20260916/paper_aligned/EVAL_SCOPE_UNITS.csv}
run() { local root=${ROOT:-$OUT}; env PYTHONPATH="$SYN" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
        HARNESS_IN_ROOT="$root" HARNESS_OUT_ROOT="$root" HARNESS_CAUSAL_BANK="$ORIG/causal_reference" \
        HARNESS_SYN_SCRIPTS="$SYN" HARNESS_SKIP_RUN_CHECKS=1 HARNESS_SCOPE_UNITS="$UNITS" nice -n 10 "$PY" "$@"; }
if [[ $(backend a) == game || $(backend b) == game ]]; then
  [[ $(backend a) == game && $(backend b) == game ]] || { echo "sessions a / b are different representations" >&2; exit 1; }
  G=$OUT/surface; rm -rf "$G"
  for s in a b; do
    m=$(surface $s) || exit 1
    "$PY" - "$m" "$G/ours/native/$s" "$s" "$PROJECT" <<'EOF'
import sys
import numpy as np
import open3d as o3d
sys.path.insert(0, sys.argv[4])
from update_layer.eval.emit_synthetic import write_mesh_native
mesh = o3d.io.read_triangle_mesh(sys.argv[1])
origin = {"a": 1_000_000_000, "b": 141_000_000_000}            # cvpr_eval_common.ORIGIN: each session's first stamp
write_mesh_native(np.asarray(mesh.vertices), np.asarray(mesh.triangles), sys.argv[2], origin[sys.argv[3]])
EOF
  done
  echo "[$(date +%T)] geometry (median-depth TSDF surface)"
  ROOT=$G run "$H/evaluate_cross_session_geometry.py" --methods ours > "$OUT/eval_logs/geometry.log" 2>&1 || { tail -20 "$OUT/eval_logs/geometry.log"; exit 1; }
  rm -rf "$OUT/geometry"; mv "$G/geometry" "$OUT/geometry"; rm -rf "$G"
else
  echo "[$(date +%T)] geometry (surfel export)"
  run "$H/evaluate_cross_session_geometry.py" --methods ours > "$OUT/eval_logs/geometry.log" 2>&1 || { tail -20 "$OUT/eval_logs/geometry.log"; exit 1; }
fi
echo "[$(date +%T)] state";    run "$H/score_complete_states.py" --ours-only > "$OUT/eval_logs/state.log" 2>&1 || { tail -20 "$OUT/eval_logs/state.log"; exit 1; }
# the native export is only an input of the two scorers above (tens of GB per chain); the scores
# stay. KEEP_INTERMEDIATES=1 keeps it for debugging; it can be rebuilt from the timelines.
[[ ${KEEP_INTERMEDIATES:-0} == 1 ]] || rm -rf "$OUT/ours/native"
echo "[$(date +%T)] done"; ls "$OUT"
