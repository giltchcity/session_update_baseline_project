#!/usr/bin/env bash
# Real room-18 A/B/C metrics for a map given as EvaluationScene timelines (any representation).
#
#   eval_real_scene.sh RUN_DIR OUT_DIR [BG_SPACING OBJ_SPACING]
#
# RUN_DIR/session_{a,b,c}/timeline.pkl are written as the harness intermediates
# (update_layer/eval/emit_gpu.py = GPU emit_harness.py: snapshots.jsonl, cleanup_probes.bin, cleanup_mesh.jsonl) and a
# surfel PLY of the final scene (update_layer/eval/emit_synthetic.surfel_triangles); then the validated
# harness Python runs unchanged: eval_state_change_h.py (D2, D3), cleanup_eval_h.py,
# cleanup_evaluate_h.py + ghost_cohort.py (ghost %), geometry_g1.py (G1) and score_geometry.py (G2).
# Not covered: official Object F1 (reads a Khronos .4dmap through exp_pipeline).
# G1 / G2 score the map's surface: a Gaussian map's (session_<s>/run.json backend "game") is its median-depth TSDF mesh
# (eval/tsdf_export.py, made here from checkpoint_<s>.pt when missing); other representations, the surfel PLY.
set -Eeuo pipefail
RUN=$(realpath "$1"); OUT=$(realpath -m "$2"); BG=${3:-0.02}; OBJ=${4:-0.01}
FT=/home/jixian/Desktop/FT
PY=${PY:-/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python}
GPY=$FT/envs/gs-cu128/bin/python          # GPU emitter (emit_gpu.py; byte-identical to the CPU writers)
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
H=$FT/results/current_version_20260921/tools/harness_real
CV=$FT/results/current_version_20260921
RESIDUAL_PROBES=$FT/results/abc_eval_final_baseline/temporal/RESIDUAL_PROBES.json
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PROJECT=$(cd "$HERE/../.." && pwd)
backend() { "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("backend", ""))' "$RUN/session_$1/run.json" 2>/dev/null || true; }
surface() {   # session -> the median-depth TSDF mesh of the Gaussian map's checkpoint (exported from it whenever it is here)
  local m=$RUN/session_$1/tsdf_final.ply ck=$RUN/checkpoint_$1.pt
  if [[ -e $ck ]]; then
    ( cd "$PROJECT" && GAME_CONFIG=${GAME_CONFIG:-$FT/baselines/GaME/configs/flat/flat.yaml} nice -n 10 "$GPY" -W ignore \
        -m update_layer.eval.tsdf_export "$ck" "$m.tmp.ply" real ) >> "$RUN/tsdf_export.log" 2>&1 \
        || { echo "session $1: TSDF export failed ($RUN/tsdf_export.log)" >&2; return 1; }
    mv -f "$m.tmp.ply" "$m"                 # replaces a symlinked mesh instead of writing through it
  elif [[ ! -s $m ]]; then
    echo "session $1: no checkpoint_$1.pt and no tsdf_final.ply" >&2; return 1
  fi
  echo "$m"
}
geometry() {  # session surface -> G1 / G2
  local SU=${1^^} GEO=$OUT/geometry/$1
  echo "[$(date +%T)] $SU geometry ($2)"
  "$PY" "$FT/eval/geometry_g1.py" score real "$SU" "$FT/results/abc_eval_v2/geometry/${SU}_reference_1cm.ply" \
      "$2" "$FT/eval/crops/real_$SU.json" "$GEO/G1.json" > "$GEO/G1.log" 2>&1 || tail -5 "$GEO/G1.log"
  "$PY" "$CV/tools/score_geometry.py" "$SU" layer_scene "$2" --json "$GEO/G2.json" > "$GEO/G2.log" 2>&1 || tail -5 "$GEO/G2.log"
}
for s in ${SESSIONS:-a b c}; do
  SU=${s^^}; EXP=$OUT/exports/$s; CH=$OUT/changes/$s; GH=$OUT/ghost/$s; GEO=$OUT/geometry/$s
  mkdir -p "$EXP" "$CH" "$GH" "$GEO"
  GAME=$( [[ $(backend $s) == game ]] && echo 1 || true )
  if [[ -n $GAME ]]; then SURF=$(surface $s) || exit 1; geometry $s "$SURF"; fi
  [[ -s $RUN/session_$s/timeline.pkl ]] || { echo "no timeline for $s"; continue; }
  if [[ ! -s $EXP/COMPLETE.json ]]; then
    echo "[$(date +%T)] $SU emit"
    CUDA_VISIBLE_DEVICES=0 nice -n 10 "$GPY" "$HERE/emit_gpu.py" "$RUN/session_$s/timeline.pkl" "$EXP" "$s" "$RESIDUAL_PROBES" \
        "$FT/results/abc_eval_v2/cleanup/probes_$s.json" "$BG" "$OBJ"
  fi
  [[ -n $GAME ]] || geometry $s "$EXP/surfels_final.ply"
  if [[ $s != a ]]; then
    echo "[$(date +%T)] $SU cleanup + D2/D3"
    "$PY" "$H/cleanup_eval_h.py" --session "$SU" --snapshots "$EXP/snapshots.jsonl" --bin "$EXP/cleanup_probes.bin" --out "$CH" > "$CH/cleanup.log" 2>&1 || tail -5 "$CH/cleanup.log"
    "$PY" "$H/eval_state_change_h.py" --session "$SU" --snapshots "$EXP/snapshots.jsonl" --cleanup "$CH/CLEANUP_PROBES_$SU.csv" --out "$CH" > "$CH/state.log" 2>&1 || tail -5 "$CH/state.log"
    echo "[$(date +%T)] $SU ghost"
    n=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["snapshots"])' "$EXP/COMPLETE.json")
    "$PY" "$H/cleanup_evaluate_h.py" --session "$SU" --mesh-jsonl "$EXP/cleanup_mesh.jsonl" --expected-snapshots "$n" --out "$GH" > "$GH/ghost.log" 2>&1 || tail -5 "$GH/ghost.log"
    "$PY" "$H/ghost_cohort.py" --summary "$GH/event_summary_${s}.csv" --out "$GH/GHOST.json" > "$GH/cohort.log" 2>&1 || tail -5 "$GH/cohort.log"
  else
    "$PY" "$H/eval_state_change_h.py" --session "$SU" --snapshots "$EXP/snapshots.jsonl" --out "$CH" > "$CH/state.log" 2>&1 || tail -5 "$CH/state.log"
  fi
done
# the harness intermediates (exports/) are only inputs of the scorers above; the scores stay.
# KEEP_INTERMEDIATES=1 keeps them for debugging; they can be rebuilt from the timelines.
# the final scene of every session (surfel PLY) is kept next to the scores
for f in "$OUT"/exports/*/surfels_final.ply; do [[ -s $f ]] && cp "$f" "$OUT/final_$(basename "$(dirname "$f")").ply"; done
[[ ${KEEP_INTERMEDIATES:-0} == 1 ]] || rm -rf "$OUT/exports"
echo "[$(date +%T)] done"
