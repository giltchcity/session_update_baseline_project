#!/bin/bash
# Temporal mesh F1 of the occupancy rows on Home (evaluate_map_vs_gt.py, scheme C unless GEO4D_RECALL=complete):
#   row3  official wavemap carrying its own map (wm5_row3_22e68d6, timelines kept in KEEP_baseline_runs_wavemap_rows13)
#   row4  ours (wm_synth_row4_7b63cc1)
# Each session's timeline.pkl is written as the native export by update_layer.eval.emit_synthetic.write_native at the
# acceptance harness spacing (0.02 m background, 0.01 m objects), as eval_synthetic_scene.sh. 4 cores, nice 19.
# usage: run_wavemap.sh [OUT_DIR]   (default: the working folder's room/; writes OUT_DIR/map_vs_gt_wavemap_{row3,row4})
set -eo pipefail
FT=/home/jixian/Desktop/FT; HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); O=${1:-$FT/wt_takeover_runs/geometry_4d_20261114/room}
PROJECT=$FT/runs/update_layer_game_T1_20261006/worktrees/acceptance_eval_25263f6
PY=/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python; L="taskset -c 8-11 ionice -c3 nice -n 19"
mkdir -p $O/logs
export PYTHONPATH=/mnt/d/3Study/ETH/FT/Synthetic_ABC/scripts PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  HARNESS_SYN_SCRIPTS=/mnt/d/3Study/ETH/FT/Synthetic_ABC/scripts \
  HARNESS_CAUSAL_BANK=$FT/datasets/synthetic_ab/evaluation_complete_20260917/causal_reference
for row in row3 row4; do
  if [[ $row == row3 ]]; then TL=$FT/KEEP_baseline_runs_wavemap_rows13/runs/update_layer_game_T1_20261006/acceptance_runs/wm5_row3_22e68d6/maps/synthetic
  else TL=$FT/runs/update_layer_game_T1_20261006/wm_synth_row4_7b63cc1; fi
  T=$(mktemp -d /dev/shm/geo4d-XXXXXX)
  for s in a b; do
    $L $PY - "$TL/session_$s/timeline.pkl" "$T/ours/native/$s" 0.02 0.01 "$PROJECT" <<'PY'
import sys
sys.path.insert(0, sys.argv[5])
from update_layer.eval.scenelist import load_timeline
from update_layer.eval.emit_synthetic import write_native
write_native(load_timeline(sys.argv[1]), sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
PY
  done
  echo "$(date '+%H:%M') wavemap $row"
  HARNESS_OUT_ROOT=$T $L $PY $HERE/evaluate_map_vs_gt.py --methods ours > $O/logs/map_vs_gt_wavemap_$row.log 2>&1
  mkdir -p $O/map_vs_gt_wavemap_$row; cp $T/geometry/MAP_VS_GT.csv $O/map_vs_gt_wavemap_$row/
  rm -rf $T
done
echo "$(date '+%H:%M') done"
