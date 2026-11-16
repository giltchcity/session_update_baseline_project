#!/bin/bash
# Temporal mesh F1 of Panoptic Mapping on Home: the main table's export (cloud eval/panoptic_rerun_synthetic_5cm/{a,b}/native)
# at the 25 query frames of panoptic_query_times.csv (Khronos's slices, as the first Panoptic score) and, separately, at
# each session's last frame (final_frames.csv: the final Mesh). First the complete-mesh score at the 25 frames as a check
# (must give 41.64, README.md), then scheme C at both frame sets. 4 cores, nice 19.
# usage: run_panoptic.sh [OUT_DIR]   (default: the working folder's room/)
set -eo pipefail
FT=/home/jixian/Desktop/FT; HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); O=${1:-$FT/wt_takeover_runs/geometry_4d_20261114/room}
PY=/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python; L="taskset -c 8-11 ionice -c3 nice -n 19"
export PYTHONPATH=/mnt/d/3Study/ETH/FT/Synthetic_ABC/scripts PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  HARNESS_SYN_SCRIPTS=/mnt/d/3Study/ETH/FT/Synthetic_ABC/scripts \
  HARNESS_CAUSAL_BANK=$FT/datasets/synthetic_ab/evaluation_complete_20260917/causal_reference
mkdir -p $O/logs; T=$(mktemp -d /dev/shm/geo4d-XXXXXX); trap 'rm -rf $T' EXIT; P="/mnt/g/我的云端硬盘/FT_archive/eval/panoptic_rerun_synthetic_5cm"
for s in a b; do mkdir -p $T/panoptic/session_$s; cp -r "$P/$s/native" $T/panoptic/session_$s/native; done
run() {  # NAME RECALL QUERY_CSV
  echo "$(date '+%H:%M') panoptic $1"
  GEO4D_RECALL=$2 GEO4D_QUERY_TIMES=$3 HARNESS_OUT_ROOT=$T $L $PY $HERE/evaluate_map_vs_gt.py --methods panoptic > $O/logs/$1.log 2>&1
  mkdir -p $O/$1; cp $T/geometry/MAP_VS_GT.csv $O/$1/; rm -rf $T/geometry
}
run check_panoptic_complete complete $HERE/panoptic_query_times.csv
run map_vs_gt_panoptic room $HERE/panoptic_query_times.csv
run map_vs_gt_panoptic_final room $HERE/final_frames.csv
echo "$(date '+%H:%M') done"
