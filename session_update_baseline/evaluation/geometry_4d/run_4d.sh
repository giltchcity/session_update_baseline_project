#!/bin/bash
# Temporal mesh F1 of TSDF ours and Khronos on Home (evaluate_map_vs_gt.py, scheme C unless GEO4D_RECALL=complete).
#   ours     TSDF, the valid-domain ablation "on" maps (cloud FT_archive/ablation_vd_20261111/on; the main table's
#            synthetic metrics), exported as run_synthetic_eval.py does (synthetic_ram.py, eval_tools_official)
#   khronos  official Khronos continuous A -> B at 5 cm (the main table's run), exported as synthetic_5cm_reruns.sh
# One job at a time, 4 cores, nice 19; exports in /dev/shm, removed at the end.
# usage: run_4d.sh [OUT_DIR]   (default: the working folder's room/; writes OUT_DIR/map_vs_gt_{ours,khronos_cont}/MAP_VS_GT.csv)
set -eo pipefail
FT=/home/jixian/Desktop/FT; HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); O=${1:-$FT/wt_takeover_runs/geometry_4d_20261114/room}
OURS="/mnt/g/我的云端硬盘/FT_archive/ablation_vd_20261111/on"; KHRONOS=$FT/wt_takeover_runs/khronos_continuous/synthetic_5cm/sessions
T=$(mktemp -d /dev/shm/geo4d-XXXXXX); trap 'rm -rf $T' EXIT
mkdir -p $O/logs
export PYTHONPATH=/mnt/d/3Study/ETH/FT/Synthetic_ABC/scripts PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  HARNESS_SYN_SCRIPTS=/mnt/d/3Study/ETH/FT/Synthetic_ABC/scripts \
  HARNESS_CAUSAL_BANK=$FT/datasets/synthetic_ab/evaluation_complete_20260917/causal_reference
L="taskset -c 8-11 ionice -c3 nice -n 19"; cd $FT
for m in ours khronos; do
  for s in a b; do
    if [[ $m == ours ]]; then map="$OURS/synthetic_$s/final.4dmap.zpk"; tools=(SYNTHETIC_TOOLS=ours SYNTHETIC_TOOLS_BIN=$FT/wt_takeover_runs/eval_tools_official)
    else map=$KHRONOS/session_$s.4dmap.zpk; tools=(SYNTHETIC_TOOLS=khronos SYNTHETIC_TOOLS_BIN=$FT/eval/synthetic_object_eval_official); fi
    cp "$map" $T/session_$s.4dmap.zpk
    echo "$(date '+%H:%M') export $m $s"
    env "${tools[@]}" $L python wt_takeover_runs/analysis/synthetic_ram.py export $T/session_$s.4dmap.zpk $T/$m/$m/native/$s > $O/logs/export_${m}_$s.log 2>&1
    rm $T/session_$s.4dmap.zpk
  done
  name=$m; [[ $m == khronos ]] && name=khronos_cont
  echo "$(date '+%H:%M') geometry $m"
  HARNESS_OUT_ROOT=$T/$m $L /home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python $HERE/evaluate_map_vs_gt.py --methods $m > $O/logs/map_vs_gt_${name}.log 2>&1
  rm -rf $O/map_vs_gt_${name}; mkdir -p $O/map_vs_gt_${name}; cp $T/$m/geometry/MAP_VS_GT.csv $O/map_vs_gt_${name}/
  rm -rf $T/$m
done
echo "$(date '+%H:%M') done"
