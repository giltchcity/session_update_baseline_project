#!/usr/bin/env bash
# 3RScan through the fixed timeline route (eval/chain_fill/timeline_export, eval/timeline_complete.py,
# eval/chain_fill/timeline_to_4dmap; the earlier tools in archive/timeline_route_before_fix_20261024), objects only,
# 0.5 m: our 290e110 maps exported in /dev/shm and scored into ROOM/ours_290e110_timeline/khronos_metrics, row by row
# against the same maps scored directly (ROOM/ours_290e110/khronos_metrics); then the Gaussian timeline of
# p3_ad408c8f. The earlier exports and metrics of ours_290e110_timeline go to the archive; the Gaussian's 2 m metrics
# are kept as gaussian_timeline/khronos_metrics_2m.
# usage: [ROOMS="ref8 ..."] timeline_fix_3rscan.sh   (ROOMS: the rooms to score, default every room of the chain list)
FT=/home/jixian/Desktop/FT; R=$FT/wt_takeover_runs/3rscan; PY=/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python
A=$FT/archive/timeline_route_before_fix_20261024
LOW=(ionice -c3 nice -n 19 taskset -c 0-9)
ENV="set +u; source /opt/ros/jazzy/setup.bash; source /home/jixian/ros2_ws/install/setup.bash; export LD_LIBRARY_PATH=$FT/wt_290e110_build/canonical/install/lib:\$LD_LIBRARY_PATH; exec \"\$@\""
step() { echo "$(date +%H:%M) $*"; }
for ref in ${ROOMS:-$(grep -v "^#" $FT/datasets/3rscan/CHAINS_P3.txt | cut -c1-8)}; do
  room=p3_$ref; d=$R/$room/ours_290e110_timeline; w=/dev/shm/timeline-fix-$room
  mkdir -p $A/$room && mv $d/visit_* $d/khronos_metrics* $A/$room/ 2>/dev/null
  n=$(python3 -c "import json; print(len(json.load(open('$FT/datasets/3rscan/mapping_configs/$room/chain.json'))['visits']))")
  rm -rf $w; for ((k = 0; k < n; k++)); do
    mkdir -p $w/visit_$k
    bash -c "$ENV" export $FT/eval/chain_fill/timeline_export $R/$room/ours_290e110/visit_$k/final.4dmap.zpk $w/visit_$k/snapshots.jsonl
  done
  THREERSCAN_OBJECTS_ONLY=1 THREERSCAN_JOBS=2 "${LOW[@]}" $PY $FT/eval/threerscan_khronos_eval.py $room timeline $w > $R/$room/khronos_metrics_ours_290e110_timeline_fixed.log 2>&1
  mv $w/khronos_metrics $d/ && rm -rf $w
  step "$room timeline: $(tail -1 $R/$room/khronos_metrics_ours_290e110_timeline_fixed.log | cut -c1-150)"
  python3 - $R/$room <<'EOF'
import csv, sys
r = sys.argv[1]
A = list(csv.DictReader(open(f'{r}/ours_290e110/khronos_metrics/results/static_objects.csv')))
B = list(csv.DictReader(open(f'{r}/ours_290e110_timeline/khronos_metrics/results/static_objects.csv')))
diff = [i for i, (a, b) in enumerate(zip(A, B)) if a != b]
print(f'   rows {len(A)}/{len(B)}, identical {len(A) - len(diff)}, differing {diff[:20]}')
EOF
done
g=$R/p3_ad408c8f/gaussian_timeline
[[ -d $g/khronos_metrics && ! -e $g/khronos_metrics_2m ]] && mv $g/khronos_metrics $g/khronos_metrics_2m
rm -rf $g/khronos_metrics
THREERSCAN_OBJECTS_ONLY=1 THREERSCAN_JOBS=2 "${LOW[@]}" $PY $FT/eval/threerscan_khronos_eval.py p3_ad408c8f timeline $g > $R/p3_ad408c8f/khronos_metrics_gaussian_timeline_m050.log 2>&1
step "p3_ad408c8f Gaussian: $(tail -1 $R/p3_ad408c8f/khronos_metrics_gaussian_timeline_m050.log | cut -c1-150)"
step "timeline fix 3RScan done"
