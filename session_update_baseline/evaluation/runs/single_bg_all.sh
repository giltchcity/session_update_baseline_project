#!/bin/bash
# The background of our single-run 4D maps at every threshold of the config (tesse_bg_history.py: the background, the
# objects and the historical objects; 11-01 the 5 cm one too), the maps rebuilt as single_run_official_mv.sh builds
# them (change_fill, chain_fill, synthetic motion_fill), into eval/single_official/NAME_mv_bghist; then the 5 cm and
# 20 cm official averages. Evaluation only.
# usage: single_bg_all.sh NAME:KIND:SESSION:RUN[:MOTION.csv] ...
set -eo pipefail
FT=/home/jixian/Desktop/FT; R=$FT/wt_takeover_runs; PY=/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python
E=$R/eval/single_official; C=$FT/eval/chain_fill
ENV="set +u; source /opt/ros/jazzy/setup.bash; source /home/jixian/ros2_ws/install/setup.bash; export LD_LIBRARY_PATH=$FT/wt_290e110_build/canonical/install/lib:\$LD_LIBRARY_PATH; exec \"\$@\""
step() { echo "$(date +%H:%M) $*"; }
w=/dev/shm/single-bg-all; mkdir -p $w
for job in "$@"; do
  IFS=: read -r name kind s run motion <<< "$job"
  bash -c "$ENV" change-fill $C/change_fill $w/changes.4dmap.zpk $R/$run/final.4dmap.zpk $R/$run/object_changes.csv > /dev/null
  bash -c "$ENV" chain-fill $C/chain_fill $w/map.4dmap.zpk $w/changes.4dmap.zpk > /dev/null
  rm -f /dev/shm/single-bg-all/changes.4dmap.zpk
  if [[ $motion ]]; then
    bash -c "$ENV" motion-fill $C/motion_fill $w/motion.4dmap.zpk $w/map.4dmap.zpk $motion > /dev/null
    mv $w/motion.4dmap.zpk $w/map.4dmap.zpk
  fi
  nice -n 19 $PY $R/analysis/tesse_bg_history.py $w/map.4dmap.zpk $E/${name}_mv $E/${name}_mv_bghist \
    $FT/eval/khronos4d_gt/${kind}_single_$s/gt_background.ply > $E/${name}_mv_bghist.log 2>&1 || { step "$name: failed"; continue; }
  rm -f /dev/shm/single-bg-all/map.4dmap.zpk
  for t in 0.05 0.2; do
    step "$name background @$t: $($PY $R/analysis/official_tables_at.py $t $E/${name}_mv_bghist 2>/dev/null | tail -1 | tr -s ' ')"
  done
done
rmdir $w
step "SINGLE BG ALL DONE"
