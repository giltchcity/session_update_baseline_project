#!/bin/bash
# Our single runs through the TESSE table's 4D evaluation (single_run_official.sh) with the run's own changes in the 4D
# map (eval/chain_fill/change_fill: closures, appearances by the official reconciler on the run's object_changes.csv,
# moves from the run's state closures beyond the ground truth's state gate), then chain_fill (synthetic: then
# motion_fill, the run's motion), into eval/single_official/ours_{real,syn}_S_mv; and the background with what the map
# holds at each time, the background, the objects and the historical objects (tesse_bg_history.py, as TESSE), into
# ..._mv_bghist (beside the next map's scoring, on its own cores). Evaluation only.
# usage: single_run_official_mv.sh
set -eo pipefail
FT=/home/jixian/Desktop/FT; R=$FT/wt_takeover_runs; PY=/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python
E=$R/eval/single_official; C=$FT/eval/chain_fill
ENV="set +u; source /opt/ros/jazzy/setup.bash; source /home/jixian/ros2_ws/install/setup.bash; export LD_LIBRARY_PATH=$FT/wt_290e110_build/canonical/install/lib:\$LD_LIBRARY_PATH; exec \"\$@\""
step() { echo "$(date +%H:%M) $*"; }
w=/dev/shm/single-official-mv; mkdir -p $w

background() {  # NAME KIND SESSION MAP
  nice -n 19 taskset -c 10-15 $PY $R/analysis/tesse_bg_history.py $4 $E/$1_mv $E/$1_mv_bghist \
    $FT/eval/khronos4d_gt/$2_single_$3/gt_background.ply > $E/$1_mv_bghist.log 2>&1 \
    && step "$1_mv_bghist: $(grep -A1 '^Data' $E/$1_mv_bghist/TABLES_4D.txt | tail -1 | tr -s ' ' | cut -d' ' -f1-4)" \
    || step "$1_mv_bghist: failed (see $E/$1_mv_bghist.log)"
  rm -f "$4"
}

one() {  # NAME KIND SESSION RUN [MOTION.csv]
  local name=$1 kind=$2 s=$3 run=$R/$4 motion=${5:-}
  step "$name: $(bash -c "$ENV" change-fill $C/change_fill $w/$name.changes.4dmap.zpk $run/final.4dmap.zpk $run/object_changes.csv | tail -1)"
  step "$name: $(bash -c "$ENV" chain-fill $C/chain_fill $w/$name.4dmap.zpk $w/$name.changes.4dmap.zpk | tail -1)"
  rm -f $w/$name.changes.4dmap.zpk
  if [[ $motion ]]; then
    step "$name: $(bash -c "$ENV" motion-fill $C/motion_fill $w/$name.motion.4dmap.zpk $w/$name.4dmap.zpk $motion | tail -1)"
    mv $w/$name.motion.4dmap.zpk $w/$name.4dmap.zpk
  fi
  [[ -e $E/${name}_mv ]] && mv $E/${name}_mv $E/${name}_mv.before.$(date +%m%d%H%M%S)
  TESSE_EVAL_CONFIG=$R/eval/khronos4d_gt_${kind}_single_$s.yaml TESSE_EVAL_JOBS=8 ionice -c3 nice -n 19 taskset -c 0-9 \
    $PY $R/analysis/tesse_online_eval.py ours4d $w/$name.4dmap.zpk $E/${name}_mv > $E/${name}_mv.log 2>&1 || true
  if [[ -s $E/${name}_mv/TABLES_4D.txt ]]; then
    step "${name}_mv: $(grep -A1 '^Data' $E/${name}_mv/TABLES_4D.txt | tail -1 | tr -s ' ')"
    background $name $kind $s $w/$name.4dmap.zpk &
  else
    step "${name}_mv: scoring failed (see $E/${name}_mv.log)"; rm -f $w/$name.4dmap.zpk
  fi
}

one ours_real_a real a real_a_states
one ours_real_b real b real_b_single
one ours_real_c real c real_c_single
one ours_syn_a synthetic a synthetic_a_single_obs $R/eval/single_live_dynamics/ours_motion_a.csv
one ours_syn_b synthetic b synthetic_b_single_obs $R/eval/single_live_dynamics/ours_motion_b.csv
wait
rmdir $w
step "SINGLE OFFICIAL MV DONE"
