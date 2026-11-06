#!/bin/bash
# TESSE ours through eval/chain_fill after it takes the run's own moves (10-29, no 50 % re-check), as 3RScan: office from
# the accepted tesse_a_5m_resupport map, apartment from tesse_apt_5m_290e110 (until now scored on the raw map, not through
# chain_fill). Objects and changes by ours4d (every map at every earlier map time, objects-only config, 0.5 m); the
# background and dynamic tables are the same map's, from its earlier evaluation; then tables.py. Evaluation only.
FT=/home/jixian/Desktop/FT; R=$FT/wt_takeover_runs; PY=/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python
ENV="set +u; source /opt/ros/jazzy/setup.bash; source /home/jixian/ros2_ws/install/setup.bash; export LD_LIBRARY_PATH=$FT/wt_290e110_build/canonical/install/lib:\$LD_LIBRARY_PATH; exec \"\$@\""
step() { echo "$(date +%H:%M) $*"; }
run() {  # run SCENE MAP OUT TABLES_FROM
  local w=/dev/shm/chain4dA-$1 out=$3 flag=(); [[ $1 == apartment ]] && flag=(--scene apartment)
  rm -rf $w; mkdir -p $w
  step "$1: $(bash -c "$ENV" chain-fill $FT/eval/chain_fill/chain_fill $w/chain.4dmap.zpk $2 | tail -1)"
  [[ -e $out ]] && mv $out $out.before.$(date +%m%d%H%M%S)
  TESSE_EVAL_CONFIG=$FT/configs/khronos/eval_$1_objects_only_local.yaml TESSE_EVAL_TABLES=static_objects.csv TESSE_EVAL_JOBS=6 \
    ionice -c3 nice -n 19 taskset -c 0-9 $PY $R/analysis/tesse_online_eval.py "${flag[@]}" ours4d $w/chain.4dmap.zpk $out > $out.log 2>&1
  cp $4/results/background_mesh.csv $4/results/dynamic_objects.csv $out/results/
  $PY $R/analysis/tesse_online_eval.py "${flag[@]}" tables $out >> $out.log 2>&1
  rm -rf $w
  step "$1 ours (chain_fill A 4D): $(grep -A1 '^Data' $out/TABLES_4D.txt | tail -1 | tr -s ' ')"
}
run office $R/tesse_a_5m_resupport/final.4dmap.zpk $R/eval/tesse_paper_qj/ours_resupport_chain4dA_5m_all_m050 $R/eval/tesse_paper_qj/ours_resupport_chain4d_5m_all_m050
run apartment $R/tesse_apt_5m_290e110/final.4dmap.zpk $R/eval/tesse_apt_qj/ours_chain4dA_5m_all_m050 $R/eval/tesse_apt_qj/ours_290e110_5m_all_m050
step "TESSE chain_fill A done"
