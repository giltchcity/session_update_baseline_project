#!/usr/bin/env bash
# TESSE, our 4D map as on 3RScan: eval/chain_fill derives it from the run's own snapshots (each the map of its moment,
# nothing remapped): a relocation the map counted keeps the old place as a closed node and the moved node is first
# observed at its first observation there. The object and change tables of that 4D map at 0.5 m (ours4d, objects only,
# in /dev/shm); the background and dynamic tables are the same map's (same snapshots), taken from its evaluation
# (office: the D1 splice); then tables.py. Into EVAL_DIR/ours_chain4d_5m_all_m050.
# usage: tesse_chain4d.sh [office] [apartment]   (default both)
FT=/home/jixian/Desktop/FT; R=$FT/wt_takeover_runs; PY=/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python
ENV="set +u; source /opt/ros/jazzy/setup.bash; source /home/jixian/ros2_ws/install/setup.bash; export LD_LIBRARY_PATH=$FT/wt_290e110_build/canonical/install/lib:\$LD_LIBRARY_PATH; exec \"\$@\""
step() { echo "$(date +%H:%M) $*"; }
run() {  # run SCENE MAP EVAL_DIR TABLES_FROM
  local w=/dev/shm/chain4d-$1 out=$3/ours_chain4d_5m_all_m050 flag=(); [[ $1 == apartment ]] && flag=(--scene apartment)
  rm -rf $w; mkdir -p $w
  step "$1: $(bash -c "$ENV" chain-fill $FT/eval/chain_fill/chain_fill $w/chain.4dmap.zpk $2 | tail -1)"
  $PY -c "import sys; sys.path.insert(0, '$FT/eval_fast/vendor/pack'); import zpk_snapshot as zs; a, b = zs.stamps('$w/chain.4dmap.zpk'), zs.stamps('$2'); print('stamps', len(a), 'of', len(b), 'same' if list(a) == list(b) else 'DIFFER')"
  [[ -e $out ]] && mv $out $out.before.$(date +%m%d%H%M%S)
  TESSE_EVAL_CONFIG=$FT/configs/khronos/eval_$1_objects_only_local.yaml TESSE_EVAL_TABLES=static_objects.csv TESSE_EVAL_JOBS=4 \
    ionice -c3 nice -n 19 taskset -c 0-9 $PY $R/analysis/tesse_online_eval.py "${flag[@]}" ours4d $w/chain.4dmap.zpk $out > $out.log 2>&1
  cp $4/results/background_mesh.csv $4/results/dynamic_objects.csv $out/results/
  $PY $R/analysis/tesse_online_eval.py "${flag[@]}" tables $out >> $out.log 2>&1
  rm -rf $w
  step "$1 ours (chain_fill 4D): $(grep -A1 '^Data' $out/TABLES_4D.txt | tail -1 | tr -s ' ')"
}
for scene in ${@:-office apartment}; do
  case $scene in
    office) run office $R/tesse_a_5m_290e110/final.4dmap.zpk $R/eval/tesse_paper_qj $R/eval/tesse_paper_qj/ours_290e110_d1ball_splice_5m_all_m050;;
    apartment) run apartment $R/tesse_apt_5m_290e110/final.4dmap.zpk $R/eval/tesse_apt_qj $R/eval/tesse_apt_qj/ours_290e110_5m_all_m050;;
  esac
done
step "TESSE chain_fill 4D done"
