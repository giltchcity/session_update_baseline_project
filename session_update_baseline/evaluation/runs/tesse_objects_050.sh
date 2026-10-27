#!/usr/bin/env bash
# TESSE at 0.5 m object matching: only the object and change tables are recomputed (eval_SCENE_objects_only_local.yaml:
# background and dynamic evaluation off; they do not depend on object matching), the background and dynamic tables
# are taken from the earlier evaluation of the same map (same snapshots, same queries), then tables.py.
# usage: tesse_objects_050.sh
R=/home/jixian/Desktop/FT/wt_takeover_runs; PY=/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python
step() { echo "$(date +%H:%M) $*"; }
run() {  # run EVAL_DIR NAME MAP SCENE
  local E=$1 old=$1/$2 out=$1/$2_m050 flag=(); [[ $4 == apartment ]] && flag=(--scene apartment)
  [[ -s $out/TABLES_4D.txt ]] && { step "$4 $2: already done"; return; }
  [[ -e $out ]] && mv $out $out.before.$(date +%H%M%S)
  TESSE_EVAL_CONFIG=/home/jixian/Desktop/FT/configs/khronos/eval_$4_objects_only_local.yaml TESSE_EVAL_TABLES=static_objects.csv \
    TESSE_EVAL_JOBS=8 ionice -c3 nice -n 19 taskset -c 0-9 $PY $R/analysis/tesse_online_eval.py "${flag[@]}" ours4d $3 $out > $out.log 2>&1
  cp $old/results/background_mesh.csv $old/results/dynamic_objects.csv $out/results/
  $PY $R/analysis/tesse_online_eval.py "${flag[@]}" tables $out >> $out.log 2>&1
  step "$4 $2: $(grep -A1 '^Data' $out/TABLES_4D.txt | tail -1 | tr -s ' ')"
}
run $R/eval/tesse_paper_qj khronos_5m_all $R/tesse/khronos_5m_cd5/state/final.4dmap.zpk office
run $R/eval/tesse_paper_qj khronos_inst_5m_all $R/tesse/khronos_inst_office/state/final.4dmap.zpk office
run $R/eval/tesse_apt_qj ours_290e110_5m_all $R/tesse_apt_5m_290e110/final.4dmap.zpk apartment
run $R/eval/tesse_apt_qj khronos_5m_all $R/tesse/khronos_apt_5m_cd5/state/final.4dmap.zpk apartment
step "TESSE 0.5 m done"
