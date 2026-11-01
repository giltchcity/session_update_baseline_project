#!/usr/bin/env bash
# TESSE apartment then office on wt_tesse_fix (branch tesse-fix-1101: the resupport version e541075 with the present
# surface at the background map's truncation, 0.15 m at 5 cm, instead of two voxels; built in wt_tesse_fix_build), the
# command of tesse_resupport.sh, into tesse_{apt,a}_5m_fix1101; then, as there, our 4D map through eval/chain_fill scored
# by the 4D evaluation (tesse_online_eval.py ours4d, configs/khronos at 0.5 m) into
# eval/tesse_{apt,paper}_qj/ours_${TAG}_chain4dAppear_5m_all_m050, and the background with the 4D map's history
# (tesse_bg_history.py) into ours_${TAG}_chain4dAppear_bghist_5m_all_m050. Nothing in here stops the run.
# 11-01 second experiment, TAG=fill1101: the truncation back as before, a fill face without evidence either way
# shown as a memory face (present_surface.cpp compose), the same build directory.
# usage: [TAG=fix1101|fill1101] tesse_fix_1101.sh [apartment] [office]
set -eo pipefail
FT=/home/jixian/Desktop/FT; R=$FT/wt_takeover_runs
WORK=$FT/wt_tesse_fix/session_update_baseline; BUILD=$FT/wt_tesse_fix_build
PY=/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python
ENV="set +u; source /opt/ros/jazzy/setup.bash; source /home/jixian/ros2_ws/install/setup.bash; export LD_LIBRARY_PATH=$FT/wt_290e110_build/canonical/install/lib:\$LD_LIBRARY_PATH; exec \"\$@\""
step() { echo "$(date +%H:%M) $*"; }
TAG=${TAG:-fix1101}
for SCENE in ${@:-apartment office}; do
  case $SCENE in
    apartment) DATA=$R/tesse/data/session_apt; CFG=$R/tesse/configs/apartment; OUT=$R/tesse_apt_5m_$TAG; EV=$R/eval/tesse_apt_qj; FLAG=(--scene apartment);;
    office) DATA=$R/tesse/data/session_a; CFG=$R/tesse/configs; OUT=$R/tesse_a_5m_$TAG; EV=$R/eval/tesse_paper_qj; FLAG=();;
  esac
  E=$EV/ours_${TAG}_chain4dAppear_5m_all_m050
  [[ -e $OUT ]] && mv $OUT $OUT.before.$(date +%m%d%H%M%S)
  start=$(python3 -c "import json; print(json.load(open('$DATA/session.json'))['start_ns'])")
  step "TESSE $SCENE $TAG: mapping"
  (cd $WORK && SESSION_UPDATE_CANONICAL_ROOT=$BUILD/canonical SESSION_UPDATE_CANONICAL_PREFIX=$BUILD/canonical/install \
    SESSION_UPDATE_CANONICAL_BUILD=$BUILD/baseline taskset -c 0-9 nice -n 10 bash scripts/run_session.sh --output-state $OUT \
    --session-id tesse_${SCENE}_5m_$TAG --run-dir $DATA/rgbd --semantic-dir $DATA/semantics --instance-dir $DATA/instance_labels \
    --session-start-ns $start --tf-settle-s .02 --play-rate 100 --image-scale 1 --finalization-timeout-s 86400 --sensor-max-range 5 \
    --ack-timeout-s 86400 --discovery-timeout-s 86400 --change-detection-every-n-backend-updates 5 \
    --mapper-config $CFG/mapper_tesse_5cm.yaml --input-config $FT/datasets/synthetic_ab/mapping_configs/input.yaml \
    --labelspace-config $CFG/label_space.yaml --physical-catalog $CFG/physical_catalog.json) > $R/logs/tesse_${SCENE}_5m_${TAG}_mapping.log 2>&1
  step "TESSE $SCENE $TAG: mapped, $(du -h $OUT/final.4dmap.zpk | cut -f1), refusion truncation $(grep -o '"truncation":[0-9.]*' $OUT/refusion_report.json)"
  w=/dev/shm/tesse-$TAG-chain; rm -rf $w; mkdir -p $w
  step "TESSE $SCENE $TAG: $(bash -c "$ENV" chain-fill $FT/eval/chain_fill/chain_fill $w/chain.4dmap.zpk $OUT/final.4dmap.zpk | tail -1)"
  [[ -e $E ]] && mv $E $E.before.$(date +%m%d%H%M%S)
  TESSE_EVAL_JOBS=8 ionice -c3 nice -n 19 taskset -c 0-9 $PY $R/analysis/tesse_online_eval.py "${FLAG[@]}" ours4d $w/chain.4dmap.zpk $E > $E.log 2>&1
  rm -rf $w
  step "TESSE $SCENE $TAG 4D (0.5 m): $(grep -A1 '^Data' $E/TABLES_4D.txt | tail -1 | tr -s ' ')"
  nice -n 19 $PY $R/analysis/tesse_bg_history.py $OUT/final.4dmap.zpk $E ${E/_5m_all/_bghist_5m_all} \
    $FT/datasets/khronos/tesse_cd/$SCENE/ground_truth/gt_background.ply > ${E/_5m_all/_bghist_5m_all}.log 2>&1
  step "TESSE $SCENE $TAG background with history: $(grep -A1 '^Data' ${E/_5m_all/_bghist_5m_all}/TABLES_4D.txt | tail -1 | tr -s ' ' | cut -d' ' -f1-4)"
done
step "TESSE $TAG DONE"
