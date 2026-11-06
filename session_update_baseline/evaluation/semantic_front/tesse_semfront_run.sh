#!/bin/bash
# TESSE with the semantic object front (user 11-03): the TESSE table's run (analysis/tesse_fix_1101.sh, TAG fill1101: build
# wt_tesse_fix_build cc262a3, mapper_tesse_5cm.yaml, the same player settings) with only the instance input changed: the
# front's ids (semantic_front/tesse_SCENE: tesse_semantic_front.py) and their catalog instead of the ground-truth instance
# ids; then, as the table: eval/chain_fill, the official 4D evaluation (tesse_online_eval.py ours4d) into
# eval/tesse_{apt,paper}_qj/ours_${TAG}_chain4dAppear_5m_all_m050, the background with objects and history at every
# threshold (tesse_bg_history.py) into ..._bghistall_..., and the table's lines: background at 5 cm, G1 (the final map's
# background at 5 cm), static / dynamic objects and changes.
# FRONT=_place TAG=semplace: the ids of tesse_semantic_front.py --place (identity by surface place).
# FRONT=_v3 TAG=semv3: the ids of tesse_semantic_front.py --place --3d (Khronos' 3-D regions, identity by surface place
# within a region of the record's class).
# usage: [FRONT=_v3 TAG=semv3] tesse_semfront_run.sh [apartment] [office]
set -eo pipefail
FT=/home/jixian/Desktop/FT; R=$FT/wt_takeover_runs; SF=$R/semantic_front
WORK=$FT/wt_tesse_fix/session_update_baseline; BUILD=$FT/wt_tesse_fix_build
PY=/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python
ENV="set +u; source /opt/ros/jazzy/setup.bash; source /home/jixian/ros2_ws/install/setup.bash; export LD_LIBRARY_PATH=$FT/wt_290e110_build/canonical/install/lib:\$LD_LIBRARY_PATH; exec \"\$@\""
step() { echo "$(date +%H:%M) $*"; }
FRONT=${FRONT:-}; TAG=${TAG:-semfront}
for SCENE in ${@:-apartment office}; do
  case $SCENE in
    apartment) DATA=$R/tesse/data/session_apt; CFG=$R/tesse/configs/apartment; OUT=$R/tesse_apt_5m_$TAG; EV=$R/eval/tesse_apt_qj; FLAG=(--scene apartment); G=apartment;;
    office) DATA=$R/tesse/data/session_a; CFG=$R/tesse/configs; OUT=$R/tesse_a_5m_$TAG; EV=$R/eval/tesse_paper_qj; FLAG=(); G=office;;
  esac
  until [[ -s $SF/tesse_$SCENE$FRONT/physical_catalog.json ]]; do sleep 20; done
  step "$SCENE: front ids $(ls $SF/tesse_$SCENE$FRONT/instance_labels | wc -l) frames, $(head -1 $SF/tesse_$SCENE$FRONT/FRONT_CHECK.txt)"
  E=$EV/ours_${TAG}_chain4dAppear_5m_all_m050; H=$EV/ours_${TAG}_chain4dAppear_bghistall_5m_all_m050
  [[ -e $OUT ]] && mv $OUT $OUT.before.$(date +%m%d%H%M%S)
  start=$(python3 -c "import json; print(json.load(open('$DATA/session.json'))['start_ns'])")
  step "$SCENE $TAG: mapping"
  (cd $WORK && export ROS_DOMAIN_ID=77 && SESSION_UPDATE_CANONICAL_ROOT=$BUILD/canonical SESSION_UPDATE_CANONICAL_PREFIX=$BUILD/canonical/install \
    SESSION_UPDATE_CANONICAL_BUILD=$BUILD/baseline taskset -c 0-9 nice -n 10 bash scripts/run_session.sh --output-state $OUT \
    --session-id tesse_${SCENE}_5m_$TAG --run-dir $DATA/rgbd --semantic-dir $DATA/semantics --instance-dir $SF/tesse_$SCENE$FRONT/instance_labels \
    --session-start-ns $start --tf-settle-s .02 --play-rate 100 --image-scale 1 --finalization-timeout-s 86400 --sensor-max-range 5 \
    --ack-timeout-s 86400 --discovery-timeout-s 86400 --change-detection-every-n-backend-updates 5 \
    --mapper-config $CFG/mapper_tesse_5cm.yaml --input-config $FT/datasets/synthetic_ab/mapping_configs/input.yaml \
    --labelspace-config $CFG/label_space.yaml --physical-catalog $SF/tesse_$SCENE$FRONT/physical_catalog.json) > $SF/tesse_${SCENE}_${TAG}_mapping.log 2>&1 || true
  [[ -s $OUT/final.4dmap.zpk ]] || { step "$SCENE $TAG: no map (see semantic_front/tesse_${SCENE}_${TAG}_mapping.log)"; continue; }
  step "$SCENE $TAG: mapped, $(du -h $OUT/final.4dmap.zpk | cut -f1)"
  w=/dev/shm/tesse-$TAG-chain; mkdir -p $w
  step "$SCENE $TAG: $(bash -c "$ENV" chain-fill $FT/eval/chain_fill/chain_fill $w/chain.4dmap.zpk $OUT/final.4dmap.zpk | tail -1)"
  [[ -e $E ]] && mv $E $E.before.$(date +%m%d%H%M%S)
  TESSE_EVAL_JOBS=8 ionice -c3 nice -n 19 taskset -c 0-9 $PY $R/analysis/tesse_online_eval.py "${FLAG[@]}" ours4d $w/chain.4dmap.zpk $E > $E.log 2>&1 || true
  rm -f /dev/shm/tesse-$TAG-chain/chain.4dmap.zpk; rmdir $w
  [[ -s $E/TABLES_4D.txt ]] || { step "$SCENE $TAG: 4D evaluation failed (see $E.log)"; continue; }
  step "$SCENE $TAG 4D (20 cm background): $(grep -A1 '^Data' $E/TABLES_4D.txt | tail -1 | tr -s ' ')"
  nice -n 19 taskset -c 0-9 $PY $R/analysis/tesse_bg_history.py $OUT/final.4dmap.zpk $E $H $FT/datasets/khronos/tesse_cd/$G/ground_truth/gt_background.ply > $H.log 2>&1 \
    && step "$SCENE $TAG background 5 cm: $($PY $R/analysis/official_tables_at.py 0.05 $H 2>/dev/null | tail -1)" \
    && step "$SCENE $TAG G1 (final map, 5 cm): $(python3 -c "
import csv; rows = list(csv.DictReader(open('$H/results/background_mesh.csv'))); r = max(rows, key=lambda r: int(r['Name']))
p, q = float(r['Accuracy@0.05']), float(r['Completeness@0.05']); print(f'P {100*p:.1f} R {100*q:.1f} F1 {200*p*q/(p+q):.1f}')")" \
    || step "$SCENE $TAG: background failed (see $H.log)"
done
step "TESSE $TAG DONE"
