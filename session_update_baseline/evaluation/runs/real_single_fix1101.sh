#!/bin/bash
# Real A, B, C mapped alone again (single runs) on the fixed build (wt_tesse_fix, cc262a3: a fill face without evidence
# either way stays shown, as a memory face; the TESSE table's build), the commands of single_run_compare.sh's real
# lane otherwise unchanged (11-01: on e541075 the background hid static surfaces beside the walking people, which
# then got no evidence: real A lost 0.57 % of its reference by the end, nine tenths of it held earlier), into
# real_{a,b,c}_single_fix1101; each scored by eval_fast (as eval/m050_single_*, SINGLE_RUN=1) into
# eval/m050_single_fix1101_S; then the TESSE table's 4D evaluation (single_run_official_mv.sh's steps) into
# eval/single_official/ours_real_S_fix1101_mv. Lanes: C on cores 0-9, A then B on 10-15. Nothing in here stops a run.
# usage: real_single_fix1101.sh
set -eo pipefail
FT=/home/jixian/Desktop/FT; R=$FT/wt_takeover_runs; L=$FT/datasets/local_ab
WORK=$FT/wt_tesse_fix/session_update_baseline; BUILD=$FT/wt_tesse_fix_build
PY=/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python; E=$R/eval/single_official; C=$FT/eval/chain_fill
CANON=(SESSION_UPDATE_CANONICAL_ROOT=$BUILD/canonical SESSION_UPDATE_CANONICAL_PREFIX=$BUILD/canonical/install SESSION_UPDATE_CANONICAL_BUILD=$BUILD/baseline)
ENV="set +u; source /opt/ros/jazzy/setup.bash; source /home/jixian/ros2_ws/install/setup.bash; export LD_LIBRARY_PATH=$FT/wt_290e110_build/canonical/install/lib:\$LD_LIBRARY_PATH; exec \"\$@\""
step() { echo "$(date +%H:%M) $*"; }
export SINGLE_RUN=1

visit() {  # SESSION CORES ROS_DOMAIN
  local s=$1 cores=$2 out=$R/real_${1}_single_fix1101 world=() prior=()
  case $s in
    a) RUN=session_a_20260809_204010_201_flat_rgbd_30hz_1080p; START=1786279210201000000;;
    b) RUN=session_b_20260810_030502_620_flat_rgbd_30hz_1080p; START=1786302302620000000; prior=(--prior $R/real_a_states/final.4dmap.zpk);;
    c) RUN=session_c_20260822_182703_387_flat_rgbd_30hz_1080p; START=1787394423387000000; prior=(--prior $R/real_b_single/final.4dmap.zpk);;
  esac
  [[ $s != a ]] && world=(--world-transform $L/alignment/session_${s}_to_session_a.txt)
  [[ -e $out ]] && mv $out $out.before.$(date +%m%d%H%M%S)
  step "real $s fix1101: mapping"
  (cd $WORK && export ROS_DOMAIN_ID=$3 && env "${CANON[@]}" taskset -c $cores nice -n 10 bash scripts/run_session.sh --output-state $out \
    --session-id real_${s}_single_fix1101 --run-dir $L/rgbd/$RUN --semantic-dir $L/semantics/session_$s \
    --instance-dir $L/instance_labels/session_$s --session-start-ns $START --tf-settle-s 0.02 --play-rate 100 --image-scale 0.5 \
    --finalization-timeout-s 86400 --sensor-max-range 5.0 --ack-timeout-s 86400 --discovery-timeout-s 86400 \
    --change-detection-every-n-backend-updates 5 --mapper-config $WORK/configs/room18_instance_5cm.yaml \
    --input-config $L/mapping_configs/nss_flat_input.yaml --labelspace-config $L/mapping_configs/nss_ade20k_room_label_space.yaml \
    --physical-catalog $L/mapping_configs/room18_physical_catalog.json "${world[@]}") > $R/logs/real_${s}_single_fix1101_mapping.log 2>&1
  [[ -s $out/final.4dmap.zpk ]] || { step "real $s fix1101: no map (see logs/real_${s}_single_fix1101_mapping.log)"; return; }
  step "real $s fix1101: mapped, $(du -h $out/final.4dmap.zpk | cut -f1)"
  [[ -e $R/eval/m050_single_fix1101_$s ]] && mv $R/eval/m050_single_fix1101_$s $R/eval/m050_single_fix1101_$s.before.$(date +%m%d%H%M%S)
  (cd $FT && SESSION_UPDATE_CANONICAL_PREFIX=$BUILD/canonical/install ionice -c3 nice -n 19 taskset -c $cores \
    /home/jixian/Desktop/miniconda3/bin/python eval_fast/run.py --map $out/final.4dmap.zpk --session $s --name m050_single_fix1101 \
    --mode full --out $R/eval/m050_single_fix1101_$s "${prior[@]}") > $R/eval/m050_single_fix1101_$s.log 2>&1 \
    && step "real $s fix1101: G1 $(python3 -c "import json,glob; d=json.load(open(glob.glob('$R/eval/m050_single_fix1101_$s/real_abc/geometry/*/geometry_viewer_G1.json')[0])); print(round(100*d['F1_05cm'],2))")" \
    || step "real $s fix1101: eval_fast failed (see eval/m050_single_fix1101_$s.log)"
}

official() {  # SESSION: the TESSE table's 4D evaluation, as single_run_official_mv.sh
  local s=$1 name=ours_real_$1_fix1101 run=$R/real_$1_single_fix1101 w=/dev/shm/real-fix1101-official
  [[ -s $run/final.4dmap.zpk ]] || return 0
  mkdir -p $w
  step "$name: $(bash -c "$ENV" change-fill $C/change_fill $w/changes.4dmap.zpk $run/final.4dmap.zpk $run/object_changes.csv | tail -1)"
  step "$name: $(bash -c "$ENV" chain-fill $C/chain_fill $w/map.4dmap.zpk $w/changes.4dmap.zpk | tail -1)"
  rm -f /dev/shm/real-fix1101-official/changes.4dmap.zpk
  [[ -e $E/${name}_mv ]] && mv $E/${name}_mv $E/${name}_mv.before.$(date +%m%d%H%M%S)
  TESSE_EVAL_CONFIG=$R/eval/khronos4d_gt_real_single_$s.yaml TESSE_EVAL_JOBS=8 ionice -c3 nice -n 19 taskset -c 0-9 \
    $PY $R/analysis/tesse_online_eval.py ours4d $w/map.4dmap.zpk $E/${name}_mv > $E/${name}_mv.log 2>&1 || true
  if [[ -s $E/${name}_mv/TABLES_4D.txt ]]; then
    step "${name}_mv: $(grep -A1 '^Data' $E/${name}_mv/TABLES_4D.txt | tail -1 | tr -s ' ')"
    nice -n 19 taskset -c 0-9 $PY $R/analysis/tesse_bg_history.py $w/map.4dmap.zpk $E/${name}_mv $E/${name}_mv_bghist \
      $FT/eval/khronos4d_gt/real_single_$s/gt_background.ply > $E/${name}_mv_bghist.log 2>&1 \
      && step "${name}_mv_bghist: $(grep -A1 '^Data' $E/${name}_mv_bghist/TABLES_4D.txt | tail -1 | tr -s ' ' | cut -d' ' -f1-4)" \
      || step "${name}_mv_bghist: failed (see $E/${name}_mv_bghist.log)"
  else step "${name}_mv: scoring failed (see $E/${name}_mv.log)"; fi
  rm -f /dev/shm/real-fix1101-official/map.4dmap.zpk; rmdir $w
}

visit c 0-9 73 &
(visit a 10-15 74; visit b 10-15 74) &
wait
for s in a b c; do official $s; done
step "REAL SINGLE FIX1101 DONE"
