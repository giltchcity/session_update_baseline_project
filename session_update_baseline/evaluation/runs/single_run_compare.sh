#!/usr/bin/env bash
# Single-run comparison (user 11-01): every visit mapped alone from an empty map, Khronos's own setting (one run).
# Real A/B/C and synthetic A/B; Flat has no change inside a visit. Already there: our real A (real_a_states) and
# synthetic A (synthetic_a_once) start empty; Khronos real A/B/C from empty is the detail page's row 1
# (khronos_pervisit_semantic_*, m050_khronos_pv_*); Khronos synthetic A from empty is the A slice of
# khronos_continuous/synthetic_5cm. Mapped here, each from empty with the command of the results table's run:
#   lane real: ours real B, then real C (real_bc_resupport.sh without --input-state), scored by eval_fast (0.5 m, no
#     prior) into eval/m050_single_{b,c};
#   lane synthetic: ours synthetic B (synthetic_b_resupport.sh without --input-state), scored with our synthetic A by
#     run_synthetic_eval.py (the one object setting, eval_tools_official) into eval/obj100_single_synthetic; then
#     Khronos synthetic B at 5 cm (synthetic_5cm_clean_b.sh) into eval/obj100_khronos_clean_synthetic_5cm.
# The two lanes run side by side on their own ROS domains and cores. Nothing in here stops a run.
# eval_fast full mode refuses B/C without --prior; the prior map only labels D3 provenance in the summary (the objects,
# D2 and G1 never read it), so the previous visit's map is passed; D3 of a single run is not used.
# REAL_ONLY=1: the real lane alone, its visits whose map exists not remapped (rerun after the --prior fix).
# usage: single_run_compare.sh
set -eo pipefail
FT=/home/jixian/Desktop/FT; R=$FT/wt_takeover_runs; L=$FT/datasets/local_ab; S=$FT/datasets/synthetic_ab
WORK=$FT/wt_resupport/session_update_baseline; BUILD=$FT/wt_resupport_build
PY=/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python
CANON=(SESSION_UPDATE_CANONICAL_ROOT=$BUILD/canonical SESSION_UPDATE_CANONICAL_PREFIX=$BUILD/canonical/install SESSION_UPDATE_CANONICAL_BUILD=$BUILD/baseline)
step() { echo "$(date +%H:%M) $*"; }

real_lane() {
  export ROS_DOMAIN_ID=71
  for s in b c; do
    case $s in
      b) RUN=session_b_20260810_030502_620_flat_rgbd_30hz_1080p; START=1786302302620000000; PRIOR=$R/real_a_states/final.4dmap.zpk;;
      c) RUN=session_c_20260822_182703_387_flat_rgbd_30hz_1080p; START=1787394423387000000; PRIOR=$R/real_b_single/final.4dmap.zpk;;
    esac
    out=$R/real_${s}_single
    if [[ ${REAL_ONLY:-} && -s $out/final.4dmap.zpk ]]; then step "real $s single: map there"; else
    [[ -e $out ]] && mv $out $out.before.$(date +%m%d%H%M%S)
    step "real $s single: mapping"
    (cd $WORK && env "${CANON[@]}" taskset -c 0-9 nice -n 10 bash scripts/run_session.sh --output-state $out \
      --session-id real_${s}_single --run-dir $L/rgbd/$RUN --semantic-dir $L/semantics/session_$s \
      --instance-dir $L/instance_labels/session_$s --session-start-ns $START --tf-settle-s 0.02 --play-rate 100 --image-scale 0.5 \
      --finalization-timeout-s 86400 --sensor-max-range 5.0 --ack-timeout-s 86400 --discovery-timeout-s 86400 \
      --change-detection-every-n-backend-updates 5 --mapper-config $WORK/configs/room18_instance_5cm.yaml \
      --input-config $L/mapping_configs/nss_flat_input.yaml --labelspace-config $L/mapping_configs/nss_ade20k_room_label_space.yaml \
      --physical-catalog $L/mapping_configs/room18_physical_catalog.json --world-transform $L/alignment/session_${s}_to_session_a.txt) \
      > $R/logs/real_${s}_single_mapping.log 2>&1
    step "real $s single: mapped, $(du -h $out/final.4dmap.zpk | cut -f1)"; fi
    [[ -e $R/eval/m050_single_$s ]] && mv $R/eval/m050_single_$s $R/eval/m050_single_$s.before.$(date +%m%d%H%M%S)
    (cd $FT && SESSION_UPDATE_CANONICAL_PREFIX=$BUILD/canonical/install ionice -c3 nice -n 19 taskset -c 0-9 \
      /home/jixian/Desktop/miniconda3/bin/python eval_fast/run.py --map $out/final.4dmap.zpk --session $s --name m050_single \
      --mode full --out $R/eval/m050_single_$s --prior $PRIOR) > $R/eval/m050_single_$s.log 2>&1
    step "real $s single: scored into eval/m050_single_$s"
  done
}

synthetic_lane() {
  export ROS_DOMAIN_ID=72
  out=$R/synthetic_b_single
  [[ -e $out ]] && mv $out $out.before.$(date +%m%d%H%M%S)
  step "synthetic b single: mapping"
  (cd $WORK && env "${CANON[@]}" taskset -c 10-15 nice -n 10 bash scripts/run_session.sh --output-state $out \
    --session-id synthetic_b_single --run-dir $S/session_b/rgbd --semantic-dir $S/session_b/semantics \
    --instance-dir $S/session_b/instance_labels --session-start-ns 141000000000 --tf-settle-s .02 --play-rate 100 --image-scale 1 \
    --finalization-timeout-s 86400 --sensor-max-range 15 --ack-timeout-s 86400 --discovery-timeout-s 86400 \
    --change-detection-every-n-backend-updates 25 --mapper-config $WORK/configs/synthetic_5cm.yaml \
    --input-config $S/mapping_configs/input.yaml --labelspace-config $S/mapping_configs/label_space.yaml \
    --physical-catalog $S/mapping_configs/physical_catalog.json) > $R/logs/synthetic_b_single_mapping.log 2>&1
  step "synthetic b single: mapped, $(du -h $out/final.4dmap.zpk | cut -f1)"
  maps=$(mktemp -d /dev/shm/synthetic-single-XXXXXX)
  ln -s $R/synthetic_a_once/final.4dmap.zpk $maps/session_a.4dmap.zpk
  ln -s $out/final.4dmap.zpk $maps/session_b.4dmap.zpk
  [[ -e $R/eval/obj100_single_synthetic ]] && mv $R/eval/obj100_single_synthetic $R/eval/obj100_single_synthetic.before.$(date +%m%d%H%M%S)
  SYN_MAPS=$maps SYN_OUT=$R/eval/obj100_single_synthetic SYN_REF=$R/eval/obj100_states_synthetic SYN_REF_LABEL=ours_chained \
    SYNTHETIC_TOOLS_BIN=$R/eval_tools_official ionice -c3 nice -n 19 taskset -c 10-15 $PY $R/analysis/run_synthetic_eval.py \
    > $R/logs/obj100_single_synthetic.log 2>&1
  rm -r $maps
  step "synthetic b single: scored into eval/obj100_single_synthetic ($(tail -1 $R/logs/obj100_single_synthetic.log))"
  [[ -e $R/khronos_continuous/synthetic_5cm_clean_b ]] && mv $R/khronos_continuous/synthetic_5cm_clean_b $R/khronos_continuous/synthetic_5cm_clean_b.before.$(date +%m%d%H%M%S)
  bash $R/analysis/synthetic_5cm_clean_b.sh > $R/logs/synthetic_5cm_clean_b.log 2>&1
  step "Khronos synthetic b single: $(tail -2 $R/logs/synthetic_5cm_clean_b.log | tr '\n' ' ')"
}

if [[ ${REAL_ONLY:-} ]]; then real_lane >> $R/logs/single_run_real.log 2>&1; exit; fi
real_lane > $R/logs/single_run_real.log 2>&1 &
synthetic_lane > $R/logs/single_run_synthetic.log 2>&1 &
wait
step "SINGLE RUN COMPARE DONE"
