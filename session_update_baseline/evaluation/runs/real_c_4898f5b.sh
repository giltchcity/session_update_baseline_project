#!/usr/bin/env bash
# Our real C session on 4898f5b (290e110 + D1; wt_d1ball, wt_d1ball_build; its player re-sends unacknowledged frames) with real_b_tk as the prior, the
# settings of the L2 record (rewrite_goal_L2_FINAL2_20261006/real_c, as takeover_case.py real_c), into real_c_4898f5b;
# then eval_fast for C (now 0.5 m object matching) into eval/m050_ours_c. Nothing in here stops the run.
# usage: real_c_4898f5b.sh
set -eo pipefail
FT=/home/jixian/Desktop/FT; R=$FT/wt_takeover_runs; L=$FT/datasets/local_ab
WORK=$FT/wt_d1ball/session_update_baseline; BUILD=$FT/wt_d1ball_build; OUT=$R/real_c_4898f5b
step() { echo "$(date +%H:%M) $*"; }
[[ -e $OUT ]] && mv $OUT $OUT.before.$(date +%m%d%H%M%S)
step "real C 4898f5b: mapping"
(cd $WORK && SESSION_UPDATE_CANONICAL_ROOT=$BUILD/canonical SESSION_UPDATE_CANONICAL_PREFIX=$BUILD/canonical/install \
  SESSION_UPDATE_CANONICAL_BUILD=$BUILD/baseline KHRONOS_REFUSION_DUMP=$R/replay_real_c_4898f5b \
  taskset -c 0-9 nice -n 10 bash scripts/run_session.sh --output-state $OUT --session-id real_c_4898f5b \
  --run-dir $L/rgbd/session_c_20260822_182703_387_flat_rgbd_30hz_1080p --semantic-dir $L/semantics/session_c \
  --instance-dir $L/instance_labels/session_c --session-start-ns 1787394423387000000 --tf-settle-s 0.02 --play-rate 100 \
  --image-scale 0.5 --finalization-timeout-s 86400 --sensor-max-range 5.0 --ack-timeout-s 86400 --discovery-timeout-s 86400 \
  --change-detection-every-n-backend-updates 5 --mapper-config $WORK/configs/room18_instance_5cm.yaml \
  --input-config $L/mapping_configs/nss_flat_input.yaml --labelspace-config $L/mapping_configs/nss_ade20k_room_label_space.yaml \
  --physical-catalog $L/mapping_configs/room18_physical_catalog.json --world-transform $L/alignment/session_c_to_session_a.txt \
  --input-state $R/real_b_tk) > $R/logs/real_c_4898f5b_mapping.log 2>&1
step "real C 4898f5b: mapped, $(du -h $OUT/final.4dmap.zpk | cut -f1)"
(cd $FT && SESSION_UPDATE_CANONICAL_PREFIX=$BUILD/canonical/install ionice -c3 nice -n 19 taskset -c 0-9 \
  /home/jixian/Desktop/miniconda3/bin/python eval_fast/run.py --map $OUT/final.4dmap.zpk --session c --name m050_ours \
  --mode full --out $R/eval/m050_ours_c --prior $R/real_b_tk/final.4dmap.zpk) > $R/eval/m050_ours_c.log 2>&1
step "real C 4898f5b: scored (0.5 m) into eval/m050_ours_c"
