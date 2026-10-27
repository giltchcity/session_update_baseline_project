#!/usr/bin/env bash
# Synthetic B with a state limiting the frames before it once, when it starts (wt_resupport, built in wt_resupport_build),
# from synthetic_a_once, with the command of synthetic_b_once.sh, into synthetic_b_states; the full synthetic evaluation
# of synthetic_a_once + this B into eval/states_synthetic against the read-once eval/once_synthetic (the same A).
# Nothing in here stops the run. usage: synthetic_b_states.sh
set -eo pipefail
FT=/home/jixian/Desktop/FT; R=$FT/wt_takeover_runs; S=$FT/datasets/synthetic_ab
WORK=$FT/wt_resupport/session_update_baseline; BUILD=$FT/wt_resupport_build; OUT=$R/synthetic_b_states
step() { echo "$(date +%H:%M) $*"; }
[[ -e $OUT ]] && mv $OUT $OUT.before.$(date +%m%d%H%M%S)
(cd $WORK && SESSION_UPDATE_CANONICAL_ROOT=$BUILD/canonical SESSION_UPDATE_CANONICAL_PREFIX=$BUILD/canonical/install \
  SESSION_UPDATE_CANONICAL_BUILD=$BUILD/baseline taskset -c 0-9 nice -n 10 bash scripts/run_session.sh --output-state $OUT \
  --session-id synthetic_b_states --run-dir $S/session_b/rgbd --semantic-dir $S/session_b/semantics \
  --instance-dir $S/session_b/instance_labels --session-start-ns 141000000000 --tf-settle-s .02 --play-rate 100 --image-scale 1 \
  --finalization-timeout-s 86400 --sensor-max-range 15 --ack-timeout-s 86400 --discovery-timeout-s 86400 \
  --change-detection-every-n-backend-updates 25 --mapper-config $WORK/configs/synthetic_5cm.yaml \
  --input-config $S/mapping_configs/input.yaml --labelspace-config $S/mapping_configs/label_space.yaml \
  --physical-catalog $S/mapping_configs/physical_catalog.json --input-state $R/synthetic_a_once) > $R/logs/synthetic_b_states_mapping.log 2>&1
step "synthetic B states: mapped, $(head -1 $OUT/control/logs/khronos.log | cut -c10-17) -> $(tail -1 $OUT/control/logs/khronos.log | cut -c10-17)"
maps=$(mktemp -d /dev/shm/synthetic-maps-XXXXXX)
ln -s $R/synthetic_a_once/final.4dmap.zpk $maps/session_a.4dmap.zpk
ln -s $OUT/final.4dmap.zpk $maps/session_b.4dmap.zpk
[[ -e $R/eval/states_synthetic ]] && mv $R/eval/states_synthetic $R/eval/states_synthetic.before.$(date +%m%d%H%M%S)
SYN_MAPS=$maps SYN_OUT=$R/eval/states_synthetic SYN_REF=$R/eval/once_synthetic SYN_REF_LABEL=read-once \
  SYNTHETIC_TOOLS_BIN=$R/eval_tools_m050 ionice -c3 nice -n 19 taskset -c 0-9 \
  /home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python $R/analysis/run_synthetic_eval.py > $R/logs/states_synthetic_eval.log 2>&1
rm -r $maps
step "synthetic states: evaluated into eval/states_synthetic (FULL_TABLE.md)"
