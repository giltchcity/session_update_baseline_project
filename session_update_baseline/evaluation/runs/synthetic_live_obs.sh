#!/usr/bin/env bash
# Our synthetic A and B, each from an empty map, with the live-track recorder (observer_resupport, analysis/
# build_ours_observer.py: the results table's version, mapping unchanged), the commands of synthetic_a_once and the
# single-run B; tracks into eval/single_live_dynamics/ours_live/{a,b}/live_tracks.jsonl for the live D1 evaluation.
# usage: synthetic_live_obs.sh
set -eo pipefail
FT=/home/jixian/Desktop/FT; R=$FT/wt_takeover_runs; S=$FT/datasets/synthetic_ab
WORK=$FT/wt_resupport/session_update_baseline; BUILD=$FT/wt_resupport_build; OBS=$R/observer_resupport
export ROS_DOMAIN_ID=73
step() { echo "$(date +%H:%M) $*"; }
for s in a b; do
  case $s in a) START=1000000000;; b) START=141000000000;; esac
  out=$R/synthetic_${s}_single_obs; live=$R/eval/single_live_dynamics/ours_live/$s
  [[ -e $out ]] && mv $out $out.before.$(date +%m%d%H%M%S)
  rm -rf $live; mkdir -p $live
  step "synthetic $s observer: mapping"
  (cd $WORK && SYN_EVAL_OUT=$live SESSION_UPDATE_CANONICAL_ROOT=$BUILD/canonical SESSION_UPDATE_CANONICAL_PREFIX=$BUILD/canonical/install \
    SESSION_UPDATE_CANONICAL_BUILD=$BUILD/baseline taskset -c 10-15 nice -n 10 bash $OBS/run_ours_session.sh --output-state $out \
    --session-id synthetic_${s}_single_obs --run-dir $S/session_$s/rgbd --semantic-dir $S/session_$s/semantics \
    --instance-dir $S/session_$s/instance_labels --session-start-ns $START --tf-settle-s .02 --play-rate 100 --image-scale 1 \
    --finalization-timeout-s 86400 --sensor-max-range 15 --ack-timeout-s 86400 --discovery-timeout-s 86400 \
    --change-detection-every-n-backend-updates 25 --mapper-config $WORK/configs/synthetic_5cm.yaml \
    --input-config $S/mapping_configs/input.yaml --labelspace-config $S/mapping_configs/label_space.yaml \
    --physical-catalog $S/mapping_configs/physical_catalog.json) > $R/logs/synthetic_${s}_single_obs_mapping.log 2>&1
  step "synthetic $s observer: mapped, $(du -h $out/final.4dmap.zpk | cut -f1), $(wc -l < $live/live_tracks.jsonl) frames recorded"
done
step "SYNTHETIC LIVE OBS DONE"
