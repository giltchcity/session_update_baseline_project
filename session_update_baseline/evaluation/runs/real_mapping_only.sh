#!/usr/bin/env bash
# Real A, B or C: mapping only, the command of real_bc_resupport.sh (wt_resupport, built in wt_resupport_build; A
# without prior, B from real_a_tk, C from real_b_resupport) into the run directory named by the second argument, for
# timing and the per-round diagnostics. Nothing in here stops the run.
# usage: real_mapping_only.sh a|b|c <run name>
set -eo pipefail
FT=/home/jixian/Desktop/FT; R=$FT/wt_takeover_runs; L=$FT/datasets/local_ab
WORK=$FT/wt_resupport/session_update_baseline; BUILD=$FT/wt_resupport_build
s=${1:?b or c}; NAME=${2:?run name}; out=$R/$NAME
prior_arg=(); world=()
case $s in
  a) run=session_a_20260809_204010_201_flat_rgbd_30hz_1080p; start=1786279210201000000;;
  b) prior=$R/real_a_tk; run=session_b_20260810_030502_620_flat_rgbd_30hz_1080p; start=1786302302620000000;;
  c) prior=$R/real_b_resupport; run=session_c_20260822_182703_387_flat_rgbd_30hz_1080p; start=1787394423387000000;;
esac
[[ $s != a ]] && prior_arg=(--input-state $prior) && world=(--world-transform $L/alignment/session_${s}_to_session_a.txt)
[[ -e $out ]] && mv $out $out.before.$(date +%m%d%H%M%S)
echo "$(date +%H:%M) $NAME: mapping"
(cd $WORK && SESSION_UPDATE_CANONICAL_ROOT=$BUILD/canonical SESSION_UPDATE_CANONICAL_PREFIX=$BUILD/canonical/install \
  SESSION_UPDATE_CANONICAL_BUILD=$BUILD/baseline taskset -c 0-9 nice -n 10 bash scripts/run_session.sh --output-state $out \
  --session-id $NAME --run-dir $L/rgbd/$run --semantic-dir $L/semantics/session_$s \
  --instance-dir $L/instance_labels/session_$s --session-start-ns $start --tf-settle-s 0.02 --play-rate 100 --image-scale 0.5 \
  --finalization-timeout-s 86400 --sensor-max-range 5.0 --ack-timeout-s 86400 --discovery-timeout-s 86400 \
  --change-detection-every-n-backend-updates 5 --mapper-config $WORK/configs/room18_instance_5cm.yaml \
  --input-config $L/mapping_configs/nss_flat_input.yaml --labelspace-config $L/mapping_configs/nss_ade20k_room_label_space.yaml \
  --physical-catalog $L/mapping_configs/room18_physical_catalog.json "${world[@]}" "${prior_arg[@]}") > $R/logs/${NAME}_mapping.log 2>&1
echo "$(date +%H:%M) $NAME: mapped, $(grep -h TIMING_PROBE $out/control/logs/player.log | tail -1)"
