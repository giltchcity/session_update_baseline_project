#!/usr/bin/env bash
# TESSE office, 5 m: mapping only, the command of tesse_resupport.sh (wt_resupport, built in wt_resupport_build) into
# the run directory named by the argument, for timing and round-by-round comparison with tesse_a_5m_resupport.
# Nothing in here stops the run.
# usage: tesse_mapping_only.sh <run name>
set -eo pipefail
FT=/home/jixian/Desktop/FT; R=$FT/wt_takeover_runs; DATA=$R/tesse/data/session_a; CFG=$R/tesse/configs
NAME=${1:?run name}; OUT=$R/$NAME
WORK=$FT/wt_resupport/session_update_baseline; BUILD=$FT/wt_resupport_build
[[ -e $OUT ]] && mv $OUT $OUT.before.$(date +%m%d%H%M%S)
start=$(python3 -c "import json; print(json.load(open('$DATA/session.json'))['start_ns'])")
echo "$(date +%H:%M) $NAME: mapping"
(cd $WORK && SESSION_UPDATE_CANONICAL_ROOT=$BUILD/canonical SESSION_UPDATE_CANONICAL_PREFIX=$BUILD/canonical/install \
  SESSION_UPDATE_CANONICAL_BUILD=$BUILD/baseline taskset -c 0-9 nice -n 10 bash scripts/run_session.sh --output-state $OUT \
  --session-id $NAME --run-dir $DATA/rgbd --semantic-dir $DATA/semantics --instance-dir $DATA/instance_labels \
  --session-start-ns $start --tf-settle-s .02 --play-rate 100 --image-scale 1 --finalization-timeout-s 86400 --sensor-max-range 5 \
  --ack-timeout-s 86400 --discovery-timeout-s 86400 --change-detection-every-n-backend-updates 5 \
  --mapper-config $CFG/mapper_tesse_5cm.yaml --input-config $FT/datasets/synthetic_ab/mapping_configs/input.yaml \
  --labelspace-config $CFG/label_space.yaml --physical-catalog $CFG/physical_catalog.json) > $R/logs/${NAME}_mapping.log 2>&1
echo "$(date +%H:%M) $NAME: mapped, $(grep -h TIMING_PROBE $OUT/control/logs/player.log | tail -1)"
