#!/usr/bin/env bash
# A Flat session on the object-memory re-support branch (wt_resupport, object-memory-resupport on 4898f5b; built in
# wt_resupport_build): the command of our Flat run of that session (analysis/tk_flat_{a,b}_run.json; B with prior
# flat_a_tk) into flat_SESSION_resupport; the refusion report in /dev/shm. Then G1 (analysis/flat_eval.py) into
# eval/g1final_flat_SESSION_resupport. Nothing in here stops the run.
# usage: flat_b_resupport.sh [a|b]   (default b)
set -eo pipefail
S=${1:-b}
FT=/home/jixian/Desktop/FT; R=$FT/wt_takeover_runs; D=$FT/datasets/flat_ab; BUILD=$FT/wt_resupport_build
OUT=$R/flat_${S}_resupport; DUMP=/dev/shm/replay_flat_${S}_resupport; PRIOR=()
[[ $S == b ]] && PRIOR=(--input-state $R/flat_a_tk)
START=$(python3 -c "import json; print(json.load(open('$D/session.json'))['$S']['start_ns'])")
step() { echo "$(date +%H:%M) $*"; }
[[ -e $OUT ]] && mv $OUT $OUT.before.$(date +%m%d%H%M%S)
step "Flat $S re-support: mapping"
(cd $FT/wt_resupport/session_update_baseline && SESSION_UPDATE_CANONICAL_ROOT=$BUILD/canonical \
  SESSION_UPDATE_CANONICAL_PREFIX=$BUILD/canonical/install SESSION_UPDATE_CANONICAL_BUILD=$BUILD/baseline \
  KHRONOS_REFUSION_DUMP=$DUMP taskset -c 0-9 nice -n 10 bash scripts/run_session.sh --output-state $OUT \
  --session-id flat_${S}_resupport --run-dir $D/session_$S/rgbd --semantic-dir $D/session_$S/semantics \
  --instance-dir $D/session_$S/instance_labels --session-start-ns $START --tf-settle-s .02 --play-rate 100 \
  --image-scale 1 --finalization-timeout-s 86400 --sensor-max-range 15 --ack-timeout-s 86400 --discovery-timeout-s 86400 \
  --change-detection-every-n-backend-updates 25 --mapper-config $R/flat/configs/mapper_flat_5cm.yaml \
  --input-config $FT/datasets/synthetic_ab/mapping_configs/input.yaml --labelspace-config $D/mapping_configs/label_space.yaml \
  --physical-catalog $R/flat/configs/physical_catalog.json "${PRIOR[@]}") > $R/logs/flat_${S}_resupport_mapping.log 2>&1
step "Flat $S re-support: mapped, $(du -h $OUT/final.4dmap.zpk | cut -f1); report $(python3 -c "import json; r=json.load(open('$DUMP/refusion_report.json')); print(json.dumps(r['memory']))")"
rm -rf $DUMP
ionice -c3 nice -n 19 taskset -c 0-9 /home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python $R/analysis/flat_eval.py \
  $S $OUT/final.4dmap.zpk $R/eval/g1final_flat_${S}_resupport | tail -1
