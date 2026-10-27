#!/usr/bin/env bash
# TESSE office, 5 m, on the object-memory-resupport branch (wt_resupport: 4898f5b with D1, the memory reading
# attribution and the snapshot cadence fix; built in wt_resupport_build): the command of the 290e110 row
# (tesse_a_5m_290e110) into tesse_a_5m_resupport; its snapshot spacing; then our 4D map as on 3RScan (eval/chain_fill,
# packed, in /dev/shm) scored by the 4D evaluation (tesse_online_eval.py ours4d, configs/khronos at 0.5 m) into
# eval/tesse_paper_qj/ours_resupport_chain4d_5m_all_m050. Nothing in here stops the run.
# usage: tesse_resupport.sh
set -eo pipefail
FT=/home/jixian/Desktop/FT; R=$FT/wt_takeover_runs; DATA=$R/tesse/data/session_a; CFG=$R/tesse/configs
OUT=$R/tesse_a_5m_resupport; E=$R/eval/tesse_paper_qj/ours_resupport_chain4d_5m_all_m050
WORK=$FT/wt_resupport/session_update_baseline; BUILD=$FT/wt_resupport_build
PY=/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python
ENV="set +u; source /opt/ros/jazzy/setup.bash; source /home/jixian/ros2_ws/install/setup.bash; export LD_LIBRARY_PATH=$FT/wt_290e110_build/canonical/install/lib:\$LD_LIBRARY_PATH; exec \"\$@\""
step() { echo "$(date +%H:%M) $*"; }
[[ -e $OUT ]] && mv $OUT $OUT.before.$(date +%m%d%H%M%S)
start=$(python3 -c "import json; print(json.load(open('$DATA/session.json'))['start_ns'])")
step "TESSE office re-support: mapping"
(cd $WORK && SESSION_UPDATE_CANONICAL_ROOT=$BUILD/canonical SESSION_UPDATE_CANONICAL_PREFIX=$BUILD/canonical/install \
  SESSION_UPDATE_CANONICAL_BUILD=$BUILD/baseline taskset -c 0-9 nice -n 10 bash scripts/run_session.sh --output-state $OUT \
  --session-id tesse_a_5m_resupport --run-dir $DATA/rgbd --semantic-dir $DATA/semantics --instance-dir $DATA/instance_labels \
  --session-start-ns $start --tf-settle-s .02 --play-rate 100 --image-scale 1 --finalization-timeout-s 86400 --sensor-max-range 5 \
  --ack-timeout-s 86400 --discovery-timeout-s 86400 --change-detection-every-n-backend-updates 5 \
  --mapper-config $CFG/mapper_tesse_5cm.yaml --input-config $FT/datasets/synthetic_ab/mapping_configs/input.yaml \
  --labelspace-config $CFG/label_space.yaml --physical-catalog $CFG/physical_catalog.json) > $R/logs/tesse_a_5m_resupport_mapping.log 2>&1
step "TESSE office re-support: mapped, $(du -h $OUT/final.4dmap.zpk | cut -f1); $($PY -c "import sys, numpy as np; sys.path.insert(0, '$FT/eval_fast/vendor/pack'); import zpk_snapshot as zs; s = np.array(zs.stamps('$OUT/final.4dmap.zpk')) / 1e9; g = np.diff(s); print(f'{len(s)} snapshots {s[0]:.1f}-{s[-1]:.1f} s, steps median {np.median(g):.2f} max {g.max():.2f} s, steps > 3 s: {(g > 3).sum()}')")"
w=/dev/shm/tesse-resupport-chain; rm -rf $w; mkdir -p $w
step "TESSE office re-support: $(bash -c "$ENV" chain-fill $FT/eval/chain_fill/chain_fill $w/chain.4dmap.zpk $OUT/final.4dmap.zpk | tail -1)"
[[ -e $E ]] && mv $E $E.before.$(date +%m%d%H%M%S)
TESSE_EVAL_JOBS=4 ionice -c3 nice -n 19 taskset -c 0-9 $PY $R/analysis/tesse_online_eval.py ours4d $w/chain.4dmap.zpk $E > $E.log 2>&1
rm -rf $w
step "TESSE office re-support 4D (0.5 m): $(grep -A1 '^Data' $E/TABLES_4D.txt | tail -1 | tr -s ' ')"
