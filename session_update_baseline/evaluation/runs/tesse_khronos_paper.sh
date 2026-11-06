#!/usr/bin/env bash
# TESSE, official Khronos (63faadde, zsnap build) exactly as the paper's Table I runs it: the official launcher
# (khronos_ros uhumans2_khronos.launch.yaml: its bag player on the official ROS2 bag, GT semantics, GT frames, sensor
# 0.1-5 m) and the official mapper config (config/mapper/uHumans2.yaml, map_update_frequency 50) at the paper's 8 cm
# (voxel 0.08 m, truncation 0.16 m: the config's two voxels), into tesse/khronos_paper_SCENE; the map saved by the
# node's finish_mapping_and_save service once the bag has played. Then the TESSE table's 4D evaluation
# (tesse_online_eval.py ours4d, configs/khronos at 0.5 m) into eval/tesse_{paper,apt}_qj/khronos_paper_8cm_all_m050.
# usage: tesse_khronos_paper.sh office|apartment
set -eo pipefail
SCENE=$1; FT=/home/jixian/Desktop/FT; R=$FT/wt_takeover_runs; T=$R/tesse
case $SCENE in
  office) DATASET=tesse_cd_office; E=$R/eval/tesse_paper_qj; FLAG=();;
  apartment) DATASET=tesse_cd_apartment; E=$R/eval/tesse_apt_qj; FLAG=(--scene apartment);;
  *) echo "usage: $0 office|apartment" >&2; exit 2;;
esac
BUILD=$FT/build_artifacts/official_khronos_63faadde_zsnap
OUT=$T/khronos_paper_$SCENE; CFG=$T/configs/khronos_uHumans2_paper_8cm.yaml
PY=/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python
step() { echo "$(date +%H:%M:%S) $*"; }
sed -e '46s/voxel_size: 0.1 # m/voxel_size: 0.08 # m (the paper: 8 cm)/' -e '47s/truncation_distance: 0.2 # m/truncation_distance: 0.16 # m/' \
  $BUILD/install/share/khronos_ros/config/mapper/uHumans2.yaml > $CFG
grep -q 'voxel_size: 0.08' $CFG && grep -q 'truncation_distance: 0.16' $CFG
[[ -e $OUT ]] && mv $OUT $OUT.before.$(date +%m%d%H%M%S)
mkdir -p $OUT/logs
set +u; source /opt/ros/jazzy/setup.bash; source /home/jixian/ros2_ws/install/setup.bash; source $BUILD/install/local_setup.bash; set -u
export LD_LIBRARY_PATH=$BUILD/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} ROS_DOMAIN_ID=61 ROS_LOG_DIR=$OUT/ros_log
step "TESSE $SCENE Khronos paper setting: launch"
setsid ros2 launch khronos_ros uhumans2_khronos.launch.yaml dataset:=$DATASET bag_dir:=/home/jixian/datasets/khronos/tesse_cd/ros2_converted \
  output_dir:=$OUT/state mapper_config:=$CFG evaluate:=false start_visualizer:=false play_rate:=1 > $OUT/logs/launch.log 2>&1 &
LAUNCH=$!
stop() { kill -INT -$LAUNCH 2>/dev/null || true; sleep 5; kill -KILL -$LAUNCH 2>/dev/null || true; }
trap stop EXIT
sleep 20
while pgrep -f "play_rosbag.*$DATASET" > /dev/null; do sleep 5; done   # the bag has played
step "TESSE $SCENE Khronos paper setting: bag played, saving"
sleep 10
timeout 1800 ros2 service call /khronos_node/finish_mapping_and_save std_srvs/srv/Empty > $OUT/logs/save.log 2>&1
for _ in $(seq 360); do [[ -s $OUT/state/final.4dmap.zpk ]] && break; sleep 5; done
stop; trap - EXIT
[[ -s $OUT/state/final.4dmap.zpk ]] || { step "TESSE $SCENE Khronos paper setting: no map saved (see $OUT/logs)"; exit 1; }
step "TESSE $SCENE Khronos paper setting: mapped, $(du -h $OUT/state/final.4dmap.zpk | cut -f1), $($PY -c "import sys; sys.path.insert(0, '$FT/eval_fast/vendor/pack'); import zpk_snapshot as zs; print(len(zs.stamps('$OUT/state/final.4dmap.zpk')), 'snapshots')")"
TESSE_EVAL_JOBS=8 ionice -c3 nice -n 19 $PY $R/analysis/tesse_online_eval.py "${FLAG[@]}" ours4d $OUT/state/final.4dmap.zpk $E/khronos_paper_8cm_all_m050 \
  > $E/khronos_paper_8cm_all_m050.log 2>&1
step "TESSE $SCENE Khronos paper setting 4D (0.5 m): $(grep -A1 '^Data' $E/khronos_paper_8cm_all_m050/TABLES_4D.txt | tail -1 | tr -s ' ')"
