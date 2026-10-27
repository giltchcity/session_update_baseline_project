#!/usr/bin/env bash
# TESSE, official Khronos (63faadde, zsnap build + instance bridge) fed our instances: the TESSE table's Khronos run
# (tesse/configs/khronos_uHumans2_cd5.yaml: 5 m, change detection every 5 frames; the scene's label space) with
# PackedSemanticInstanceForwarding (min_cluster_size 50, max_range 5 m, as the TESSE runs' ConnectedSemantics) on the
# instance images and physical catalog our runs use, as tesse/khronos_inst_office; into tesse/khronos_inst_SCENE.
# Then the 4D evaluation of the TESSE table (tesse_online_eval.py ours4d, configs/khronos at 0.5 m) into
# eval/tesse_{paper,apt}_qj/khronos_inst_5m_all_m050 and the shared grid against ours (290e110, 0.5 m).
# Nothing in here ever stops the run: no memory, swap, disk or progress checks, generous player timeouts.
# usage: tesse_khronos_inst.sh office|apartment
set -eo pipefail
SCENE=$1; FT=/home/jixian/Desktop/FT; R=$FT/wt_takeover_runs; T=$R/tesse
case $SCENE in
  office) TD=$T/data/session_a; TC=$T/configs; E=$R/eval/tesse_paper_qj; FLAG=();;
  apartment) TD=$T/data/session_apt; TC=$T/configs/apartment; E=$R/eval/tesse_apt_qj; FLAG=(--scene apartment);;
  *) echo "usage: $0 office|apartment" >&2; exit 2;;
esac
OUT=$T/khronos_inst_$SCENE; MAP=$OUT/state/final.4dmap.zpk
ROOT=$FT/session_update_baseline_project/session_update_baseline
PLAYER=$R/analysis/nss_flat_ros2_player_resend.py   # the player with the unacknowledged-frame resend
BUILD=$FT/build_artifacts/official_khronos_63faadde_zsnap
PY=/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python
step() { echo "$(date +%H:%M) $*"; }
[[ -e $OUT ]] && mv $OUT $OUT.before.$(date +%m%d%H%M%S)

STAGING=$OUT.incomplete.$$; CONTROL=$STAGING/control; STATE=$STAGING/state
mkdir -p $CONTROL/logs $STAGING/ros_home $STAGING/ros_log
export ROS_HOME=$STAGING/ros_home ROS_LOG_DIR=$STAGING/ros_log
set +u; source /opt/ros/jazzy/setup.bash; source /home/jixian/ros2_ws/install/setup.bash; source $BUILD/install/local_setup.bash; set -u
export LD_LIBRARY_PATH=$BUILD/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export ROS_DOMAIN_ID=57 RMW_IMPLEMENTATION=rmw_fastrtps_cpp RMW_FASTRTPS_USE_QOS_FROM_XML=1 FASTRTPS_DEFAULT_PROFILES_FILE=$ROOT/configs/fastdds_session_update_ack.xml
unset FASTDDS_DEFAULT_PROFILES_FILE || true

step "TESSE $SCENE Khronos instances: start"
taskset -c 0-9 $BUILD/adapter_inst/official_khronos_session_node \
  --config-utilities-file "$ROOT/configs/nss_flat_input.yaml@input" --config-utilities-file $T/configs/khronos_uHumans2_cd5.yaml \
  --config-utilities-file $TC/label_space.yaml \
  --config-utilities-yaml "{robot_id: 0, odom_frame: odom, robot_frame: robot_0, map_frame: map}" \
  --config-utilities-yaml "{sensor_frame: left_cam, store_visualization_details: true, experiment: {output_dir: '$STATE', overwrite: true, save_every_n_frames: 0, save_full_state: false, log_timing: false, log_timing_details: false, exit_after_clock: false}}" \
  --config-utilities-yaml "{input: {inputs: {left_cam: {sensor: {min_range: 0.1, max_range: 5.0, extrinsics: {sensor_frame: left_cam}}}}}}" \
  --config-utilities-yaml "{active_window: {object_detector: {type: PackedSemanticInstanceForwarding, min_cluster_size: 50, max_range: 5}}}" \
  --ros-args -r "~/input/left_cam/depth_registered/image_rect:=/nss/depth/image_raw" -r "~/input/left_cam/rgb/image_raw:=/nss/rgb/image_raw" \
  -r "~/input/left_cam/rgb/camera_info:=/nss/rgb/camera_info" -r "~/input/left_cam/semantic/image_raw:=/nss/semantic/image_raw" \
  > $CONTROL/logs/khronos.log 2>&1 &
MAPPER=$!
/usr/bin/python3 -u $PLAYER --run-dir $TD/rgbd --label-dir $TD/semantics --instance-dir $TD/instance_labels \
  --physical-catalog $TC/physical_catalog.json --session-start-ns $(python3 -c "import json; print(json.load(open('$TD/session.json'))['start_ns'])") \
  --tf-settle-s 0.02 --play-rate 100 --image-scale 1 --flow-control ack --ack-timeout-s 86400 --finish-timeout-s 86400 \
  --discovery-timeout-s 86400 --post-wait-s 0 --frame-limit 0 --manifest $CONTROL/playback_v0.json > $CONTROL/logs/player_v0.log 2>&1
wait $MAPPER
[[ -s $STATE/final.4dmap.zpk ]] || { step "TESSE $SCENE Khronos instances: no map saved (see $CONTROL/logs/khronos.log)"; exit 1; }
mv $STAGING $OUT
step "TESSE $SCENE Khronos instances: mapped, $(du -h $MAP | cut -f1), bridge $(grep -h INSTANCE_BRIDGE $OUT/control/logs/khronos.log | tail -1 | grep -o 'observed_instances=[0-9]*')"

TESSE_EVAL_JOBS=4 ionice -c3 nice -n 19 taskset -c 0-9 $PY $R/analysis/tesse_online_eval.py "${FLAG[@]}" ours4d $MAP $E/khronos_inst_5m_all_m050 \
  > $R/logs/tesse_khronos_inst_eval_$SCENE.log 2>&1
step "TESSE $SCENE Khronos instances 4D (0.5 m): $(grep -A1 '^Data' $E/khronos_inst_5m_all_m050/TABLES_4D.txt | tail -1 | tr -s ' ')"
$PY $R/analysis/tesse_shared_grid.py $E/khronos_inst_5m_all_m050 $E/ours_290e110_5m_all_m050 > $E/SHARED_GRID_khronos_inst_m050.txt 2>&1
cat $E/SHARED_GRID_khronos_inst_m050.txt
step "TESSE $SCENE Khronos instances: done"
