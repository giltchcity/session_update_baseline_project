#!/usr/bin/env bash
# Official Khronos (MIT-SPARK/Khronos@63faadde) on the Flat dataset (datasets/flat_ab), one session from scratch, with
# the same protocol as the real A/B/C per-visit baseline runs: the official adapter, uHumans2 mapper config with the
# common threshold overrides (5 cm background, 2 cm objects), semantic labels only. Only the dataset paths, label space,
# sensor range (15 m, as our Flat run) and image scale (1) differ. OBJECT_RESOLUTION (default 0.02, the matched protocol)
# sets the object reconstruction voxel. The zsnap build (the same official source, its 4D-map snapshots held
# zstd-compressed and saved packed: final.4dmap.zpk) and the player that re-sends unacknowledged frames.
# Then G1 of the session's final map (analysis/flat_eval.py) into eval/g1final_khronos_flat_SESSION.
# usage: [OBJECT_RESOLUTION=0.05] official_khronos_flat.sh a|b OUTPUT_DIR
set -eo pipefail
SESSION="${1:-}"; OUTPUT="${2:-}"
[[ "$SESSION" == a || "$SESSION" == b ]] && [[ -n "$OUTPUT" ]] || { echo "usage: $0 a|b OUTPUT_DIR" >&2; exit 2; }
[[ ! -e "$OUTPUT" ]] || { echo "output already exists: $OUTPUT" >&2; exit 3; }
FT=/home/jixian/Desktop/FT; R=$FT/wt_takeover_runs
OFFICIAL_BUILD=$FT/build_artifacts/official_khronos_63faadde_zsnap
ADAPTER=$OFFICIAL_BUILD/adapter/official_khronos_session_node
MAPPER_CONFIG=$OFFICIAL_BUILD/src/khronos_ros/config/mapper/uHumans2.yaml
CANON=$FT/wt_takeover/session_update_baseline
INPUT_CONFIG=$FT/datasets/synthetic_ab/mapping_configs/input.yaml
LABELSPACE_CONFIG=$FT/datasets/flat_ab/mapping_configs/label_space.yaml
FASTRTPS_PROFILE=$CANON/configs/fastdds_session_update_ack.xml
PLAYER=$R/analysis/nss_flat_ros2_player_resend.py
RUN_DIR=$FT/datasets/flat_ab/session_$SESSION/rgbd
SEMANTIC_DIR=$R/flat/semantics8/session_$SESSION   # 8-bit copy: the semantic-only player publishes mono8
SESSION_START_NS=$(python3 -c "import json; print(json.load(open('$FT/datasets/flat_ab/session.json'))['$SESSION']['start_ns'])")
STAGING="$OUTPUT.incomplete.$$"; CONTROL="$STAGING/control"; STATE="$STAGING/state"
mkdir -p "$CONTROL/logs" "$STAGING/ros_home" "$STAGING/ros_log"
export ROS_HOME="$STAGING/ros_home" ROS_LOG_DIR="$STAGING/ros_log"
set +u
source /opt/ros/jazzy/setup.bash
source /home/jixian/ros2_ws/install/setup.bash
source "$OFFICIAL_BUILD/install/local_setup.bash"
set -u
export LD_LIBRARY_PATH="$OFFICIAL_BUILD/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ROS_DOMAIN_ID="${OFFICIAL_ROS_DOMAIN_ID:-48}" RMW_IMPLEMENTATION=rmw_fastrtps_cpp RMW_FASTRTPS_USE_QOS_FROM_XML=1
export FASTRTPS_DEFAULT_PROFILES_FILE="$FASTRTPS_PROFILE"; unset FASTDDS_DEFAULT_PROFILES_FILE || true
COMMON_THRESHOLDS="{active_window: {min_output_separation: 0.4, volumetric_map: {voxel_size: 0.05, truncation_distance: 0.15}, projective_integrator: {interpolation_method: {max_depth_difference_m: 0.05}}, tracking_integrator: {temporal_buffer: 2.0}, motion_detector: {min_cluster_size: 3000}, object_detector: {grid_size: 0.05}, object_extractor: {object_reconstruction_resolution: ${OBJECT_RESOLUTION:-0.02}, projective_integrator: {num_threads: 16}}, frame_data_buffer: {max_buffer_size: 100, store_every_n_frames: 3}}, frontend: {pgmo: {mesh_resolution: 0.005, time_horizon: 15.0, d_graph_resolution: 2.5}}, backend: {type: Backend, fix_input_poses: true, change_detection: {run_every_n_frames: 5, ray_verificator: {radial_tolerance: 0.08, depth_tolerance: 0.3, active_window_duration: 3.0}, ray_change_detector: {temporal_resolution: 5.0, window_size: 5, use_relative_confidence: true, absence_confidence: 0.6, presence_confidence: 0.5}, objects: {time_filtering_threshold: 3.0}}}}"
"$ADAPTER" \
  --config-utilities-file "$INPUT_CONFIG@input" \
  --config-utilities-file "$MAPPER_CONFIG" \
  --config-utilities-file "$LABELSPACE_CONFIG" \
  --config-utilities-yaml "{robot_id: 0, odom_frame: odom, robot_frame: robot_0, map_frame: map}" \
  --config-utilities-yaml "{sensor_frame: left_cam, store_visualization_details: true, experiment: {output_dir: '$STATE', overwrite: true, save_every_n_frames: 0, save_full_state: false, log_timing: false, log_timing_details: false, exit_after_clock: false}}" \
  --config-utilities-yaml "{input: {inputs: {left_cam: {sensor: {min_range: 0.1, max_range: 15.0, extrinsics: {sensor_frame: left_cam}}}}}}" \
  --config-utilities-yaml "$COMMON_THRESHOLDS" \
  --ros-args \
  -r "~/input/left_cam/depth_registered/image_rect:=/nss/depth/image_raw" \
  -r "~/input/left_cam/rgb/image_raw:=/nss/rgb/image_raw" \
  -r "~/input/left_cam/rgb/camera_info:=/nss/rgb/camera_info" \
  -r "~/input/left_cam/semantic/image_raw:=/nss/semantic/image_raw" \
  >"$CONTROL/logs/khronos.log" 2>&1 &
MAPPER_PID=$!
trap 'kill -TERM $MAPPER_PID 2>/dev/null || true' EXIT INT TERM
/usr/bin/python3 -u "$PLAYER" --run-dir "$RUN_DIR" --label-dir "$SEMANTIC_DIR" \
  --session-start-ns "$SESSION_START_NS" --tf-settle-s 0.02 --play-rate 100 --image-scale 1 --flow-control ack \
  --ack-timeout-s 86400 --finish-timeout-s 86400 --discovery-timeout-s 86400 --post-wait-s 0 --frame-limit 0 \
  --manifest "$CONTROL/playback_manifest.json" >"$CONTROL/logs/player.log" 2>&1
wait "$MAPPER_PID"
trap - EXIT INT TERM
[[ -s "$STATE/final.4dmap.zpk" ]] || { echo "missing final.4dmap.zpk" >&2; exit 5; }
mv "$STAGING" "$OUTPUT"
echo "$(date +%H:%M) OFFICIAL_FLAT_COMPLETE session=$SESSION map=$OUTPUT/state/final.4dmap.zpk"
ionice -c3 nice -n 19 taskset -c 0-9 /home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python $R/analysis/flat_eval.py \
  $SESSION $OUTPUT/state/final.4dmap.zpk $R/eval/g1final_khronos_flat_$SESSION | tail -1
