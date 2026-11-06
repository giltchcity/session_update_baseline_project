#!/usr/bin/env bash
# Official Khronos (MIT-SPARK/Khronos@63faadde) over consecutive sessions
# in ONE mapper process: real A -> B -> C, Flat A -> B or synthetic A -> B, each session played at its own acquisition
# stamps (real B, C in A's frame through their alignment files). Khronos cannot load an earlier map; kept running, it carries the map
# across sessions itself. Mapper, input and thresholds exactly as the single-session launchers
# (scripts/run_official_khronos_matched_session.sh, analysis/official_khronos_flat.sh with 5 cm objects as its khronos5
# runs, synthetic_ab/evaluation_complete_20260917/run_official_session.sh);
# only the sessions before the last end on a segment topic that segment_relay.py subscribes and acknowledges, so
# Khronos saves once, after the last.
# Every dataset runs on build_artifacts/official_khronos_63faadde_zsnap: the official source with its 4D-map snapshots
# held zstd-compressed and saved packed (uncompressed, real A -> B alone took 22 GB); synthetic through its observer
# rebuilt against it (analysis/build_khronos_observer_zsnap.sh; until 10-31 on official_khronos_63faadde_abc, the
# build the published observer is linked against). Synthetic at 5 cm, truncation 0.15 m, as our maps (user 10-31,
# every method at the same resolution; until then 10 cm, 0.3 m: khronos_continuous/synthetic).
# 3rscan: a 3RScan room's visits in order (THREERSCAN_ROOM, datasets/3rscan/mapping_configs/ROOM/chain.json), rescans in
# the reference frame through their transforms, visit k at 1 + 200 k s as our chained runs; the thresholds of the main
# table's 3RScan Khronos runs (the real ones with min_cluster_size 1778 for the 960x540 frames); its label space with
# the room's thing classes as object classes (p3_khronos_label_space.py: mapping_configs/ROOM/khronos_label_space.yaml).
# KHRONOS_INSTANCES=1 (3rscan, real): the instance control (real: KHRONOS_INSTANCE_DIR picks the instance images,
# default instance_labels; KHRONOS_TRACKER=external uses upstream ExternalTracker, min_num_observations 15). Khronos gets the ground-truth instances our runs get: the player
# publishes packed semantic << 16 | instance labels (--instance-dir, --physical-catalog) and the object detector is
# PackedSemanticInstanceForwarding (upstream InstanceForwarding behind the synthetic runs' input-format bridge, adapter
# built from adapter_inst_src), min_cluster_size 50 as the synthetic runs, max_range 5 m.
# tesse: the TESSE instance control (TESSE_SCENE=office|apartment): our player on tesse/data/session_a|session_apt
# with the instance images of analysis/tesse_instances.py (the ones our runs use) and its physical catalog, through
# the same bridge (PackedSemanticInstanceForwarding, min_cluster_size 50 and max_range 5 m as the TESSE runs'
# ConnectedSemantics); otherwise the TESSE table's run: tesse/configs/khronos_uHumans2_cd5.yaml (5 m, change detection
# every 5 frames), the scene's label space.
# usage: [KHRONOS_SESSIONS="a b"] [THREERSCAN_ROOM=ROOM] [TESSE_SCENE=SCENE] khronos_continuous.sh real|flat|synthetic|3rscan|tesse OUTPUT_DIR [FRAME_LIMIT per session, 0 = all]
set -eo pipefail
DATASET="${1:-}"; OUTPUT="${2:-}"; FRAME_LIMIT="${3:-0}"
[[ "$DATASET" == real || "$DATASET" == flat || "$DATASET" == synthetic || "$DATASET" == 3rscan || "$DATASET" == tesse ]] && [[ -n "$OUTPUT" && ! -e "$OUTPUT" ]] \
  || { echo "usage: $0 real|flat|synthetic|3rscan|tesse NEW_OUTPUT_DIR" >&2; exit 2; }
ROOT=/home/jixian/Desktop/FT/session_update_baseline_project/session_update_baseline
PLAYER=/home/jixian/Desktop/FT/wt_resupport/session_update_baseline/scripts/nss_flat_ros2_player.py   # 6b6f7fa: waits never negative
FASTRTPS_PROFILE="$ROOT/configs/fastdds_session_update_ack.xml"
L=/home/jixian/Desktop/FT/datasets/local_ab; S=/home/jixian/Desktop/FT/datasets/synthetic_ab; F=/home/jixian/Desktop/FT/datasets/flat_ab
matched_thresholds() {  # matched_thresholds OBJECT_RESOLUTION: the real/Flat matched protocol
  echo "{active_window: {min_output_separation: 0.4, volumetric_map: {voxel_size: 0.05, truncation_distance: 0.15}, projective_integrator: {interpolation_method: {max_depth_difference_m: 0.05}}, tracking_integrator: {temporal_buffer: 2.0}, motion_detector: {min_cluster_size: 3000}, object_detector: {grid_size: 0.05}, object_extractor: {object_reconstruction_resolution: $1, projective_integrator: {num_threads: 16}}, frame_data_buffer: {max_buffer_size: 100, store_every_n_frames: 3}}, frontend: {pgmo: {mesh_resolution: 0.005, time_horizon: 15.0, d_graph_resolution: 2.5}}, backend: {type: Backend, fix_input_poses: true, change_detection: {run_every_n_frames: 5, ray_verificator: {radial_tolerance: 0.08, depth_tolerance: 0.3, active_window_duration: 3.0}, ray_change_detector: {temporal_resolution: 5.0, window_size: 5, use_relative_confidence: true, absence_confidence: 0.6, presence_confidence: 0.5}, objects: {time_filtering_threshold: 3.0}}}}"
}
if [[ $DATASET == real ]]; then
  OFFICIAL_BUILD=/home/jixian/Desktop/FT/build_artifacts/official_khronos_63faadde_zsnap; MAP_FILE=final.4dmap.zpk
  ADAPTER="$OFFICIAL_BUILD/adapter/official_khronos_session_node"
  INPUT_CONFIG="$ROOT/configs/nss_flat_input.yaml"; LABELSPACE_CONFIG="$ROOT/configs/nss_ade20k_room_label_space.yaml"
  MAX_RANGE=5.0; EXTRA_FILE=""
  COMMON_THRESHOLDS=$(matched_thresholds 0.02)
  SESSIONS=(a b c)
  if [[ ${KHRONOS_INSTANCES:-0} == 1 ]]; then        # instance control, as the 3rscan branch: our instances and catalog
    ADAPTER="$OFFICIAL_BUILD/adapter_inst/official_khronos_session_node"
    COMMON_THRESHOLDS=$(echo "$COMMON_THRESHOLDS" | sed 's/object_detector: {grid_size: 0.05}/object_detector: {type: PackedSemanticInstanceForwarding, min_cluster_size: 50, max_range: 5}/')
    [[ $COMMON_THRESHOLDS == *PackedSemanticInstanceForwarding* ]] || { echo "instance detector not set" >&2; exit 2; }
    REAL_INSTANCES="--physical-catalog $L/mapping_configs/room18_physical_catalog.json"
    REAL_INSTANCE_DIR=${KHRONOS_INSTANCE_DIR:-$L/instance_labels}     # e.g. $L/instance_labels_validdepth
    if [[ ${KHRONOS_TRACKER:-} == external ]]; then                    # official ExternalTracker, as khronos_single_5cm_20260909
      COMMON_THRESHOLDS=$(echo "$COMMON_THRESHOLDS" | sed 's/^{active_window: {/{active_window: {tracker: {type: ExternalTracker, min_num_observations: 15}, /')
      [[ $COMMON_THRESHOLDS == *ExternalTracker* ]] || { echo "tracker not set" >&2; exit 2; }
    fi
  fi
  run_dir()   { case $1 in a) echo $L/rgbd/session_a_20260809_204010_201_flat_rgbd_30hz_1080p;; b) echo $L/rgbd/session_b_20260810_030502_620_flat_rgbd_30hz_1080p;; c) echo $L/rgbd/session_c_20260822_182703_387_flat_rgbd_30hz_1080p;; esac; }
  start_ns()  { case $1 in a) echo 1786279210201000000;; b) echo 1786302302620000000;; c) echo 1787394423387000000;; esac; }
  player_args() {  # player_args SESSION
    local w=""; [[ $1 != a ]] && w="--world-transform $L/alignment/session_$1_to_session_a.txt"
    [[ -n ${REAL_INSTANCES:-} ]] && w="$w --instance-dir $REAL_INSTANCE_DIR/session_$1 $REAL_INSTANCES"
    echo "$w --run-dir $(run_dir $1) --label-dir $L/semantics/session_$1 --session-start-ns $(start_ns $1) --tf-settle-s 0.02 --play-rate 100 --image-scale 0.5 --flow-control ack --ack-timeout-s 180 --finish-timeout-s 1800 --discovery-timeout-s 180 --post-wait-s 0 --frame-limit $FRAME_LIMIT"
  }
elif [[ $DATASET == 3rscan ]]; then
  OFFICIAL_BUILD=/home/jixian/Desktop/FT/build_artifacts/official_khronos_63faadde_zsnap; MAP_FILE=final.4dmap.zpk
  ADAPTER="$OFFICIAL_BUILD/adapter/official_khronos_session_node"
  D3=/home/jixian/Desktop/FT/datasets/3rscan; C3=$D3/mapping_configs/${THREERSCAN_ROOM:?THREERSCAN_ROOM}
  INPUT_CONFIG="$ROOT/configs/nss_flat_input.yaml"; LABELSPACE_CONFIG="$C3/khronos_label_space.yaml"
  MAX_RANGE=5.0; EXTRA_FILE=""
  COMMON_THRESHOLDS=$(matched_thresholds 0.02 | sed 's/min_cluster_size: 3000/min_cluster_size: 1778/')
  INSTANCES=""
  if [[ ${KHRONOS_INSTANCES:-0} == 1 ]]; then
    ADAPTER="$OFFICIAL_BUILD/adapter_inst/official_khronos_session_node"
    COMMON_THRESHOLDS=$(echo "$COMMON_THRESHOLDS" | sed 's/object_detector: {grid_size: 0.05}/object_detector: {type: PackedSemanticInstanceForwarding, min_cluster_size: 50, max_range: 5}/')
    [[ $COMMON_THRESHOLDS == *PackedSemanticInstanceForwarding* ]] || { echo "instance detector not set" >&2; exit 2; }
    INSTANCES=1
    if [[ ${KHRONOS_TRACKER:-} == external ]]; then   # official ExternalTracker, as the real branch
      COMMON_THRESHOLDS=$(echo "$COMMON_THRESHOLDS" | sed 's/^{active_window: {/{active_window: {tracker: {type: ExternalTracker, min_num_observations: 15}, /')
      [[ $COMMON_THRESHOLDS == *ExternalTracker* ]] || { echo "tracker not set" >&2; exit 2; }
    fi
  fi
  read -r -a VISITS <<< "$(python3 -c "import json; print(' '.join(json.load(open('$C3/chain.json'))['visits']))")"
  SESSIONS=(); for k in "${!VISITS[@]}"; do SESSIONS+=(v$k); done
  player_args() {  # player_args vK
    local k=${1#v} scan w=""; scan=${VISITS[$k]}; [[ $k != 0 ]] && w="--world-transform $C3/world_transform_${scan}_to_${VISITS[0]}.txt"
    [[ -n $INSTANCES ]] && w="$w --instance-dir $D3/sessions/$scan/instance_labels --physical-catalog $C3/physical_catalog.json"
    echo "$w --run-dir $D3/sessions/$scan/rgbd --label-dir $D3/sessions/$scan/semantics --session-start-ns $((1000000000 + 200000000000 * k)) --tf-settle-s 0.02 --play-rate 100 --image-scale 1 --flow-control ack --ack-timeout-s 180 --finish-timeout-s 1800 --discovery-timeout-s 180 --post-wait-s 0 --frame-limit $FRAME_LIMIT"
  }
elif [[ $DATASET == tesse ]]; then
  OFFICIAL_BUILD=/home/jixian/Desktop/FT/build_artifacts/official_khronos_63faadde_zsnap; MAP_FILE=final.4dmap.zpk
  ADAPTER="$OFFICIAL_BUILD/adapter_inst/official_khronos_session_node"
  T=/home/jixian/Desktop/FT/wt_takeover_runs/tesse
  case ${TESSE_SCENE:-office} in office) TD=$T/data/session_a; TC=$T/configs;; apartment) TD=$T/data/session_apt; TC=$T/configs/apartment;; *) exit 2;; esac
  INPUT_CONFIG="$ROOT/configs/nss_flat_input.yaml"; LABELSPACE_CONFIG="$TC/label_space.yaml"
  MAPPER_CONFIG="$T/configs/khronos_uHumans2_cd5.yaml"
  MAX_RANGE=5.0; EXTRA_FILE=""
  COMMON_THRESHOLDS="{active_window: {object_detector: {type: PackedSemanticInstanceForwarding, min_cluster_size: 50, max_range: 5}}}"
  SESSIONS=(v0)
  player_args() {
    echo "--run-dir $TD/rgbd --label-dir $TD/semantics --instance-dir $TD/instance_labels --physical-catalog $TC/physical_catalog.json --session-start-ns $(python3 -c "import json; print(json.load(open('$TD/session.json'))['start_ns'])") --tf-settle-s 0.02 --play-rate 100 --image-scale 1 --flow-control ack --ack-timeout-s 300 --finish-timeout-s 1800 --discovery-timeout-s 180 --post-wait-s 0 --frame-limit $FRAME_LIMIT"
  }
elif [[ $DATASET == flat ]]; then
  OFFICIAL_BUILD=/home/jixian/Desktop/FT/build_artifacts/official_khronos_63faadde_zsnap; MAP_FILE=final.4dmap.zpk
  ADAPTER="$OFFICIAL_BUILD/adapter/official_khronos_session_node"
  INPUT_CONFIG="$S/mapping_configs/input.yaml"; LABELSPACE_CONFIG="$F/mapping_configs/label_space.yaml"
  MAX_RANGE=15.0; EXTRA_FILE=""
  COMMON_THRESHOLDS=$(matched_thresholds 0.05)
  SESSIONS=(a b)
  start_ns() { python3 -c "import json, sys; print(json.load(open('$F/session.json'))[sys.argv[1]]['start_ns'])" $1; }
  player_args() {  # the semantic-only player publishes mono8: the 8-bit label copy
    echo "--run-dir $F/session_$1/rgbd --label-dir /home/jixian/Desktop/FT/wt_takeover_runs/flat/semantics8/session_$1 --session-start-ns $(start_ns $1) --tf-settle-s 0.02 --play-rate 100 --image-scale 1 --flow-control ack --ack-timeout-s 180 --finish-timeout-s 1800 --discovery-timeout-s 180 --post-wait-s 0 --frame-limit $FRAME_LIMIT"
  }
else
  OFFICIAL_BUILD=/home/jixian/Desktop/FT/build_artifacts/official_khronos_63faadde_zsnap; MAP_FILE=final.4dmap.zpk
  ADAPTER="$OFFICIAL_BUILD/synthetic_observer/khronos_observer"
  INPUT_CONFIG="$S/mapping_configs/input.yaml"; LABELSPACE_CONFIG="$S/mapping_configs/label_space.yaml"
  MAX_RANGE=15.0; EXTRA_FILE="$S/evaluation_complete_20260917/khronos_background_ray_overlay.yaml"
  COMMON_THRESHOLDS="{active_window: {min_output_separation: 0.4, volumetric_map: {voxel_size: 0.05, truncation_distance: 0.15}, projective_integrator: {interpolation_method: {max_depth_difference_m: 0.05}}, tracking_integrator: {temporal_buffer: 2.0}, motion_detector: {min_cluster_size: 3000, max_range: 15}, object_detector: {type: PackedSemanticInstanceForwarding, min_cluster_size: 50, max_range: 15}, object_extractor: {object_reconstruction_resolution: 0.05, projective_integrator: {num_threads: 16}}, frame_data_buffer: {max_buffer_size: 100, store_every_n_frames: 3}}, frontend: {pgmo: {mesh_resolution: 0.005, time_horizon: 15.0, d_graph_resolution: 2.5}}, backend: {type: Backend, fix_input_poses: true, change_detection: {run_every_n_frames: 25, ray_verificator: {radial_tolerance: 0.08, depth_tolerance: 0.3, active_window_duration: 3.0}, ray_change_detector: {temporal_resolution: 5.0, window_size: 5, use_relative_confidence: true, absence_confidence: 0.6, presence_confidence: 0.5}, objects: {time_filtering_threshold: 3.0}}}}"
  SESSIONS=(a b)
  start_ns() { case $1 in a) echo 1000000000;; b) echo 141000000000;; esac; }
  player_args() {
    echo "--instance-dir $S/session_$1/instance_labels --physical-catalog $S/mapping_configs/physical_catalog.json --run-dir $S/session_$1/rgbd --label-dir $S/session_$1/semantics --session-start-ns $(start_ns $1) --tf-settle-s 0.02 --play-rate 100 --image-scale 1 --flow-control ack --ack-timeout-s 300 --finish-timeout-s 1800 --discovery-timeout-s 180 --post-wait-s 0 --frame-limit $FRAME_LIMIT"
  }
fi
[[ -n "${KHRONOS_SESSIONS:-}" ]] && read -r -a SESSIONS <<< "$KHRONOS_SESSIONS"
MAPPER_CONFIG="${MAPPER_CONFIG:-$OFFICIAL_BUILD/src/khronos_ros/config/mapper/uHumans2.yaml}"
STAGING="$OUTPUT.incomplete.$$"; CONTROL="$STAGING/control"; STATE="$STAGING/state"
mkdir -p "$CONTROL/logs" "$STAGING/ros_home" "$STAGING/ros_log"
export ROS_HOME="$STAGING/ros_home" ROS_LOG_DIR="$STAGING/ros_log"
set +u; source /opt/ros/jazzy/setup.bash; source /home/jixian/ros2_ws/install/setup.bash; source "$OFFICIAL_BUILD/install/local_setup.bash"; set -u
export LD_LIBRARY_PATH="$OFFICIAL_BUILD/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ $DATASET == synthetic ]]; then
  # The observer's live tracks (read by the synthetic D1 evaluation), one file over all sessions.
  export SYN_EVAL_OUT="$STAGING/live"; mkdir -p "$SYN_EVAL_OUT"
fi
export ROS_DOMAIN_ID=57 RMW_IMPLEMENTATION=rmw_fastrtps_cpp RMW_FASTRTPS_USE_QOS_FROM_XML=1 FASTRTPS_DEFAULT_PROFILES_FILE="$FASTRTPS_PROFILE"
unset FASTDDS_DEFAULT_PROFILES_FILE || true
EXTRA=(); [[ -n "$EXTRA_FILE" ]] && EXTRA=(--config-utilities-file "$EXTRA_FILE")
# Mapper, ACK relay and players in one process group, stopped as a whole on any exit.
"$ADAPTER" \
  --config-utilities-file "$INPUT_CONFIG@input" --config-utilities-file "$MAPPER_CONFIG" --config-utilities-file "$LABELSPACE_CONFIG" \
  --config-utilities-yaml "{robot_id: 0, odom_frame: odom, robot_frame: robot_0, map_frame: map}" \
  --config-utilities-yaml "{sensor_frame: left_cam, store_visualization_details: true, experiment: {output_dir: '$STATE', overwrite: true, save_every_n_frames: 0, save_full_state: false, log_timing: false, log_timing_details: false, exit_after_clock: false}}" \
  --config-utilities-yaml "{input: {inputs: {left_cam: {sensor: {min_range: 0.1, max_range: $MAX_RANGE, extrinsics: {sensor_frame: left_cam}}}}}}" \
  --config-utilities-yaml "$COMMON_THRESHOLDS" "${EXTRA[@]}" \
  --ros-args -r "~/input/left_cam/depth_registered/image_rect:=/nss/depth/image_raw" -r "~/input/left_cam/rgb/image_raw:=/nss/rgb/image_raw" \
  -r "~/input/left_cam/rgb/camera_info:=/nss/rgb/camera_info" -r "~/input/left_cam/semantic/image_raw:=/nss/semantic/image_raw" \
  > "$CONTROL/logs/khronos.log" 2>&1 &
MAPPER=$!
RELAY=""
stop() { kill -TERM $MAPPER $RELAY 2>/dev/null || true; wait $MAPPER 2>/dev/null || true; }
trap stop EXIT INT TERM
last=${SESSIONS[-1]}
for s in "${SESSIONS[@]}"; do
  FINISH=()
  if [[ $s != $last ]]; then
    /usr/bin/python3 -u /home/jixian/Desktop/FT/wt_takeover_runs/analysis/segment_relay.py /khronos_continuous/segment_end \
      /khronos_continuous/segment_saved > "$CONTROL/logs/relay_$s.log" 2>&1 & RELAY=$!
    FINISH=(--finish-topic /khronos_continuous/segment_end --finish-ack-topic /khronos_continuous/segment_saved)
  fi
  echo "$(date +%H:%M) play $s"
  /usr/bin/python3 -u "$PLAYER" $(player_args $s) "${FINISH[@]}" --manifest "$CONTROL/playback_$s.json" > "$CONTROL/logs/player_$s.log" 2>&1
  # the relay can ignore TERM after the segment (rclpy spin): 10 s, then KILL, so the next visit is never blocked
  [[ -n "$RELAY" ]] && { kill -TERM $RELAY 2>/dev/null || true; for _ in $(seq 10); do kill -0 $RELAY 2>/dev/null || break; sleep 1; done;
    kill -KILL $RELAY 2>/dev/null || true; wait $RELAY 2>/dev/null || true; RELAY=""; }
  echo "$(date +%H:%M) played $s: $(grep -c PLAYBACK_PROGRESS "$CONTROL/logs/player_$s.log") progress lines"
done
wait $MAPPER; trap - EXIT INT TERM
[[ -s "$STATE/$MAP_FILE" ]] || { echo "missing $MAP_FILE" >&2; exit 5; }
mv "$STAGING" "$OUTPUT"
echo "$(date +%H:%M) KHRONOS_CONTINUOUS_COMPLETE $DATASET map=$OUTPUT/state/$MAP_FILE"
