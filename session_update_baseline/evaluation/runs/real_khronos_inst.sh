#!/usr/bin/env bash
# Real room, official Khronos (63faadde, zsnap build + instance bridge) fed our instances (valid-depth instance images,
# PackedSemanticInstanceForwarding, min_cluster_size 50, max_range 5 m), continuous A -> B -> C in one mapper process;
# then the three visits scored in parallel with eval_fast (B against A, C against B as the prior).
# Nothing in here ever stops the run: no memory, swap, disk or progress checks, generous player timeouts.
# usage: real_khronos_inst.sh maxiou|external
set -eo pipefail
TRACKER=$1; [[ $TRACKER == maxiou || $TRACKER == external ]] || { echo "usage: $0 maxiou|external" >&2; exit 2; }
FT=/home/jixian/Desktop/FT; R=$FT/wt_takeover_runs; L=$FT/datasets/local_ab
TAG=real_inst_$TRACKER; OUT=$R/khronos_continuous/$TAG; MAP=$OUT/state/final.4dmap.zpk
ROOT=$FT/session_update_baseline_project/session_update_baseline
PLAYER=$R/analysis/nss_flat_ros2_player_resend.py   # the player with the unacknowledged-frame resend
BUILD=$FT/build_artifacts/official_khronos_63faadde_zsnap
step() { echo "$(date +%H:%M) $*"; }
[[ -e $OUT ]] && mv $OUT $OUT.before.$(date +%m%d%H%M%S)

# thresholds of the semantic continuous run, with the instance detector (and ExternalTracker if asked)
TH="{active_window: {min_output_separation: 0.4, volumetric_map: {voxel_size: 0.05, truncation_distance: 0.15}, projective_integrator: {interpolation_method: {max_depth_difference_m: 0.05}}, tracking_integrator: {temporal_buffer: 2.0}, motion_detector: {min_cluster_size: 3000}, object_detector: {type: PackedSemanticInstanceForwarding, min_cluster_size: 50, max_range: 5}, object_extractor: {object_reconstruction_resolution: 0.02, projective_integrator: {num_threads: 16}}, frame_data_buffer: {max_buffer_size: 100, store_every_n_frames: 3}}, frontend: {pgmo: {mesh_resolution: 0.005, time_horizon: 15.0, d_graph_resolution: 2.5}}, backend: {type: Backend, fix_input_poses: true, change_detection: {run_every_n_frames: 5, ray_verificator: {radial_tolerance: 0.08, depth_tolerance: 0.3, active_window_duration: 3.0}, ray_change_detector: {temporal_resolution: 5.0, window_size: 5, use_relative_confidence: true, absence_confidence: 0.6, presence_confidence: 0.5}, objects: {time_filtering_threshold: 3.0}}}}"
if [[ $TRACKER == external ]]; then
  TH=$(echo "$TH" | sed 's/^{active_window: {/{active_window: {tracker: {type: ExternalTracker, min_num_observations: 15}, /')
fi

run_dir()  { case $1 in a) echo $L/rgbd/session_a_20260809_204010_201_flat_rgbd_30hz_1080p;; b) echo $L/rgbd/session_b_20260810_030502_620_flat_rgbd_30hz_1080p;; c) echo $L/rgbd/session_c_20260822_182703_387_flat_rgbd_30hz_1080p;; esac; }
start_ns() { case $1 in a) echo 1786279210201000000;; b) echo 1786302302620000000;; c) echo 1787394423387000000;; esac; }
player_args() {
  local w=""; [[ $1 != a ]] && w="--world-transform $L/alignment/session_$1_to_session_a.txt"
  echo "$w --instance-dir $L/instance_labels_validdepth/session_$1 --physical-catalog $L/mapping_configs/room18_physical_catalog.json --run-dir $(run_dir $1) --label-dir $L/semantics/session_$1 --session-start-ns $(start_ns $1) --tf-settle-s 0.02 --play-rate 100 --image-scale 0.5 --flow-control ack --ack-timeout-s 86400 --finish-timeout-s 86400 --discovery-timeout-s 86400 --post-wait-s 0 --frame-limit 0"
}

STAGING=$OUT.incomplete.$$; CONTROL=$STAGING/control; STATE=$STAGING/state
mkdir -p $CONTROL/logs $STAGING/ros_home $STAGING/ros_log
export ROS_HOME=$STAGING/ros_home ROS_LOG_DIR=$STAGING/ros_log
set +u; source /opt/ros/jazzy/setup.bash; source /home/jixian/ros2_ws/install/setup.bash; source $BUILD/install/local_setup.bash; set -u
export LD_LIBRARY_PATH=$BUILD/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export ROS_DOMAIN_ID=57 RMW_IMPLEMENTATION=rmw_fastrtps_cpp RMW_FASTRTPS_USE_QOS_FROM_XML=1 FASTRTPS_DEFAULT_PROFILES_FILE=$ROOT/configs/fastdds_session_update_ack.xml
unset FASTDDS_DEFAULT_PROFILES_FILE || true

step "$TAG: start"
taskset -c 0-9 $BUILD/adapter_inst/official_khronos_session_node \
  --config-utilities-file "$ROOT/configs/nss_flat_input.yaml@input" --config-utilities-file $BUILD/src/khronos_ros/config/mapper/uHumans2.yaml \
  --config-utilities-file $ROOT/configs/nss_ade20k_room_label_space.yaml \
  --config-utilities-yaml "{robot_id: 0, odom_frame: odom, robot_frame: robot_0, map_frame: map}" \
  --config-utilities-yaml "{sensor_frame: left_cam, store_visualization_details: true, experiment: {output_dir: '$STATE', overwrite: true, save_every_n_frames: 0, save_full_state: false, log_timing: false, log_timing_details: false, exit_after_clock: false}}" \
  --config-utilities-yaml "{input: {inputs: {left_cam: {sensor: {min_range: 0.1, max_range: 5.0, extrinsics: {sensor_frame: left_cam}}}}}}" \
  --config-utilities-yaml "$TH" \
  --ros-args -r "~/input/left_cam/depth_registered/image_rect:=/nss/depth/image_raw" -r "~/input/left_cam/rgb/image_raw:=/nss/rgb/image_raw" \
  -r "~/input/left_cam/rgb/camera_info:=/nss/rgb/camera_info" -r "~/input/left_cam/semantic/image_raw:=/nss/semantic/image_raw" \
  > $CONTROL/logs/khronos.log 2>&1 &
MAPPER=$!
for s in a b c; do
  FINISH=(); RELAY=""
  if [[ $s != c ]]; then   # A and B end on a segment topic the relay acknowledges, so Khronos saves once, after C
    /usr/bin/python3 -u $R/analysis/segment_relay.py /khronos_continuous/segment_end /khronos_continuous/segment_saved > $CONTROL/logs/relay_$s.log 2>&1 & RELAY=$!
    FINISH=(--finish-topic /khronos_continuous/segment_end --finish-ack-topic /khronos_continuous/segment_saved)
  fi
  step "$TAG: play $s"
  /usr/bin/python3 -u $PLAYER $(player_args $s) "${FINISH[@]}" --manifest $CONTROL/playback_$s.json > $CONTROL/logs/player_$s.log 2>&1
  # the relay's job is done once the visit is played; it does not exit by itself
  if [[ -n $RELAY ]]; then kill -TERM $RELAY 2>/dev/null || true; sleep 10; kill -KILL $RELAY 2>/dev/null || true; wait $RELAY 2>/dev/null || true; fi
done
wait $MAPPER
[[ -s $STATE/final.4dmap.zpk ]] || { step "$TAG: no map saved (see $CONTROL/logs/khronos.log)"; exit 1; }
mv $STAGING $OUT
step "$TAG: mapped, $(du -h $MAP | cut -f1)"

# the three visits scored in parallel
python3 $R/analysis/khronos_continuous_slices.py $MAP --moments real > $R/logs/khronos_continuous_moments_$TAG.txt
prev=""
while read -r s first last; do
  prior=(); [[ -n $prev ]] && prior=(--prior $MAP)
  (cd $FT && SESSION_UPDATE_CANONICAL_PREFIX=$FT/wt_takeover_build/canonical/install ionice -c3 nice -n 19 taskset -c 0-9 \
    /home/jixian/Desktop/miniconda3/bin/python eval_fast/run.py --map $MAP --moments $first:$last --session $s --name khronos_cont \
    --mode full --out $R/eval/khronos_cont_${TAG}_$s "${prior[@]}" > $R/logs/eval_khronos_cont_${TAG}_$s.log 2>&1 \
    && step "$TAG: scored $s" || step "$TAG: scoring $s failed (logs/eval_khronos_cont_${TAG}_$s.log)") &
  prev=$s
done < $R/logs/khronos_continuous_moments_$TAG.txt
wait
step "$TAG: done"
