/** -----------------------------------------------------------------------------
 * Task-2 confirmation test (read-only audit follow-up).
 *
 * Question: does a settled moved object's current reconstruction use ONLY the
 * observations after last_motion_seen (the settle tail), and how many frames
 * is that in the accepted-A configuration (store_every_n_frames=3,
 * max_buffer_size=100)?
 *
 * Setup: a track with has_dynamic_history=true and last_motion_seen=3s.
 * Observations every 0.5 s from 1 s to 4 s. The shared FrameDataBuffer stores
 * every 3rd frame (accepted-A parameters).
 *
 * Prints the actual frame count/timestamps used for reconstruction; asserts
 * that every used frame is strictly after last_motion_seen and that the count
 * is bounded by the settle tail.
 * -------------------------------------------------------------------------- */

#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <hydra/input/input_data.h>

#include "khronos/active_window/data/frame_data.h"
#include "khronos/active_window/data/frame_data_buffer.h"
#include "khronos/active_window/data/track.h"
#include "khronos/active_window/object_extraction/mesh_object_extractor.h"

namespace {

using khronos::FrameData;
using khronos::FrameDataBuffer;
using khronos::MeshObjectExtractor;
using khronos::Observation;
using khronos::TimeStamp;
using khronos::Track;

constexpr TimeStamp kSecond = 1'000'000'000ULL;

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << "\n";
    std::exit(EXIT_FAILURE);
  }
}

FrameData::Ptr makeFrame(TimeStamp stamp) {
  hydra::InputData input(nullptr);
  input.timestamp_ns = stamp;
  return std::make_shared<FrameData>(input);
}

}  // namespace

int main() {
  // Accepted-A configuration (runs/session_update_accepted_20260814/session_a/config.txt).
  FrameDataBuffer::Config buffer_config;
  buffer_config.max_buffer_size = 100;
  buffer_config.store_every_n_frames = 3;
  FrameDataBuffer buffer(buffer_config);

  // Observation stream: 1.0, 1.5, 2.0, ..., 4.0 s (the camera keeps seeing the
  // object); motion ends at 3.0 s.
  const std::vector<TimeStamp> all_stamps = {1 * kSecond,
                                             1500000000ULL,
                                             2 * kSecond,
                                             2500000000ULL,
                                             3 * kSecond,
                                             3500000000ULL,
                                             4 * kSecond};
  for (const TimeStamp stamp : all_stamps) {
    buffer.storeData(makeFrame(stamp));
  }

  Track track;
  track.id = 1;
  track.physical_instance_id = 7;
  track.confidence = 1.0f;
  track.first_seen = all_stamps.front();
  track.last_seen = all_stamps.back();
  track.has_dynamic_history = true;
  track.last_motion_seen = 3 * kSecond;  // settle_time=1s: static only after 3s
  for (const TimeStamp stamp : all_stamps) {
    track.observations.emplace_back(stamp, 1, -1);  // semantic_cluster_id=1
  }

  // Mirror extractStaticObject's after_motion filter
  // (mesh_object_extractor.cpp:318-321).
  const auto after_motion = track.has_dynamic_history && track.last_motion_seen > 0
                                ? std::optional<TimeStamp>(track.last_motion_seen)
                                : std::nullopt;
  const auto used_frames = MeshObjectExtractor::collectSemanticFrames(
      track, buffer, after_motion);

  std::cout << "observations in track: " << track.observations.size() << "\n";
  std::cout << "frames retained in shared buffer: " << buffer.size() << "\n";
  std::cout << "frames used for current reconstruction (after_motion="
            << track.last_motion_seen / kSecond << "s): " << used_frames.size() << "\n";
  for (const auto& [frame, segment_id] : used_frames) {
    std::cout << "  used frame stamp: " << frame->input.timestamp_ns / kSecond << "s\n";
  }

  for (const auto& [frame, segment_id] : used_frames) {
    require(frame->input.timestamp_ns > track.last_motion_seen,
            "every reconstruction frame is strictly after last_motion_seen");
    require(buffer.getData(frame->input.timestamp_ns) != nullptr,
            "every reconstruction frame still exists in the shared buffer");
  }

  // Control: without dynamic history the filter is nullopt and the full stored
  // set is used (including pre-motion frames).
  Track static_track = track;
  static_track.has_dynamic_history = false;
  static_track.last_motion_seen = 0;
  const auto all_frames = MeshObjectExtractor::collectSemanticFrames(
      static_track, buffer, std::nullopt);
  require(all_frames.size() > used_frames.size(),
          "without motion history more frames are used (control)");

  std::cout << "PASS: reconstruction uses only the post-motion tail; "
            << used_frames.size() << " frames (accepted-A params)\n";
  return 0;
}
