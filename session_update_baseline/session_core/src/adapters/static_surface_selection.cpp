/** -----------------------------------------------------------------------------
 * Copyright (c) 2024 Massachusetts Institute of Technology.
 * All Rights Reserved.
 *
 * AUTHORS:      Lukas Schmid <lschmid@mit.edu>, Marcus Abate <mabate@mit.edu>,
 *               Yun Chang <yunchang@mit.edu>, Luca Carlone <lcarlone@mit.edu>
 * AFFILIATION:  MIT SPARK Lab, Massachusetts Institute of Technology
 * YEAR:         2024
 * SOURCE:       https://github.com/MIT-SPARK/Khronos
 * LICENSE:      BSD 3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * -------------------------------------------------------------------------- */

#include "khronos/active_window/object_extraction/mesh_object_extractor.h"
#include <algorithm>
#include <cmath>
#include <set>
#include <glog/logging.h>
namespace khronos {
namespace {

struct SurfaceCompatibility {
  size_t supported = 0;
  size_t free = 0;
  size_t sampled = 0;  // surface samples of the source frame that could be tested
};

// Compare measured surfaces in world coordinates, not image centroids. Camera
// motion therefore does not look like object motion. Test both directions:
// a nearer surface may just occlude the query; only measured space behind it
// disproves a common static surface. RGB and GT are not used here.
SurfaceCompatibility compareSurfaceFrames(
    const std::pair<FrameData::Ptr, int>& source,
    const std::pair<FrameData::Ptr, int>& target, float tolerance) {
  SurfaceCompatibility result;
  const auto& a = *source.first;
  const auto& b = *target.first;
  if (a.input.vertex_map.empty() || b.input.range_image.empty() ||
      b.object_image.empty()) return result;
  const auto cluster = std::find_if(a.semantic_clusters.begin(), a.semantic_clusters.end(),
      [&](const auto& c) { return c.id == source.second; });
  if (cluster == a.semantic_clusters.end()) return result;
  const Eigen::Isometry3f world_T_a = a.input.getSensorPose().cast<float>();
  const Eigen::Isometry3f b_T_world = b.input.getSensorPose().inverse().cast<float>();
  const size_t stride = std::max<size_t>(1, (cluster->pixels.size() + 511) / 512);
  std::set<std::pair<int, int>> sampled;
  for (size_t i = 0; i < cluster->pixels.size(); i += stride) {
    const auto& px = cluster->pixels[i];
    if (!px.isInImage(a.input.vertex_map)) continue;
    const auto& v = a.input.vertex_map.at<InputData::VertexType>(px.v, px.u);
    Point world(v[0], v[1], v[2]);
    if (!world.array().isFinite().all()) continue;
    if (!a.input.points_in_world_frame) world = world_T_a * world;
    ++result.sampled;
    const Point sensor = b_T_world * world;
    const float distance = sensor.norm();
    int u, row;
    if (!std::isfinite(distance) || !b.input.inRange(distance) ||
        !b.input.getSensor().projectPointToImagePlane(sensor, u, row)) continue;
    // The 3x3 footprint tolerates subpixel rounding and object silhouettes.
    // Unknown neighbors or a foreground occluder veto a free-space vote.
    if (u < 1 || row < 1 || u + 1 >= b.input.range_image.cols ||
        row + 1 >= b.input.range_image.rows || !sampled.emplace(u, row).second) continue;
    bool support = false, all_free = true;
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        const float depth = b.input.range_image.at<float>(row + dy, u + dx);
        if (!std::isfinite(depth) || depth <= 0.f || !b.input.inRange(depth)) {
          all_free = false;
          continue;
        }
        const float delta = depth - distance;
        if (delta <= tolerance) all_free = false;
        if (std::abs(delta) <= tolerance &&
            b.object_image.at<int>(row + dy, u + dx) == target.second) support = true;
      }
    }
    if (support) ++result.supported;
    else if (all_free) ++result.free;
  }
  return result;
}

}  // namespace

std::vector<std::pair<FrameData::Ptr, int>> MeshObjectExtractor::selectStaticFrames(
    const Track& track, const FrameDataBuffer& frame_data,
    std::optional<TimeStamp> after_stamp) const {
  auto frames = collectSemanticFrames(track, frame_data, after_stamp);
  if (!track.physical_instance_id || frames.size() < 2) return frames;
  std::stable_sort(frames.begin(), frames.end(), [](const auto& a, const auto& b) {
    return a.first->input.timestamp_ns < b.first->input.timestamp_ns;
  });
  const size_t original_size = frames.size();
  // Anchor to the newest measured state rather than adjacent frames; many
  // small steps must not accumulate into a large undetected displacement.
  for (size_t offset = frames.size() - 1; offset > 0; --offset) {
    const auto forward = compareSurfaceFrames(frames[offset - 1], frames.back(),
                                              config.static_consistency_tolerance);
    const auto reverse = compareSurfaceFrames(frames.back(), frames[offset - 1],
                                              config.static_consistency_tolerance);
    // A pair of frames may split a track only if the later frame actually judged
    // a real share of the earlier surface: a sliver at the image border or in a
    // depth hole says nothing about the object as a whole (same coverage rule as
    // the observed-absence test of the state machine).
    const auto conflicts = [&](const SurfaceCompatibility& value) {
      const size_t count = value.supported + value.free;
      return value.free >= static_cast<size_t>(config.static_consistency_min_pixels) &&
             count * 2 >= value.sampled &&  // one pair decides at once: it must have judged the majority
             static_cast<float>(value.free) >
                 config.static_consistency_max_free_fraction * count;
    };
    // Diagnostics (no decision change): every compared pair, for estimating the static and
    // moved distributions of the free share (P30-P33).
    LOG(INFO) << "STATIC_PAIR inst=" << *track.physical_instance_id
              << " earlier=" << frames[offset - 1].first->input.timestamp_ns
              << " newest=" << frames.back().first->input.timestamp_ns
              << " f_sup=" << forward.supported << " f_free=" << forward.free << " f_sampled=" << forward.sampled
              << " r_sup=" << reverse.supported << " r_free=" << reverse.free << " r_sampled=" << reverse.sampled
              << " conflict=" << (conflicts(forward) || conflicts(reverse));
    if (!conflicts(forward) && !conflicts(reverse)) continue;
    LOG(INFO) << "STATIC_SURFACE_BOUNDARY inst=" << *track.physical_instance_id
              << " rejected_stamp=" << frames[offset - 1].first->input.timestamp_ns
              << " current_stamp=" << frames.back().first->input.timestamp_ns
              << " forward_support=" << forward.supported << " forward_free=" << forward.free
              << " reverse_support=" << reverse.supported << " reverse_free=" << reverse.free
              << " forward_sampled=" << forward.sampled << " reverse_sampled=" << reverse.sampled;
    frames.erase(frames.begin(), frames.begin() + offset);
    break;
  }
  LOG(INFO) << "STATIC_SURFACE_FRAMES inst=" << *track.physical_instance_id
            << " candidates=" << original_size << " selected=" << frames.size()
            << " first=" << frames.front().first->input.timestamp_ns
            << " last=" << frames.back().first->input.timestamp_ns;
  return frames;
}


std::optional<Track> MeshObjectExtractor::preparePhysicalTrack(
    const Track& track, const FrameDataBuffer& frame_data) const {
  std::optional<Track> static_fallback;
  if (track.physical_instance_id &&
      (track.is_dynamic || track.has_dynamic_history) &&
      computeDynamicDisplacement(track, frame_data) < config.min_dynamic_displacement) {
    static_fallback = track;
    static_fallback->is_dynamic = false;
    static_fallback->has_dynamic_history = false;
    static_fallback->last_motion_seen = 0;
  }
  return static_fallback;
}
}  // namespace khronos
