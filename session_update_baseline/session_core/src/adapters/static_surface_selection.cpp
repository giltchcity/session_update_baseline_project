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
#include <map>
#include <stdexcept>
#include <tuple>

#include <glog/logging.h>

#include "session_core/evidence/physical_evidence_store.h"
#include "session_core/evidence/predictive_surface.h"
#include "session_core/evidence/range_encoding.h"

namespace khronos {
namespace {

using MeasurementSource = std::tuple<TimeStamp, int, int>;

struct SourceGroup {
  bool has_classified = false;
  bool all_free = true;
};

struct SurfaceComparison {
  std::map<MeasurementSource, SourceGroup> sources;
  size_t unknown_without_source = 0;
};

// The two directions contribute actual target-pixel measurements to one
// comparison. A pixel contributes once even when several source points project
// onto it; an unknown member cannot turn that shared measurement into absence.
void compareSurfaceFrames(const std::pair<FrameData::Ptr, int>& source,
                          const std::pair<FrameData::Ptr, int>& target,
                          float tolerance,
                          SurfaceComparison& result) {
  const auto& a = *source.first;
  const auto& b = *target.first;
  if (a.input.vertex_map.empty()) return;
  const auto cluster = std::find_if(a.semantic_clusters.begin(), a.semantic_clusters.end(),
      [&](const auto& c) { return c.id == source.second; });
  if (cluster == a.semantic_clusters.end()) return;
  const Eigen::Isometry3f world_T_a = a.input.getSensorPose().cast<float>();
  const Eigen::Isometry3f b_T_world = b.input.getSensorPose().inverse().cast<float>();
  // Preserve the extraction's fixed source-pixel candidate budget.
  const size_t stride = cluster->pixels.empty() ? 1 : 1 + (cluster->pixels.size() - 1) / 512;
  for (size_t i = 0; i < cluster->pixels.size(); i += stride) {
    const auto& px = cluster->pixels[i];
    if (!px.isInImage(a.input.vertex_map)) continue;
    const auto& v = a.input.vertex_map.at<InputData::VertexType>(px.v, px.u);
    Point world(v[0], v[1], v[2]);
    if (!world.allFinite()) continue;
    if (!a.input.points_in_world_frame) world = world_T_a * world;
    const Point sensor = b_T_world * world;
    const float distance = sensor.norm();
    int u = 0, row = 0;
    if (!std::isfinite(distance) || distance <= 0.f ||
        !b.input.getSensor().projectPointToImagePlane(sensor, u, row) ||
        u < 0 || row < 0 || u >= b.input.range_image.cols ||
        row >= b.input.range_image.rows) {
      ++result.unknown_without_source;
      continue;
    }

    ProjectedEndpointEvidence projected;
    projected.query_range_m = distance;
    if (b.input.inRange(distance)) {
      auto& endpoint = projected.endpoint;
      // README (6a), (6e): every consumer classifies the same encoded
      // endpoint interval, including this view of the native float image.
      const auto range_code = measurement::encodeRange(
          b.input.range_image.at<float>(row, u),
          b.input.getSensor().min_range(), b.input.getSensor().max_range());
      endpoint.measured_depth_m = static_cast<float>(range_code) / 1000.f;
      if (!range_code) {
        endpoint.type = EndpointClass::kInvalid;
      } else {
        const int identity = row < b.object_image.rows && u < b.object_image.cols
            ? b.object_image.at<int>(row, u) : -1;
        endpoint.physical_id = identity;
        endpoint.type = identity > 0 ? EndpointClass::kPhysical
            : identity == 0 ? EndpointClass::kBackground : EndpointClass::kUnidentifiedObject;
      }
    }
    const auto vote = measurement::classifySurfaceMeasurement(
        projected, static_cast<size_t>(target.second), tolerance);
    const bool absent = vote == measurement::SurfaceVote::Free ||
        vote == measurement::SurfaceVote::Background || vote == measurement::SurfaceVote::Other;
    const bool classified = absent || vote == measurement::SurfaceVote::Supported;
    auto& group = result.sources[{b.input.timestamp_ns, row, u}];
    group.has_classified = group.has_classified || classified;
    group.all_free = group.all_free && absent;
  }
}

}  // namespace

std::vector<std::pair<FrameData::Ptr, int>> MeshObjectExtractor::selectStaticFrames(
    const Track& track, const FrameDataBuffer& frame_data,
    std::optional<TimeStamp> after_stamp) const {
  const float alpha = config.static_consistency_max_free_fraction;
  if (!std::isfinite(alpha) || alpha < 0.f || alpha > 1.f ||
      !std::isfinite(config.static_consistency_tolerance) ||
      config.static_consistency_tolerance <= 0.f || config.static_consistency_min_pixels < 0) {
    throw std::invalid_argument("Invalid static surface comparison contract");
  }
  auto frames = collectSemanticFrames(track, frame_data, after_stamp);
  // alpha=1 leaves no incompatible hypothesis; preserve the existing no-cut setting.
  if (!track.physical_instance_id || frames.size() < 2 || alpha == 1.f) return frames;
  std::stable_sort(frames.begin(), frames.end(), [](const auto& a, const auto& b) {
    return a.first->input.timestamp_ns < b.first->input.timestamp_ns;
  });
  const size_t original_size = frames.size();
  // Anchor to the newest measured state. The first incompatible older frame
  // defines a contiguous retained suffix; small adjacent motions cannot chain
  // arbitrarily different poses into one static reconstruction.
  for (size_t offset = frames.size() - 1; offset > 0; --offset) {
    SurfaceComparison comparison;
    compareSurfaceFrames(frames[offset - 1], frames.back(),
                         config.static_consistency_tolerance, comparison);
    compareSurfaceFrames(frames.back(), frames[offset - 1],
                         config.static_consistency_tolerance, comparison);
    size_t measured = 0, absent = 0, unknown = comparison.unknown_without_source;
    for (const auto& [source, group] : comparison.sources) {
      (void)source;
      if (group.has_classified) {
        ++measured;
        absent += group.all_free;
      } else {
        ++unknown;
      }
    }
    // This fixed acquisition qualification counts classified sources, not free
    // votes. Unknown sources do not establish the required observation budget.
    if (measured < static_cast<size_t>(config.static_consistency_min_pixels)) continue;
    const auto completed = measurement::retainingCompletion(absent, measured, unknown);
    // Both directions share the declared mismatch-rate model. With no static
    // calibration, the reference is the fixed uniform Beta(1,1), conditioned on
    // [0,alpha]; the common predictor and 99:1 loss decide the single comparison.
    const auto log_odds = measurement::surfaceLogOdds(completed, alpha);
    if (!measurement::favorsExit(log_odds)) continue;
    LOG(INFO) << "STATIC_SURFACE_BOUNDARY inst=" << *track.physical_instance_id
              << " rejected_stamp=" << frames[offset - 1].first->input.timestamp_ns
              << " current_stamp=" << frames.back().first->input.timestamp_ns
              << " measured_sources=" << measured << " absent_sources=" << absent
              << " unknown_sources=" << unknown << " completed_absent=" << completed.absent
              << " completed_total=" << completed.total << " log_odds=" << log_odds;
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
