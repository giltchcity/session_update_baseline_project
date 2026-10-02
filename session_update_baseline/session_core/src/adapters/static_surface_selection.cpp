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

#include "session_core/model/model_math.h"

namespace khronos {
namespace {

// Computation budget of the frame-pair test: at most this many pixels of an object in a frame are
// carried to the other frame (standard error of a proportion <= 0.5 / sqrt(512) = 2.2%).
constexpr size_t kPairSamples = 512;

const MeasurementCluster* clusterOf(const FrameData& frame, int id) {
  const auto it = std::find_if(frame.semantic_clusters.begin(), frame.semantic_clusters.end(),
                               [id](const auto& cluster) { return cluster.id == id; });
  return it == frame.semantic_clusters.end() ? nullptr : &*it;
}

struct PairCounts {
  double n = 0.0, f = 0.0;  // decided verdicts and see-throughs
  double predicted = 0.0;   // sum of the see-through rates the measurement model predicts (m_0)
};

// README principle 5, (6b), principle 4: the object's pixels of `from` are carried to the frame `to`
// and the range read there is classified T / H by psi (the residual scale of the readings against
// the fused surface, which does not depend on the time difference of the two frames); occluded and
// invalid readings carry likelihood ratio one.
void carryPixels(const FrameData& from, const MeasurementCluster& cluster, const FrameData& to,
                 const model::RangeModel& psi, PairCounts& counts) {
  const auto& sensor = to.input.getSensor();
  const Eigen::Isometry3f sensor_T_world = to.input.getSensorPose().cast<float>().inverse();
  const auto& vertices = from.input.vertex_map;
  const auto& ranges = to.input.range_image;
  if (vertices.empty() || ranges.empty()) return;
  const size_t stride = std::max<size_t>(1, cluster.pixels.size() / kPairSamples);
  for (size_t i = 0; i < cluster.pixels.size(); i += stride) {
    const Pixel& pixel = cluster.pixels[i];
    if (pixel.u < 0 || pixel.v < 0 || pixel.u >= vertices.cols || pixel.v >= vertices.rows) continue;
    const auto& vertex = vertices.at<InputData::VertexType>(pixel.v, pixel.u);
    const Eigen::Vector3f world(vertex[0], vertex[1], vertex[2]);
    if (!world.allFinite()) continue;
    const Eigen::Vector3f local = sensor_T_world * world;
    int u, v;
    if (!sensor.projectPointToImagePlane(local, u, v) || u < 0 || v < 0 || u >= ranges.cols ||
        v >= ranges.rows) {
      continue;
    }
    const double reading = ranges.at<InputData::RangeType>(v, u);
    const double rho = local.norm();
    const double sigma = psi.sigmaEff(rho, 0.0, 0.0, false);
    const auto kind = model::classifyRange(psi, reading, rho, sigma, sensor.min_range(),
                                           sensor.max_range());
    if (kind == model::RangeClass::kHit || kind == model::RangeClass::kThrough) {
      counts.n += 1.0;
      if (kind == model::RangeClass::kThrough) counts.f += 1.0;
      counts.predicted += psi.predictedSeeThrough(rho, sigma, sensor.max_range());
    }
  }
}

}  // namespace

// README (4.0) P5, (6b), principle 5: a static reconstruction fuses the observations attributed to
// one placement. Frames acquired at or before the right end of (5t) of a closed placement belong
// to that placement. Within the rest, every earlier frame i is compared with the newest frame j: the
// object's pixels of i are carried to j and those of j to i, and the look ratio (7) of the decided
// verdicts (F see-through of n, the object's in-place distribution against the uniform ended
// distribution) says whether the object changed between the two. The frames of a reconstruction are
// a representation output (5e) without a prior, so the maximum a posteriori choice: a frame whose
// ratio exceeds 1 and all frames before it are cut; the observation domain of the placement starts
// after the cut (6b).
std::vector<std::pair<FrameData::Ptr, int>> MeshObjectExtractor::selectStaticFrames(
    const Track& track, const FrameDataBuffer& frame_data,
    std::optional<TimeStamp> after_stamp, std::vector<FramePairDecision>* decisions) const {
  const auto snapshot = attribution_ ? attribution_->snapshot() : nullptr;
  if (snapshot && track.physical_instance_id) {
    const TimeStamp closed = attribution_->closedThrough(*track.physical_instance_id);
    if (closed > 0 && (!after_stamp || *after_stamp < closed)) after_stamp = closed;
  }
  auto frames = collectSemanticFrames(track, frame_data, after_stamp);
  if (!snapshot || !track.physical_instance_id || frames.size() < 2 ||
      !snapshot->rounds) {
    return frames;
  }
  const size_t id = static_cast<size_t>(*track.physical_instance_id);
  const auto& anchor = frames.back();
  const auto* anchor_cluster = clusterOf(*anchor.first, anchor.second);
  if (!anchor_cluster) return frames;
  for (size_t i = frames.size() - 1; i-- > 0;) {
    const auto& [frame, cluster_id] = frames[i];
    const auto* cluster = clusterOf(*frame, cluster_id);
    if (!cluster) continue;
    if (!(anchor.first->input.timestamp_ns > frame->input.timestamp_ns)) continue;
    PairCounts counts;
    carryPixels(*frame, *cluster, *anchor.first, snapshot->psi, counts);
    carryPixels(*anchor.first, *anchor_cluster, *frame, snapshot->psi, counts);
    if (!(counts.n > 0.0)) continue;
    const auto in_place = snapshot->rounds->inPlace(id, counts.predicted / counts.n);
    const double log_lr = model::RoundModel::logLikelihoodRatio(in_place, counts.n, counts.f);
    // The frames of a reconstruction are a representation output, recomputed at every extraction
    // (5e): no deferral, so the maximum a posteriori choice -- changed iff the ratio reaches 1.
    const bool cut = model::representationHolds(log_lr);
    if (decisions) decisions->push_back({frame->input.timestamp_ns, counts.n, counts.f, log_lr, cut});
    if (cut) {
      frames.erase(frames.begin(), frames.begin() + static_cast<std::ptrdiff_t>(i) + 1);
      break;
    }
  }
  return frames;
}

}  // namespace khronos
