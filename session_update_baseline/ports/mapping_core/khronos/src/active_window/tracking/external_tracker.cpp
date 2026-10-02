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

#include "khronos/active_window/tracking/external_tracker.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <unordered_set>

#include <config_utilities/config.h>
#include <config_utilities/factory.h>
#include <config_utilities/validation.h>

#include "khronos/utils/geometry_utils.h"
#include "session_core/model/model_math.h"

namespace khronos {

namespace {
// README (5r): C' = max(0, C + l), the CUSUM of the per-frame log likelihood ratio l (every pixel of
// the object is judged in every frame: the weight is one).
double cusum(double c, double log_lr) { return std::max(0.0, c + log_lr); }

static const auto registration =
    config::RegistrationWithConfig<Tracker, ExternalTracker, ExternalTracker::Config>(
        "ExternalTracker");

}  // namespace

void declare_config(ExternalTracker::Config& config) {
  using namespace config;
  name("ExternalTracker");
  field(config.verbosity, "verbosity");
  field(config.temporal_window, "temporal_window", "s");
  field(config.min_num_observations, "min_num_observations", "frames");
  check(config.temporal_window, GT, 0.f, "temporal_window");
  check(config.min_num_observations, GT, 0, "min_num_observations");
}

ExternalTracker::ExternalTracker(const Config& config) : config(config::checkValid(config)) {}

void ExternalTracker::processInput(FrameData& data) {
  processing_stamp_ = data.input.timestamp_ns;
  Timer timer("tracking/all", processing_stamp_);

  // Compute the bounding boxes for all clusters for visualization, physical to
  // motion association, and dynamic-only tracking.
  for (auto& cluster : data.semantic_clusters) {
    cluster.bounding_box =
        BoundingBox(utils::VertexMapAdaptor(cluster.pixels, data.input.vertex_map));
  }
  for (auto& cluster : data.dynamic_clusters) {
    cluster.bounding_box =
        BoundingBox(utils::VertexMapAdaptor(cluster.pixels, data.input.vertex_map));
  }

  // Associate current objects to existing tracks and create new tracks for
  // unassociated objects.
  associateTracks(data);

  // Update which tracks are still active. Tracks labeled inactive will be removed by
  // the active window.
  updateTrackingDuration();
}

void ExternalTracker::associateTracks(const FrameData& data) {
  std::unordered_set<int> used_dynamic_clusters;
  associatePhysicalTracks(data, used_dynamic_clusters);
  associateDynamicTracks(data, used_dynamic_clusters);
}

void ExternalTracker::associatePhysicalTracks(
    const FrameData& data,
    std::unordered_set<int>& used_dynamic_clusters) {
  std::unordered_set<int> used_physical_ids;
  for (const auto& observation : data.semantic_clusters) {
    if (!used_physical_ids.insert(observation.id).second) {
      LOG(WARNING) << "Duplicate physical instance id " << observation.id
                   << " in one frame; ignoring the duplicate cluster.";
      continue;
    }

    // README principle 5: the pixels of the object that lie in motion clusters are the evidence;
    // the cluster covering most of them is the object's own motion cluster.
    const auto overlap = motionOverlap(data, observation);
    size_t covered = 0;
    const MeasurementCluster* best_dynamic = nullptr;
    size_t best_overlap = 0;
    for (const auto& [dynamic_id, pixels] : overlap) {
      covered += pixels;
      if (used_dynamic_clusters.count(dynamic_id) || pixels <= best_overlap) continue;
      const auto it = std::find_if(data.dynamic_clusters.begin(), data.dynamic_clusters.end(),
                                   [dynamic_id](const auto& c) { return c.id == dynamic_id; });
      if (it == data.dynamic_clusters.end()) continue;
      best_overlap = pixels;
      best_dynamic = &*it;
    }

    auto track_it = std::find_if(tracks_.begin(), tracks_.end(), [&](const Track& track) {
      return track.id == observation.id && track.id < kFirstGeneratedDynamicTrackId;
    });
    Track& track = track_it == tracks_.end() ? addPhysicalTrack(observation) : *track_it;
    updatePhysicalTrack(observation, best_dynamic, covered, track);
    if (best_dynamic) {
      used_dynamic_clusters.insert(best_dynamic->id);
    }
  }
}

void ExternalTracker::associateDynamicTracks(
    const FrameData& data,
    std::unordered_set<int>& used_dynamic_clusters) {
  // Associate dynamic-only tracks by centroid distance. Physical tracks are
  // handled above even after they become dynamic, because their ID remains the
  // authoritative external identity.
  for (Track& track : tracks_) {
    if (!track.is_dynamic || track.id < kFirstGeneratedDynamicTrackId) {
      continue;
    }

    const MeasurementCluster* best_dynamic = nullptr;
    // README principle 5: the (1 - alpha) chi-square gate of the centroid jitter and the speed of
    // committed motion; unbounded until both are known.
    const double elapsed =
        track.last_seen > 0 && processing_stamp_ > track.last_seen
            ? static_cast<double>(processing_stamp_ - track.last_seen) * 1e-9 : 0.0;
    const double gate = attribution_ ? attribution_->motion().associationGate(elapsed)
                                     : std::numeric_limits<double>::infinity();
    float best_distance = std::isfinite(gate) ? static_cast<float>(gate)
                                              : std::numeric_limits<float>::infinity();
    for (const auto& dynamic : data.dynamic_clusters) {
      if (used_dynamic_clusters.count(dynamic.id)) {
        continue;
      }
      const float distance =
          (dynamic.bounding_box.world_P_center - track.last_centroid).norm();
      if (distance <= best_distance) {
        best_distance = distance;
        best_dynamic = &dynamic;
      }
    }
    if (best_dynamic) {
      updateDynamicTrack(*best_dynamic, track);
      used_dynamic_clusters.insert(best_dynamic->id);
    }
  }

  for (const auto& dynamic : data.dynamic_clusters) {
    if (used_dynamic_clusters.insert(dynamic.id).second) {
      addDynamicTrack(dynamic);
    }
  }
}

Track& ExternalTracker::addPhysicalTrack(const MeasurementCluster& observation) {
  auto& track = tracks_.emplace_back();
  track.is_dynamic = false;
  track.id = observation.id;
  track.physical_instance_id = observation.id;
  track.first_seen = processing_stamp_;
  return track;
}

Track& ExternalTracker::addDynamicTrack(const MeasurementCluster& observation) {
  auto& track = tracks_.emplace_back();
  track.is_dynamic = true;
  track.has_dynamic_history = true;
  track.last_motion_seen = processing_stamp_;
  track.id = next_dynamic_track_id_++;
  track.first_seen = processing_stamp_;
  updateDynamicTrack(observation, track);
  return track;
}

void ExternalTracker::updatePhysicalTrack(
    const MeasurementCluster& observation,
    const MeasurementCluster* dynamic_observation,
    const size_t covered_pixels,
    Track& track) const {
  // README principle 5, (5r): the frame's share k/n of the object's pixels covered by motion enters
  // the CUSUM of the current state, without a prior (nothing seen, nothing changes). Static: "started
  // to move"; moving: "settled" (the inverse ratio). While the object is static the share is its
  // normal (learned) share, while it moves it carries no information.
  const double dt = track.last_frame > 0 && processing_stamp_ > track.last_frame
      ? static_cast<double>(processing_stamp_ - track.last_frame) * 1e-9 : 0.0;
  const auto snapshot = attribution_ ? attribution_->snapshot() : nullptr;
  if (attribution_ && snapshot && track.physical_instance_id && dt > 0.0 &&
      !observation.pixels.empty()) {
    auto& motion = attribution_->motion();
    const size_t id = static_cast<size_t>(*track.physical_instance_id);
    const double n = static_cast<double>(observation.pixels.size());
    const double k = static_cast<double>(std::min(covered_pixels, observation.pixels.size()));
    const double log_lr = motion.logMovingRatio(id, n, k);
    const double threshold = std::log(model::closeOdds());
    if (!track.is_dynamic) {
      track.motion_cusum = cusum(track.motion_cusum, log_lr);
      if (track.motion_cusum > threshold) {
        // Committed visible motion (D1 begins): the placement ends, the trajectory starts.
        track.is_dynamic = true;
        track.has_dynamic_history = true;
        track.motion_since = processing_stamp_;
        track.settle_cusum = 0.0;
        track.motion_cusum = 0.0;
      } else if (!(track.motion_cusum > 0.0)) {
        // A frame that left the statistic at 0 showed the placement static: its share is a normal
        // share of the object, learned after the frame was judged.
        motion.addStaticFrame(id, n, k);
        if (dynamic_observation) {
          motion.addCentroidOffset(
              (dynamic_observation->bounding_box.world_P_center -
               observation.bounding_box.world_P_center).norm());
        }
      }
    } else {
      track.settle_cusum = cusum(track.settle_cusum, -log_lr);
      if (dynamic_observation) {
        // The speed of the committed motion.
        motion.addSpeed((dynamic_observation->bounding_box.world_P_center - track.last_centroid).norm() / dt);
      }
      if (track.settle_cusum > threshold) {
        // The motion has ended: the placement that follows is reconstructed from the frames after
        // this commitment only.
        track.is_dynamic = false;
        track.motion_cusum = 0.0;
        track.settle_cusum = 0.0;
      }
      track.last_motion_seen = processing_stamp_;
    }
  }
  track.last_frame = processing_stamp_;
  track.updateSemantics(observation.semantics);
  track.last_bounding_box = observation.bounding_box;
  track.last_centroid = dynamic_observation && track.is_dynamic
                            ? dynamic_observation->bounding_box.world_P_center
                            : observation.bounding_box.world_P_center;
  track.last_seen = processing_stamp_;
  track.observations.emplace_back(processing_stamp_,
                                  observation.id,
                                  dynamic_observation ? dynamic_observation->id : -1);
  track.confidence = std::min(
      static_cast<float>(track.observations.size()) / (config.min_num_observations * 2), 1.f);
}

void ExternalTracker::updateDynamicTrack(const MeasurementCluster& observation,
                                         Track& track) const {
  if (attribution_ && track.last_seen > 0 && processing_stamp_ > track.last_seen) {
    attribution_->motion().addSpeed(
        (observation.bounding_box.world_P_center - track.last_centroid).norm() /
        (static_cast<double>(processing_stamp_ - track.last_seen) * 1e-9));
  }
  track.is_dynamic = true;
  track.has_dynamic_history = true;
  track.last_motion_seen = processing_stamp_;
  track.updateSemantics(observation.semantics);
  track.last_bounding_box = observation.bounding_box;
  track.last_centroid = observation.bounding_box.world_P_center;
  track.last_seen = processing_stamp_;
  track.observations.emplace_back(processing_stamp_, -1, observation.id);
  track.confidence = std::min(
      static_cast<float>(track.observations.size()) / (config.min_num_observations * 2), 1.f);
}

std::map<int, size_t> ExternalTracker::motionOverlap(const FrameData& data,
                                                     const MeasurementCluster& physical) {
  std::map<int, size_t> overlap;
  if (!data.dynamic_image.empty()) {
    for (const Pixel& pixel : physical.pixels) {
      if (!pixel.isInImage(data.dynamic_image)) continue;
      const int id = data.dynamic_image.at<FrameData::DynamicImageType>(pixel.v, pixel.u);
      if (id != 0) ++overlap[id];
    }
  } else {
    // Direct tests and custom detectors may omit the raster images. Keep a deterministic fallback
    // without imposing its allocation cost on runtime.
    const std::set<Pixel> physical_pixels(physical.pixels.begin(), physical.pixels.end());
    for (const auto& dynamic : data.dynamic_clusters) {
      for (const Pixel& pixel : dynamic.pixels) {
        if (physical_pixels.count(pixel)) ++overlap[dynamic.id];
      }
    }
  }
  return overlap;
}

void ExternalTracker::updateTrackingDuration() {
  // Label tracks that exit the temporal window as inactive.
  const TimeStamp window = fromSeconds(config.temporal_window);
  const TimeStamp min_time = processing_stamp_ > window ? processing_stamp_ - window : 0;
  for (Track& track : tracks_) {
    track.is_active = track.last_seen >= min_time;
  }
}

}  // namespace khronos
