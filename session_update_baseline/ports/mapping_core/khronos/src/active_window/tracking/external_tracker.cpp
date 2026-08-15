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
#include <set>
#include <unordered_set>

#include <config_utilities/config.h>
#include <config_utilities/factory.h>
#include <config_utilities/validation.h>

#include "khronos/utils/geometry_utils.h"

namespace khronos {

namespace {
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
  field(config.min_cross_iou, "min_cross_iou");
  field(config.max_dynamic_distance, "max_dynamic_distance", "m");
  field(config.settle_time, "settle_time", "s");
  check(config.temporal_window, GT, 0.f, "temporal_window");
  check(config.min_num_observations, GT, 0, "min_num_observations");
  checkInRange(config.min_cross_iou, 0.0f, 1.0f, "min_cross_iou");
  check(config.max_dynamic_distance, GT, 0.0f, "max_dynamic_distance");
  check(config.settle_time, GE, 0.0f, "settle_time");
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

    const MeasurementCluster* best_dynamic = nullptr;
    float best_iou = config.min_cross_iou;
    for (const auto& dynamic : data.dynamic_clusters) {
      if (used_dynamic_clusters.count(dynamic.id)) {
        continue;
      }
      const float iou = pixelIoU(data, observation, dynamic);
      if (iou >= best_iou && iou > 0.0f) {
        best_iou = iou;
        best_dynamic = &dynamic;
      }
    }

    auto track_it = std::find_if(tracks_.begin(), tracks_.end(), [&](const Track& track) {
      return track.id == observation.id && track.id < kFirstGeneratedDynamicTrackId;
    });
    Track& track = track_it == tracks_.end() ? addPhysicalTrack(observation) : *track_it;
    updatePhysicalTrack(observation, best_dynamic, track);
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
    float best_distance = config.max_dynamic_distance;
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
    Track& track) const {
  if (dynamic_observation) {
    track.is_dynamic = true;
    track.has_dynamic_history = true;
    track.last_motion_seen = processing_stamp_;
  } else if (track.is_dynamic && track.physical_instance_id &&
             processing_stamp_ >= track.last_motion_seen + fromSeconds(config.settle_time)) {
    // The object is still identified in the semantic/instance stream but no
    // longer overlaps a motion cluster. Re-enter static-current reconstruction
    // at its new pose; has_dynamic_history preserves the D1 trajectory.
    track.is_dynamic = false;
  }
  track.updateSemantics(observation.semantics);
  track.last_bounding_box = observation.bounding_box;
  track.last_centroid = dynamic_observation
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

float ExternalTracker::pixelIoU(const FrameData& data,
                                const MeasurementCluster& physical,
                                const MeasurementCluster& dynamic) {
  std::size_t intersection = 0;
  if (physical.pixels.size() <= dynamic.pixels.size() && !data.dynamic_image.empty()) {
    for (const Pixel& pixel : physical.pixels) {
      if (pixel.isInImage(data.dynamic_image) &&
          data.dynamic_image.at<FrameData::DynamicImageType>(pixel.v, pixel.u) == dynamic.id) {
        ++intersection;
      }
    }
  } else if (!data.object_image.empty()) {
    for (const Pixel& pixel : dynamic.pixels) {
      if (pixel.isInImage(data.object_image) &&
          data.object_image.at<FrameData::ObjectImageType>(pixel.v, pixel.u) == physical.id) {
        ++intersection;
      }
    }
  } else {
    // Direct tests and custom detectors may omit the raster images. Keep a
    // deterministic fallback without imposing its allocation cost on runtime.
    const std::set<Pixel> physical_pixels(physical.pixels.begin(), physical.pixels.end());
    const std::set<Pixel> dynamic_pixels(dynamic.pixels.begin(), dynamic.pixels.end());
    for (const Pixel& pixel : physical_pixels) {
      intersection += dynamic_pixels.count(pixel);
    }
  }
  const std::size_t union_size =
      physical.pixels.size() + dynamic.pixels.size() - intersection;
  return union_size == 0
             ? 0.0f
             : static_cast<float>(intersection) / static_cast<float>(union_size);
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
