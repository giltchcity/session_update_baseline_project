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

#pragma once

#include <unordered_set>

#include <hydra/common/global_info.h>

#include "khronos/active_window/data/frame_data.h"
#include "khronos/active_window/tracking/tracker.h"

namespace khronos {

/**
 * @brief Tracker for externally supplied physical IDs plus dynamic detections.
 *
 * Physical semantic clusters retain their externally supplied ID. Free-space
 * motion and promoted dynamic-semantic clusters are associated in parallel;
 * motion overlapping a physical cluster marks that same physical track dynamic
 * instead of allocating a second object.
 */
class ExternalTracker : public Tracker {
 public:
  struct Config {
    int verbosity = hydra::GlobalInfo::instance().getConfig().default_verbosity;

    // Duration [s] until tracks become deactivated, leaving the active window.
    float temporal_window = 3.f;

    // Number of times a track has to be observed to be considered existent.
    int min_num_observations = 20;

    // Minimum pixel IoU for associating a motion cluster with a physical
    // instance observed in the same frame.
    float min_cross_iou = 0.1f;

    // Maximum centroid displacement for associating dynamic-only detections
    // between consecutive observations.
    float max_dynamic_distance = 1.0f;

    // A physical object becomes static-current again after this many seconds
    // without overlapping motion evidence. Its dynamic history remains stored.
    float settle_time = 1.0f;
  } const config;

  // Construction.
  explicit ExternalTracker(const Config& config);
  virtual ~ExternalTracker() = default;

  // Inputs.
  void processInput(FrameData& data) override;

 protected:
  // Processing.
  void associateTracks(const FrameData& data);
  void associatePhysicalTracks(const FrameData& data,
                               std::unordered_set<int>& used_dynamic_clusters);
  void associateDynamicTracks(const FrameData& data,
                              std::unordered_set<int>& used_dynamic_clusters);
  void updateTrackingDuration();
  Track& addPhysicalTrack(const MeasurementCluster& observation);
  Track& addDynamicTrack(const MeasurementCluster& observation);
  void updatePhysicalTrack(const MeasurementCluster& observation,
                           const MeasurementCluster* dynamic_observation,
                           Track& track) const;
  void updateDynamicTrack(const MeasurementCluster& observation, Track& track) const;
  static float pixelIoU(const FrameData& data,
                        const MeasurementCluster& physical,
                        const MeasurementCluster& dynamic);

 private:
  static constexpr int kFirstGeneratedDynamicTrackId = 1 << 16;

  TimeStamp processing_stamp_;
  int next_dynamic_track_id_ = kFirstGeneratedDynamicTrackId;
};

void declare_config(ExternalTracker::Config& config);

}  // namespace khronos
