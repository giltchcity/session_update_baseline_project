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

#include <spatial_hash/grid.h>

#include "khronos/active_window/data/frame_data.h"
#include "khronos/active_window/data/reconstruction_types.h"
#include "khronos/active_window/data/track.h"

namespace khronos {

/**
 * @brief Interface class for trackers. Trackers use the motion and semantic detections
 * each frame and associate them between frames and between each other.
 */
class Tracker {
 public:
  // Construction.
  Tracker() = default;
  virtual ~Tracker() = default;

  // Inputs.
  /**
   * @brief Use the motion and semantic detections each frame and associate them between
   * frames and between each other.
   * @param map The active window map.
   * @param data The current frame data.
   */
  virtual void processInput(const VolumetricMap& /* map */, FrameData& /* data */) {}

  // Access.
  Tracks& getTracks() { return tracks_; }
  const Tracks& getTracks() const { return tracks_; }

 protected:
  /**
   * @brief A track's observation in the current frame. With a DynamicCluster it also measures
   * that cluster now (see Observation::dynamic_centroid).
   */
  static Observation observe(const VolumetricMap& map,
                             const FrameData& data,
                             int semantic_cluster_id,
                             const MeasurementCluster* dynamic_cluster) {
    Observation observation(data.input.timestamp_ns,
                            semantic_cluster_id,
                            dynamic_cluster ? dynamic_cluster->id : -1);
    if (!dynamic_cluster || dynamic_cluster->pixels.empty()) {
      return observation;
    }
    const bool keep_points =
        hydra::GlobalInfo::instance().getConfig().store_visualization_details;
    const float voxel_size_inv = 1.f / map.config.voxel_size;
    struct Sum {
      Point point = Point::Zero();
      int count = 0;
    };
    GlobalIndexMap<Sum> voxels;
    for (const Pixel& pixel : dynamic_cluster->pixels) {
      const auto& vertex = data.input.vertex_map.at<InputData::VertexType>(pixel.v, pixel.u);
      const Point point(vertex[0], vertex[1], vertex[2]);
      observation.dynamic_centroid += point;
      if (keep_points) {
        Sum& voxel = voxels[spatial_hash::indexFromPoint<GlobalIndex>(point, voxel_size_inv)];
        voxel.point += point;
        ++voxel.count;
      }
    }
    observation.dynamic_centroid /= static_cast<float>(dynamic_cluster->pixels.size());
    observation.dynamic_bounding_box = dynamic_cluster->bounding_box;
    observation.dynamic_points.reserve(voxels.size());
    for (const auto& [index, voxel] : voxels) {
      observation.dynamic_points.push_back(voxel.point / static_cast<float>(voxel.count));
    }
    return observation;
  }

  Tracks tracks_;
};

}  // namespace khronos
