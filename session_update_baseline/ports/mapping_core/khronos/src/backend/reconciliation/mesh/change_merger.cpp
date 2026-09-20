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

#include "khronos/backend/reconciliation/mesh/change_merger.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <config_utilities/config_utilities.h>
#include <hydra/utils/nearest_neighbor_utilities.h>

namespace khronos {

void declare_config(ChangeMerger::Config& config) {
  using namespace config;
  name("ChangeMerger");
  base<MeshMerger::Config>(config);
}

ChangeMerger::ChangeMerger(const Config& config)
    : MeshMerger(config), config(config::checkValid(config)) {}

void ChangeMerger::merge(DynamicSceneGraph& dsg, const BackgroundChanges& changes) {
  Timer timer("merge_mesh/all", 0);
  const auto& vertices = dsg.mesh()->points;
  const size_t num_prev_vertices = vertices.size();
  const size_t num_prev_faces = dsg.mesh()->numFaces();

  // Setup object surface search if required.
  Points object_points;
  if (config.remove_objects_from_background) {
    const auto& nodes = dsg.getLayer(DsgLayers::OBJECTS).nodes();
    for (const auto& [id, node] : nodes) {
      const auto& attrs = node->attributes<KhronosObjectAttributes>();
      object_points.reserve(object_points.size() + attrs.mesh.numVertices());
      for (const auto& vertex : attrs.mesh.points) {
        object_points.emplace_back(attrs.bounding_box.pointToWorldFrame(vertex));
      }
    }
  }
  const hydra::PointNeighborSearch search(object_points);
  const float distance_threshold =
      config.object_proximity_threshold * config.object_proximity_threshold;

  // Observation priority. A vertex first built in THIS session is removed only
  // when verified absent. A vertex inherited from an earlier session is memory.
  // Absence removes it as usual. When this session's rays verify its site as
  // present (persistent) and this session also built its own surface there,
  // both are estimates of one surface and THIS SESSION'S MEASUREMENTS decide:
  //  - no session surface within the ray association tolerance: this session
  //    did not rebuild the surface, so memory keeps representing it;
  //  - the estimates agree within half a voxel: a duplicate below TSDF
  //    quantization, kept (it costs nothing and completes the surface);
  //  - they disagree, and the measurements support the inherited estimate at
  //    least as well as this session's: memory is the better estimate, kept;
  //  - they disagree and the measurements do not support memory (its projected
  //    residual exceeds half a voxel and is worse than this session's): the
  //    inherited vertex is a stale, misregistered estimate of a surface that
  //    this session has measured and rebuilt. It is retired from the current
  //    map and remains in the earlier 4D time steps.
  // Without measurement evidence no inherited vertex is retired on geometry
  // alone; only measured absence removes it.
  const auto& mesh = *dsg.mesh();
  const bool track_memory = inherited_horizon_ > 0 && mesh.has_timestamps &&
                            mesh.stamps.size() == vertices.size();
  Points session_points;
  if (track_memory) {
    for (size_t i = 0; i < vertices.size(); ++i) {
      if (mesh.stamps[i] > inherited_horizon_) {
        session_points.push_back(vertices[i]);
      }
    }
  }
  const hydra::PointNeighborSearch session_search(session_points);
  // Agreement scale: half a voxel. A voxel-hashed TSDF localizes a surface to
  // within about half a cell, so two reconstructions of one surface closer than
  // that are the same surface at this map's resolution and memory is kept. The
  // scale is the map's own resolution, not a tuned distance; measured F1 is flat
  // for any value from 0 to half a voxel and only degrades beyond one voxel.
  const float agree = 0.5f * surface_resolution_;
  const float agree_sq = agree * agree;
  const float assoc_sq = association_tolerance_ * association_tolerance_;
  // Median projected |measured - expected| over the frames that actually
  // observed this point; the verificator's spatial index supplies them, the
  // same rays that decided the persistent state. A negative result means this
  // session measured the point in no frame.
  const auto residual = [&](const Point& point) {
    if (!verificator_ || !evidence_) return -1.f;
    const auto observed = verificator_->check(point, inherited_horizon_ + 1);
    std::vector<float> errors;
    for (const TimeStamp t : observed.present) {
      const auto projected = evidence_->project(t, point);
      const auto& endpoint = projected.endpoint;
      if (endpoint.type == EndpointClass::kUnavailable ||
          endpoint.type == EndpointClass::kInvalid ||
          !std::isfinite(endpoint.measured_depth_m) || endpoint.measured_depth_m <= 0.f ||
          !std::isfinite(projected.query_range_m)) {
        continue;
      }
      errors.push_back(std::abs(endpoint.measured_depth_m - projected.query_range_m));
      if (errors.size() >= 5) break;
    }
    if (errors.empty()) return -1.f;
    const size_t middle = errors.size() / 2;
    std::nth_element(errors.begin(), errors.begin() + middle, errors.end());
    return errors[middle];
  };
  std::unordered_set<uint64_t> vertices_to_delete;
  size_t retired_memory = 0;
  for (size_t i = 0; i < vertices.size(); ++i) {
    if (changes.size() > i) {
      if (changes[i] == ChangeState::kAbsent) {
        vertices_to_delete.insert(i);
        continue;
      }
      const bool inherited = track_memory && mesh.stamps[i] <= inherited_horizon_;
      if (inherited && changes[i] == ChangeState::kPersistent && !session_points.empty()) {
        float distance_sq = std::numeric_limits<float>::max();
        size_t nearest = 0;
        session_search.search(vertices[i], distance_sq, nearest);
        if (distance_sq > agree_sq && distance_sq <= assoc_sq) {
          // Two differently meshed reconstructions of one surface can disagree
          // by a cell without either being wrong, so geometry alone does not
          // retire memory: this session's measurements must actually place the
          // surface elsewhere. Retire only when they put it more than half a
          // voxel from the inherited vertex AND agree better with this
          // session's own surface. Unmeasured points keep memory.
          const float memory_error = residual(vertices[i]);
          const float session_error = residual(session_points[nearest]);
          const bool measurements_reject_memory =
              memory_error > agree && session_error >= 0.f && memory_error > session_error;
          if (measurements_reject_memory) {
            ++retired_memory;
            vertices_to_delete.insert(i);
            continue;
          }
        }
      }
    }

    // Check if close to an object.
    if (!config.remove_objects_from_background) {
      continue;
    }
    float distance_sqaured = std::numeric_limits<float>::max();
    size_t index;
    search.search(vertices[i], distance_sqaured, index);
    if (distance_sqaured < distance_threshold) {
      vertices_to_delete.insert(i);
    }
  }

  // Write the new mesh.
  dsg.mesh()->eraseVertices(vertices_to_delete);
  LOG(INFO) << "[MemoryRetirement] background agreement_m=" << agree
            << " retired_inherited=" << retired_memory << " removed_vertices="
            << num_prev_vertices - dsg.mesh()->numVertices() << " of "
            << num_prev_vertices;
}

}  // namespace khronos
