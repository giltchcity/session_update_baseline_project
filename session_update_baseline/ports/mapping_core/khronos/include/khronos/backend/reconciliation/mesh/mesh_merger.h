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

#include <optional>
#include <utility>
#include <vector>

#include <hydra/common/global_info.h>

#include "khronos/backend/change_detection/physical_evidence_store.h"
#include "khronos/backend/change_detection/ray_verificator.h"
#include "khronos/backend/change_state.h"
#include "khronos/common/common_types.h"

namespace khronos {

/**
 * @brief Interface class for operations to reconcile (merge) meshes.
 */
class MeshMerger {
 public:
  // Config.
  struct Config {
    int verbosity = hydra::GlobalInfo::instance().getConfig().default_verbosity;

    // If true erase all background vertices that are already part of an object.
    bool remove_objects_from_background = false;

    // The maximum distance between vertices to be considered part of an object in meters.
    float object_proximity_threshold = 0.05f;
  } const config;

  /**
   * @brief Reconcile the given mesh.
   * @param dsg DSG whose mesh is to be reconciled.
   * @param changes The changes identified by the change detector.
   */
  virtual void merge(DynamicSceneGraph& dsg, const BackgroundChanges& changes);

  /**
   * @brief Declare which background vertices are memory inherited from earlier
   * sessions: every vertex whose timestamp is not later than `stamp`. Zero (the
   * default) means the map has no inherited memory.
   */
  void setInheritedHorizon(uint64_t stamp) { inherited_horizon_ = stamp; }

  /**
   * @brief Scales that decide whether an inherited vertex and this session's
   * surface are two estimates of the same surface. `resolution` is the TSDF
   * voxel size (two reconstructions agree within half a voxel);
   * `association_tolerance` is the change detector's ray depth tolerance
   * (beyond it this session built no surface that could replace the vertex).
   */
  void setSurfaceScales(float resolution, float association_tolerance) {
    surface_resolution_ = resolution;
    association_tolerance_ = association_tolerance;
  }

  /**
   * @brief This session's endpoint measurements. They arbitrate between an
   * inherited surface estimate and this session's own estimate of the same
   * surface: memory is only retired where the measurements do not support it.
   */
  void setMeasurementEvidence(std::optional<PhysicalEvidenceStore::Snapshot> evidence,
                              RayVerificator::ConstPtr verificator = nullptr) {
    evidence_ = std::move(evidence);
    verificator_ = std::move(verificator);
  }

  // Construction.
  explicit MeshMerger(const Config& config);
  virtual ~MeshMerger() = default;

 protected:
  void removeObjectsFromBackground(DynamicSceneGraph& dsg);

  // Latest timestamp of the inherited prior map; 0 when there is none.
  uint64_t inherited_horizon_ = 0;
  float surface_resolution_ = 0.f;
  float association_tolerance_ = 0.f;
  std::optional<PhysicalEvidenceStore::Snapshot> evidence_;
  RayVerificator::ConstPtr verificator_;
};

void declare_config(MeshMerger::Config& config);

}  // namespace khronos
