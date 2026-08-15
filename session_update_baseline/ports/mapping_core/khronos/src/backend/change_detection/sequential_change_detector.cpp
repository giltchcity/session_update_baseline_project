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

#include "khronos/backend/change_detection/sequential_change_detector.h"

namespace khronos {

namespace {

bool sameMerges(const RPGOMerges& lhs, const RPGOMerges& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i].from_node != rhs[i].from_node ||
        lhs[i].to_node != rhs[i].to_node ||
        lhs[i].is_valid != rhs[i].is_valid) {
      return false;
    }
  }
  return true;
}

}  // namespace

void declare_config(SequentialChangeDetector::Config& config) {
  using namespace config;
  name("SequentialChangeDetector");

  // Member configs.
  field(config.ray_verificator, "ray_verificator");
  field(config.ray_change_detector, "ray_change_detector");

  config.objects.setOptional();
  field(config.objects, "objects");
  config.background.setOptional();
  field(config.background, "background");
}

SequentialChangeDetector::SequentialChangeDetector(const Config& config)
    : config(config::checkValid(config)) {
  // Setup members.
  ray_verificator_ = std::make_shared<RayVerificator>(config.ray_verificator);
  ray_change_detector_ = std::make_shared<RayChangeDetector>(config.ray_change_detector);
  object_change_detector_ = config.objects.create(ray_verificator_, ray_change_detector_);
  if (!object_change_detector_) {
    object_change_detector_ = std::make_unique<ObjectChangeDetector>();
  }
  background_change_detector_ = config.background.create(ray_verificator_, ray_change_detector_);
  if (!background_change_detector_) {
    background_change_detector_ = std::make_unique<BackgroundChangeDetector>();
  }
}

void SequentialChangeDetector::setPhysicalEvidenceStore(
    PhysicalEvidenceStore::Ptr store) {
  ray_verificator_->setPhysicalEvidenceStore(std::move(store));
}

RayVerificator::UpdateMode SequentialChangeDetector::setDsg(
    std::shared_ptr<const DynamicSceneGraph> dsg) {
  dsg_ = std::move(dsg);
  last_binding_mode_ = ray_verificator_->setDsg(dsg_);
  if (last_binding_mode_ == RayVerificator::UpdateMode::kFullReset) {
    require_full_change_recompute_ = true;
  }
  return last_binding_mode_;
}

const Changes& SequentialChangeDetector::detectChanges(const RPGOMerges& rpgo_merges,
                                                       TimeStamp stamp,
                                                       bool had_loopclosure) {
  const bool predicted_full_recompute = had_loopclosure || require_full_change_recompute_;
  const std::string timer_suffix =
      predicted_full_recompute ? "_recompute" : "_incremental";
  Timer timer("change_detection" + timer_suffix + "/all", stamp);

  // Setup the ray verificator.
  Timer detail_timer("change_detection" + timer_suffix + "/update_ray_verificator", stamp);
  bool full_recompute = require_full_change_recompute_;
  // A genuine loop closure/deformation invalidates source and target geometry;
  // rebuilding only the block hash is insufficient. Rebuild rays as well.
  if (had_loopclosure) {
    // setDsg() may already have rejected the prefix and rebuilt this exact DSG.
    // Do not immediately rebuild the same state a second time.
    if (!require_full_change_recompute_) {
      ray_verificator_->forceReset();
    }
    full_recompute = true;
  } else if (ray_verificator_->updateDsg() ==
             RayVerificator::UpdateMode::kFullReset) {
    // The DSG may have been mutated in place after setDsg(). The verificator
    // independently re-checks the prefix and reports the conservative fallback.
    full_recompute = true;
  }

  if (full_recompute) {
    changes_.object_changes.clear();
    changes_.background_changes.clear();
  }
  require_full_change_recompute_ = false;

  // Merge proposals are object state. If an existing proposal changes, retain
  // the incremental ray index/background states but recompute every object so
  // the result remains identical to a fresh detector.
  if (have_previous_rpgo_merges_ &&
      !sameMerges(previous_rpgo_merges_, rpgo_merges)) {
    changes_.object_changes.clear();
  }
  previous_rpgo_merges_ = rpgo_merges;
  have_previous_rpgo_merges_ = true;

  // Perform object-level change detection.
  detail_timer.reset("change_detection" + timer_suffix + "/objects");
  object_change_detector_->detectChanges(*dsg_, rpgo_merges, changes_.object_changes);

  // Perform background-level change detection.
  detail_timer.reset("change_detection" + timer_suffix + "/background");
  background_change_detector_->detectChanges(*dsg_, changes_.background_changes);

  return changes_;
}

}  // namespace khronos
