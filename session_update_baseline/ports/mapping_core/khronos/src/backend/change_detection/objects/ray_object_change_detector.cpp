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

#include "khronos/backend/change_detection/objects/ray_object_change_detector.h"

#include <algorithm>
#include <numeric>
#include <optional>
#include <string>

#include <config_utilities/config_utilities.h>

#include "khronos/utils/khronos_attribute_utils.h"

namespace khronos {

namespace {

std::optional<size_t> physicalInstanceId(const KhronosObjectAttributes& attrs) {
  const auto iter = attrs.details.find("instance_id");
  if (iter == attrs.details.end() || iter->second.size() != 1 ||
      iter->second.front() == 0) {
    return std::nullopt;
  }
  return iter->second.front();
}

}  // namespace

void declare_config(RayObjectChangeDetector::Config& config) {
  using namespace config;
  name("RayObjectChangeDetector");
  field(config.time_filtering_threshold, "time_filtering_threshold", "s");
  field(config.query_subsampling, "query_subsampling");
  check(config.time_filtering_threshold, GE, 0.f, "time_filtering_threshold");
}

RayObjectChangeDetector::RayObjectChangeDetector(const Config& config,
                                                 RayVerificator::ConstPtr ray_verificator,
                                                 RayChangeDetector::ConstPtr ray_change_detector)
    : config(config::checkValid(config)),
      ray_verificator_(std::move(ray_verificator)),
      ray_change_detector_(std::move(ray_change_detector)),
      time_filtering_threshold_ns_(config.time_filtering_threshold * 1e9) {}

void RayObjectChangeDetector::detectChanges(const DynamicSceneGraph& dsg,
                                            const RPGOMerges& rpgo_merges,
                                            ObjectChanges& changes) {
  // Freeze one evidence version for this entire reducer pass. D2 and D3 use
  // exactly this same reducer; starting a new session merely starts a fresh
  // session-local evidence store while the previous scene state comes from the
  // loaded DSG.
  const auto physical_evidence = ray_verificator_->physicalEvidenceSnapshot();

  // Generic objects retain the incremental contract and are recomputed only
  // when their indexed object snapshot changed. Physical evidence is supplied
  // by a frame store independent of the DSG append-only prefix, so every
  // physical record must be recomputed against the latest immutable snapshot.
  for (const NodeId node_id : ray_verificator_->getReobservedObjects()) {
    auto it = changes.find(node_id);
    if (it != changes.end()) {
      changes.erase(it);
    }
  }
  const auto& object_layer = dsg.getLayer(DsgLayers::OBJECTS);
  for (const auto& [node_id, node] : object_layer.nodes()) {
    const auto& attrs = node->attributes<KhronosObjectAttributes>();
    if (!physicalInstanceId(attrs)) {
      continue;
    }
    auto it = changes.find(node_id);
    if (it != changes.end()) {
      changes.erase(it);
    }
  }

  std::unordered_set<NodeId> already_existing_objects;
  for (const ObjectChange& change : changes) {
    already_existing_objects.insert(change.node_id);
  }

  // Iterate over all objects and chompute their current change state.
  for (const auto& object : object_layer.nodes()) {
    // Ignore existing objects.
    if (already_existing_objects.count(object.first)) {
      continue;
    }

    auto& attrs = object.second->attributes<KhronosObjectAttributes>();

    // Setup the object change.
    ObjectChange& change = changes.emplace_back();
    change.node_id = object.first;

    const auto physical_id = physicalInstanceId(attrs);
    // A stable physical ID is reduced only after every visibility segment has
    // independently received its site evidence and presence estimate. Never
    // let an RPGO proposal merge physical segments inside Reconciler first.
    // Geometry-only objects keep the upstream verified merge path.
    if (!physical_id) {
      checkObjectMerge(rpgo_merges, change);
    }

    // A trajectory is historical metadata, not the negation of current
    // geometry. Run ray presence checks only when a current surface exists, but
    // keep the change record above even for a trajectory-only segment.
    if (hasCurrentObjectMesh(attrs)) {
      checkObjectObservation(attrs, change, physical_id, physical_evidence);
    }
  }

  // Incremental re-observation erases and appends records. Keep the externally
  // visible result in the same deterministic node order as a fresh full pass.
  std::sort(changes.begin(), changes.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.node_id < rhs.node_id;
  });
}

void RayObjectChangeDetector::checkObjectMerge(const RPGOMerges& rpgo_merges,
                                               ObjectChange& change) const {
  // NOTE(lschmid): This does currently not use invalid merge proposals as evidence of absence.
  // Could think more about.
  const auto it =
      std::find_if(rpgo_merges.begin(), rpgo_merges.end(), [change](const RPGOMerge& merge) {
        return merge.from_node == change.node_id;
      });
  if (it != rpgo_merges.end() && it->is_valid) {
    change.merged_id = it->to_node;
  }
}

void RayObjectChangeDetector::checkObjectObservation(
    KhronosObjectAttributes& attrs,
    ObjectChange& change,
    std::optional<size_t> physical_id,
    const RayVerificator::PhysicalEvidenceSnapshot& physical_evidence) const {
  if (attrs.mesh.points.empty() || attrs.first_observed_ns.empty() ||
      attrs.last_observed_ns.empty()) {
    return;
  }

  // first_observed_ns/last_observed_ns are estimated presence intervals after
  // reconciliation.  They are not sensor timestamps: an object that is merely
  // still current has last_observed_ns=max().  Always anchor evidence queries
  // to the persisted real observation bounds so a D2 revisit and a D3 revisit
  // after save/load execute the same hidden-change transition.
  const auto first_observed = observationFirstStamp(attrs);
  const auto last_observed = observationLastStamp(attrs);
  const auto before_latest =
      first_observed > time_filtering_threshold_ns_
          ? first_observed - time_filtering_threshold_ns_
          : 0;
  const auto after_earliest =
      last_observed > std::numeric_limits<TimeStamp>::max() - time_filtering_threshold_ns_
          ? std::numeric_limits<TimeStamp>::max()
          : last_observed + time_filtering_threshold_ns_;

  // Compute all ray presence/absence stamps of all vertices before and after the object was
  // actually observed.
  RayVerificator::CheckResult before_data, after_data;
  for (size_t i = 0; i < attrs.mesh.numVertices(); i += config.query_subsampling) {
    const Point point = attrs.mesh.pos(i) + attrs.bounding_box.world_P_center;

    // TODO(lschmid): This double query could be simplified into a single double-ended query.
    const auto before_check =
        physical_id
            ? ray_verificator_->checkPhysical(
                  point, *physical_id, physical_evidence, 0ul, before_latest)
            : ray_verificator_->check(point, 0ul, before_latest);
    const auto after_check =
        physical_id
            ? ray_verificator_->checkPhysical(
                  point, *physical_id, physical_evidence, after_earliest)
            : ray_verificator_->check(point, after_earliest);
    before_data.merge(before_check);
    after_data.merge(after_check);
  }

  // Perform change detection.
  const auto coverage_mode = physical_id ? RayChangeDetector::CoverageMode::kPhysical
                                         : RayChangeDetector::CoverageMode::kDecisiveOnly;
  const auto before_result =
      ray_change_detector_->detectChanges(before_data, false, coverage_mode);
  const auto after_result =
      ray_change_detector_->detectChanges(after_data, true, coverage_mode);
  change.first_absent = before_result.closest_absent.value_or(0ul);
  change.last_absent = after_result.closest_absent.value_or(0ul);
  change.first_persistent = before_result.furthest_persistent.value_or(0ul);
  change.last_persistent = after_result.furthest_persistent.value_or(0ul);

  // Optionally store visualization details.
  const bool store_visualization_details =
      hydra::GlobalInfo::instance().getConfig().store_visualization_details;
  if (store_visualization_details) {
    const auto store_stamps = [&attrs, physical_id](const std::string& key,
                                                     const auto& stamps) {
      if (physical_id) {
        // Physical records are intentionally refreshed every pass, so their
        // diagnostics must describe this snapshot only rather than accumulate
        // stale observations from previous passes.
        attrs.details[key] = stamps;
      } else {
        attrs.details[key].insert(
            attrs.details[key].end(), stamps.begin(), stamps.end());
      }
    };
    store_stamps("cd_before_present", before_data.present);
    store_stamps("cd_before_absent", before_data.absent);
    store_stamps("cd_after_present", after_data.present);
    store_stamps("cd_after_absent", after_data.absent);
    attrs.details["cd_before_inconclusive"] = before_data.inconclusive;
    attrs.details["cd_after_inconclusive"] = after_data.inconclusive;

    const auto store_reason_counts = [&attrs](const std::string& prefix,
                                               const auto& reasons) {
      attrs.details[prefix + "same_id"] = {reasons.same_id};
      attrs.details[prefix + "different_id"] = {reasons.different_id};
      attrs.details[prefix + "unidentified_object"] = {reasons.unidentified_object};
      attrs.details[prefix + "background_replacement"] = {
          reasons.background_replacement};
      attrs.details[prefix + "free_space"] = {reasons.free_space};
      attrs.details[prefix + "geometric_occlusion"] = {reasons.geometric_occlusion};
      attrs.details[prefix + "unavailable"] = {reasons.unavailable};
      attrs.details[prefix + "invalid"] = {reasons.invalid};
      attrs.details[prefix + "no_overlap"] = {reasons.no_overlap};
    };
    store_reason_counts("cd_before_reason_", before_data.reasons);
    store_reason_counts("cd_after_reason_", after_data.reasons);
  }
}

}  // namespace khronos
