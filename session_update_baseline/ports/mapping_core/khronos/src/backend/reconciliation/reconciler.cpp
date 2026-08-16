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

#include "khronos/backend/reconciliation/reconciler.h"

#include <filesystem>
#include <stdexcept>

#include "khronos/common/common_types.h"
#include "khronos/utils/khronos_attribute_utils.h"

namespace khronos {

void declare_config(Reconciler::Config& config) {
  using namespace config;
  name("Reconciler");
  field(config.verbosity, "verbosity");
  field(config.time_estimates_conservative, "time_estimates_conservative");
  field(config.allow_overestimation, "allow_overestimation");
  field(config.merge_object_meshes, "merge_object_meshes");
  config.mesh_merger.setOptional();
  field(config.mesh_merger, "mesh_merger");
}

Reconciler::Reconciler(const Config& config) : config(config::checkValid(config)) {
  // Load the mesh merger.
  if (config.mesh_merger) {
    mesh_merger_ = config.mesh_merger.create();
  }
}

void Reconciler::reconcile(DynamicSceneGraph& dsg,
                           const Changes& changes,
                           TimeStamp stamp,
                           std::map<NodeId, ObjectReconciliationDetail>* object_details) {
  Timer timer("reconcile/all", stamp);

  VLOG(4) << "Reconciler has " << changes.object_changes.size() << " object changes.";
  for (const auto& change : changes.object_changes) {
    VLOG(4) << "\tChange: " << NodeSymbol(change.node_id).str() << " -> "
            << NodeSymbol(change.merged_id).str();
  }

  // Reconcile the background mesh.
  if (mesh_merger_) {
    Timer bg_timer("reconcile/mesh", stamp);
    mesh_merger_->merge(dsg, changes.background_changes);
  }

  // Reconcile the objects based on the observed changes.
  Timer obj_timer("reconcile/objects", stamp);
  reconcileObjects(changes, dsg, object_details);

  // Give object private meshes the same free-space treatment the background mesh just got. An
  // object's surface is not exempt from "the robot looked straight through that spot": whatever
  // a later observation sees through has to go, wherever it is stored.
  const size_t culled = cullAbsentObjectVertices(dsg);
  if (culled > 0) {
    CLOG(3) << "[Reconciler] Culled " << culled
            << " object mesh vertices contradicted by later free-space evidence.";
  }
}

size_t Reconciler::cullAbsentObjectVertices(DynamicSceneGraph& dsg) const {
  if (!ray_verificator_ || !dsg.hasLayer(DsgLayers::OBJECTS)) {
    return 0;
  }

  size_t removed_total = 0;
  for (const auto& [node_id, node] : dsg.getLayer(DsgLayers::OBJECTS).nodes()) {
    (void)node_id;
    auto* attrs = node->tryAttributes<KhronosObjectAttributes>();
    if (!attrs || attrs->mesh.numVertices() == 0) {
      continue;
    }

    // Same decision rule the background path uses: a vertex goes when later rays passed through
    // it and nothing newer re-hit it. Requiring present.empty() keeps this strictly conservative
    // -- a surface that is still being seen is never removed, so an object that did not move
    // cannot be thinned away by this.
    std::unordered_set<size_t> to_delete;
    for (size_t i = 0; i < attrs->mesh.numVertices(); ++i) {
      const Point world = attrs->bounding_box.pointToWorldFrame(attrs->mesh.pos(i));
      const auto check = ray_verificator_->check(world);
      if (!check.absent.empty() && check.present.empty()) {
        to_delete.insert(i);
      }
    }
    if (to_delete.empty()) {
      continue;
    }
    // Never let this empty an object outright: total disappearance is the node-level absent
    // decision's call, made from presence evidence, not a side effect of vertex culling.
    if (to_delete.size() >= attrs->mesh.numVertices()) {
      continue;
    }
    attrs->mesh.eraseVertices(to_delete);
    removed_total += to_delete.size();
  }
  return removed_total;
}

void Reconciler::ObjectReconciliationDetail::merge(const ObjectReconciliationDetail& other) {
  // Merge observed times.
  observed_times.insert(
      observed_times.end(), other.observed_times.begin(), other.observed_times.end());

  // Merge estimated times.
  estimated_times.insert(
      estimated_times.end(), other.estimated_times.begin(), other.estimated_times.end());
}

std::string Reconciler::ObjectReconciliationDetail::serializeTimestamps(
    const std::vector<uint64_t>& timestamps) const {
  std::string result;
  for (auto it = timestamps.begin(); it != timestamps.end(); ++it) {
    if (it != timestamps.begin()) {
      result += ":";
    }
    result += std::to_string(*it);
  }
  return result;
}

void Reconciler::reconcileObjects(const Changes& changes,
                                  DynamicSceneGraph& dsg,
                                  std::map<NodeId, Reconciler::ObjectReconciliationDetail>*
                                      object_details) {
  // Check which objects will be merged, and which have no merges.
  std::stringstream logging_info;
  ChangeMap all_changes;
  std::set<NodeId> final_merged_nodes, final_unmerged_nodes, merged_away_nodes;
  for (const ObjectChange& change : changes.object_changes) {
    all_changes.insert({change.node_id, change});
    if (change.merged_id == 0ul) {
      if (!final_merged_nodes.count(change.node_id)) {
        final_unmerged_nodes.insert(change.node_id);
      }
    } else {
      logging_info << NodeSymbol(change.node_id).str() << " -> "
                   << NodeSymbol(change.merged_id).str() << "\n";
      final_merged_nodes.insert(change.merged_id);
      final_merged_nodes.erase(change.node_id);
      merged_away_nodes.insert(change.node_id);
      final_unmerged_nodes.erase(change.node_id);
      final_unmerged_nodes.erase(change.merged_id);
    }
  }

  // Log merge details if reqquired.
  if (config.verbosity >= 4) {
    logging_info << "Final merged nodes:\n";
    for (const NodeId node_id : final_merged_nodes) {
      logging_info << NodeSymbol(node_id).str() << "\n";
    }
    logging_info << "Final unmerged nodes:\n";
    for (const NodeId node_id : final_unmerged_nodes) {
      logging_info << NodeSymbol(node_id).str() << "\n";
    }
    logging_info << "Merged away nodes:\n";
    for (const NodeId node_id : merged_away_nodes) {
      logging_info << NodeSymbol(node_id).str() << "\n";
    }
    logging_info << "Num nodes: merged_away: " << merged_away_nodes.size()
                 << ", final_merged: " << final_merged_nodes.size()
                 << ", final_unmerged: " << final_unmerged_nodes.size()
                 << ", changes: " << changes.object_changes.size();
    LOG(INFO) << "Received input changes:\nMerges:\n" << logging_info.str();
  }

  // Estimate updated presence times for all observations based on change detection results.
  for (const auto& [node_id, change] : all_changes) {
    if (object_details && !object_details->count(node_id)) {
      object_details->insert({node_id, ObjectReconciliationDetail()});
    }
    const SceneGraphNode& node = dsg.getNode(node_id);
    KhronosObjectAttributes& attrs = node.attributes<KhronosObjectAttributes>();
    if (object_details) {
      auto& detail = object_details->at(node_id);
      detail.observed_times.push_back(attrs.first_observed_ns.front());
      detail.observed_times.push_back(attrs.last_observed_ns.front());
    }
    estimatePresenceTimes(change, attrs);
    if (object_details) {
      auto& detail = object_details->at(node_id);
      detail.estimated_times.push_back(attrs.first_observed_ns.front());
      detail.estimated_times.push_back(attrs.last_observed_ns.front());
    }
  }

  // Merge all nodes with verified merges, and their change observations.
  for (const ObjectChange& change : changes.object_changes) {
    if (change.merged_id == 0ul) {
      continue;
    }

    if (!all_changes.count(change.merged_id)) {
      continue;
    }

    // Merge the nodes and their change info together.
    mergeObjects(change, dsg);
    if (object_details) {
      object_details->at(change.merged_id).merge(object_details->at(change.node_id));
    }

    const SceneGraphNode& node = dsg.getNode(change.merged_id);
    const KhronosObjectAttributes& attrs = node.attributes<KhronosObjectAttributes>();
    mergeObjectChanges(change,
                       all_changes.at(change.merged_id),
                       attrs.first_observed_ns.front(),
                       attrs.last_observed_ns.back());
  }
}

void Reconciler::estimatePresenceTimes(const ObjectChange& change,
                                       KhronosObjectAttributes& attrs) const {
  // Preserve the sensor observation bounds independently. The vectors below
  // are rewritten into estimated presence intervals and may become [0,max()].
  // A later session must not mistake those estimates for measurement times.
  persistObservationBounds(attrs);
  const uint64_t first_seen = observationFirstStamp(attrs);
  const uint64_t last_seen = observationLastStamp(attrs);
  const auto physical_id = attrs.details.find("instance_id");
  const bool has_physical_identity =
      physical_id != attrs.details.end() && physical_id->second.size() == 1 &&
      physical_id->second.front() != 0;

  // Estimate the presence time before the observation.
  bool first_absent_valid = change.first_absent > 0ul && change.first_absent < first_seen;
  bool first_persistent_valid =
      change.first_persistent > 0ul && change.first_persistent < first_seen;

  if (first_absent_valid && first_persistent_valid &&
      change.first_absent >= change.first_persistent) {
    // NOTE(lschmid): For now only consider the closest options. Could support arbitrary sequences
    // in the future.
    first_persistent_valid = false;
  }

  const uint64_t first_evidence =
      first_persistent_valid ? std::min(change.first_persistent, first_seen) : first_seen;
  if (first_absent_valid) {
    // There is evidence of absence, estimate the appearance time via minimum expected risk.
    attrs.first_observed_ns = {
        change.first_absent + (first_evidence - change.first_absent) / 2};
  } else {
    // A stable physical ID does not exist before its first direct observation
    // merely because no camera ray covered that site. For physical objects,
    // direct instance-labelled evidence is the prior: a genuinely new ID starts
    // at its first direct segment, while a repeated same-ID segment starts from
    // the preceding direct segment supplied by the physical reducer. Preserve
    // the legacy configurable prior for objects without an explicit identity.
    attrs.first_observed_ns = {
        has_physical_identity || config.time_estimates_conservative
            ? first_evidence
            : 0ul};
  }

  // Estimate the presence time after the observation.
  bool last_absent_valid = change.last_absent > 0ul && change.last_absent > last_seen;
  bool last_persistent_valid = change.last_persistent > 0ul && change.last_persistent > last_seen;

  if (last_absent_valid && last_persistent_valid && change.last_absent <= change.last_persistent) {
    last_persistent_valid = false;
  }

  const uint64_t last_evidence =
      last_persistent_valid ? std::max(change.last_persistent, last_seen) : last_seen;
  if (last_absent_valid) {
    // There is evidence of absence, estimate the disappearance time via minimum expected risk.
    attrs.last_observed_ns = {
        last_evidence + (change.last_absent - last_evidence) / 2};
  } else {
    // No later site observation means unobserved, not absent. An explicit
    // physical object therefore remains current/open until actual absence or a
    // later direct segment says otherwise.
    attrs.last_observed_ns = {
        has_physical_identity || !config.time_estimates_conservative
            ? std::numeric_limits<uint64_t>::max()
            : last_evidence};
  }
}

void Reconciler::mergeObjects(const ObjectChange& change, DynamicSceneGraph& dsg) const {
  if (change.node_id == change.merged_id) {
    LOG(WARNING) << "Cannot merge node '" << change.node_id << "' into '" << change.merged_id
                 << "': Same node.";
    return;
  }
  if (!dsg.hasNode(change.node_id) || !dsg.hasNode(change.merged_id)) {
    LOG(WARNING) << "Cannot merge node '" << change.node_id << "' into '" << change.merged_id
                 << "': Not all nodes exist.";
    return;
  }

  // Merge the attributes, nodes, and details.
  auto& from_node = dsg.getNode(change.node_id);
  auto& into_node = dsg.getNode(change.merged_id);
  auto& into_attrs = into_node.attributes<KhronosObjectAttributes>();
  auto& from_attrs = from_node.attributes<KhronosObjectAttributes>();
  if (!config.allow_overestimation) {
    KhronosObjectAttributes from_copy = from_attrs;
    clampOverestimation(into_attrs, from_attrs);
    clampOverestimation(from_copy, into_attrs);
  }
  mergeObjectAttributes(from_attrs, into_attrs);
  if (!dsg.mergeNodes(change.node_id, change.merged_id)) {
    LOG(WARNING) << "Failed to merge node '" << change.node_id << "' into '" << change.merged_id
                 << "'.";
  }
}

void Reconciler::mergeObjectChanges(const ObjectChange& from,
                                    ObjectChange& into,
                                    const uint64_t first_seen,
                                    const uint64_t last_seen) const {
  // Aggregate the first / last present / absent times into a single change struct by taking the
  // extrema at both ends of the changes. This loses information about any gaps in between, but
  // these should be sufficiently represented through the node merges already.

  // First absent.
  bool into_valid = into.first_absent > 0ul && into.first_absent < first_seen;
  bool from_valid = from.first_absent > 0ul && from.first_absent < first_seen;
  if ((!into_valid && from_valid) ||
      (into_valid && from_valid && from.first_absent < into.first_absent)) {
    into.first_absent = from.first_absent;
  }

  // First persistent.
  into_valid = into.first_persistent > 0ul && into.first_persistent < first_seen;
  from_valid = from.first_persistent > 0ul && from.first_persistent < first_seen;
  if ((!into_valid && from_valid) ||
      (into_valid && from_valid && from.first_persistent < into.first_persistent)) {
    into.first_persistent = from.first_persistent;
  }

  // Last absent.
  into_valid = into.last_absent > 0ul && into.last_absent > last_seen;
  from_valid = from.last_absent > 0ul && from.last_absent > last_seen;
  if ((!into_valid && from_valid) ||
      (into_valid && from_valid && from.last_absent > into.last_absent)) {
    into.last_absent = from.last_absent;
  }

  // Last persistent.
  into_valid = into.last_persistent > 0ul && into.last_persistent > last_seen;
  from_valid = from.last_persistent > 0ul && from.last_persistent > last_seen;
  if ((!into_valid && from_valid) ||
      (into_valid && from_valid && from.last_persistent > into.last_persistent)) {
    into.last_persistent = from.last_persistent;
  }
}

void Reconciler::mergeObjectAttributes(const KhronosObjectAttributes& from,
                                       KhronosObjectAttributes& into) const {
  if (from.first_observed_ns.size() != from.last_observed_ns.size() ||
      into.first_observed_ns.size() != into.last_observed_ns.size()) {
    throw std::invalid_argument(
        "Cannot merge object attributes with unpaired presence vectors");
  }
  const auto get_instance_id = [](const KhronosObjectAttributes& attrs) -> std::optional<size_t> {
    const auto iter = attrs.details.find("instance_id");
    if (iter == attrs.details.end() || iter->second.size() != 1 || iter->second.front() == 0) {
      return std::nullopt;
    }
    return iter->second.front();
  };
  const auto from_instance = get_instance_id(from);
  const auto into_instance = get_instance_id(into);
  const bool same_physical_instance = from_instance && into_instance &&
                                      *from_instance == *into_instance;

  // A stable physical identity may move between observations or sessions.  Its previous geometry
  // belongs to the earlier 4D time step; the current DSG must contain the newer private mesh, not a
  // union of old and new positions.  For overlapping observations of the same stationary object we
  // still accumulate surfaces as usual.
  const bool physical_object_moved =
      same_physical_instance && !from.bounding_box.intersects(into.bounding_box);
  if (physical_object_moved) {
    const uint64_t from_time = from.last_observed_ns.empty() ? 0 : from.last_observed_ns.back();
    const uint64_t into_time = into.last_observed_ns.empty() ? 0 : into.last_observed_ns.back();
    if (from_time >= into_time) {
      into.mesh = from.mesh;
      into.bounding_box = from.bounding_box;
      into.position = from.position;
    }
  } else if (config.merge_object_meshes) {
    mergeObjectMeshes(from, into);
  } else if (from.mesh.numVertices() > into.mesh.numVertices()) {
    // Keep the larger mesh.
    into.mesh = std::move(from.mesh);
    into.bounding_box = std::move(from.bounding_box);
  }

  // Trajectory and optional visualization point frames are a single aligned
  // history. Merge all three together so a settled object can carry current
  // geometry without corrupting historical sample indices.
  mergeTrajectoryHistory({&into, &from}, into);

  // Also add up the seen time stamps.
  for (size_t i = 0; i < from.first_observed_ns.size(); ++i) {
    addPresenceDuration(into, from.first_observed_ns[i], from.last_observed_ns[i]);
  }

  if (!into_instance && from_instance) {
    into.details["instance_id"] = {*from_instance};
  }
}

void Reconciler::mergeObjectMeshes(const KhronosObjectAttributes& from,
                                   KhronosObjectAttributes& into) const {
  // Update the bounding box. Do this first as it is the reference for the mesh.
  BoundingBox new_box = into.bounding_box;
  new_box.merge(from.bounding_box);

  // Adjust the old vertex positions.
  for (Point& vertex : into.mesh.points) {
    vertex = new_box.pointToBoxFrame(into.bounding_box.pointToWorldFrame(vertex));
  }

  // Merge incoming vertices.
  const size_t num_previous_vertices = into.mesh.numVertices();
  into.mesh.resizeVertices(num_previous_vertices + from.mesh.numVertices());
  for (size_t i = 0; i < from.mesh.numVertices(); ++i) {
    into.mesh.setPos(
        num_previous_vertices + i,
        new_box.pointToBoxFrame(from.bounding_box.pointToWorldFrame(from.mesh.pos(i))));
    into.mesh.setColor(num_previous_vertices + i, from.mesh.color(i));
  }

  // Merges incoming faces.
  const size_t idx_offset = num_previous_vertices;
  into.mesh.faces.reserve(from.mesh.faces.size() + into.mesh.faces.size());
  for (const auto& face : from.mesh.faces) {
    spark_dsg::Mesh::Face new_face = face;
    for (size_t i = 0; i < new_face.size(); ++i) {
      new_face[i] += idx_offset;
    }
    into.mesh.faces.emplace_back(new_face);
  }

  into.bounding_box = new_box;
}

void Reconciler::clampOverestimation(const KhronosObjectAttributes& reference,
                                     KhronosObjectAttributes& to_clamp) const {
  // Clamp all timings that were not observed to not extend the observed ones of the reference.
  // Assumes the first/last seen stamps are sorted.
  uint64_t first_evidence = 0;
  uint64_t last_evidence = 0;
  for (uint64_t time : reference.first_observed_ns) {
    if (time > 0ul) {
      first_evidence = time;
      break;
    }
  }
  for (uint64_t time : reference.last_observed_ns) {
    if (time < std::numeric_limits<uint64_t>::max() && time > last_evidence) {
      last_evidence = time;
    }
  }

  // Clamp the first/last seen times.
  if (first_evidence != 0ul) {
    for (uint64_t& time : to_clamp.first_observed_ns) {
      if (time == 0ul) {
        time = first_evidence;
      }
    }
  }
  if (last_evidence != 0ul) {
    for (uint64_t& time : to_clamp.last_observed_ns) {
      if (time == std::numeric_limits<uint64_t>::max()) {
        time = last_evidence;
      }
    }
  }
}

}  // namespace khronos
