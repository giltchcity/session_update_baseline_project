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

#include "khronos/backend/reconciliation/persistent_object_state.h"

#include <algorithm>
#include <tuple>

#include <glog/logging.h>

#include "khronos/backend/update_khronos_objects_functor.h"
#include "khronos/utils/khronos_attribute_utils.h"

namespace khronos {

namespace {

// Matches UpdateKhronosObjectsFunctor::mergeObjectAttributes' physical-move
// threshold: a stationary re-observation whose box does not intersect the
// reference and whose center is at least this far away is a real
// displacement of the same physical identity, not the same footprint.
constexpr float kPhysicalMoveDisplacementM = 1.0f;

size_t detailValue(const KhronosObjectAttributes& attrs, const char* key) {
  const auto iter = attrs.details.find(key);
  return (iter == attrs.details.end() || iter->second.empty()) ? 0 : iter->second.front();
}

bool hasMotionEvidence(const KhronosObjectAttributes& attrs) {
  return detailValue(attrs, kHasDynamicHistoryDetail) != 0;
}

bool isDisplacedFrom(const BoundingBox& reference, const BoundingBox& candidate) {
  return !reference.intersects(candidate) &&
         (candidate.world_P_center - reference.world_P_center).norm() >=
             kPhysicalMoveDisplacementM;
}

// Reproject `mesh`'s vertices from `from` frame into `to` frame in place.
void reprojectMeshFrame(spark_dsg::Mesh& mesh, const BoundingBox& from, const BoundingBox& to) {
  for (auto& vertex : mesh.points) {
    vertex = to.pointToBoxFrame(from.pointToWorldFrame(vertex));
  }
}

// Append `add_mesh` (expressed in `add_bbox` frame) into `into_mesh`
// (expressed in `into_bbox` frame, reprojected to `union_bbox` alongside the
// new vertices). Mismatched optional-field layouts (colors/timestamps/labels)
// are defensively skipped per-field rather than throwing: this only happens
// for hand-built test meshes with inconsistent flags, never for the uniform
// production MeshObjectExtractor output.
void appendMeshUnion(spark_dsg::Mesh& into_mesh,
                     BoundingBox& into_bbox,
                     const spark_dsg::Mesh& add_mesh,
                     const BoundingBox& add_bbox) {
  BoundingBox union_bbox = into_bbox;
  union_bbox.merge(add_bbox);

  reprojectMeshFrame(into_mesh, into_bbox, union_bbox);

  const size_t offset = into_mesh.numVertices();
  into_mesh.resizeVertices(offset + add_mesh.numVertices());
  for (size_t i = 0; i < add_mesh.numVertices(); ++i) {
    into_mesh.setPos(offset + i, union_bbox.pointToBoxFrame(add_bbox.pointToWorldFrame(add_mesh.pos(i))));
    // Per-field copies must also be size-defensive, not just flag-defensive:
    // production object meshes (utils::combineMeshLayer) are built with the
    // default Mesh flags (has_timestamps=true) while their blocks were
    // extracted from a with_tracking=false map, leaving `stamps` (and friends)
    // empty. Reading such a field through the flagged getter would throw
    // vector::at on an empty vector.
    const bool add_has_colors =
        add_mesh.has_colors && add_mesh.colors.size() == add_mesh.points.size();
    const bool add_has_timestamps =
        add_mesh.has_timestamps && add_mesh.stamps.size() == add_mesh.points.size();
    const bool add_has_first_seen = add_mesh.has_first_seen_stamps &&
                                    add_mesh.first_seen_stamps.size() == add_mesh.points.size();
    const bool add_has_labels =
        add_mesh.has_labels && add_mesh.labels.size() == add_mesh.points.size();
    if (into_mesh.has_colors) {
      into_mesh.setColor(offset + i, add_has_colors ? add_mesh.color(i) : Color());
    }
    if (into_mesh.has_timestamps) {
      into_mesh.setTimestamp(offset + i, add_has_timestamps ? add_mesh.timestamp(i) : 0);
    }
    if (into_mesh.has_first_seen_stamps) {
      into_mesh.setFirstSeenTimestamp(
          offset + i, add_has_first_seen ? add_mesh.firstSeenTimestamp(i) : 0);
    }
    if (into_mesh.has_labels) {
      into_mesh.setLabel(offset + i, add_has_labels ? add_mesh.label(i) : 0);
    }
  }
  if ((into_mesh.has_colors != add_mesh.has_colors) ||
      (into_mesh.has_timestamps != add_mesh.has_timestamps) ||
      (into_mesh.has_first_seen_stamps != add_mesh.has_first_seen_stamps) ||
      (into_mesh.has_labels != add_mesh.has_labels)) {
    VLOG(2) << "PersistentObjectState: accumulating meshes with mismatched "
               "optional-field layouts; missing fields defaulted to zero.";
  }

  const size_t idx_offset = offset;
  into_mesh.faces.reserve(into_mesh.faces.size() + add_mesh.faces.size());
  for (const auto& face : add_mesh.faces) {
    auto new_face = face;
    for (auto& index : new_face) {
      index += idx_offset;
    }
    into_mesh.faces.emplace_back(new_face);
  }

  into_bbox = union_bbox;
}

struct Segment {
  NodeId node_id;
  const KhronosObjectAttributes* attrs;
};

std::vector<Segment> collectSegments(const DynamicSceneGraph& graph,
                                     const std::vector<NodeId>& nodes) {
  std::vector<Segment> segments;
  segments.reserve(nodes.size());
  for (const auto node_id : nodes) {
    if (!graph.hasNode(node_id)) {
      continue;
    }
    const auto* attrs = graph.getNode(node_id).tryAttributes<KhronosObjectAttributes>();
    if (attrs) {
      segments.push_back({node_id, attrs});
    }
  }
  std::sort(segments.begin(), segments.end(), [](const Segment& lhs, const Segment& rhs) {
    return std::make_tuple(
               observationFirstStamp(*lhs.attrs), observationLastStamp(*lhs.attrs), lhs.node_id) <
           std::make_tuple(
               observationFirstStamp(*rhs.attrs), observationLastStamp(*rhs.attrs), rhs.node_id);
  });
  return segments;
}

}  // namespace

void PersistentObjectState::applyPhysicalGeometry(const DynamicSceneGraph& graph,
                                                   const std::vector<NodeId>& nodes,
                                                   KhronosObjectAttributes& merged) {
  const auto instance_id = UpdateKhronosObjectsFunctor::physicalInstanceId(merged);
  if (!instance_id) {
    return;
  }

  const auto segments = collectSegments(graph, nodes);
  if (segments.empty()) {
    return;
  }

  State& state = states_[*instance_id];
  // Captured before this round mutates state: distinguishes "establishing
  // this ID's canonical geometry for the first time" from "this ID already
  // has a persisted canonical shape to protect".
  const bool had_prior_canonical = !state.ingested_intervals.empty();

  // Idempotence locks. Skip the anchor (last round's own merge target) and
  // any segment whose exact observation interval was already ingested.
  std::vector<const Segment*> to_process;
  to_process.reserve(segments.size());
  for (size_t i = 0; i < segments.size(); ++i) {
    const auto& segment = segments[i];
    const auto first = observationFirstStamp(*segment.attrs);
    const auto last = observationLastStamp(*segment.attrs);
    if (had_prior_canonical && i == 0 && first == state.canonical_observation_first) {
      continue;  // anchor lock
    }
    if (state.ingested_intervals.count({first, last}) > 0) {
      continue;  // interval lock
    }
    to_process.push_back(&segment);
  }

  if (!to_process.empty()) {
    // had_prior_canonical implies the composite already holds this ID's
    // established canonical geometry. Guard on a valid bbox too: if every
    // prior round for this ID was trajectory-only (no geometry-bearing
    // segment has ever been ingested), there is nothing established yet to
    // protect from a "move", so the first geometry-bearing segment below
    // still takes the plain-initialization path.
    bool have_composite = had_prior_canonical && state.canonical_bbox.isValid();
    // Fixed for the whole round: whether there was a genuinely established
    // shape (from a *previous* round or initializeFromObjects) to protect
    // from a move. A segment that establishes have_composite for the first
    // time mid-round must not retroactively be treated as "established"
    // for a later segment's move classification within this same round.
    const bool established_before_round = have_composite;

    for (const auto* segment_ptr : to_process) {
      const auto& segment = *segment_ptr;
      const auto* attrs = segment.attrs;
      const auto first = observationFirstStamp(*attrs);
      const auto last = observationLastStamp(*attrs);
      state.ingested_intervals.insert({first, last});

      if (attrs->mesh.points.empty()) {
        // Trajectory-only observation: contributes no geometry. Current
        // canonical mesh/pose remain exactly as established.
        continue;
      }

      if (!have_composite) {
        // First-ever geometry for this ID (this round establishes it).
        state.canonical_mesh = attrs->mesh;
        state.canonical_bbox = attrs->bounding_box;
        state.canonical_position = attrs->position;
        state.reconstruction_frames = detailValue(*attrs, kReconstructionFramesDetail);
        state.has_dynamic_history = hasMotionEvidence(*attrs);
        have_composite = true;
        continue;
      }

      const bool moved =
          hasMotionEvidence(*attrs) || isDisplacedFrom(state.canonical_bbox, attrs->bounding_box);

      if (moved) {
        state.has_dynamic_history = true;
        state.canonical_position = attrs->position;
        state.reconstruction_frames = detailValue(*attrs, kReconstructionFramesDetail);
        if (established_before_round) {
          // An established persistent shape survives a move: only the
          // pose/bbox change, the canonical local-frame geometry is left
          // untouched (rigid translation is implicit, since the mesh stays
          // expressed relative to the new bounding box's origin).
          state.canonical_bbox = attrs->bounding_box;
        } else {
          // Still establishing this ID for the first time: no established
          // shape exists yet to protect, so the moved segment's own
          // reconstruction becomes the (new) running composite, matching
          // the original single-round winner-takes-all reduction.
          state.canonical_mesh = attrs->mesh;
          state.canonical_bbox = attrs->bounding_box;
        }
      } else {
        // Stationary re-observation: accumulate regardless of relative
        // reconstruction support -- every visibility segment of the same
        // static object is evidence of more of its surface, not a
        // competing claim to be the "real" current mesh.
        appendMeshUnion(state.canonical_mesh, state.canonical_bbox, attrs->mesh, attrs->bounding_box);
        state.canonical_position = state.canonical_bbox.world_P_center.cast<double>();
        state.reconstruction_frames += detailValue(*attrs, kReconstructionFramesDetail);
      }
    }
  }

  state.canonical_observation_first = observationFirstStamp(merged);

  // The canonical composite is authoritative only if this ID has actually
  // established geometry (valid bbox). An ID whose segments are all
  // trajectory-only never enters the have_composite path, so its state
  // remains the default-constructed values; overwriting the merge result
  // with those (INVALID bbox, zero position, cleared dynamic flag) would
  // corrupt a valid trajectory-derived node.
  if (state.canonical_bbox.isValid()) {
    merged.mesh = state.canonical_mesh;
    merged.bounding_box = state.canonical_bbox;
    merged.position = state.canonical_position;
    merged.details[kReconstructionFramesDetail] = {state.reconstruction_frames};
    merged.details[kHasDynamicHistoryDetail] = {state.has_dynamic_history ? 1u : 0u};
  }
}

void PersistentObjectState::initializeFromObjects(const DynamicSceneGraph& dsg) {
  if (!dsg.hasLayer(DsgLayers::OBJECTS)) {
    return;
  }
  const auto& objects = dsg.getLayer(DsgLayers::OBJECTS);
  for (const auto& [node_id, node] : objects.nodes()) {
    (void)node_id;
    const auto* attrs = node->tryAttributes<KhronosObjectAttributes>();
    if (!attrs) {
      continue;
    }
    const auto instance_id = UpdateKhronosObjectsFunctor::physicalInstanceId(*attrs);
    if (!instance_id) {
      continue;
    }
    State& state = states_[*instance_id];
    state.canonical_mesh = attrs->mesh;
    state.canonical_bbox = attrs->bounding_box;
    state.canonical_position = attrs->position;
    state.canonical_observation_first = observationFirstStamp(*attrs);
    state.ingested_intervals.clear();
    state.ingested_intervals.insert(
        {observationFirstStamp(*attrs), observationLastStamp(*attrs)});
    state.reconstruction_frames = detailValue(*attrs, kReconstructionFramesDetail);
    state.has_dynamic_history = hasMotionEvidence(*attrs);
  }
}

void PersistentObjectState::clear() { states_.clear(); }

size_t PersistentObjectState::numStates() const { return states_.size(); }

bool PersistentObjectState::hasState(size_t physical_instance_id) const {
  return states_.count(physical_instance_id) > 0;
}

}  // namespace khronos
