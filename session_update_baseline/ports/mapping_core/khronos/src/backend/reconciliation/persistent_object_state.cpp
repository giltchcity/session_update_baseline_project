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
#include <cmath>
#include <set>
#include <tuple>

#include <glog/logging.h>

#include "khronos/backend/update_khronos_objects_functor.h"
#include "khronos/utils/khronos_attribute_utils.h"

namespace khronos {

namespace {

size_t detailValue(const KhronosObjectAttributes& attrs, const char* key) {
  const auto iter = attrs.details.find(key);
  return (iter == attrs.details.end() || iter->second.empty()) ? 0 : iter->second.front();
}

bool hasMotionEvidence(const KhronosObjectAttributes& attrs) {
  return detailValue(attrs, kHasDynamicHistoryDetail) != 0;
}

// The map's own reconstruction scale. Two surfaces that occupy the same space at this resolution
// are the same site seen twice; this is the sensor/model scale, not a tunable "moved far enough"
// distance.
constexpr float kMapResolutionM = 0.05f;

// SUPPORT evidence between two surfaces: do they occupy any common voxel? Both meshes are stored
// in their own bounding-box frame, so each is lifted to world first.
//
// True means SAME_STATE: these are two views of one surface. False means only "this observation
// does not support the state we hold" -- never "the object moved". Two views of one static object
// routinely share no surface at all (a wardrobe's front and its back), so the caller must treat a
// false here as UNRESOLVED until real contradiction evidence arrives.
bool surfacesShareSpace(const spark_dsg::Mesh& current,
                        const BoundingBox& current_box,
                        const spark_dsg::Mesh& candidate,
                        const BoundingBox& candidate_box,
                        float resolution) {
  if (current.points.empty() || candidate.points.empty()) {
    return false;
  }
  const auto key = [resolution](const Point& p) {
    return std::make_tuple(static_cast<int64_t>(std::floor(p.x() / resolution)),
                           static_cast<int64_t>(std::floor(p.y() / resolution)),
                           static_cast<int64_t>(std::floor(p.z() / resolution)));
  };
  std::set<std::tuple<int64_t, int64_t, int64_t>> occupied;
  for (const auto& local : current.points) {
    occupied.insert(key(current_box.pointToWorldFrame(local)));
  }
  for (const auto& local : candidate.points) {
    if (occupied.count(key(candidate_box.pointToWorldFrame(local)))) {
      return true;
    }
  }
  return false;
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

PersistentObjectState::FragmentView PersistentObjectState::viewOf(const Fragment& fragment) {
  FragmentView view;
  view.geometry = &fragment.geometry;
  view.bbox = &fragment.bbox;
  view.position = fragment.position;
  view.birth_time = fragment.birth_time;
  view.last_support_time = fragment.last_support_time;
  view.death_time = fragment.death_time;
  view.reconstruction_frames = fragment.reconstruction_frames;
  return view;
}

std::vector<PersistentObjectState::FragmentView> PersistentObjectState::viewsOf(
    const std::vector<Fragment>& fragments) {
  std::vector<FragmentView> views;
  views.reserve(fragments.size());
  for (const auto& fragment : fragments) {
    views.push_back(viewOf(fragment));
  }
  return views;
}

PersistentObjectState::Fragment PersistentObjectState::makeFragment(
    const KhronosObjectAttributes& attrs, const TimeStamp first, const TimeStamp last) {
  Fragment fragment;
  // Provenance rule: a fragment's geometry is exactly what was observed of *this* state. It is
  // never a previous fragment's mesh re-anchored to a new box, and never a union across states.
  fragment.geometry = attrs.mesh;
  fragment.bbox = attrs.bounding_box;
  fragment.position = attrs.position;
  fragment.birth_time = first;
  fragment.last_support_time = last;
  fragment.reconstruction_frames = detailValue(attrs, kReconstructionFramesDetail);
  return fragment;
}

void PersistentObjectState::absorbReachableCandidates(PhysicalState& state) {
  if (!state.current) {
    return;
  }
  bool absorbed_any = true;
  while (absorbed_any && !state.unresolved.empty()) {
    absorbed_any = false;
    for (auto it = state.unresolved.begin(); it != state.unresolved.end(); ++it) {
      Fragment& target = state.fragments[*state.current];
      if (!surfacesShareSpace(
              target.geometry, target.bbox, it->geometry, it->bbox, kMapResolutionM)) {
        continue;
      }
      appendMeshUnion(target.geometry, target.bbox, it->geometry, it->bbox);
      target.position = target.bbox.world_P_center.cast<double>();
      target.reconstruction_frames += it->reconstruction_frames;
      target.last_support_time = std::max(target.last_support_time, it->last_support_time);
      target.birth_time = std::min(target.birth_time, it->birth_time);
      state.unresolved.erase(it);
      absorbed_any = true;
      break;  // `it` is invalidated, and CURRENT has grown -- rescan from the start.
    }
  }
}

void PersistentObjectState::closeCurrent(PhysicalState& state, const TimeStamp stamp) {
  if (!state.current) {
    return;
  }
  Fragment& current = state.fragments[*state.current];
  // Upper bound, not a measured instant: the state ended somewhere in (last_support, stamp]. Never
  // record a death preceding the last moment the fragment was actually supported.
  current.death_time = std::max(stamp, current.last_support_time);
  state.current.reset();
  state.has_dynamic_history = true;
}

void PersistentObjectState::promoteNewestCandidate(PhysicalState& state) {
  if (state.current || state.unresolved.empty()) {
    return;
  }
  // The most recently supported candidate is the best available account of where the object is
  // now. It is promoted with the geometry it was observed with; nothing is carried over from the
  // fragment that was just closed.
  const auto newest = std::max_element(
      state.unresolved.begin(), state.unresolved.end(), [](const Fragment& a, const Fragment& b) {
        return a.last_support_time < b.last_support_time;
      });
  state.fragments.push_back(std::move(*newest));
  state.unresolved.erase(newest);
  state.current = state.fragments.size() - 1;
  absorbReachableCandidates(state);
}

void PersistentObjectState::ingestObservation(PhysicalState& state,
                                              const KhronosObjectAttributes& attrs,
                                              const TimeStamp first,
                                              const TimeStamp last) {
  // Nothing established yet: this observation opens the first fragment. No state is being
  // displaced, so no contradiction evidence is required.
  if (!state.current) {
    state.fragments.push_back(makeFragment(attrs, first, last));
    state.current = state.fragments.size() - 1;
    if (hasMotionEvidence(attrs)) {
      state.has_dynamic_history = true;
    }
    absorbReachableCandidates(state);
    return;
  }

  const size_t current_index = *state.current;
  const bool supports_current = surfacesShareSpace(state.fragments[current_index].geometry,
                                                   state.fragments[current_index].bbox,
                                                   attrs.mesh,
                                                   attrs.bounding_box,
                                                   kMapResolutionM);

  // A non-overlapping observation is still SAME_STATE when CURRENT is known to have been present
  // at the moment that observation was made: one physical ID cannot be in two places at one
  // instant, so the two surfaces are two sides of one object. That confirmation comes from real
  // measurements (reportCurrentSupported), never from proximity. Successive partial scans of a
  // static object -- a wardrobe's front, then its back -- take this path.
  const bool current_confirmed_present =
      state.fragments[current_index].last_support_time >= first;

  if (supports_current || current_confirmed_present) {
    // SAME_STATE: another view of the site we already hold. Refine it, then re-check whether the
    // enlarged surface now reaches any candidate that was previously disjoint from it.
    Fragment& target = state.fragments[current_index];
    appendMeshUnion(target.geometry, target.bbox, attrs.mesh, attrs.bounding_box);
    target.position = target.bbox.world_P_center.cast<double>();
    target.reconstruction_frames += detailValue(attrs, kReconstructionFramesDetail);
    target.last_support_time = std::max(target.last_support_time, last);
    absorbReachableCandidates(state);
    return;
  }

  // NEW_STATE requires direct evidence that the state we hold no longer holds. Tracker motion
  // evidence is exactly that: the object was watched leaving (D1). Geometric disjointness is not,
  // which is why it is not consulted here.
  if (hasMotionEvidence(attrs)) {
    closeCurrent(state, first);
    state.fragments.push_back(makeFragment(attrs, first, last));
    state.current = state.fragments.size() - 1;
    state.has_dynamic_history = true;
    return;
  }

  // UNRESOLVED: neither supported nor contradicted. Held as its own candidate and materialized
  // nowhere. Candidates that support each other merge, so repeated observations of one new site
  // accumulate into one candidate instead of piling up.
  for (auto& candidate : state.unresolved) {
    if (!surfacesShareSpace(
            candidate.geometry, candidate.bbox, attrs.mesh, attrs.bounding_box, kMapResolutionM)) {
      continue;
    }
    appendMeshUnion(candidate.geometry, candidate.bbox, attrs.mesh, attrs.bounding_box);
    candidate.position = candidate.bbox.world_P_center.cast<double>();
    candidate.reconstruction_frames += detailValue(attrs, kReconstructionFramesDetail);
    candidate.last_support_time = std::max(candidate.last_support_time, last);
    return;
  }
  state.unresolved.push_back(makeFragment(attrs, first, last));
}

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

  PhysicalState& state = states_[*instance_id];
  // Captured before this round mutates state: distinguishes "this ID has been processed before"
  // from "first time it is seen at all". Trajectory-only rounds count as processed.
  const bool processed_before = !state.ingested_intervals.empty();

  // Idempotence locks. Skip the anchor (last round's own merge target) and any segment whose exact
  // observation interval was already ingested.
  std::vector<const Segment*> to_process;
  to_process.reserve(segments.size());
  for (size_t i = 0; i < segments.size(); ++i) {
    const auto& segment = segments[i];
    const auto first = observationFirstStamp(*segment.attrs);
    const auto last = observationLastStamp(*segment.attrs);
    if (processed_before && i == 0 && first == state.last_merged_observation_first) {
      continue;  // anchor lock
    }
    if (state.ingested_intervals.count({first, last}) > 0) {
      continue;  // interval lock
    }
    to_process.push_back(&segment);
  }

  for (const auto* segment_ptr : to_process) {
    const auto* attrs = segment_ptr->attrs;
    const auto first = observationFirstStamp(*attrs);
    const auto last = observationLastStamp(*attrs);
    state.ingested_intervals.insert({first, last});

    if (attrs->mesh.points.empty()) {
      // Trajectory-only observation: contributes no geometry. CURRENT stays exactly as established.
      continue;
    }
    // Note: a degenerate bounding box (a sliver whose points are collinear, so the box has zero
    // extent in some axis) is still real observed geometry and is ingested like any other. Only a
    // segment with no mesh at all is skipped, above.
    ingestObservation(state, *attrs, first, last);
  }

  state.last_merged_observation_first = observationFirstStamp(merged);

  // Only the CURRENT fragment is materialized. With no CURRENT (never established, or contradicted
  // with no candidate to promote) the merge result is left untouched: whether the object is gone is
  // the node-level absence decision, not this function's.
  if (state.current) {
    const Fragment& current = state.fragments[*state.current];
    merged.mesh = current.geometry;
    merged.bounding_box = current.bbox;
    merged.position = current.position;
    merged.details[kReconstructionFramesDetail] = {current.reconstruction_frames};
    merged.details[kHasDynamicHistoryDetail] = {state.has_dynamic_history ? 1u : 0u};
  } else if (!state.fragments.empty()) {
    // Every state we held has been contradicted and nothing has been observed since. The object is
    // not currently anywhere we know of, so CURRENT materializes no geometry. Leaving the closed
    // fragment's mesh on the node is precisely the ghost this design exists to remove -- "we still
    // have detail for the old site" is never a reason to keep publishing it as current. The
    // geometry is not lost: it stays in the history, and the node's presence/trajectory are
    // untouched so node-level absence still owns whether the object exists at all.
    merged.mesh = spark_dsg::Mesh(merged.mesh.has_colors,
                                  merged.mesh.has_timestamps,
                                  merged.mesh.has_labels,
                                  merged.mesh.has_first_seen_stamps);
    merged.details[kReconstructionFramesDetail] = {0};
    merged.details[kHasDynamicHistoryDetail] = {state.has_dynamic_history ? 1u : 0u};
  }
}

bool PersistentObjectState::reportCurrentContradicted(const size_t physical_instance_id,
                                                      const TimeStamp stamp) {
  const auto it = states_.find(physical_instance_id);
  if (it == states_.end() || !it->second.current) {
    return false;
  }
  // Closing and promoting are one operation, which is what makes the outcome independent of
  // whether the contradiction or the new-site observation was processed first.
  closeCurrent(it->second, stamp);
  promoteNewestCandidate(it->second);
  return true;
}

bool PersistentObjectState::reportCurrentSupported(const size_t physical_instance_id,
                                                   const TimeStamp stamp) {
  const auto it = states_.find(physical_instance_id);
  if (it == states_.end() || !it->second.current) {
    return false;
  }
  PhysicalState& state = it->second;
  state.fragments[*state.current].last_support_time =
      std::max(state.fragments[*state.current].last_support_time, stamp);

  // Retroactive coexistence. A candidate that was observed at a moment CURRENT is now confirmed to
  // have been present cannot be a *later* state of this object -- one physical ID is not in two
  // places at one instant -- so it is another part of the same one. This is what resolves a
  // candidate that was parked before the confirming measurement arrived, and it is why support
  // evidence and observations may be processed in either order.
  for (auto it_candidate = state.unresolved.begin(); it_candidate != state.unresolved.end();) {
    if (it_candidate->birth_time > stamp) {
      ++it_candidate;
      continue;
    }
    Fragment& target = state.fragments[*state.current];
    appendMeshUnion(target.geometry, target.bbox, it_candidate->geometry, it_candidate->bbox);
    target.position = target.bbox.world_P_center.cast<double>();
    target.reconstruction_frames += it_candidate->reconstruction_frames;
    target.last_support_time = std::max(target.last_support_time, it_candidate->last_support_time);
    target.birth_time = std::min(target.birth_time, it_candidate->birth_time);
    it_candidate = state.unresolved.erase(it_candidate);
  }
  return true;
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
    const auto first = observationFirstStamp(*attrs);
    const auto last = observationLastStamp(*attrs);

    // A DSG node carries only the CURRENT materialization, so a seeded ID starts with exactly one
    // fragment. The previous session's history is not recoverable from the node, and is not
    // invented here.
    PhysicalState& state = states_[*instance_id];
    state.fragments.clear();
    state.unresolved.clear();
    state.current.reset();
    if (attrs->bounding_box.isValid()) {
      state.fragments.push_back(makeFragment(*attrs, first, last));
      state.current = 0;
    }
    state.last_merged_observation_first = first;
    state.ingested_intervals.clear();
    state.ingested_intervals.insert({first, last});
    state.has_dynamic_history = hasMotionEvidence(*attrs);
  }
}

void PersistentObjectState::clear() { states_.clear(); }

size_t PersistentObjectState::numStates() const { return states_.size(); }

bool PersistentObjectState::hasState(const size_t physical_instance_id) const {
  return states_.count(physical_instance_id) > 0;
}

std::vector<size_t> PersistentObjectState::trackedIds() const {
  std::vector<size_t> ids;
  ids.reserve(states_.size());
  for (const auto& [id, state] : states_) {
    (void)state;
    ids.push_back(id);
  }
  return ids;
}

std::optional<PersistentObjectState::FragmentView> PersistentObjectState::currentFragment(
    const size_t physical_instance_id) const {
  const auto it = states_.find(physical_instance_id);
  if (it == states_.end() || !it->second.current) {
    return std::nullopt;
  }
  return viewOf(it->second.fragments[*it->second.current]);
}

std::vector<PersistentObjectState::FragmentView> PersistentObjectState::historyFragments(
    const size_t physical_instance_id) const {
  const auto it = states_.find(physical_instance_id);
  return it == states_.end() ? std::vector<FragmentView>{} : viewsOf(it->second.fragments);
}

std::vector<PersistentObjectState::FragmentView> PersistentObjectState::unresolvedCandidates(
    const size_t physical_instance_id) const {
  const auto it = states_.find(physical_instance_id);
  return it == states_.end() ? std::vector<FragmentView>{} : viewsOf(it->second.unresolved);
}

}  // namespace khronos
