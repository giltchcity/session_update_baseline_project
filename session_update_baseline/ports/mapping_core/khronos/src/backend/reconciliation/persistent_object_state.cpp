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


// Number of surface points in `current` that occupy the same map voxel (or a
// directly neighbouring voxel) as a surface point of `candidate`. This is
// geometric co-observation, not an object-level threshold: it counts evidence
// that two sessions sampled the same physical surface.
size_t sharedSurfaceSamples(const spark_dsg::Mesh& current,
                            const BoundingBox& current_box,
                            const spark_dsg::Mesh& candidate,
                            const BoundingBox& candidate_box,
                            float resolution) {
  if (current.points.empty() || candidate.points.empty()) {
    return 0;
  }
  const auto key = [resolution](const Point& p) {
    return std::make_tuple(static_cast<int64_t>(std::floor(p.x() / resolution)),
                           static_cast<int64_t>(std::floor(p.y() / resolution)),
                           static_cast<int64_t>(std::floor(p.z() / resolution)));
  };
  std::set<std::tuple<int64_t, int64_t, int64_t>> candidate_voxels;
  for (const auto& local : candidate.points) {
    candidate_voxels.insert(key(candidate_box.pointToWorldFrame(local)));
  }
  size_t shared = 0;
  for (const auto& local : current.points) {
    const auto voxel = key(current_box.pointToWorldFrame(local));
    bool found = false;
    for (int dx = -1; dx <= 1 && !found; ++dx) {
      for (int dy = -1; dy <= 1 && !found; ++dy) {
        for (int dz = -1; dz <= 1; ++dz) {
          if (candidate_voxels.count(std::make_tuple(
                  std::get<0>(voxel) + dx,
                  std::get<1>(voxel) + dy,
                  std::get<2>(voxel) + dz)) > 0) {
            ++shared;
            found = true;
            break;
          }
        }
      }
    }
  }
  return shared;
}

// Reproject `mesh`'s vertices from `from` frame into `to` frame in place.'s vertices from `from` frame into `to` frame in place.
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

// The moveability prior is deliberately generic and config-driven:
//   observed D1 history (has_dynamic_history)
//   + past relocation frequency (closed temporal fragments)
//   + a semantic ontology list supplied from the mapping configuration.
// There is no physical-ID table and no hardcoded furniture class list here;
// a semantic prior is a weak hint, never a correctness decision.
bool PersistentObjectState::isHighMobility(const PhysicalState& state,
                                           const Fragment& current) const {
  if (state.has_dynamic_history) {
    return true;
  }
  for (const auto& fragment : state.fragments) {
    if (fragment.death_time) {
      return true;
    }
  }
  return high_mobility_semantic_labels_.count(current.semantic_label) > 0;
}

void PersistentObjectState::setHighMobilitySemanticLabels(
    const std::vector<int>& labels) {
  high_mobility_semantic_labels_.clear();
  high_mobility_semantic_labels_.insert(labels.begin(), labels.end());
}

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
  // A direct observation is support for the state it observed. For fragments
  // restored from a previous session this is reset by initializeFromObjects:
  // A's observation timestamps are not evidence in B.
  fragment.last_confirmed_support = last;
  fragment.requires_current_session_support = false;
  fragment.semantic_label = attrs.semantic_label;
  fragment.reconstruction_frames = detailValue(attrs, kReconstructionFramesDetail);
  return fragment;
}

void PersistentObjectState::mergeObservationIntoFragment(Fragment& target,
                                                        const KhronosObjectAttributes& attrs,
                                                        const TimeStamp first,
                                                        const TimeStamp last) {
  appendMeshUnion(target.geometry, target.bbox, attrs.mesh, attrs.bounding_box);
  target.position = target.bbox.world_P_center.cast<double>();
  target.reconstruction_frames += detailValue(attrs, kReconstructionFramesDetail);
  target.last_support_time = std::max(target.last_support_time, last);
  target.last_confirmed_support =
      std::max(target.last_confirmed_support, last);
  target.birth_time = std::min(target.birth_time, first);
}

void PersistentObjectState::mergeObservedNew(PhysicalState& state,
                                             const KhronosObjectAttributes& attrs,
                                             const TimeStamp first,
                                             const TimeStamp last) {
  // One slot, not competing candidates. Every observation that did not belong
  // to CURRENT is unioned here. A later "which candidate should win" step does
  // not exist, so geometry that pure-B would have accumulated cannot be lost.
  if (!state.observed_new) {
    state.observed_new = makeFragment(attrs, first, last);
    return;
  }
  mergeObservationIntoFragment(*state.observed_new, attrs, first, last);
}

void PersistentObjectState::absorbObservedThrough(PhysicalState& state,
                                                  const TimeStamp stamp) {
  if (!state.current || !state.observed_new) {
    return;
  }
  // Precondition: a real measurement confirmed CURRENT present through `stamp`.
  // One physical ID cannot be in two places at one instant, so the accumulated
  // non-current observations are more views of the same state.
  if (state.observed_new->birth_time > stamp) {
    return;
  }
  Fragment& current = state.fragments[*state.current];
  appendMeshUnion(current.geometry, current.bbox,
                  state.observed_new->geometry, state.observed_new->bbox);
  current.position = current.bbox.world_P_center.cast<double>();
  current.reconstruction_frames += state.observed_new->reconstruction_frames;
  current.last_support_time =
      std::max(current.last_support_time, state.observed_new->last_support_time);
  current.last_confirmed_support =
      std::max(current.last_confirmed_support,
               state.observed_new->last_confirmed_support);
  current.birth_time = std::min(current.birth_time, state.observed_new->birth_time);
  state.observed_new.reset();
}

void PersistentObjectState::promoteObservedNew(PhysicalState& state) {
  if (!state.observed_new || state.current) {
    return;
  }
  state.fragments.push_back(std::move(*state.observed_new));
  state.observed_new.reset();
  state.current = state.fragments.size() - 1;
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

void PersistentObjectState::archiveSessionState(PhysicalState& state,
                                                TimeStamp stamp) {
  if (!state.b_session) {
    return;
  }
  PhysicalState& b = *state.b_session;
  // Identity conflict or different-site candidate: keep both hypotheses as
  // closed history fragments. Never union them, never delete them.
  if (b.current) {
    Fragment& fragment = b.fragments[*b.current];
    fragment.death_time = std::max(stamp, fragment.last_support_time);
    state.fragments.push_back(std::move(fragment));
    b.current.reset();
  }
  if (b.observed_new) {
    b.observed_new->death_time = stamp;
    state.fragments.push_back(std::move(*b.observed_new));
    b.observed_new.reset();
  }
  state.b_session.reset();
}

void PersistentObjectState::ingestObservation(PhysicalState& state,
                                              const KhronosObjectAttributes& attrs,
                                              const TimeStamp first,
                                              const TimeStamp last,
                                              const size_t physical_instance_id,
                                              const float map_resolution) {
  LOG(INFO) << "INGEST inst=" << physical_instance_id
            << " first=" << (first / 1000000000ULL)
            << "s seg_verts=" << attrs.mesh.numVertices()
            << " cur_verts=" << (state.current ? state.fragments[*state.current].geometry.numVertices() : 0)
            << " observed_verts=" << (state.observed_new ? state.observed_new->geometry.numVertices() : 0);

  // Nothing established yet: this observation opens the first fragment. No state is being
  // displaced, so no contradiction evidence is required.
  if (!state.current) {
    if (state.observed_new) {
      mergeObservedNew(state, attrs, first, last);
      return;
    }
    state.fragments.push_back(makeFragment(attrs, first, last));
    state.current = state.fragments.size() - 1;
    if (hasMotionEvidence(attrs)) {
      state.has_dynamic_history = true;
    }
    return;
  }

  // NEW_STATE requires direct evidence that the state we hold no longer holds. Tracker motion
  // evidence is exactly that: the object was watched leaving (D1). Any observations accumulated
  // while the old state was still CURRENT are not mixed into the new state: motion identifies the
  // new state directly.
  if (hasMotionEvidence(attrs)) {
    closeCurrent(state, first);
    state.fragments.push_back(makeFragment(attrs, first, last));
    state.current = state.fragments.size() - 1;
    state.observed_new.reset();
    state.pending_absence_stamp = 0;
    state.has_dynamic_history = true;
    return;
  }

  const Fragment& current = state.fragments[*state.current];
  if (current.requires_current_session_support) {
    // B observations never merge into A's old mesh online. They run their own
    // mini D1/D2 session state; the inherited-vs-session comparison happens
    // once at terminal finalization.
    LOG(INFO) << "INGEST_DECIDE inst=" << physical_instance_id
              << " inherited_session_deferred=true";
    if (!state.b_session) {
      state.b_session = std::make_unique<PhysicalState>();
    }
    ingestObservation(*state.b_session, attrs, first, last,
                      physical_instance_id, map_resolution);
    return;
  }
  // Within one session, two surface maps of the same site refine each other
  // directly when they actually share surface. A stale support timestamp must
  // not merge a later observation from a different site; that decision belongs
  // to the ray evidence in resolveCurrentEvidence.
  const bool same_session_overlap =
      surfacesShareSpace(current.geometry, current.bbox, attrs.mesh,
                         attrs.bounding_box, map_resolution);
  LOG(INFO) << "INGEST_DECIDE inst=" << physical_instance_id
            << " same_session_overlap=" << same_session_overlap;
  if (same_session_overlap) {
    state.pending_absence_stamp = 0;
    mergeObservationIntoFragment(state.fragments[*state.current], attrs, first, last);
    return;
  }

  // Neither current support nor current-session surface overlap. Put the
  // observation in the one replacement slot; do not decide whether the object
  // moved yet.
  mergeObservedNew(state, attrs, first, last);
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
    ingestObservation(state, *attrs, first, last, *instance_id,
                     map_resolution_);
  }

  state.last_merged_observation_first = observationFirstStamp(merged);

  // Only the CURRENT fragment is materialized. While an inherited A state and
  // an independent B-session state coexist, materialize their union so the
  // timeline shows A+B refinement online; the registry still owns them
  // separately and can still close A later without losing B geometry.
  if (state.current) {
    const Fragment& current = state.fragments[*state.current];
    if (current.requires_current_session_support &&
        state.b_session && state.b_session->current) {
      const bool already_absent = inheritedEvidenceAbsent(
          state,
          current,
          state.last_support_rays,
          state.last_contradiction_rays,
          state.last_geometric_support,
          state.last_surface_samples);
      const Fragment& b_current =
          state.b_session->fragments[*state.b_session->current];
      const size_t shared = sharedSurfaceSamples(
          current.geometry, current.bbox,
          b_current.geometry, b_current.bbox, map_resolution_);
      // Different-location fragments are never unioned. Static identities may
      // accumulate disjoint views (a wardrobe's front and back), but a movable
      // identity's B state is the same physical surface only when it actually
      // shares surface with the inherited state.
      const bool same_site =
          !isHighMobility(state, current) || shared > 0;
      LOG(INFO) << "MATERIALIZE inst=" << *instance_id
                << " inherited_verts=" << current.geometry.numVertices()
                << " session_verts=" << b_current.geometry.numVertices()
                << " shared=" << shared
                << " high_mobility=" << isHighMobility(state, current)
                << " already_absent=" << already_absent;
      if (!already_absent && same_site) {
        // Same physical state: A+B refinement is visible online.
        merged.mesh = current.geometry;
        merged.bounding_box = current.bbox;
        appendMeshUnion(merged.mesh, merged.bounding_box,
                        b_current.geometry, b_current.bbox);
        merged.position = merged.bounding_box.world_P_center.cast<double>();
        merged.details[kReconstructionFramesDetail] = {
            current.reconstruction_frames + b_current.reconstruction_frames};
        merged.details[kHasDynamicHistoryDetail] = {
            state.has_dynamic_history ? 1u : 0u};
      } else {
        // Old site contradicted, or the B-session state occupies a different
        // site. Materialize the inherited state alone; the next evidence round
        // performs the atomic handoff, or the terminal round archives the
        // session state separately.
        merged.mesh = current.geometry;
        merged.bounding_box = current.bbox;
        merged.position = current.position;
        merged.details[kReconstructionFramesDetail] = {
            current.reconstruction_frames};
        merged.details[kHasDynamicHistoryDetail] = {
            state.has_dynamic_history ? 1u : 0u};
      }
    } else {
      merged.mesh = current.geometry;
      merged.bounding_box = current.bbox;
      merged.position = current.position;
      merged.details[kReconstructionFramesDetail] = {current.reconstruction_frames};
      merged.details[kHasDynamicHistoryDetail] = {state.has_dynamic_history ? 1u : 0u};
    }
  } else if (!state.fragments.empty()) {
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
  closeCurrent(it->second, stamp);
  promoteObservedNew(it->second);
  return true;
}

bool PersistentObjectState::reportCurrentSupported(const size_t physical_instance_id,
                                                   const TimeStamp stamp) {
  const auto it = states_.find(physical_instance_id);
  if (it == states_.end() || !it->second.current) {
    return false;
  }
  PhysicalState& state = it->second;
  Fragment& current = state.fragments[*state.current];
  current.last_support_time = std::max(current.last_support_time, stamp);
  current.last_confirmed_support =
      std::max(current.last_confirmed_support, stamp);
  absorbObservedThrough(state, stamp);
  return true;
}

bool PersistentObjectState::inheritedEvidenceAbsent(const PhysicalState& state,
                             const Fragment& current,
                             size_t support,
                             size_t contradiction,
                             size_t geometric,
                             size_t samples) {
  const double scale = samples > 0 ? static_cast<double>(samples) : 1.0;
  const double contradiction_rate =
      static_cast<double>(contradiction) / scale;
  // Generic moveability prior: for a movable identity, surface overlap with B
  // is not evidence that A's old site still exists (a moved object may graze
  // its old footprint, and its B state may be far away). For a static identity
  // overlap is strong co-observation even across disjoint viewpoints.
  const bool trust_geometric_overlap = !isHighMobility(state, current);
  const double support_rate =
      (static_cast<double>(support) +
       (trust_geometric_overlap ? static_cast<double>(geometric) : 0.0)) /
      scale;
  return contradiction_rate > support_rate;
}

size_t PersistentObjectState::finalizePendingAbsences(const TimeStamp stamp) {
  size_t closed = 0;
  for (auto& [id, state] : states_) {
    (void)id;
    if (!state.current) {
      promoteObservedNew(state);
      state.pending_absence_stamp = 0;
      continue;
    }

    Fragment& current = state.fragments[*state.current];
    if (!current.requires_current_session_support) {
      if (state.pending_absence_stamp != 0 && !state.observed_new) {
        closeCurrent(state, stamp);
        ++closed;
      }
      state.pending_absence_stamp = 0;
      continue;
    }

    // Compare the frozen inherited state with the independent B-session state.
    const size_t support = state.last_support_rays;
    const size_t contradiction = state.last_contradiction_rays;
    const size_t geometric = state.last_geometric_support;
    const size_t samples = state.last_surface_samples;
    const double scale = samples > 0 ? static_cast<double>(samples) : 1.0;
    const double contradiction_rate =
        static_cast<double>(contradiction) / scale;
    const bool trust_geometric_overlap = !isHighMobility(state, current);
    const double support_rate =
        (static_cast<double>(support) +
         (trust_geometric_overlap ? static_cast<double>(geometric) : 0.0)) /
        scale;

    const bool have_b_current =
        state.b_session && state.b_session->current;
    const bool inherited_absent =
        contradiction_rate > support_rate && (have_b_current || support == 0);

    if (inherited_absent) {
      closeCurrent(state, stamp);
      if (have_b_current) {
        // Move B's fully resolved current into the top-level fragments.
        PhysicalState& b = *state.b_session;
        state.fragments.push_back(
            std::move(b.fragments[*b.current]));
        b.current.reset();
        state.current = state.fragments.size() - 1;
        if (b.observed_new) {
          // Any leftover candidate is a different site: archive, never union.
          b.observed_new->death_time = stamp;
          state.fragments.push_back(std::move(*b.observed_new));
          b.observed_new.reset();
        }
      }
      ++closed;
    } else if (have_b_current) {
      PhysicalState& b = *state.b_session;
      const Fragment& b_current = b.fragments[*b.current];
      const size_t shared = sharedSurfaceSamples(
          current.geometry, current.bbox,
          b_current.geometry, b_current.bbox, map_resolution_);
      // Static A+B completion is only safe when the two fragments actually
      // co-observe the same surface, or when the identity is static (disjoint
      // viewpoints of one wardrobe still refine each other). A movable
      // identity whose B state does not touch the inherited site is kept as a
      // separate hypothesis, never merged.
      const bool same_site = !isHighMobility(state, current) || shared > 0;
      LOG(INFO) << "FINALIZE inst=" << id
                << " shared=" << shared
                << " same_site=" << same_site
                << " inherited_verts=" << current.geometry.numVertices()
                << " session_verts=" << b_current.geometry.numVertices();
      if (same_site) {
        appendMeshUnion(current.geometry, current.bbox,
                        b_current.geometry, b_current.bbox);
        current.position = current.bbox.world_P_center.cast<double>();
        current.reconstruction_frames += b_current.reconstruction_frames;
        current.last_support_time =
            std::max(current.last_support_time, b_current.last_support_time);
        current.last_confirmed_support =
            std::max(current.last_confirmed_support, b_current.last_confirmed_support);
        current.birth_time = std::min(current.birth_time, b_current.birth_time);
      } else {
        // Different site and not absent: identity conflict or a hidden move.
        // Keep the inherited state CURRENT; archive the B-session hypotheses
        // as closed fragments instead of merging or deleting them.
        archiveSessionState(state, stamp);
      }
    }
    state.b_session.reset();
    state.pending_absence_stamp = 0;
  }
  return closed;
}

bool PersistentObjectState::resolveCurrentEvidence(
    const size_t physical_instance_id,
    const SurfaceEvidence& inherited_evidence,
    const SurfaceEvidence& session_evidence,
    const TimeStamp stamp) {
  const auto it = states_.find(physical_instance_id);
  if (it == states_.end()) {
    return false;
  }
  PhysicalState& state = it->second;

  LOG(INFO) << "EVIDENCE inst=" << physical_instance_id
            << " inherited_support=" << inherited_evidence.support_rays
            << " inherited_contradiction="
            << inherited_evidence.contradiction_rays
            << " session_support=" << session_evidence.support_rays
            << " session_contradiction=" << session_evidence.contradiction_rays
            << " cur_verts=" << (state.current ? state.fragments[*state.current].geometry.numVertices() : 0)
            << " observed_verts=" << (state.observed_new ? state.observed_new->geometry.numVertices() : 0);

  // Resolve the independent B-session mini state first. Its D2 decisions are
  // allowed online because both the old and the new observations belong to B.
  if (state.b_session) {
    PhysicalState& b = *state.b_session;
    const size_t support = session_evidence.support_rays;
    const size_t contradiction = session_evidence.contradiction_rays;
    const size_t samples = session_evidence.surface_samples;

    if (b.current) {
      const size_t geom =
          b.observed_new
              ? sharedSurfaceSamples(b.fragments[*b.current].geometry,
                                     b.fragments[*b.current].bbox,
                                     b.observed_new->geometry,
                                     b.observed_new->bbox,
                                     map_resolution_)
              : 0;
      LOG(INFO) << "SESSION_EVIDENCE inst=" << physical_instance_id
                << " support=" << support
                << " contradiction=" << contradiction
                << " geometric=" << geom
                << " samples=" << samples
                << " current_verts="
                << b.fragments[*b.current].geometry.numVertices()
                << " observed_verts="
                << (b.observed_new ? b.observed_new->geometry.numVertices() : 0);

      const double scale = samples > 0 ? static_cast<double>(samples) : 1.0;
      const double support_rate = static_cast<double>(support) / scale;
      const double contradiction_rate =
          static_cast<double>(contradiction) / scale;
      const double geometric_rate = static_cast<double>(geom) / scale;

      // The map follows the real world: a candidate at a different site is the
      // object's current place as soon as the old site is no longer actively
      // ray-supported (the camera sees the object elsewhere and nothing
      // confirms it at the old site). The old site is preserved as a closed
      // history fragment -- never deleted by the new position. Contradiction
      // dominance (the old site was seen empty) also closes it; this remains
      // the only path for objects that disappear without a replacement.
      if (b.observed_new &&
          (contradiction_rate > support_rate + geometric_rate ||
           support_rate == 0.0)) {
        closeCurrent(b, stamp);
        promoteObservedNew(b);
        b.has_dynamic_history = true;
      } else if (support_rate > 0.0 || geom > 0) {
        Fragment& current_b = b.fragments[*b.current];
        current_b.last_support_time =
            std::max(current_b.last_support_time, stamp);
        current_b.last_confirmed_support =
            std::max(current_b.last_confirmed_support, stamp);
        // Absorb the accumulated candidate only when it is actually the same
        // site. A movable identity's candidate at a different location (an
        // in-session move, cabinet X->Y) must stay a separate hypothesis until
        // free-space evidence closes the current site.
        const bool same_site =
            !isHighMobility(b, current_b) || geom > 0;
        LOG(INFO) << "SESSION_ABSORB inst=" << physical_instance_id
                  << " geom=" << geom
                  << " high_mobility=" << isHighMobility(b, current_b)
                  << " absorb=" << same_site;
        if (same_site) {
          absorbObservedThrough(b, stamp);
        }
      }
    }
  }

  // The inherited fragment is frozen until terminal finalization. Store the
  // latest cumulative evidence against it.
  if (state.current &&
      state.fragments[*state.current].requires_current_session_support) {
    Fragment& inherited = state.fragments[*state.current];
    state.last_support_rays = inherited_evidence.support_rays;
    state.last_contradiction_rays = inherited_evidence.contradiction_rays;
    state.last_surface_samples = inherited_evidence.surface_samples;
    state.last_geometric_support =
        state.b_session && state.b_session->current
            ? sharedSurfaceSamples(
                  inherited.geometry, inherited.bbox,
                  state.b_session->fragments[*state.b_session->current].geometry,
                  state.b_session->fragments[*state.b_session->current].bbox,
                  map_resolution_)
            : 0;

    // Online D2/D3 transition: as soon as the B-session state exists and A's
    // old surface is seen through, switch CURRENT to the B state. Do not wait
    // until the end of the session.
    const bool inherited_absent = inheritedEvidenceAbsent(
        state,
        inherited,
        inherited_evidence.support_rays,
        inherited_evidence.contradiction_rays,
        state.last_geometric_support,
        inherited_evidence.surface_samples);
    if (inherited_absent && state.b_session &&
        state.b_session->current) {
      PhysicalState& b = *state.b_session;
      closeCurrent(state, stamp);
      state.fragments.push_back(
          std::move(b.fragments[*b.current]));
      b.current.reset();
      state.current = state.fragments.size() - 1;
      if (b.observed_new) {
        // A leftover candidate is a different site: archive, never union.
        b.observed_new->death_time = stamp;
        state.fragments.push_back(std::move(*b.observed_new));
        b.observed_new.reset();
      }
      state.b_session.reset();
      state.pending_absence_stamp = 0;
      return true;
    }
    state.pending_absence_stamp = stamp;
    return false;
  }

  // A session-local top-level current uses the same support-dominance rule as
  // the mini B state above. After an online promotion the current is a normal
  // top-level fragment, so its evidence arrives in `inherited_evidence`
  // (the only non-empty measurement slot).
  if (state.current) {
    const bool use_inherited_slot =
        session_evidence.surface_samples == 0 &&
        inherited_evidence.surface_samples > 0;
    const SurfaceEvidence& evidence =
        use_inherited_slot ? inherited_evidence : session_evidence;
    PhysicalState& b = state;
    const size_t support = evidence.support_rays;
    const size_t contradiction = evidence.contradiction_rays;
    const size_t samples = evidence.surface_samples;
    const size_t geom =
        b.observed_new
            ? sharedSurfaceSamples(b.fragments[*b.current].geometry,
                                   b.fragments[*b.current].bbox,
                                   b.observed_new->geometry,
                                   b.observed_new->bbox,
                                   map_resolution_)
            : 0;
    const double scale = samples > 0 ? static_cast<double>(samples) : 1.0;
    const double contradiction_rate =
        static_cast<double>(contradiction) / scale;
    const double support_rate = static_cast<double>(support) / scale;
    const double geometric_rate = static_cast<double>(geom) / scale;

    if (b.observed_new &&
        contradiction_rate > support_rate + geometric_rate) {
      closeCurrent(b, stamp);
      promoteObservedNew(b);
      return true;
    }
    if (support_rate > 0.0 || geom > 0) {
      Fragment& current = b.fragments[*b.current];
      current.last_support_time = std::max(current.last_support_time, stamp);
      current.last_confirmed_support =
          std::max(current.last_confirmed_support, stamp);
      // Same rule as the mini B state: a movable identity's different-site
      // candidate is never absorbed into the current site's geometry.
      const bool same_site =
          !isHighMobility(b, current) || geom > 0;
      LOG(INFO) << "TOP_ABSORB inst=" << physical_instance_id
                << " geom=" << geom
                << " high_mobility=" << isHighMobility(b, current)
                << " absorb=" << same_site;
      if (same_site) {
        absorbObservedThrough(b, stamp);
      }
    }
  }
  return false;
}

void PersistentObjectState::setMapResolution(const float resolution) {
  if (!(resolution > 0.0f)) {
    throw std::invalid_argument("PersistentObjectState map resolution must be positive");
  }
  map_resolution_ = resolution;
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
    state.observed_new.reset();
    state.pending_absence_stamp = 0;
    state.current.reset();
    if (attrs->bounding_box.isValid()) {
      state.fragments.push_back(makeFragment(*attrs, first, last));
      state.current = 0;
      // A's observation is the state we inherit, but it is not a B-ray
      // measurement. Until B itself sees this surface, an overlapping new
      // observation must not be treated as "CURRENT still confirmed present".
      state.fragments.back().last_confirmed_support = 0;
      state.fragments.back().requires_current_session_support = true;
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

std::optional<PersistentObjectState::FragmentView>
PersistentObjectState::sessionCurrentFragment(
    const size_t physical_instance_id) const {
  const auto it = states_.find(physical_instance_id);
  if (it == states_.end() || !it->second.b_session ||
      !it->second.b_session->current) {
    return std::nullopt;
  }
  const PhysicalState& b = *it->second.b_session;
  return viewOf(b.fragments[*b.current]);
}

std::vector<PersistentObjectState::FragmentView> PersistentObjectState::historyFragments(
    const size_t physical_instance_id) const {
  const auto it = states_.find(physical_instance_id);
  return it == states_.end() ? std::vector<FragmentView>{} : viewsOf(it->second.fragments);
}

std::optional<PersistentObjectState::FragmentView> PersistentObjectState::observedNew(
    const size_t physical_instance_id) const {
  const auto it = states_.find(physical_instance_id);
  if (it == states_.end() || !it->second.observed_new) {
    return std::nullopt;
  }
  return viewOf(*it->second.observed_new);
}

std::vector<PersistentObjectState::FragmentView>
PersistentObjectState::unresolvedCandidates(
    const size_t physical_instance_id) const {
  const auto observed = observedNew(physical_instance_id);
  return observed ? std::vector<FragmentView>{*observed}
                  : std::vector<FragmentView>{};
}

}  // namespace khronos
