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

#include "session_core/state/persistent_object_state.h"
#include "session_core/adapters/physical_object_access.h"
#include "session_core/adapters/extraction_source.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iterator>
#include <limits>
#include <set>
#include <tuple>
#include <unordered_map>

#include <glog/logging.h>

#include "khronos/backend/update_khronos_objects_functor.h"
#include "khronos/utils/khronos_attribute_utils.h"

namespace khronos {
using session_detail::inputFirstStamp;
using session_detail::inputLastStamp;

namespace {

std::atomic<uint64_t> next_fragment_evidence_key{1};

// The current materialization and the latest input observation have distinct
// provenance. These fields travel with the existing DSG node, README section 7.
constexpr auto kCurrentExists = "session_current_exists";
constexpr auto kCurrentBirth = "session_current_birth";
constexpr auto kCurrentSupport = "session_current_support";
constexpr auto kCurrentTrackFirst = "session_current_track_first";
// README (5b): the interval (last support, first valid contradiction] of the latest closed state.
constexpr auto kChangeAfter = "session_change_after";
constexpr auto kChangeBefore = "session_change_before";


size_t detailValue(const KhronosObjectAttributes& attrs, const char* key);

TimeStamp trackFirstSeen(const KhronosObjectAttributes& attrs, const TimeStamp first) {
  const size_t value = detailValue(attrs, kTrackFirstSeenDetail);
  return value > 0 ? static_cast<TimeStamp>(value) : first;
}

size_t detailValue(const KhronosObjectAttributes& attrs, const char* key) {
  const auto iter = attrs.details.find(key);
  return (iter == attrs.details.end() || iter->second.empty()) ? 0 : iter->second.front();
}

bool hasMotionEvidence(const KhronosObjectAttributes& attrs) {
  return hasTrajectoryHistory(attrs) || detailValue(attrs, kHasDynamicHistoryDetail) != 0;
}

// Express the same world surface in the merged bounding-box frame.
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

std::string sourceKey(const Segment& segment) {
  if (const auto source = session_detail::extractionSource(*segment.attrs)) {
    return "request:" + std::to_string(source->scope[0]) + ":" +
        std::to_string(source->scope[1]) + ":" + std::to_string(source->generation);
  }
  // Raw maps of the old format (README 7.1 "old-format import") preserve their node identity and
  // observation interval.
  return "old-format:" + std::to_string(segment.node_id) + ":" +
      std::to_string(inputFirstStamp(*segment.attrs)) + ":" +
      std::to_string(inputLastStamp(*segment.attrs));
}

auto sourceOrder(const Segment& segment) {
  if (const auto source = session_detail::extractionSource(*segment.attrs)) {
    return std::make_tuple(source->input_through,source->scope[0],source->scope[1],
                           source->generation,segment.node_id);
  }
  return std::make_tuple(inputLastStamp(*segment.attrs),uint64_t{0},uint64_t{0},
                         inputFirstStamp(*segment.attrs),segment.node_id);
}

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
  std::sort(segments.begin(),segments.end(),[](const Segment& left,const Segment& right) {
    return sourceOrder(left) < sourceOrder(right);
  });
  return segments;
}

}  // namespace

namespace {

constexpr TimeStamp kOpenEnd = std::numeric_limits<TimeStamp>::max();

// Union of the presence intervals of the placements, README (5f): overlapping or adjacent
// intervals merge, finite gaps stay.
std::vector<std::pair<TimeStamp, TimeStamp>> mergeIntervals(
    std::vector<std::pair<TimeStamp, TimeStamp>> intervals) {
  std::sort(intervals.begin(), intervals.end());
  std::vector<std::pair<TimeStamp, TimeStamp>> merged;
  for (const auto& interval : intervals) {
    if (!merged.empty() && interval.first <= merged.back().second) {
      merged.back().second = std::max(merged.back().second, interval.second);
    } else {
      merged.push_back(interval);
    }
  }
  return merged;
}

}  // namespace

void PersistentObjectState::reserveEvidenceKeys(uint64_t maximum) {
  if (maximum == std::numeric_limits<uint64_t>::max()) {
    throw std::overflow_error("Exhausted fragment identity space");
  }
  auto next = next_fragment_evidence_key.load(std::memory_order_relaxed);
  while (next <= maximum && !next_fragment_evidence_key.compare_exchange_weak(
      next, maximum + 1, std::memory_order_relaxed)) {}
}

TimeStamp PersistentObjectState::latestSupport(const Fragment& fragment) {
  return std::max(fragment.last_support_time, fragment.last_confirmed_support);
}

bool PersistentObjectState::isEligibleSuccessor(const PhysicalState& state,
                                                const Fragment& candidate) {
  TimeStamp frontier = state.succession_floor;
  if (state.current) frontier = std::max(frontier, latestSupport(state.fragments[*state.current]));
  return (!state.current && frontier == 0) || latestSupport(candidate) > frontier;
}

bool PersistentObjectState::ownsEvidence(const Fragment& fragment, const RoundInput& evidence) {
  return evidence.evidence_key != 0 && evidence.evidence_key == fragment.evidence_key &&
      evidence.geometry_revision == fragment.geometry_revision &&
      evidence.measured_through >= std::max({fragment.birth_time, fragment.input_boundary,
                                             latestSupport(fragment)});
}

PersistentObjectState::FragmentView PersistentObjectState::viewOf(const Fragment& fragment) {
  FragmentView view;
  view.geometry = &fragment.geometry;
  view.bbox = &fragment.bbox;
  view.position = fragment.position;
  view.evidence_key = fragment.evidence_key;
  view.geometry_revision = fragment.geometry_revision;
  view.inherited = fragment.inherited;
  view.semantic_label = fragment.semantic_label;
  view.birth_time = fragment.birth_time;
  view.presence_begin = fragment.presence_begin;
  view.track_first_seen = fragment.track_first_seen;
  view.last_support_time = fragment.last_support_time;
  view.input_boundary = fragment.input_boundary;
  view.last_confirmed_support = fragment.last_confirmed_support;
  view.death_time = fragment.death_time;
  view.change_left = fragment.change_left;
  view.reconstruction_frames = fragment.reconstruction_frames;
  view.cusum = fragment.cusum.value();
  view.ended_since = fragment.ended_since;
  view.zeta_e = fragment.zeta_e;
  view.direct_since = fragment.direct_since;
  view.direct_looks = fragment.direct_looks;
  return view;
}

std::vector<PersistentObjectState::FragmentView> PersistentObjectState::viewsOf(
    const std::vector<Fragment>& fragments) {
  std::vector<FragmentView> views;
  views.reserve(fragments.size());
  for (const auto& fragment : fragments) views.push_back(viewOf(fragment));
  return views;
}

PersistentObjectState::Fragment PersistentObjectState::makeFragment(
    const KhronosObjectAttributes& attrs, const TimeStamp first, const TimeStamp last) {
  Fragment fragment;
  fragment.evidence_key = next_fragment_evidence_key.fetch_add(1, std::memory_order_relaxed);
  // Provenance rule: a fragment's geometry is exactly what was observed of *this* state. It is
  // never a previous fragment's mesh re-anchored to a new box, and never a union across states.
  fragment.geometry = attrs.mesh;
  fragment.bbox = attrs.bounding_box;
  fragment.position = attrs.position;
  fragment.birth_time = first;
  fragment.presence_begin = first;
  fragment.track_first_seen = trackFirstSeen(attrs, first);
  fragment.last_support_time = last;
  // A direct observation is support for the state it observed. For fragments restored from a
  // previous session this is reset by initializeFromObjects: A's observation timestamps are not
  // evidence in B.
  fragment.last_confirmed_support = last;
  fragment.looked_through = last;
  fragment.last_decisive = last;
  fragment.semantic_label = attrs.semantic_label;
  fragment.reconstruction_frames = detailValue(attrs, kReconstructionFramesDetail);
  // README (4): the distinct capture-frame keys of the reconstruction, when the extractor recorded
  // them; the count is then their number.
  const auto keys = attrs.details.find(kFrameStampsDetail);
  if (keys != attrs.details.end() && !keys->second.empty()) {
    fragment.frame_keys.assign(keys->second.begin(), keys->second.end());
    std::sort(fragment.frame_keys.begin(), fragment.frame_keys.end());
    fragment.frame_keys.erase(std::unique(fragment.frame_keys.begin(), fragment.frame_keys.end()),
                              fragment.frame_keys.end());
    fragment.reconstruction_frames = fragment.frame_keys.size();
  }
  return fragment;
}

// README (4): all compatible fragment merges use this reduction.
void PersistentObjectState::mergeFragments(Fragment& target, const Fragment& observation) {
  if (target.geometry_revision == std::numeric_limits<uint64_t>::max()) {
    throw std::overflow_error("Exhausted fragment geometry revisions");
  }
  appendMeshUnion(target.geometry, target.bbox, observation.geometry, observation.bbox);
  ++target.geometry_revision;
  target.position = target.bbox.world_P_center.cast<double>();
  // README (4): the frame count is the number of distinct capture-frame keys; a fragment of an
  // older map without keys adds its stored count.
  if (!target.frame_keys.empty() && !observation.frame_keys.empty()) {
    std::vector<TimeStamp> keys;
    std::set_union(target.frame_keys.begin(), target.frame_keys.end(),
                   observation.frame_keys.begin(), observation.frame_keys.end(),
                   std::back_inserter(keys));
    target.frame_keys = std::move(keys);
    target.reconstruction_frames = target.frame_keys.size();
  } else {
    target.frame_keys.clear();
    target.reconstruction_frames += observation.reconstruction_frames;
  }
  target.last_support_time = std::max(target.last_support_time, observation.last_support_time);
  target.input_boundary = std::max(target.input_boundary, observation.input_boundary);
  target.last_confirmed_support =
      std::max(target.last_confirmed_support, observation.last_confirmed_support);
  target.birth_time = std::min(target.birth_time, observation.birth_time);
  target.presence_begin = std::min(target.presence_begin, observation.presence_begin);
  target.track_first_seen = std::min(target.track_first_seen, observation.track_first_seen);
  // t_L: the first time the placement was directly seen in place, whichever view saw it first.
  if (observation.direct_since > 0 &&
      (target.direct_since == 0 || observation.direct_since < target.direct_since)) {
    target.direct_since = observation.direct_since;
  }
}

// README (13) (a): a placement directly seen in place at or after the start of a pending candidate
// cannot have ended before that candidate began, so the candidate is another view of it. A
// candidate established elsewhere (13 (f): directly seen in k_min looks, all of its surface outside
// the placement's extent) while the placement is also supported is not folded in: one of the two
// identity readings is a label error and the candidate stays undecided.
void PersistentObjectState::absorbCommitted(PhysicalState& state, size_t k_min) {
  if (!state.current) return;
  for (size_t i = 0; i < state.observed_new.size();) {
    const auto& current = state.fragments[*state.current];
    const auto& candidate = state.observed_new[i];
    if (establishedElsewhere(candidate, k_min)) {
      ++i;
    } else if (current.last_confirmed_support >= candidate.birth_time &&
               current.birth_time <= candidate.last_support_time) {
      absorb(state, i);
    } else {
      ++i;
    }
  }
}

// README (13) (f): the candidate was directly seen in place in at least k_min looks and all of its
// surface lies outside the extent of the current placement.
bool PersistentObjectState::establishedElsewhere(const Fragment& candidate, size_t k_min) {
  return candidate.direct_looks >= k_min && candidate.extent_tested > 0 && candidate.extent_inside == 0;
}

void PersistentObjectState::absorb(PhysicalState& state, size_t pending_index) {
  mergeFragments(state.fragments[*state.current], state.observed_new[pending_index]);
  state.observed_new.erase(state.observed_new.begin() + pending_index);
}

size_t PersistentObjectState::latestPendingIndex(const PhysicalState& state) {
  const auto& pending = state.observed_new;
  if (pending.empty()) throw std::logic_error("No pending observation to select");
  const auto found = std::max_element(pending.begin(), pending.end(), [](const auto& a, const auto& b) {
    return std::make_tuple(latestSupport(a), a.birth_time) <
        std::make_tuple(latestSupport(b), b.birth_time);
  });
  return static_cast<size_t>(found - pending.begin());
}

void PersistentObjectState::promoteObservedNew(size_t id, PhysicalState& state) {
  (void)id;
  if (state.observed_new.empty() || state.current) return;
  const auto index = latestPendingIndex(state);
  if (!isEligibleSuccessor(state, state.observed_new[index])) return;
  Fragment promoted = std::move(state.observed_new[index]);
  state.observed_new.erase(state.observed_new.begin() + index);
  // README (5f): a successor begins where the predecessor's change interval ends.
  if (!promoted.inherited && state.closed_through > 0 && state.closed_through <= promoted.birth_time) {
    promoted.presence_begin = state.closed_through;
  }
  state.fragments.push_back(std::move(promoted));
  state.current = state.fragments.size() - 1;
}

// README principle 2 (2): the way the gap before the next decisive look of a placement was observed.
model::Gap PersistentObjectState::gapOf(const Fragment& fragment, TimeStamp stamp) const {
  (void)stamp;
  if (fragment.gap_session) return model::Gap::kSession;
  // Adjacent rounds both saw the placement: the previous round was its last decisive look.
  return fragment.last_decisive >= previous_round_ && previous_round_ > 0
      ? model::Gap::kContinuous : model::Gap::kWithinSession;
}

// README principle 2 (6): the gap before a decisive look is decided by that look -- seen in place:
// not changed; ended with the interval (5t) in the gap: changed.
void PersistentObjectState::decideGap(size_t id, Fragment& fragment, TimeStamp stamp, bool changed) {
  prior_.addOutcome(id, fragment.semantic_label, gapOf(fragment, stamp), changed);
  fragment.gap_session = false;
  fragment.last_decisive = stamp;
}

// README (5e), (5t): the commitment that a placement ended. Its end time lies in the interval
// (last support, close]; the right end is the upper bound the single-time interface uses.
void PersistentObjectState::closeCurrent(size_t id, PhysicalState& state, TimeStamp commit_stamp,
                                         TimeStamp left, TimeStamp right) {
  if (!state.current) return;
  Fragment& current = state.fragments[*state.current];
  const TimeStamp death = std::max(right, current.last_support_time);
  current.death_time = death;
  current.change_left = std::min(left, death);
  current.ended_since = commit_stamp;
  state.succession_floor = std::max(state.succession_floor, latestSupport(current));
  state.closed_through = std::max(state.closed_through, death);
  // Pi (principle 2): the gap before this look ended with a change.
  decideGap(id, current, commit_stamp, true);
  state.current.reset();
  state.has_dynamic_history = true;
}

void PersistentObjectState::ingestObservation(PhysicalState& state,
                                              const KhronosObjectAttributes& attrs,
                                              const TimeStamp first, const TimeStamp last,
                                              const size_t physical_instance_id) {
  VLOG(1) << "INGEST inst=" << physical_instance_id << " first=" << (first / 1000000000ULL)
          << "s seg_verts=" << attrs.mesh.numVertices() << " cur_verts="
          << (state.current ? state.fragments[*state.current].geometry.numVertices() : 0)
          << " pending=" << state.observed_new.size();

  // Nothing established yet: this observation opens the first placement. No state is displaced,
  // so no evidence is required.
  if (!state.current) {
    if (!state.observed_new.empty() || last <= state.succession_floor) {
      state.observed_new.push_back(makeFragment(attrs, first, last));
      promoteObservedNew(physical_instance_id, state);
      return;
    }
    Fragment fragment = makeFragment(attrs, first, last);
    // README (5f): a successor begins where the predecessor's change interval ends.
    if (state.closed_through > 0 && state.closed_through <= first) {
      fragment.presence_begin = state.closed_through;
    }
    state.fragments.push_back(std::move(fragment));
    state.current = state.fragments.size() - 1;
    if (hasMotionEvidence(attrs)) state.has_dynamic_history = true;
    return;
  }

  // README (13): the new segment is another view of the placement, its successor, or undecided;
  // until the decision of resolveRound it stays a pending candidate that keeps its geometry.
  state.observed_new.push_back(makeFragment(attrs, first, last));
  absorbCommitted(state, rounds_.minHits());
}

void PersistentObjectState::ingestSegments(const DynamicSceneGraph& graph,
    const std::vector<NodeId>& nodes, size_t id) {
  auto& state = states_[id];
  for (const auto& segment : collectSegments(graph, nodes)) {
    const auto& attrs = *segment.attrs;
    if (attrs.details.count(kCurrentExists)) continue;
    if (!state.ingested_sources.insert(sourceKey(segment)).second) continue;
    const auto first = inputFirstStamp(attrs), last = inputLastStamp(attrs);
    if (first > last) throw std::invalid_argument("Invalid raw observation time interval");
    if (attrs.mesh.points.empty()) {
      if (!state.current) promoteObservedNew(id, state);
    } else {
      ingestObservation(state, attrs, first, last, id);
    }
  }
}

void PersistentObjectState::ingestObjects(const DynamicSceneGraph& graph) {
  if (!graph.hasLayer(DsgLayers::OBJECTS)) return;
  std::map<size_t, std::vector<NodeId>> groups;
  for (const auto& [node_id, node] : graph.getLayer(DsgLayers::OBJECTS).nodes()) {
    const auto* attrs = node->tryAttributes<KhronosObjectAttributes>();
    if (!attrs) continue;
    if (const auto id = session_detail::getPhysicalInstanceId(*attrs)) groups[*id].push_back(node_id);
  }
  for (const auto& [id, nodes] : groups) ingestSegments(graph, nodes, id);
}

void PersistentObjectState::applyPhysicalGeometry(const DynamicSceneGraph& graph,
    const std::vector<NodeId>& nodes, KhronosObjectAttributes& merged) {
  const auto id = UpdateKhronosObjectsFunctor::physicalInstanceId(merged);
  if (!id) return;
  ingestSegments(graph, nodes, *id);
  session_detail::setInputBounds(merged, inputFirstStamp(merged), inputLastStamp(merged));
  materializeState(states_.at(*id), merged);
}

void PersistentObjectState::materializeState(
    const PhysicalState& state, KhronosObjectAttributes& merged) const {
  if (state.current) {
    const Fragment& current = state.fragments[*state.current];
    merged.mesh = current.geometry;
    merged.bounding_box = current.bbox;
    merged.position = current.position;
    merged.details[kReconstructionFramesDetail] = {current.reconstruction_frames};
    if (current.frame_keys.empty()) merged.details.erase(kFrameStampsDetail);
    else merged.details[kFrameStampsDetail] =
        std::vector<size_t>(current.frame_keys.begin(), current.frame_keys.end());
    merged.details.erase(kChangeAfter);
    merged.details.erase(kChangeBefore);
    // Native D2 receives the observation bounds of the geometry it queries.
    const TimeStamp support = std::max(current.last_support_time, current.last_confirmed_support);
    setObservationBounds(merged, current.birth_time, support);
    merged.details[kCurrentBirth] = {current.birth_time};
    merged.details[kCurrentSupport] = {support};
    merged.details[kCurrentTrackFirst] = {current.track_first_seen};
    merged.details[kHasDynamicHistoryDetail] = {state.has_dynamic_history ? 1u : 0u};
  } else {
    merged.mesh = spark_dsg::Mesh(merged.mesh.has_colors, merged.mesh.has_timestamps,
                                 merged.mesh.has_labels, merged.mesh.has_first_seen_stamps);
    merged.details[kReconstructionFramesDetail] = {0};
    merged.details.erase(kFrameStampsDetail);
    merged.details[kHasDynamicHistoryDetail] = {state.has_dynamic_history ? 1u : 0u};
    // README (5t): the change time of the latest closed placement lies in its (1 - alpha)
    // interval; the right end is the upper bound of the single-time interface.
    const Fragment* closed = nullptr;
    for (const auto& fragment : state.fragments) {
      if (!fragment.death_time) continue;
      if (!closed || latestSupport(fragment) >= latestSupport(*closed)) closed = &fragment;
    }
    if (closed) {
      merged.details[kChangeAfter] = {closed->change_left};
      merged.details[kChangeBefore] = {*closed->death_time};
    }
  }
  merged.details[kCurrentExists] = {state.current.has_value() ? 1u : 0u};

  // README (5f): the presence of the identity is the union of the intervals [b_hat, d_hat) of its
  // placements; a placement that is still current has an open right end.
  std::vector<std::pair<TimeStamp, TimeStamp>> intervals;
  for (const auto& fragment : state.fragments) {
    intervals.emplace_back(fragment.presence_begin, fragment.death_time.value_or(kOpenEnd));
  }
  if (!intervals.empty()) {
    merged.first_observed_ns.clear();
    merged.last_observed_ns.clear();
    for (const auto& [begin, end] : mergeIntervals(std::move(intervals))) {
      merged.first_observed_ns.push_back(begin);
      merged.last_observed_ns.push_back(end);
    }
  }
}

void PersistentObjectState::materialize(DynamicSceneGraph& graph) const {
  if (!graph.hasLayer(DsgLayers::OBJECTS)) return;
  for (const auto& [node_id, node] : graph.getLayer(DsgLayers::OBJECTS).nodes()) {
    auto* attrs = dynamic_cast<KhronosObjectAttributes*>(&node->attributes());
    if (!attrs) continue;
    const auto id = UpdateKhronosObjectsFunctor::physicalInstanceId(*attrs);
    const auto found = id ? states_.find(*id) : states_.end();
    if (found == states_.end()) continue;
    session_detail::setInputBounds(*attrs, inputFirstStamp(*attrs), inputLastStamp(*attrs));
    materializeState(found->second, *attrs);
  }
}

// README (7s): a sample votes iff it was proven a real surface and never seen through while the
// placement was directly seen in place (h_e >= k_min and v_e = 0). A placement restored from a
// previous session was built by that session's valid observations (h_e >= k_min), and only v_e is
// accumulated again; a placement never directly seen in place in this session lets all samples vote.
bool PersistentObjectState::reliableSample(const Fragment& fragment, uint64_t key, size_t k_min) {
  if (!fragment.recognized) return true;
  const auto found = fragment.elements.find(key);
  const float through = found == fragment.elements.end() ? 0.f : found->second.through;
  if (fragment.inherited) return through == 0.f;
  return found != fragment.elements.end() && found->second.hits >= static_cast<float>(k_min) &&
         through == 0.f;
}

double PersistentObjectState::reliableTotal(const Fragment& fragment, size_t num_elements, size_t k_min) {
  if (!fragment.recognized) return static_cast<double>(num_elements);
  if (fragment.inherited) {
    double vetoed = 0.0;
    for (const auto& [key, element] : fragment.elements) {
      (void)key;
      if (element.through > 0.f) vetoed += 1.0;
    }
    return std::max(0.0, static_cast<double>(num_elements) - vetoed);
  }
  double total = 0.0;
  for (const auto& [key, element] : fragment.elements) {
    if (element.hits >= static_cast<float>(k_min) && element.through == 0.f) total += 1.0;
    (void)key;
  }
  return total;
}

// README (7s), (7): the reliable samples of a look, their counts n, F and the samples not yet
// judged in the accumulation.
PersistentObjectState::Look PersistentObjectState::lookOf(const Fragment& fragment,
                                                          const ElementRound& round,
                                                          size_t k_min) const {
  Look look;
  look.all_vote = !fragment.recognized;
  for (const auto& verdict : round.verdicts) {
    if (!reliableSample(fragment, verdict.key, k_min)) continue;
    look.n += 1.0;
    if (verdict.through) look.f += 1.0;
    look.predicted += verdict.predicted;
    if (!fragment.counted.count(verdict.key)) look.fresh.push_back(verdict.key);
  }
  if (look.n > 0.0) look.predicted /= look.n;
  look.reliable_total = reliableTotal(fragment, round.num_elements, k_min);
  return look;
}

// README (7): the look ratio of the object against its in-place distribution; a look without
// reliable verdicts is not a geometric look.
double PersistentObjectState::lookLogRatio(size_t id, const Look& look,
                                           model::RoundModel::InPlace* in_place) const {
  if (!(look.n > 0.0)) return 0.0;
  const auto distribution = rounds_.inPlace(id, look.predicted);
  if (in_place) *in_place = distribution;
  return model::RoundModel::logLikelihoodRatio(distribution, look.n, look.f);
}

// README (5o): the posterior odds H_new : H_same of a pending candidate, prior odds q/(1 - q) of a
// change over the gap times the likelihood ratio of the evidence of both directions.
double PersistentObjectState::candidateOdds(size_t id, const Fragment& current,
                                            const Fragment& candidate, TimeStamp stamp,
                                            CandidateRecord* record) const {
  const model::Gap gap = gapOf(current, stamp);
  const double q = prior_.changeProbability(id, current.semantic_label, gap);
  double odds = q / (1.0 - q);
  const bool known = candidate.exclusion_known && candidate.exclusion_current == current.evidence_key &&
                     candidate.exclusion_revision == candidate.geometry_revision;
  if (known) odds *= std::exp(candidate.exclusion_log_lr);
  if (record) {
    const auto report = prior_.report(id, current.semantic_label, gap);
    record->q = q;
    record->q_class = report.q_class;
    record->gaps_changed = report.changed;
    record->gaps_judged = report.judged;
    record->q_prior = !report.identified;
    record->silent = !(known && candidate.evidence_samples > 0.0);
    record->ln_lr = known ? candidate.exclusion_log_lr : 0.0;
    record->ln_odds = std::log(odds);
  }
  return odds;
}

PersistentObjectState::RoundResult PersistentObjectState::resolveRound(
    const size_t id, const RoundInput& input, const std::vector<PairInput>& pending,
    const TimeStamp stamp) {
  RoundResult result;
  const auto found = states_.find(id);
  if (found == states_.end() || input.session_start == kOpenEnd) return result;
  PhysicalState& state = found->second;
  if (stamp != round_stamp_) {
    previous_round_ = round_stamp_;
    round_stamp_ = stamp;
  }
  const size_t k_min = rounds_.minHits();

  if (state.current) {
    Fragment& current = state.fragments[*state.current];
    // An inherited placement's recursion begins with this session.
    if (current.looked_through == 0) current.looked_through = input.session_start;
  }

  // README (5o), (13) (f): what the frames said about each candidate in this round.
  if (state.current) {
    const Fragment& current = state.fragments[*state.current];
    for (auto& candidate : state.observed_new) {
      for (const auto& pair : pending) {
        if (pair.evidence_key != candidate.evidence_key ||
            pair.geometry_revision != candidate.geometry_revision ||
            pair.current_key != current.evidence_key) {
          continue;
        }
        if (pair.look_measured) {
          // The candidate's own look; the labels of one look are one independent unit.
          candidate.extent_tested = pair.candidate_samples;
          candidate.extent_inside = pair.candidate_in_extent;
          if (pair.candidate_look.recognized) {
            ++candidate.direct_looks;
            if (candidate.direct_since == 0) candidate.direct_since = pair.candidate_look.first_own;
            if (!pair.current_known || pair.current_supported) candidate.current_supported_while_seen = true;
          }
        }
        if (!pair.exclusion_evaluated) continue;
        // E_{h->o}: every element of the candidate counts -- its elements rest on its own
        // construction frames and are stable; E_{o->h}: the reliable elements of h.
        Look excluded;
        for (const auto& v : pair.exclusion.verdicts) {
          excluded.n += 1.0;
          if (v.through) excluded.f += 1.0;
          excluded.predicted += v.predicted;
        }
        if (excluded.n > 0.0) excluded.predicted /= excluded.n;
        Look supported = lookOf(current, pair.support, k_min);
        double log_lr = lookLogRatio(id, excluded) + lookLogRatio(id, supported);
        candidate.exclusion_log_lr = log_lr;
        candidate.exclusion_known = true;
        candidate.exclusion_current = current.evidence_key;
        candidate.exclusion_revision = candidate.geometry_revision;
        candidate.total_elements = pair.candidate_elements;
        candidate.away_elements = pair.candidate_away;
        candidate.cross_band = pair.cross_session_band;
        candidate.cold_mean = excluded.n > 0.0 ? excluded.predicted : supported.predicted;
        candidate.evidence_samples = excluded.n + supported.n;
      }
    }
  }

  // The look of the current placement: the CUSUM (5r) with its identity channel, learning after the
  // judgement (7), and the commitment (5e).
  if (state.current) {
    Fragment& current = state.fragments[*state.current];
    if (ownsEvidence(current, input) && input.measured_through == stamp && stamp > current.looked_through) {
      const auto& round = input.elements;
      auto& record = result.look;
      record.own_samples = round.own_samples;
      record.through_samples = round.through_samples;
      record.foreign_samples = round.foreign_samples;
      record.extent_own_samples = round.extent_own;
      record.directly_seen = round.recognized;
      record.delta_plus = round.delta_plus;
      record.k_min = k_min;
      const bool is_look = !round.verdicts.empty() || round.own_samples > 0;
      Look look;
      if (is_look) {
        record.judged = true;
        record.eps_hat = rounds_.foreignShare();
        record.eps_prior = !(rounds_.labelledSamples() > 0.0);
        // (5r): once every reliable sample of the placement is counted the next look starts a new
        // pass of coverage; C is kept.
        const double total = reliableTotal(current, round.num_elements, k_min);
        double counted_reliable = 0.0;
        for (const auto key : current.counted) {
          if (reliableSample(current, key, k_min)) counted_reliable += 1.0;
        }
        if (total > 0.0 && counted_reliable >= total) {
          current.counted.clear();
          record.coverage_restarted = true;
        }
        look = lookOf(current, round, k_min);
        model::RoundModel::InPlace in_place;
        const double log_lr = lookLogRatio(id, look, &in_place);
        const double weight =
            look.reliable_total > 0.0
                ? std::min(1.0, static_cast<double>(look.fresh.size()) / look.reliable_total) : 0.0;
        // Identity channel: a look that directly saw the placement in place is a support.
        const double identity = round.recognized ? rounds_.logForeignShare() : 0.0;
        record.all_vote = look.all_vote;
        record.reliable_samples = look.reliable_total;
        record.n = look.n;
        record.f = look.f;
        record.predicted = look.predicted;
        record.in_place = in_place;
        record.in_place_prior = look.n > 0.0 && in_place.looks == 0;
        record.ln_lr = log_lr;
        record.weight = weight;
        record.ln_identity = identity;
        record.step = current.cusum.update(weight, log_lr, identity);
        current.counted.insert(look.fresh.begin(), look.fresh.end());
        // C back at 0 is a new accumulation: the judged marks are cleared.
        if (!(current.cusum.value() > 0.0)) current.counted.clear();
        record.exceeded = current.cusum.exceeded();
        if (round.recognized) {
          // The look directly saw the placement in place: it is learned (after it was judged).
          if (look.n > 0.0) rounds_.addInPlaceLook(id, current.semantic_label, look.n, look.f);
          rounds_.addLabels(static_cast<double>(round.foreign_samples), round.labelled_samples);
          for (const auto& learned : round.learning) {
            auto& element = current.elements[learned.key];
            if (learned.own) {
              element.hits += 1.f;
              if (!current.inherited) {
                // rho_e: the mean range of the looks that make the element (principle 12).
                element.range = (element.range * element.range_count + learned.range) /
                                (element.range_count + 1.f);
                element.range_count += 1.f;
              }
            }
            if (learned.vetoed) element.through += 1.f;
          }
          current.recognized = true;
          if (current.direct_since == 0) current.direct_since = round.first_own;
          current.last_confirmed_support = std::max(current.last_confirmed_support,
                                                    std::min(round.latest_own, stamp));
          decideGap(id, current, stamp, false);
          result.confirmed = true;
        }
      }
      current.looked_through = stamp;
      record.last_support = latestSupport(current);
      // README (5r), principle 3: the evidence that the placement ended exceeds the Wald boundary.
      if (current.cusum.exceeded()) {
        const TimeStamp left = latestSupport(current);
        TimeStamp right = stamp;
        // A candidate that begins inside the interval bounds the end time from above.
        for (const auto& candidate : state.observed_new) {
          if (candidate.birth_time >= left && candidate.birth_time < right) right = candidate.birth_time;
        }
        closeCurrent(id, state, stamp, left, right);
        result.closed = true;
      }
    }
  }

  if (state.current) {
    absorbCommitted(state, k_min);
    // README principle 13: the candidates against the current placement.
    for (size_t i = 0; state.current && i < state.observed_new.size();) {
      Fragment& current = state.fragments[*state.current];
      Fragment& candidate = state.observed_new[i];
      const TimeStamp left = latestSupport(current);
      const auto close = [&]() {
        TimeStamp right = std::max(left, candidate.birth_time);
        if (candidate.birth_time < left) right = stamp;
        closeCurrent(id, state, stamp, left, std::min(right, stamp));
        result.closed = true;
      };
      CandidateRecord record;
      record.k_min = k_min;
      record.direct_looks = candidate.direct_looks;
      record.in_extent = candidate.extent_inside;
      record.tested = candidate.extent_tested;
      record.current_supported = candidate.current_supported_while_seen;
      record.elements = candidate.total_elements;
      record.away = candidate.away_elements;
      // (f) the same identity is established elsewhere: the candidate was directly seen in place in
      // at least k_min looks, all of its surface lies outside the extent of the placement, and in
      // those looks the placement had no support (not directly seen, no own-identity reading in its
      // extent): all the identity readings of the candidate would have to be label errors, with
      // probability at most eps^k_min <= alpha. If the placement was supported, one of the two is a
      // label error and the candidate stays undecided.
      if (establishedElsewhere(candidate, k_min)) {
        if (!candidate.current_supported_while_seen) {
          record.rule = CandidateRecord::Rule::kF;
          result.candidates.push_back(record);
          close();
          break;
        }
        record.rule = CandidateRecord::Rule::kUndecided;
        result.candidates.push_back(record);
        ++i;
        continue;
      }
      // (g) across sessions: this session's own reconstruction of the identity is a decisive look
      // away from the surface of the inherited placement.
      if (current.inherited && !current.recognized && !candidate.inherited && candidate.cross_band &&
          candidate.cold_mean > 0.0 && candidate.total_elements > 0 &&
          candidate.total_elements >= model::RoundModel::decisiveSamples(
              rounds_.classInPlace(current.semantic_label, candidate.cold_mean)) &&
          2 * candidate.away_elements > candidate.total_elements) {
        const double q = prior_.changeProbability(id, current.semantic_label, model::Gap::kSession);
        record.q = q;
        record.q_prior = !prior_.fit(model::Gap::kSession).identified;
        if (q > model::kAlpha) {
          record.rule = CandidateRecord::Rule::kG;
          result.candidates.push_back(record);
          close();
          break;
        }
        record.rule = CandidateRecord::Rule::kSame;
        result.candidates.push_back(record);
        absorb(state, i);
        result.absorbed = true;
        continue;
      }
      // (5o): the prior odds of a change over the gap, and the evidence of both directions.
      const double odds = candidateOdds(id, current, candidate, stamp, &record);
      // README (13) (e): with no evidence at all the observation is silent and the prior only
      // decides whether the new view joins the placement (Pr(H_same) = 1 - q >= 1 - alpha);
      // otherwise it stays undecided -- the prior alone never ends a placement.
      switch (model::decide(odds)) {
        case model::Commitment::kCommitNotH:
          record.rule = CandidateRecord::Rule::kSame;
          result.candidates.push_back(record);
          absorb(state, i);
          result.absorbed = true;
          break;
        case model::Commitment::kCommitH:
          if (record.silent) {
            record.rule = CandidateRecord::Rule::kUndecided;
            result.candidates.push_back(record);
            ++i;
            break;
          }
          record.rule = CandidateRecord::Rule::kNew;
          result.candidates.push_back(record);
          close();
          break;
        case model::Commitment::kDefer:
          record.rule = CandidateRecord::Rule::kUndecided;
          result.candidates.push_back(record);
          ++i;
          break;
      }
      if (result.closed) break;
    }
  }
  if (!state.current) promoteObservedNew(id, state);
  return result;
}

// README (13) (e), principle 3: the session-end map is a representation output; a pending
// candidate whose posterior favours H_same (odds H_new : H_same at most 1, i.e. 1 - q >= 1/2 where
// the observation is silent) is displayed with the placement. The registry is not changed.
std::optional<PersistentObjectState::DisplayGeometry> PersistentObjectState::displayGeometry(
    size_t physical_instance_id) const {
  const auto it = states_.find(physical_instance_id);
  if (it == states_.end() || !it->second.current) return std::nullopt;
  const PhysicalState& state = it->second;
  Fragment merged = state.fragments[*state.current];
  size_t count = 0;
  for (const auto& candidate : state.observed_new) {
    if (establishedElsewhere(candidate, rounds_.minHits())) continue;
    const double odds = candidateOdds(physical_instance_id, state.fragments[*state.current], candidate,
                                      round_stamp_, nullptr);
    if (odds <= 1.0) {
      mergeFragments(merged, candidate);
      ++count;
    }
  }
  if (count == 0) return std::nullopt;
  DisplayGeometry display;
  display.geometry = std::move(merged.geometry);
  display.bbox = merged.bbox;
  display.position = merged.position;
  display.merged = count;
  return display;
}

void PersistentObjectState::finalizePendingAbsences(const TimeStamp stamp) {
  (void)stamp;
  for (auto& [id, state] : states_) {
    if (!state.current) promoteObservedNew(id, state);
  }
}

void PersistentObjectState::setPreviousScale(double zeta) {
  previous_zeta_ = zeta;
  for (auto& [id, state] : states_) {
    (void)id;
    for (auto& fragment : state.fragments) {
      if (fragment.inherited && !fragment.zeta_recorded) fragment.zeta_e = zeta;
    }
    for (auto& fragment : state.observed_new) {
      if (fragment.inherited && !fragment.zeta_recorded) fragment.zeta_e = zeta;
    }
  }
}

void PersistentObjectState::setMapResolution(const float resolution) {
  if (!std::isfinite(resolution) || !(resolution > 0.0f)) {
    throw std::invalid_argument("PersistentObjectState map resolution must be positive");
  }
  map_resolution_ = resolution;
}

void PersistentObjectState::initializeFromObjects(const DynamicSceneGraph& dsg, const TimeStamp boundary) {
  decltype(states_) restored;
  if (dsg.hasLayer(DsgLayers::OBJECTS)) {
    for (const auto& [node_id, node] : dsg.getLayer(DsgLayers::OBJECTS).nodes()) {
      const auto* attrs = node->tryAttributes<KhronosObjectAttributes>();
      if (!attrs) continue;
      const auto id = UpdateKhronosObjectsFunctor::physicalInstanceId(*attrs);
      if (!id) continue;
      auto& state = restored[*id];
      const bool materialized = attrs->details.count(kCurrentExists) != 0;
      if (!materialized && !state.ingested_sources.insert(sourceKey({node_id, attrs})).second) continue;
      state.has_dynamic_history = state.has_dynamic_history || hasMotionEvidence(*attrs);
      if (!attrs->bounding_box.isValid() || attrs->mesh.points.empty()) continue;
      const auto first = materialized ? detailValue(*attrs, kCurrentBirth) : observationFirstStamp(*attrs);
      auto supported = materialized ? detailValue(*attrs, kCurrentSupport) : observationLastStamp(*attrs);
      if (supported == std::numeric_limits<TimeStamp>::max()) supported = first;
      if (supported < first || supported > boundary)
        throw std::invalid_argument("Imported support lies outside the saved state boundary");
      auto fragment = makeFragment(*attrs, first, supported);
      if (materialized) fragment.track_first_seen = detailValue(*attrs, kCurrentTrackFirst);
      fragment.last_confirmed_support = 0;
      fragment.input_boundary = boundary;
      // README principle 12: a restored placement starts this session with C = 0 and no record of
      // reliable samples; the gap before its first decisive look is a session gap.
      fragment.inherited = true;
      fragment.gap_session = true;
      fragment.zeta_e = previous_zeta_;
      fragment.looked_through = 0;  // the recursion starts with the first round of the session
      fragment.last_decisive = 0;
      // Native presence remains Khronos' decision. The actual observation fields describe
      // support; its estimated interval determines seed activity.
      const bool active = materialized ? detailValue(*attrs, kCurrentExists) != 0
          : isPresent(*attrs, boundary);
      if (active) {
        state.observed_new.push_back(std::move(fragment));
      } else if (!materialized) {
        const auto departed = lastDisappearedBefore(*attrs, boundary);
        if (departed) {
          if (*departed < supported)
            throw std::invalid_argument("Imported departure precedes actual support");
          fragment.death_time = *departed;
          fragment.change_left = supported;
          state.succession_floor = std::max(state.succession_floor, supported);
          state.closed_through = std::max(state.closed_through, *departed);
          state.has_dynamic_history = true;
          state.fragments.push_back(std::move(fragment));
        }
      }
    }
  }
  // README (5b): latest supported current, with other imported candidates retained.
  for (auto& [id, state] : restored) promoteObservedNew(id, state);
  states_ = std::move(restored);
}

void PersistentObjectState::clear() {
  states_.clear();
  prior_ = model::PersistencePrior();
  rounds_ = model::RoundModel();
  previous_round_ = round_stamp_ = 0;
}

std::map<size_t, TimeStamp> PersistentObjectState::successionFloors() const {
  std::map<size_t, TimeStamp> floors;
  for (const auto& [id, state] : states_) {
    if (state.closed_through > 0) floors.emplace(id, state.closed_through);
  }
  return floors;
}

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
  if (it == states_.end() || !it->second.current) return std::nullopt;
  return viewOf(it->second.fragments[*it->second.current]);
}

std::vector<PersistentObjectState::FragmentView> PersistentObjectState::historyFragments(
    const size_t physical_instance_id) const {
  const auto it = states_.find(physical_instance_id);
  if (it == states_.end()) return {};
  auto result = viewsOf(it->second.fragments);
  std::stable_sort(result.begin(), result.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.birth_time < rhs.birth_time;
  });
  return result;
}

std::optional<PersistentObjectState::FragmentView> PersistentObjectState::observedNew(
    const size_t physical_instance_id) const {
  const auto it = states_.find(physical_instance_id);
  if (it == states_.end() || it->second.observed_new.empty()) return std::nullopt;
  return viewOf(it->second.observed_new[latestPendingIndex(it->second)]);
}

std::vector<PersistentObjectState::FragmentView>
PersistentObjectState::unresolvedCandidates(const size_t physical_instance_id) const {
  const auto it = states_.find(physical_instance_id);
  return it == states_.end() ? std::vector<FragmentView>{} : viewsOf(it->second.observed_new);
}

std::vector<PersistentObjectState::FragmentView> PersistentObjectState::pendingNeedingExclusion(
    const size_t physical_instance_id) const {
  std::vector<FragmentView> result;
  const auto it = states_.find(physical_instance_id);
  if (it == states_.end() || !it->second.current) return result;
  const auto& state = it->second;
  const auto& current = state.fragments[*state.current];
  for (const auto& candidate : state.observed_new) {
    if (!candidate.exclusion_known || candidate.exclusion_current != current.evidence_key ||
        candidate.exclusion_revision != candidate.geometry_revision) {
      result.push_back(viewOf(candidate));
    }
  }
  return result;
}

const std::unordered_map<uint64_t, float>* PersistentObjectState::elementRanges(
    const size_t physical_instance_id) const {
  const auto it = states_.find(physical_instance_id);
  if (it == states_.end() || !it->second.current) return nullptr;
  const Fragment& current = it->second.fragments[*it->second.current];
  current.range_cache.clear();
  for (const auto& [key, element] : current.elements) {
    if (element.range > 0.f) current.range_cache[key] = element.range;
  }
  return &current.range_cache;
}

std::vector<PersistentObjectState::ClosedSurface> PersistentObjectState::closedSurfaces() const {
  std::vector<ClosedSurface> result;
  for (const auto& [id, state] : states_) {
    for (const auto& fragment : state.fragments) {
      if (!fragment.death_time || fragment.geometry.numVertices() == 0) continue;
      ClosedSurface surface;
      surface.physical_id = id;
      surface.semantic_label = fragment.semantic_label;
      for (const auto& point : fragment.geometry.points) {
        surface.vertices.push_back(fragment.bbox.pointToWorldFrame(point));
      }
      for (const auto& face : fragment.geometry.faces) {
        surface.faces.push_back({static_cast<uint32_t>(face[0]), static_cast<uint32_t>(face[1]),
                                 static_cast<uint32_t>(face[2])});
      }
      result.push_back(std::move(surface));
    }
  }
  return result;
}

std::set<std::pair<size_t, uint64_t>> PersistentObjectState::liveEvidenceKeys() const {
  std::set<std::pair<size_t, uint64_t>> result;
  for (const auto& [id, state] : states_) {
    if (state.current) result.emplace(id, state.fragments.at(*state.current).evidence_key);
    for (const auto& pending : state.observed_new) result.emplace(id, pending.evidence_key);
  }
  return result;
}

}  // namespace khronos
