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
  // Legacy raw maps preserve their node identity and observation interval.
  return "legacy:" + std::to_string(segment.node_id) + ":" +
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

double seconds(TimeStamp from, TimeStamp to) {
  return to > from ? static_cast<double>(to - from) * 1e-9 : 0.0;
}

// README (13), E_{h->o}: the counts of a candidate's exclusion verdicts. Every element of the
// candidate counts: a candidate's elements rest on its own construction frames and are stable.
model::RoundModel::Counts allVerdictCounts(const ElementRound& round) {
  model::RoundModel::Counts counts;
  for (const auto& v : round.verdicts) {
    counts.n += 1.0;
    if (v.through) {
      counts.f += 1.0;
    } else if (v.label != ElementVerdict::kNoLabel) {
      counts.n_labeled += 1.0;
      if (v.label == ElementVerdict::kForeign) counts.g += 1.0;
    }
  }
  return counts;
}

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
  view.closure_odds = fragment.closure_odds;
  view.reconstruction_frames = fragment.reconstruction_frames;
  view.odds = fragment.filter.odds();
  view.ended_since = fragment.ended_since;
  view.ended_processed = fragment.ended_processed;
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
  fragment.filter_time = last;
  fragment.exposure_clock = first;
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
}

model::ChangeFilter PersistentObjectState::newFilter(size_t id, Fragment& fragment) const {
  // README (5r): a placement born in this session starts at odds 0; an inherited one at
  // q^gap / (1 - q^gap), the posterior mean change probability of one session gap.
  if (!fragment.inherited) return model::ChangeFilter();
  const double q = prior_.gapChangeProbability(id, fragment.semantic_label);
  fragment.gap_prior = q;
  fragment.elements_at_start = fragment.elements;
  return model::ChangeFilter(q / (1.0 - q), fragment.input_boundary, fragment.input_boundary);
}

// README (13) (a): a placement committed in place at or after the start of a pending candidate
// cannot have ended before that candidate began, so the candidate is another view of it.
void PersistentObjectState::absorbCommitted(PhysicalState& state) {
  if (!state.current) return;
  for (size_t i = 0; i < state.observed_new.size();) {
    const auto& current = state.fragments[*state.current];
    const auto& candidate = state.observed_new[i];
    if (latestSupport(current) >= candidate.birth_time &&
        current.birth_time <= candidate.last_support_time) {
      absorb(state, i);
    } else {
      ++i;
    }
  }
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

void PersistentObjectState::closePending(PhysicalState& state, const TimeStamp stamp) {
  auto write = state.observed_new.begin();
  for (auto it = state.observed_new.begin(); it != state.observed_new.end(); ++it) {
    if (latestSupport(*it) <= stamp) {
      // README (5b): every closed fragment contributes to the successor frontier.
      state.succession_floor = std::max(state.succession_floor, latestSupport(*it));
      it->death_time = stamp;
      it->change_left = latestSupport(*it);
      state.closed_through = std::max(state.closed_through, stamp);
      state.fragments.push_back(std::move(*it));
    } else {
      if (write != it) *write = std::move(*it);
      ++write;
    }
  }
  state.observed_new.erase(write, state.observed_new.end());
}

void PersistentObjectState::promoteObservedNew(size_t id, PhysicalState& state) {
  if (state.observed_new.empty() || state.current) return;
  const auto index = latestPendingIndex(state);
  if (!isEligibleSuccessor(state, state.observed_new[index])) return;
  Fragment promoted = std::move(state.observed_new[index]);
  state.observed_new.erase(state.observed_new.begin() + index);
  // README (5f): a successor begins where the predecessor's change interval ends.
  if (!promoted.inherited && state.closed_through > 0 && state.closed_through <= promoted.birth_time) {
    promoted.presence_begin = state.closed_through;
  }
  promoted.filter = newFilter(id, promoted);
  state.fragments.push_back(std::move(promoted));
  state.current = state.fragments.size() - 1;
}

// README (5e), (5t): the commitment that a placement changed. Its change time is the (1 - alpha)
// interval [left, right]; the right end is the upper bound the single-time interface uses.
void PersistentObjectState::closeCurrent(size_t id, PhysicalState& state, TimeStamp commit_stamp,
                                         TimeStamp left, TimeStamp right, double odds) {
  if (!state.current) return;
  Fragment& current = state.fragments[*state.current];
  const TimeStamp death = std::max(right, current.last_support_time);
  current.death_time = death;
  current.closure_odds = odds;
  current.change_left = std::min(left, death);
  current.ended_since = commit_stamp;
  current.ended_processed = commit_stamp;
  state.succession_floor = std::max(state.succession_floor, latestSupport(current));
  state.closed_through = std::max(state.closed_through, death);
  // Pi (principle 2): the committed change is an event, the time at risk its exposure.
  prior_.addEvent(id, current.semantic_label, seconds(current.exposure_clock, death));
  current.exposure_clock = death;
  if (current.gap_pending) {
    prior_.addGapOutcome(id, current.semantic_label, true);
    current.gap_pending = false;
  }
  state.current.reset();
  state.has_dynamic_history = true;
}

// README (5), principle 5: accepted native motion (committed by the visible-motion recursion)
// ends the placement at its actual sensor time.
bool PersistentObjectState::consumeMotion(size_t id, PhysicalState& state,
                                          const KhronosObjectAttributes& attrs) {
  if (!hasMotionEvidence(attrs)) return false;
  state.has_dynamic_history = true;
  TimeStamp motion = 0;
  const size_t n = std::min(attrs.trajectory_timestamps.size(), attrs.trajectory_positions.size());
  for (size_t i = 0; i < n; ++i) {
    const auto t = attrs.trajectory_timestamps[i];
    if (t != std::numeric_limits<TimeStamp>::max()) motion = std::max(motion, t);
  }
  if (motion <= state.last_motion_consumed) return false;
  state.last_motion_consumed = motion;
  bool closed = false;
  if (state.current) {
    const auto& current = state.fragments[*state.current];
    if (motion > latestSupport(current)) {
      closeCurrent(id, state, motion, latestSupport(current), motion, model::closeOdds());
      closed = true;
    }
  }
  closePending(state, motion);
  return closed;
}

void PersistentObjectState::ingestObservation(PhysicalState& state,
                                              const KhronosObjectAttributes& attrs,
                                              const TimeStamp first, const TimeStamp last,
                                              const size_t physical_instance_id) {
  VLOG(1) << "INGEST inst=" << physical_instance_id << " first=" << (first / 1000000000ULL)
          << "s seg_verts=" << attrs.mesh.numVertices() << " cur_verts="
          << (state.current ? state.fragments[*state.current].geometry.numVertices() : 0)
          << " pending=" << state.observed_new.size();

  consumeMotion(physical_instance_id, state, attrs);

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
    fragment.filter = newFilter(physical_instance_id, fragment);
    state.fragments.push_back(std::move(fragment));
    state.current = state.fragments.size() - 1;
    if (hasMotionEvidence(attrs)) state.has_dynamic_history = true;
    return;
  }

  // README (13): the new segment is another view of the placement, its successor, or undecided;
  // until the decision of resolveRound it stays a pending candidate that keeps its geometry.
  state.observed_new.push_back(makeFragment(attrs, first, last));
  absorbCommitted(state);
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
      consumeMotion(id, state, attrs);
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

// README (7s), (7): the stable elements of the round and their counts (S = n - f hits, f see-throughs,
// the labelled hits n_labeled with g foreign).
model::RoundModel::Counts PersistentObjectState::roundCounts(size_t id, const Fragment& fragment,
                                                             const ElementRound& round) const {
  const auto prior = rounds_.elementPrior(id);
  model::RoundModel::Counts counts;
  for (const auto& v : round.verdicts) {
    ElementHistory history;
    if (const auto found = fragment.elements.find(v.key); found != fragment.elements.end()) {
      history = found->second;
    }
    if (!model::RoundModel::elementStable(prior, construction_hits_, history.hits, history.through)) {
      continue;
    }
    counts.n += 1.0;
    if (v.through) {
      counts.f += 1.0;
    } else if (v.label != ElementVerdict::kNoLabel) {
      counts.n_labeled += 1.0;
      if (v.label == ElementVerdict::kForeign) counts.g += 1.0;
    }
  }
  return counts;
}

// A round committed in place enters the statistics: the element histories (7s) and the round
// counts of the object (7h). Each source enters once (the round consumed its watermark).
void PersistentObjectState::foldRound(size_t id, Fragment& fragment, const ElementRound& round,
                                      const model::RoundModel::Counts& counts) {
  for (const auto& v : round.verdicts) {
    auto& history = fragment.elements[v.key];
    const double before_j = history.through, before_n = history.hits + history.through;
    if (v.through) history.through += 1.f;
    else history.hits += 1.f;
    rounds_.moveElementHistory(before_j, before_n, history.through, history.hits + history.through);
  }
  rounds_.addInPlaceRound(id, counts);
}

PersistentObjectState::RoundResult PersistentObjectState::resolveRound(
    const size_t id, const RoundInput& input, const std::vector<PairInput>& pending,
    const TimeStamp stamp) {
  RoundResult result;
  const auto found = states_.find(id);
  if (found == states_.end() || input.session_start == kOpenEnd) return result;
  PhysicalState& state = found->second;

  if (state.current) {
    Fragment& current = state.fragments[*state.current];
    if (current.filter_time == 0) {
      // An inherited placement's recursion begins with this session: the gap ended at its first
      // input, and committed-in-place time accrues from there.
      current.filter_time = input.session_start;
      current.exposure_clock = input.session_start;
      current.filter.setGapEnd(input.session_start);
    }
  }

  // E_{h->o}: the exclusion evidence of the candidates against the current placement.
  if (state.current) {
    const Fragment& current = state.fragments[*state.current];
    for (auto& candidate : state.observed_new) {
      for (const auto& pair : pending) {
        if (pair.evidence_key != candidate.evidence_key ||
            pair.geometry_revision != candidate.geometry_revision ||
            pair.current_key != current.evidence_key) {
          continue;
        }
        const auto counts = allVerdictCounts(pair.exclusion);
        candidate.exclusion_log_lr = counts.n > 0.0 ? rounds_.logLikelihoodRatio(id, counts) : 0.0;
        candidate.exclusion_known = true;
        candidate.exclusion_current = current.evidence_key;
        candidate.exclusion_revision = candidate.geometry_revision;
      }
    }
  }

  // The round of the current placement: (5r) and (5e).
  if (state.current) {
    Fragment& current = state.fragments[*state.current];
    if (ownsEvidence(current, input) && input.measured_through == stamp) {
      const auto counts = roundCounts(id, current, input.elements);
      // A round with verdicts advances the recursion even where none of its elements is stable
      // yet (likelihood ratio one: the prior hazard alone); a round without any source does not.
      if (!input.elements.verdicts.empty() && stamp > current.filter_time) {
        const TimeStamp begin = current.filter_time;
        const double log_lr = rounds_.logLikelihoodRatio(id, counts);
        // A placement born in this session is in place from its first observation: the time it has
        // been observed since the last credited exposure is exposure too.
        const double uncredited = current.inherited ? 0.0 : seconds(current.exposure_clock, begin);
        const double q = prior_.changeProbability(id, current.semantic_label, seconds(begin, stamp),
                                                  uncredited);
        current.filter.update(begin, stamp, q, log_lr);
        current.filter_time = stamp;
        switch (current.filter.commitment()) {
          case model::Commitment::kCommitNotH: {
            // In place: the statistics, the exposure, the anchor of (5t).
            foldRound(id, current, input.elements, counts);
            prior_.addExposure(id, current.semantic_label, seconds(current.exposure_clock, stamp));
            current.exposure_clock = stamp;
            current.filter.anchorAtConfirmation();
            if (input.elements.latest_hit > 0) {
              current.last_confirmed_support = std::max(current.last_confirmed_support,
                                                        std::min(input.elements.latest_hit, stamp));
            }
            if (current.gap_pending) {
              prior_.addGapOutcome(id, current.semantic_label, false);
              current.gap_pending = false;
            }
            result.confirmed = true;
            break;
          }
          case model::Commitment::kCommitH: {
            auto interval = current.filter.changeInterval();
            TimeStamp right = interval.right;
            // A candidate that begins inside the interval bounds the change time from above.
            for (const auto& candidate : state.observed_new) {
              if (candidate.birth_time >= interval.left && candidate.birth_time < right) {
                right = candidate.birth_time;
              }
            }
            closeCurrent(id, state, stamp, interval.left, right, current.filter.odds());
            result.closed = true;
            break;
          }
          case model::Commitment::kDefer:
            break;
        }
      }
    }
  }

  if (state.current) {
    absorbCommitted(state);
    // README (5o): the candidates against the current placement. With no exclusion evidence the
    // odds equal the placement's own (the prior hazard over the time since its last evidence).
    for (size_t i = 0; state.current && i < state.observed_new.size();) {
      const Fragment& current = state.fragments[*state.current];
      const auto& candidate = state.observed_new[i];
      const double uncredited =
          current.inherited ? 0.0 : seconds(current.exposure_clock, current.filter_time);
      const double q = prior_.changeProbability(id, current.semantic_label,
                                                seconds(current.filter_time, stamp), uncredited);
      double odds = (current.filter.odds() + q) / (1.0 - q);
      if (candidate.exclusion_known && candidate.exclusion_current == current.evidence_key &&
          candidate.exclusion_revision == candidate.geometry_revision) {
        odds *= std::exp(candidate.exclusion_log_lr);
      }
      switch (model::decide(odds)) {
        case model::Commitment::kCommitNotH:
          absorb(state, i);
          result.absorbed = true;
          break;
        case model::Commitment::kCommitH: {
          auto interval = current.filter.changeInterval();
          if (interval.right == 0) interval = {latestSupport(current), candidate.birth_time};
          TimeStamp right = interval.right;
          if (candidate.birth_time >= interval.left && candidate.birth_time < right) {
            right = candidate.birth_time;
          }
          closeCurrent(id, state, stamp, std::min(interval.left, right), right, odds);
          result.closed = true;
          break;
        }
        case model::Commitment::kDefer:
          ++i;
          break;
      }
    }
  }
  if (!state.current) promoteObservedNew(id, state);
  return result;
}

void PersistentObjectState::addEndedRound(size_t id, uint64_t evidence_key,
                                          const ElementRound& round, TimeStamp stamp) {
  const auto found = states_.find(id);
  if (found == states_.end()) return;
  for (auto& fragment : found->second.fragments) {
    if (fragment.evidence_key != evidence_key || !fragment.death_time || fragment.ended_since == 0) {
      continue;
    }
    const auto counts = roundCounts(id, fragment, round);
    if (counts.n > 0.0) rounds_.addEndedRound(counts);
    fragment.ended_processed = std::max(fragment.ended_processed, stamp);
  }
}

void PersistentObjectState::finalizePendingAbsences(const TimeStamp stamp) {
  (void)stamp;
  for (auto& [id, state] : states_) {
    if (!state.current) promoteObservedNew(id, state);
  }
}

void PersistentObjectState::setMapResolution(const float resolution) {
  if (!std::isfinite(resolution) || !(resolution > 0.0f)) {
    throw std::invalid_argument("PersistentObjectState map resolution must be positive");
  }
  map_resolution_ = resolution;
}

void PersistentObjectState::setConstructionHits(const double hits) {
  if (!std::isfinite(hits) || hits < 0.0) throw std::invalid_argument("Invalid construction hits");
  construction_hits_ = hits;
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
      fragment.inherited = true;
      fragment.gap_pending = true;
      fragment.filter_time = 0;     // the recursion starts with the first round of the session
      fragment.exposure_clock = 0;
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
  background_obligations_.clear();
}

void PersistentObjectState::clear() {
  states_.clear();
  background_obligations_.clear();
  prior_ = model::PersistencePrior();
  rounds_ = model::RoundModel();
  background_removed_ = background_judged_ = 0.0;
}

std::map<size_t, TimeStamp> PersistentObjectState::successionFloors() const {
  std::map<size_t, TimeStamp> floors;
  for (const auto& [id, state] : states_) {
    if (state.closed_through > 0) floors.emplace(id, state.closed_through);
  }
  return floors;
}

std::map<size_t, model::PersistencePrior::Hazard> PersistentObjectState::motionPriors() const {
  std::map<size_t, model::PersistencePrior::Hazard> priors;
  for (const auto& [id, state] : states_) {
    int cls = -1;
    if (state.current) cls = state.fragments[*state.current].semantic_label;
    else if (!state.fragments.empty()) cls = state.fragments.back().semantic_label;
    else if (!state.observed_new.empty()) cls = state.observed_new.back().semantic_label;
    priors.emplace(id, prior_.hazard(id, cls));
  }
  return priors;
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

std::vector<std::pair<size_t, PersistentObjectState::FragmentView>>
PersistentObjectState::endedWatch() const {
  std::vector<std::pair<size_t, FragmentView>> result;
  for (const auto& [id, state] : states_) {
    for (const auto& fragment : state.fragments) {
      if (fragment.death_time && fragment.ended_since > 0 && fragment.geometry.numVertices() > 0) {
        result.emplace_back(id, viewOf(fragment));
      }
    }
  }
  return result;
}

std::optional<PersistentObjectState::StartOfSessionPrior> PersistentObjectState::startOfSessionPrior(
    const size_t physical_instance_id) const {
  const auto it = states_.find(physical_instance_id);
  if (it == states_.end() || !it->second.current) return std::nullopt;
  const Fragment& current = it->second.fragments[*it->second.current];
  if (!current.inherited) return std::nullopt;
  StartOfSessionPrior prior;
  prior.change_probability = current.gap_prior;
  for (const auto& [key, history] : current.elements_at_start) {
    prior.histories[key] = {history.hits, history.through};
  }
  return prior;
}

double PersistentObjectState::backgroundGapProbability() const {
  return (background_removed_ + 0.5) / (background_judged_ + 1.0);
}

void PersistentObjectState::recordBackgroundOutcome(double removed, double judged) {
  if (!(removed >= 0.0) || !(judged >= removed)) throw std::invalid_argument("Invalid background outcome");
  background_removed_ += removed;
  background_judged_ += judged;
}

std::vector<PersistentObjectState::ClosedSurface> PersistentObjectState::closedSurfaces() const {
  std::vector<ClosedSurface> result;
  for (const auto& [id, state] : states_) {
    (void)id;
    for (const auto& fragment : state.fragments) {
      if (!fragment.death_time || !(fragment.closure_odds > 0.0) ||
          fragment.geometry.numVertices() == 0) {
        continue;
      }
      ClosedSurface surface;
      surface.odds = fragment.closure_odds;
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
