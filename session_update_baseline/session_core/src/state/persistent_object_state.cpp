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

// README (3.2): the scene's declared class table chi(c) in {0,1}. A class in the movable list has
// chi = 0, one in the static list chi = 1; an undeclared class (or a missing label) has no chi and
// receives no semantic weight. The table only weights the soft update of (5).
std::optional<int> PersistentObjectState::chiOf(const Fragment& fragment) const {
  if (movable_labels_.count(fragment.semantic_label) > 0) return 0;
  if (static_labels_.count(fragment.semantic_label) > 0) return 1;
  return std::nullopt;
}

void PersistentObjectState::setHighMobilitySemanticLabels(const std::vector<int>& labels) {
  movable_labels_.clear();
  movable_labels_.insert(labels.begin(), labels.end());
}

void PersistentObjectState::setStaticSemanticLabels(const std::vector<int>& labels) {
  static_labels_.clear();
  static_labels_.insert(labels.begin(), labels.end());
}

// README (5), (5p), (5h): one POCD soft update. The observation gives the likelihoods L_in and
// L_out of a compatible and an anomalous next observation; the class adds the semantic power
// factor k_sem = 3 only in the direction the geometry of this round already points.
PersistentObjectState::PlacementUpdate PersistentObjectState::updatePlacement(
    double alpha, double beta, std::optional<int> chi, double l_in, double l_out) {
  if (!(alpha > 0 && beta > 0) || !std::isfinite(alpha) || !std::isfinite(beta) ||
      !(l_in >= 0 && l_out >= 0) || !std::isfinite(l_in) || !std::isfinite(l_out) ||
      (chi && *chi != 0 && *chi != 1)) {
    throw std::invalid_argument("Invalid placement update input");
  }
  const bool semantic = chi && ((*chi == 1 && l_in > l_out) || (*chi == 0 && l_out > l_in));
  const double k = semantic ? 3.0 : 0.0;
  const int c = chi.value_or(0);
  const double a = alpha + k * c, b = beta + k * (1 - c);
  const double denominator = a * l_in + b * l_out;
  if (!(denominator > 0)) throw std::invalid_argument("A round needs a positive likelihood");
  const double omega = a * l_in / denominator;
  const double m = (a + omega) / (a + b + 1);
  const double m2 = (omega * (a + 1) * (a + 2) + (1 - omega) * a * (a + 1)) /
                    ((a + b + 1) * (a + b + 2));
  const double variance = m2 - m * m;
  const double nu = m * (1 - m) / variance - 1;
  if (!(variance > 0) || !(nu > 0) || !std::isfinite(nu)) {
    throw std::runtime_error("The soft update moments do not define a Beta");
  }
  return {m * nu, (1 - m) * nu, omega};
}

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

bool PersistentObjectState::ownsEvidence(const Fragment& fragment,
                                          const SurfaceEvidence& evidence) {
  return evidence.evidence_key != 0 && evidence.evidence_key == fragment.evidence_key &&
      evidence.geometry_revision == fragment.geometry_revision &&
      evidence.measured_through >= std::max({fragment.birth_time, fragment.input_boundary,
                                             latestSupport(fragment)}) &&
      evidence.latest_support_stamp <= evidence.measured_through;
}

PersistentObjectState::FragmentView PersistentObjectState::viewOf(const Fragment& fragment) {
  FragmentView view;
  view.geometry = &fragment.geometry;
  view.bbox = &fragment.bbox;
  view.position = fragment.position;
  view.evidence_key = fragment.evidence_key;
  view.geometry_revision = fragment.geometry_revision;
  view.inherited = fragment.requires_current_session_support;
  view.birth_time = fragment.birth_time;
  view.track_first_seen = fragment.track_first_seen;
  view.last_support_time = fragment.last_support_time;
  view.input_boundary = fragment.input_boundary;
  view.last_confirmed_support = fragment.last_confirmed_support;
  view.death_time = fragment.death_time;
  view.reconstruction_frames = fragment.reconstruction_frames;
  view.alpha = fragment.alpha;
  view.beta = fragment.beta;
  view.prior_alpha = fragment.prior_alpha;
  view.prior_beta = fragment.prior_beta;
  view.first_contradiction = fragment.first_contradiction;
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
  fragment.evidence_key = next_fragment_evidence_key.fetch_add(1, std::memory_order_relaxed);
  // Provenance rule: a fragment's geometry is exactly what was observed of *this* state. It is
  // never a previous fragment's mesh re-anchored to a new box, and never a union across states.
  fragment.geometry = attrs.mesh;
  fragment.bbox = attrs.bounding_box;
  fragment.position = attrs.position;
  fragment.birth_time = first;
  fragment.track_first_seen = trackFirstSeen(attrs, first);
  fragment.last_support_time = last;
  // A direct observation is support for the state it observed. For fragments
  // restored from a previous session this is reset by initializeFromObjects:
  // A's observation timestamps are not evidence in B.
  fragment.last_confirmed_support = last;
  fragment.requires_current_session_support = false;
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
void PersistentObjectState::mergeFragments(Fragment& target,
                                           const Fragment& observation) {
  if (target.geometry_revision == std::numeric_limits<uint64_t>::max()) {
    throw std::overflow_error("Exhausted fragment geometry revisions");
  }
  appendMeshUnion(target.geometry, target.bbox,
                  observation.geometry, observation.bbox);
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
  target.track_first_seen = std::min(target.track_first_seen, observation.track_first_seen);
}

void PersistentObjectState::mergeObservedNew(PhysicalState& state,
                                             const KhronosObjectAttributes& attrs,
                                             const TimeStamp first,
                                             const TimeStamp last) {
  state.observed_new.push_back(makeFragment(attrs, first, last));
}

// README (5b): ingestion and measured support share one absorption authorization: the actual
// support time lies in the observation's own acquisition interval.
bool PersistentObjectState::canAbsorb(const Fragment& observation, const TimeStamp stamp) const {
  return observation.birth_time <= stamp && stamp <= observation.last_support_time;
}

void PersistentObjectState::absorbObservedThrough(PhysicalState& state, const TimeStamp stamp) {
  if (!state.current) return;
  auto& current = state.fragments[*state.current];
  auto& pending = state.observed_new;
  for (auto it = pending.begin(); it != pending.end();) {
    if (canAbsorb(*it, stamp)) {
      mergeFragments(current, *it);
      it = pending.erase(it);
    } else {
      ++it;
    }
  }
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
      state.fragments.push_back(std::move(*it));
    } else {
      if (write != it) *write = std::move(*it);
      ++write;
    }
  }
  state.observed_new.erase(write,state.observed_new.end());
}

void PersistentObjectState::promoteObservedNew(PhysicalState& state) {
  if (state.observed_new.empty() || state.current) return;
  const auto index = latestPendingIndex(state);
  if (!isEligibleSuccessor(state, state.observed_new[index])) return;
  state.fragments.push_back(std::move(state.observed_new[index]));
  state.observed_new.erase(state.observed_new.begin() + index);
  state.current = state.fragments.size() - 1;
}

void PersistentObjectState::closeCurrent(PhysicalState& state, const TimeStamp stamp) {
  if (!state.current) {
    return;
  }
  Fragment& current = state.fragments[*state.current];
  // README (4): d_f = max(t_decision, s_f), an upper bound of the departure time, s_f being the
  // last direct support. The successor watermark of (5b) uses the latest actual support.
  current.death_time = std::max(stamp, current.last_support_time);
  state.succession_floor = std::max(state.succession_floor, latestSupport(current));
  state.current.reset();
  state.has_dynamic_history = true;
}

// README (6c), (5b): consume accepted native motion once, using its actual sensor time.
bool PersistentObjectState::consumeMotion(PhysicalState& state,
                                           const KhronosObjectAttributes& attrs) {
  if (!hasMotionEvidence(attrs)) return false;
  state.has_dynamic_history = true;
  TimeStamp motion = 0;
  const size_t n = std::min(attrs.trajectory_timestamps.size(), attrs.trajectory_positions.size());
  for (size_t i = 0; i < n; ++i) {
    const auto t = attrs.trajectory_timestamps[i];
    if (t != std::numeric_limits<TimeStamp>::max()) motion = std::max(motion,t);
  }
  if (motion <= state.last_motion_consumed) return false;
  state.last_motion_consumed = motion;
  if (state.b_session) consumeMotion(*state.b_session,attrs);
  bool closed = false;
  if (state.current) {
    const auto& current = state.fragments[*state.current];
    if (motion > latestSupport(current)) {
      closeCurrent(state,motion);
      closed = true;
    }
  }
  closePending(state,motion);
  return closed;
}

// README (5b), P13: moving a session preserves all its resolved fragments.
bool PersistentObjectState::settleSession(PhysicalState& state,
                                          const StateRelation relation,
                                          const TimeStamp stamp) {
  const bool closed = relation == StateRelation::kReplace && state.current.has_value();
  if (relation == StateRelation::kReplace) closeCurrent(state, stamp);
  auto session = std::move(state.b_session);
  if (!session) return closed;

  const auto active = session->current;
  if (active && relation == StateRelation::kRefine && state.current) {
    mergeFragments(state.fragments[*state.current], session->fragments[*active]);
  }
  state.has_dynamic_history = state.has_dynamic_history || session->has_dynamic_history;
  state.succession_floor = std::max(state.succession_floor,session->succession_floor);
  state.last_motion_consumed = std::max(state.last_motion_consumed,session->last_motion_consumed);
  for (size_t i = 0; i < session->fragments.size(); ++i) {
    auto& fragment = session->fragments[i];
    if (active && i == *active) {
      if (relation == StateRelation::kRefine && state.current) continue;
      if (relation == StateRelation::kReplace && isEligibleSuccessor(state, fragment)) {
        state.current = state.fragments.size();
      } else {
        state.observed_new.push_back(std::move(fragment));
        continue;
      }
    }
    state.fragments.push_back(std::move(fragment));
  }
  for (auto& candidate : session->observed_new) {
    state.observed_new.push_back(std::move(candidate));
  }
  state.inherited_compatible = false;
  return closed;
}

void PersistentObjectState::ingestObservation(PhysicalState& state,
                                              const KhronosObjectAttributes& attrs,
                                              const TimeStamp first,
                                              const TimeStamp last,
                                              const size_t physical_instance_id) {
  LOG(INFO) << "INGEST inst=" << physical_instance_id
            << " first=" << (first / 1000000000ULL)
            << "s seg_verts=" << attrs.mesh.numVertices()
            << " cur_verts=" << (state.current ? state.fragments[*state.current].geometry.numVertices() : 0)
            << " observed_verts=" << (state.observed_new.empty() ? 0 : state.observed_new[latestPendingIndex(state)].geometry.numVertices());

  consumeMotion(state,attrs);
  if (!state.current && state.b_session) settleSession(state,StateRelation::kReplace,last);

  // Nothing established yet: this observation opens the first fragment. No state is being
  // displaced, so no contradiction evidence is required.
  if (!state.current) {
    if (!state.observed_new.empty() || last <= state.succession_floor) {
      mergeObservedNew(state, attrs, first, last);
      promoteObservedNew(state);
      return;
    }
    state.fragments.push_back(makeFragment(attrs, first, last));
    state.current = state.fragments.size() - 1;
    if (hasMotionEvidence(attrs)) {
      state.has_dynamic_history = true;
    }
    return;
  }

  const Fragment& current = state.fragments[*state.current];
  if (current.requires_current_session_support) {
    // Keep inherited and session observations independent until measured
    // evidence resolves their relationship online.
    LOG(INFO) << "INGEST_DECIDE inst=" << physical_instance_id
              << " inherited_session_deferred=true";
    if (!state.b_session) {
      state.b_session = std::make_unique<PhysicalState>();
    }
    ingestObservation(*state.b_session, attrs, first, last,
                      physical_instance_id);
    return;
  }
  // Keep both actual support clocks: a later confirmation does not erase a
  // direct support that lies inside this observation's interval (README 5b).
  auto observation = makeFragment(attrs, first, last);
  if (canAbsorb(observation, current.last_support_time) ||
      canAbsorb(observation, current.last_confirmed_support)) {
    mergeFragments(state.fragments[*state.current], observation);
  } else {
    state.observed_new.push_back(std::move(observation));
  }
}

void PersistentObjectState::ingestSegments(const DynamicSceneGraph& graph,
    const std::vector<NodeId>& nodes, size_t id) {
  auto& state = states_[id];
  for (const auto& segment : collectSegments(graph,nodes)) {
    const auto& attrs = *segment.attrs;
    if (attrs.details.count(kCurrentExists)) continue;
    if (!state.ingested_sources.insert(sourceKey(segment)).second) continue;
    const auto first = inputFirstStamp(attrs), last = inputLastStamp(attrs);
    if (first > last) throw std::invalid_argument("Invalid raw observation time interval");
    if (attrs.mesh.points.empty()) {
      consumeMotion(state,attrs);
      if (!state.current && state.b_session) settleSession(state,StateRelation::kReplace,last);
    } else {
      ingestObservation(state,attrs,first,last,id);
    }
  }
}

void PersistentObjectState::ingestObjects(const DynamicSceneGraph& graph) {
  if (!graph.hasLayer(DsgLayers::OBJECTS)) return;
  std::map<size_t,std::vector<NodeId>> groups;
  for (const auto& [node_id,node] : graph.getLayer(DsgLayers::OBJECTS).nodes()) {
    const auto* attrs = node->tryAttributes<KhronosObjectAttributes>();
    if (!attrs) continue;
    if (const auto id = session_detail::getPhysicalInstanceId(*attrs)) groups[*id].push_back(node_id);
  }
  for (const auto& [id,nodes] : groups) ingestSegments(graph,nodes,id);
}

void PersistentObjectState::applyPhysicalGeometry(const DynamicSceneGraph& graph,
    const std::vector<NodeId>& nodes, KhronosObjectAttributes& merged) {
  const auto id = UpdateKhronosObjectsFunctor::physicalInstanceId(merged);
  if (!id) return;
  ingestSegments(graph,nodes,*id);
  session_detail::setInputBounds(merged,inputFirstStamp(merged),inputLastStamp(merged));
  materializeState(states_.at(*id),merged);
}

void PersistentObjectState::materializeState(
    const PhysicalState& state, KhronosObjectAttributes& merged) const {
  // README (5): materialization consumes the same relation as final settlement.
  if (state.current) {
    const Fragment& current = state.fragments[*state.current];
    merged.mesh = current.geometry;
    merged.bounding_box = current.bbox;
    merged.position = current.position;
    size_t frames = current.reconstruction_frames;
    std::vector<TimeStamp> frame_keys = current.frame_keys;
    TimeStamp materialized_support = std::max(current.last_support_time, current.last_confirmed_support);
    TimeStamp materialized_track_first = current.track_first_seen;
    if (current.requires_current_session_support && state.b_session &&
        state.b_session->current && inheritedRelation(state) == StateRelation::kRefine) {
      const Fragment& session = state.b_session->fragments[*state.b_session->current];
      appendMeshUnion(merged.mesh, merged.bounding_box, session.geometry, session.bbox);
      merged.position = merged.bounding_box.world_P_center.cast<double>();
      if (!frame_keys.empty() && !session.frame_keys.empty()) {
        std::vector<TimeStamp> keys;
        std::set_union(frame_keys.begin(), frame_keys.end(), session.frame_keys.begin(),
                       session.frame_keys.end(), std::back_inserter(keys));
        frame_keys = std::move(keys);
        frames = frame_keys.size();
      } else {
        frame_keys.clear();
        frames += session.reconstruction_frames;
      }
      materialized_support = std::max({materialized_support, session.last_support_time, session.last_confirmed_support});
      materialized_track_first = std::min(materialized_track_first, session.track_first_seen);
    }
    merged.details[kReconstructionFramesDetail] = {frames};
    if (frame_keys.empty()) merged.details.erase(kFrameStampsDetail);
    else merged.details[kFrameStampsDetail] = std::vector<size_t>(frame_keys.begin(), frame_keys.end());
    merged.details.erase(kChangeAfter);
    merged.details.erase(kChangeBefore);
    // Native D2 now receives the observation bounds of the geometry it queries.
    setObservationBounds(merged, current.birth_time, materialized_support);
    merged.details[kCurrentBirth] = {current.birth_time};
    merged.details[kCurrentSupport] = {materialized_support};
    merged.details[kCurrentTrackFirst] = {materialized_track_first};
    merged.details[kHasDynamicHistoryDetail] = {state.has_dynamic_history ? 1u : 0u};
  } else {
    merged.mesh = spark_dsg::Mesh(merged.mesh.has_colors, merged.mesh.has_timestamps,
                                 merged.mesh.has_labels, merged.mesh.has_first_seen_stamps);
    merged.details[kReconstructionFramesDetail] = {0};
    merged.details.erase(kFrameStampsDetail);
    merged.details[kHasDynamicHistoryDetail] = {state.has_dynamic_history ? 1u : 0u};
    // README (5b): the change time of the closed placement lies between its latest actual support
    // and its first valid contradiction; the decision time stays the upper bound of the single-time
    // interface (the presence interval).
    const Fragment* closed = nullptr;
    for (const auto& fragment : state.fragments) {
      if (!fragment.death_time) continue;
      if (!closed || latestSupport(fragment) >= latestSupport(*closed)) closed = &fragment;
    }
    if (closed) {
      const TimeStamp after = latestSupport(*closed);
      const TimeStamp before = closed->first_contradiction > 0 ? closed->first_contradiction
                                                              : *closed->death_time;
      merged.details[kChangeAfter] = {after};
      merged.details[kChangeBefore] = {std::max(after, before)};
    }
  }
  merged.details[kCurrentExists] = {state.current.has_value() ? 1u : 0u};
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

bool PersistentObjectState::reportCurrentContradicted(const size_t physical_instance_id,
                                                      const TimeStamp stamp) {
  const auto it = states_.find(physical_instance_id);
  if (it == states_.end() || !it->second.current) {
    return false;
  }
  auto& state = it->second;
  if (state.b_session) settleSession(state, StateRelation::kReplace, stamp);
  else closeCurrent(state, stamp);
  promoteObservedNew(state);
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
  current.last_confirmed_support =
      std::max(current.last_confirmed_support, stamp);
  absorbObservedThrough(state, stamp);
  return true;
}

// README (6b), P13: the inherited current and the session's own observations of the same identity
// are one placement when this round judged the current compatible (omega > 1/2); one identity has
// one placement at a time. Otherwise the session's observations stay separate and unresolved.
PersistentObjectState::StateRelation PersistentObjectState::inheritedRelation(
    const PhysicalState& state) const {
  if (!state.current) return StateRelation::kReplace;
  const bool session = state.b_session && state.b_session->current;
  return session && state.inherited_compatible ? StateRelation::kRefine
                                               : StateRelation::kSeparate;
}

// README (5), (5p), (5h), P2: one measured round updates the placement's stationarity once. A
// round without a valid source is the unit factor and changes nothing.
PersistentObjectState::LocalOutcome PersistentObjectState::applyRound(
    Fragment& fragment, const SurfaceEvidence& evidence, const TimeStamp stamp) const {
  LocalOutcome outcome;
  if (!ownsEvidence(fragment, evidence) || evidence.measured_through != stamp ||
      !evidence.informative) {
    return outcome;
  }
  const auto update =
      updatePlacement(fragment.alpha, fragment.beta, chiOf(fragment), evidence.l_in, evidence.l_out);
  fragment.alpha = update.alpha;
  fragment.beta = update.beta;
  outcome.applied = true;
  outcome.compatible = update.omega > 0.5;
  // README (5b): remember the first valid contradiction after the last support; a later support
  // shows it was transient.
  if (evidence.contradiction_rays > 0 && evidence.first_contradiction_stamp > 0 &&
      (fragment.first_contradiction == 0 ||
       evidence.first_contradiction_stamp < fragment.first_contradiction)) {
    fragment.first_contradiction = evidence.first_contradiction_stamp;
  }
  if (outcome.compatible && evidence.support_rays > 0 &&
      evidence.latest_support_stamp > fragment.first_contradiction) {
    fragment.first_contradiction = 0;
  }
  // README (5e): the placement ends when the expected stationarity is below 1/2; a tie continues.
  outcome.closed = fragment.alpha / (fragment.alpha + fragment.beta) < 0.5;
  return outcome;
}

// README (5): this reducer handles both a top-level local state and b_session.
PersistentObjectState::LocalOutcome PersistentObjectState::resolveLocalEvidence(
    PhysicalState& state, const SurfaceEvidence& evidence, const TimeStamp stamp) {
  if (!state.current) return {};
  Fragment& current = state.fragments[*state.current];
  auto outcome = applyRound(current, evidence, stamp);
  if (!outcome.applied) return outcome;
  if (outcome.closed) {
    closeCurrent(state, stamp);
    promoteObservedNew(state);
    return outcome;
  }
  if (outcome.compatible && evidence.support_rays > 0) {
    current.last_confirmed_support = std::max(current.last_confirmed_support,
                                             std::min(evidence.latest_support_stamp, stamp));
    absorbObservedThrough(state, current.last_confirmed_support);
  }
  return outcome;
}

size_t PersistentObjectState::finalizePendingAbsences(const TimeStamp stamp) {
  size_t closed = 0;
  for (auto& [id, state] : states_) {
    (void)id;
    if (!state.current) {
      settleSession(state, StateRelation::kReplace, stamp);
      promoteObservedNew(state);
    } else if (state.fragments[*state.current].requires_current_session_support) {
      closed += settleSession(state, inheritedRelation(state), stamp);
    }
  }
  return closed;
}

PersistentObjectState::RoundResult PersistentObjectState::resolveCurrentEvidence(
    const size_t physical_instance_id,
    const SurfaceEvidence& inherited_evidence,
    const SurfaceEvidence& session_evidence,
    const TimeStamp stamp) {
  RoundResult result;
  const auto it = states_.find(physical_instance_id);
  if (it == states_.end()) return result;
  auto& state = it->second;
  if (state.b_session) {
    const auto outcome = resolveLocalEvidence(*state.b_session, session_evidence, stamp);
    result.session_normal = outcome.applied && outcome.compatible && !outcome.closed;
    state.has_dynamic_history = state.has_dynamic_history || state.b_session->has_dynamic_history;
  }
  if (!state.current) return result;

  auto& current = state.fragments[*state.current];
  if (current.requires_current_session_support) {
    // The inherited current is measured against this session's inputs; the session's own
    // observations stay independent until the relation of README (6b) is decided. Only an
    // informative round changes the judgement; a round without a valid source keeps it.
    const auto outcome = applyRound(current, inherited_evidence, stamp);
    if (outcome.applied && outcome.closed) {
      result.closed = settleSession(state, StateRelation::kReplace, stamp);
      return result;
    }
    if (outcome.applied) state.inherited_compatible = outcome.compatible;
    result.inherited_normal = outcome.applied && outcome.compatible;
    if (result.inherited_normal && inherited_evidence.support_rays > 0) {
      current.last_confirmed_support = std::max(current.last_confirmed_support,
                                                inherited_evidence.latest_support_stamp);
    }
    return result;
  }

  // A placement established in this session carries its own evidence.
  const bool own_session = session_evidence.measured_through == stamp &&
      ownsEvidence(current, session_evidence);
  const auto outcome = resolveLocalEvidence(state, own_session ? session_evidence : inherited_evidence,
                                            stamp);
  result.closed = outcome.closed;
  result.inherited_normal = outcome.applied && outcome.compatible && !outcome.closed;
  return result;
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
    for (const auto& [node_id,node] : dsg.getLayer(DsgLayers::OBJECTS).nodes()) {
      const auto* attrs = node->tryAttributes<KhronosObjectAttributes>();
      if (!attrs) continue;
      const auto id = UpdateKhronosObjectsFunctor::physicalInstanceId(*attrs);
      if (!id) continue;
      auto& state = restored[*id];
      const bool materialized = attrs->details.count(kCurrentExists) != 0;
      if (!materialized && !state.ingested_sources.insert(sourceKey({node_id,attrs})).second) continue;
      state.has_dynamic_history = state.has_dynamic_history || hasMotionEvidence(*attrs);
      if (!attrs->bounding_box.isValid() || attrs->mesh.points.empty()) continue;
      const auto first = materialized ? detailValue(*attrs,kCurrentBirth) : observationFirstStamp(*attrs);
      auto supported = materialized ? detailValue(*attrs,kCurrentSupport) : observationLastStamp(*attrs);
      if (supported == std::numeric_limits<TimeStamp>::max()) supported = first;
      if (supported < first || supported > boundary)
        throw std::invalid_argument("Imported support lies outside the saved state boundary");
      auto fragment = makeFragment(*attrs,first,supported);
      if (materialized) fragment.track_first_seen = detailValue(*attrs,kCurrentTrackFirst);
      fragment.last_confirmed_support = 0;
      fragment.input_boundary = boundary;
      fragment.requires_current_session_support = true;
      // Native presence remains Khronos' decision. The actual observation
      // fields describe support; its estimated interval determines seed activity.
      const bool active = materialized ? detailValue(*attrs,kCurrentExists) != 0
          : isPresent(*attrs,boundary);
      if (active) {
        state.observed_new.push_back(std::move(fragment));
      } else if (!materialized) {
        const auto departed = lastDisappearedBefore(*attrs,boundary);
        if (departed) {
          if (*departed < supported)
            throw std::invalid_argument("Imported departure precedes actual support");
          fragment.death_time = *departed;
          state.succession_floor = std::max(state.succession_floor,supported);
          state.has_dynamic_history = true;
          state.fragments.push_back(std::move(fragment));
        }
      }
    }
  }
  // README (5b): latest supported current, with other imported candidates retained.
  for (auto& [id,state] : restored) {
    (void)id;
    promoteObservedNew(state);
  }
  states_ = std::move(restored);
  background_obligations_.clear();
}

void PersistentObjectState::clear() { states_.clear(); background_obligations_.clear(); }

std::map<size_t, TimeStamp> PersistentObjectState::successionFloors() const {
  std::map<size_t, TimeStamp> floors;
  for (const auto& [id, state] : states_) {
    TimeStamp floor = state.succession_floor;
    if (state.b_session) floor = std::max(floor, state.b_session->succession_floor);
    if (floor > 0) floors.emplace(id, floor);
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
  if (it == states_.end()) return {};
  auto result = viewsOf(it->second.fragments);
  if (it->second.b_session) {
    const auto session = viewsOf(it->second.b_session->fragments);
    result.insert(result.end(), session.begin(), session.end());
  }
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
PersistentObjectState::unresolvedCandidates(
    const size_t physical_instance_id) const {
  const auto it = states_.find(physical_instance_id);
  return it == states_.end() ? std::vector<FragmentView>{} : viewsOf(it->second.observed_new);
}

std::set<std::pair<size_t,uint64_t>> PersistentObjectState::liveEvidenceKeys() const {
  std::set<std::pair<size_t,uint64_t>> result;
  for (const auto& [id,state] : states_) {
    const auto collect = [&](const auto& self, const PhysicalState& value) -> void {
      if (value.current) result.emplace(id,value.fragments.at(*value.current).evidence_key);
      for (const auto& pending : value.observed_new) result.emplace(id,pending.evidence_key);
      if (value.b_session) self(self,*value.b_session);
    };
    collect(collect,state);
  }
  return result;
}

}  // namespace khronos
