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

#include <array>
#include <map>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <set>
#include <tuple>
#include <unordered_map>

#include <glog/logging.h>
#include <boost/math/special_functions/beta.hpp>

#include "khronos/backend/update_khronos_objects_functor.h"
#include "khronos/utils/khronos_attribute_utils.h"

namespace khronos {

namespace {

constexpr const char* kMobilityChangesDetail = "session_mobility_changes";
constexpr const char* kMobilityContinuationsDetail = "session_mobility_continuations";

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
  return detailValue(attrs, kHasDynamicHistoryDetail) != 0;
}

// README M1c: one quantization witness per occupied voxel. Lack of shared
// support remains unresolved; it never establishes movement or absence.
double sharedSpaceProbability(const spark_dsg::Mesh& current,
                              const BoundingBox& current_box,
                              const spark_dsg::Mesh& candidate,
                              const BoundingBox& candidate_box,
                              float resolution, double decision_probability) {
  if (current.points.empty() || candidate.points.empty()) return 0.0;
  using Key = std::tuple<int64_t, int64_t, int64_t>;
  struct Cell {
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    size_t count = 0;
  };
  const auto key = [resolution](const Point& p) {
    return Key(static_cast<int64_t>(std::floor(p.x() / resolution)),
               static_cast<int64_t>(std::floor(p.y() / resolution)),
               static_cast<int64_t>(std::floor(p.z() / resolution)));
  };
  const auto group = [&](const spark_dsg::Mesh& mesh, const BoundingBox& box) {
    std::map<Key, Cell> cells;
    for (const auto& local : mesh.points) {
      const Point world = box.pointToWorldFrame(local);
      auto& cell = cells[key(world)];
      cell.sum += world.cast<double>();
      ++cell.count;
    }
    return cells;
  };
  const auto reference = group(current, current_box);
  const auto observed = group(candidate, candidate_box);
  double log_no_shared_support = 0.0;
  for (const auto& [k, cell] : observed) {
    const Eigen::Vector3d point = cell.sum / cell.count;
    double correspondence = 0.0;
    for (int dx = -1; dx <= 1; ++dx)
      for (int dy = -1; dy <= 1; ++dy)
        for (int dz = -1; dz <= 1; ++dz) {
          const auto other = reference.find(Key(std::get<0>(k) + dx,
                                                std::get<1>(k) + dy,
                                                std::get<2>(k) + dz));
          if (other == reference.end()) continue;
          const Eigen::Vector3d delta =
              (point - other->second.sum / other->second.count).cwiseAbs();
          const double probability =
              (1.0 - delta.array() / resolution).max(0.0).prod();
          correspondence = std::max(correspondence, probability);
        }
    if (correspondence == 1.0) return 1.0;
    log_no_shared_support += std::log1p(-correspondence);
    const double probability = -std::expm1(log_no_shared_support);
    // A monotone lower bound suffices for exactly the same final decision.
    if (probability > decision_probability) return probability;
  }
  return -std::expm1(log_no_shared_support);
}


// Probability that the world AABBs share occupied cells under the same
// uniform grid-phase model as sharedSpaceProbability (README M1d).
// Overlapping/degenerate equal extents retain probability one. The original
// no-active-support / measured-absence conditions still govern closing.
double extentSameSiteProbability(const spark_dsg::Mesh& current,
                                  const BoundingBox& current_box,
                                  const spark_dsg::Mesh& candidate,
                                  const BoundingBox& candidate_box,
                                  float resolution) {
  if (current.points.empty() || candidate.points.empty()) {
    return 0.0;
  }
  const auto extent = [](const spark_dsg::Mesh& mesh, const BoundingBox& box,
                         Point& lo, Point& hi) {
    lo = Point::Constant(std::numeric_limits<float>::max());
    hi = Point::Constant(std::numeric_limits<float>::lowest());
    for (const auto& local : mesh.points) {
      const Point p = box.pointToWorldFrame(local);
      lo = lo.cwiseMin(p);
      hi = hi.cwiseMax(p);
    }
  };
  Point a_lo, a_hi, b_lo, b_hi;
  extent(current, current_box, a_lo, a_hi);
  extent(candidate, candidate_box, b_lo, b_hi);
  double probability = 1.0;
  for (int axis = 0; axis < 3; ++axis) {
    const double gap = std::max({0.0,
        static_cast<double>(a_lo[axis]) - b_hi[axis],
        static_cast<double>(b_lo[axis]) - a_hi[axis]});
    probability *= std::max(0.0, 1.0 - gap / resolution);
  }
  return probability;
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

void PersistentObjectState::beginObservationEvent(const TimeStamp stamp) {
  if (event_open_ && stamp == event_stamp_) return;  // same evidence snapshot
  event_history_.clear();
  for (const auto& [id, state] : states_) {
    HistoryRecord& record = event_history_[id];
    record.changes = state.mobility_changes;
    record.continuations = state.mobility_continuations;
    if (state.b_session) {
      record.session_changes = state.b_session->mobility_changes;
      record.session_continuations = state.b_session->mobility_continuations;
    }
    if (!state.fragments.empty()) {
      record.has_fragments = true;
      record.semantic_label = state.current ? state.fragments[*state.current].semantic_label
                                            : state.fragments.back().semantic_label;
    }
  }
  event_stamp_ = stamp;
  event_open_ = true;
}

double PersistentObjectState::changePriorLogOdds(const size_t physical_instance_id,
                                                 const int state_slot) const {
  // q is the prior of one relation between consecutive states (README M2a), not an in-session
  // hazard per round, so only the inherited state's cross-session relation carries it.
  if (state_slot != 0) return 0.0;
  const auto it = states_.find(physical_instance_id);
  if (it == states_.end() || !it->second.current) return 0.0;
  const PhysicalState& state = it->second;
  const Fragment& current = state.fragments[*state.current];
  if (!current.requires_current_session_support) return 0.0;
  const double q = stateChangeProbability(state, current);
  return std::log(q) - std::log1p(-q);
}

// README M2a: ontology groups the prior population; only resolved historical
// relations update the Bernoulli probability. The target never trains its own prior.
double PersistentObjectState::stateChangeProbability(const PhysicalState& state,
                                                     const Fragment& current) const {
  const PhysicalState* owner = &state;
  size_t instance_id = 0;
  bool registered = false;
  for (const auto& [id, root] : states_) {
    if (&root == &state || root.b_session.get() == &state) {
      owner = &root;
      instance_id = id;
      registered = true;
      break;
    }
  }
  const bool session = owner != &state;
  const bool group = high_mobility_semantic_labels_.count(current.semantic_label) > 0;
  // One configured ontology judgment contributes one prior opinion, not a
  // measured movement. Empty ontology contributes no directional information.
  const double ontology_opinions = high_mobility_semantic_labels_.empty() ? 0.0 : 1.0;
  const double prior_mass = 1.0 + ontology_opinions;
  struct Counts {
    double own_m = 0.0, own_u = 0.0, class_m = 0.0, class_u = 0.0, group_m = 0.0, group_u = 0.0;
    void population(const int label, const int target, const bool target_group,
                    const bool label_group, const double m, const double u) {
      if (label == target) {
        class_m += m;
        class_u += u;
      } else if (label_group == target_group) {
        group_m += m;
        group_u += u;
      }
    }
  };
  const auto probability_of = [&](const Counts& c, double& alpha, double& beta) {
    const double group_mean = (c.group_m + 0.5 + ontology_opinions * group) /
                              (c.group_m + c.group_u + prior_mass);
    alpha = prior_mass * group_mean + c.class_m;
    beta = prior_mass * (1.0 - group_mean) + c.class_u;
    return (alpha + c.own_m) / (alpha + beta + c.own_m + c.own_u);
  };
  // README M2a: condition on the history resolved before this observation event (an identity
  // first registered inside the event has none yet); outside an event, on the live registry.
  Counts counts;
  if (event_open_) {
    const auto own = registered ? event_history_.find(instance_id) : event_history_.end();
    if (own != event_history_.end()) {
      counts.own_m = own->second.changes;
      counts.own_u = own->second.continuations;
      if (session) {
        counts.own_m += own->second.session_changes;
        counts.own_u += own->second.session_continuations;
      }
    }
    for (const auto& [id, record] : event_history_) {
      if ((registered && id == instance_id) || !record.has_fragments) continue;
      counts.population(record.semantic_label, current.semantic_label, group,
                        high_mobility_semantic_labels_.count(record.semantic_label) > 0,
                        record.changes, record.continuations);
    }
  } else {
    counts.own_m = state.mobility_changes;
    counts.own_u = state.mobility_continuations;
    if (session) {
      counts.own_m += owner->mobility_changes;
      counts.own_u += owner->mobility_continuations;
    }
    for (const auto& [id, other] : states_) {
      (void)id;
      if (&other == owner || other.fragments.empty()) continue;
      const auto& fragment = other.current ? other.fragments[*other.current]
                                          : other.fragments.back();
      counts.population(fragment.semantic_label, current.semantic_label, group,
                        high_mobility_semantic_labels_.count(fragment.semantic_label) > 0,
                        other.mobility_changes, other.mobility_continuations);
    }
  }
  double alpha = 0.0, beta = 0.0;
  const double probability = probability_of(counts, alpha, beta);
  LOG(INFO) << "MOBILITY_PRIOR inst=" << instance_id
            << " session=" << session
            << " class=" << current.semantic_label
            << " changes=" << counts.own_m << " continuations=" << counts.own_u
            << " alpha=" << alpha << " beta=" << beta << " q=" << probability
            << " event_stamp=" << (event_open_ ? event_stamp_ : 0);
  return probability;
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
  view.uid = fragment.uid;
  view.track_first_seen = fragment.track_first_seen;
  view.last_support_time = fragment.last_support_time;
  view.last_confirmed_support = fragment.last_confirmed_support;
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
  static std::atomic<uint64_t> next_uid{1};
  fragment.uid = next_uid.fetch_add(1);
  fragment.track_first_seen = trackFirstSeen(attrs, first);
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

double PersistentObjectState::motionGeometryBayesFactor(const double off,
                                                        const double effective_cells) {
  const double alpha = 0.5 + effective_cells * off;
  const double beta = 0.5 + effective_cells * (1.0 - off);
  return 2.0 * boost::math::ibetac(alpha, beta, 0.5);
}

// The 27 neighbour offsets of the M1g search grid, own cell first.
constexpr std::array<std::array<int, 3>, 27> kOwnCellFirst = [] {
  std::array<std::array<int, 3>, 27> offsets{};
  size_t i = 1;
  offsets[0] = {0, 0, 0};
  for (int dx = -1; dx <= 1; ++dx)
    for (int dy = -1; dy <= 1; ++dy)
      for (int dz = -1; dz <= 1; ++dz)
        if (dx != 0 || dy != 0 || dz != 0) offsets[i++] = {dx, dy, dz};
  return offsets;
}();

double PersistentObjectState::offStateShare(const spark_dsg::Mesh& copy, const BoundingBox& copy_box,
                                            const spark_dsg::Mesh& reference,
                                            const BoundingBox& reference_box, const float tolerance,
                                            double& effective_cells) const {
  effective_cells = 0.0;
  if (copy.points.empty()) return 0.0;
  // M1g: radial cell-phase uncertainty has support [-resolution, resolution].
  // No point beyond tolerance + resolution can contribute correspondence mass.
  const double resolution = map_resolution_;
  const double radius = tolerance + resolution;
  using Key = std::tuple<int64_t, int64_t, int64_t>;
  struct KeyHash {
    size_t operator()(const Key& k) const {
      return std::hash<int64_t>()(std::get<0>(k) * 73856093 ^ std::get<1>(k) * 19349663 ^
                                  std::get<2>(k) * 83492791);
    }
  };
  const auto cell = [radius](const Point& p) {
    return Key(static_cast<int64_t>(std::floor(p.x() / radius)),
               static_cast<int64_t>(std::floor(p.y() / radius)),
               static_cast<int64_t>(std::floor(p.z() / radius)));
  };
  std::unordered_map<Key, std::vector<Point>, KeyHash> grid;
  for (const auto& local : reference.points) {
    const Point p = reference_box.pointToWorldFrame(local);
    grid[cell(p)].push_back(p);
  }
  const double radius2 = radius * radius;
  const double certain_radius = tolerance - resolution;
  const double certain_radius2 = certain_radius * certain_radius;
  double expected_off = 0.0;
  // M1h: preserve vertex weights; one correlated spatial group per existing r cell.
  std::unordered_map<Key, size_t, KeyHash> copy_cells;
  for (const auto& local : copy.points) {
    const Point p = copy_box.pointToWorldFrame(local);
    ++copy_cells[Key(static_cast<int64_t>(std::floor(p.x() / resolution)),
                     static_cast<int64_t>(std::floor(p.y() / resolution)),
                     static_cast<int64_t>(std::floor(p.z() / resolution)))];
    const Key k = cell(p);
    double nearest2 = radius2;
    bool certain = false;
    // The vertex's own cell first: a surface re-observation usually has its
    // certain correspondence there. Visiting order cannot change any output:
    // without a point inside the certain radius the minimum runs over all 27
    // cells; with one, the probability is one and the hard test agrees.
    for (const auto& offset : kOwnCellFirst) {
      if (certain) break;
      const auto it = grid.find(Key(std::get<0>(k) + offset[0], std::get<1>(k) + offset[1],
                                    std::get<2>(k) + offset[2]));
      if (it == grid.end()) continue;
      for (const auto& q : it->second) {
        nearest2 = std::min(nearest2, static_cast<double>((q - p).squaredNorm()));
        if (certain_radius >= 0.0 && nearest2 <= certain_radius2) {
          certain = true;
          break;
        }
      }
    }
    const double u = (tolerance - std::sqrt(nearest2)) / resolution;
    // These endpoints are the exact support of the difference of two uniforms.
    const double same_probability = certain || u >= 1.0 ? 1.0 :
        u <= -1.0 ? 0.0 :
        u < 0.0 ? 0.5 * (1.0 + u) * (1.0 + u) :
                  1.0 - 0.5 * (1.0 - u) * (1.0 - u);
    expected_off += 1.0 - same_probability;
  }
  double squared_cell_counts = 0.0;
  for (const auto& entry : copy_cells) {
    const double count = static_cast<double>(entry.second);
    squared_cell_counts += count * count;
  }
  const double vertices = static_cast<double>(copy.points.size());
  effective_cells = vertices * vertices / squared_cell_counts;
  return expected_off / vertices;
}

PersistentObjectState::SameStatePosterior PersistentObjectState::sameStatePosterior(
    const double q, const Fragment& measured, const Fragment& shape) const {
  SameStatePosterior p;
  p.q = q;
  if (!measured.geometry.points.empty() && !shape.geometry.points.empty()) {
    p.factor = kGeometryFactorBound;
    p.evaluated = 1.0 - q < q * p.factor;
    if (p.evaluated) {
      p.off = offStateShare(measured.geometry, measured.bbox, shape.geometry, shape.bbox,
                            kStateTolerance, p.effective_cells);
      p.factor = motionGeometryBayesFactor(p.off, p.effective_cells);
    }
  }
  p.same = 1.0 - q >= q * p.factor;
  return p;
}

static std::ostream& operator<<(std::ostream& out,
                                const PersistentObjectState::SameStatePosterior& p) {
  out << " change_prior=" << p.q << " geometry_factor_or_bound=" << p.factor
      << " effective_cells=" << p.effective_cells << " off_share=";
  if (p.evaluated) out << p.off; else out << "unmeasured";
  return out;
}

double PersistentObjectState::copyInvalidityTerm(const Fragment& copy, bool& calibrated,
                                                 size_t& measured_looks, double& log_ratio) {
  // An invalid copy is a surface the rays pass through: each calibrated look
  // already carries log p(look | absent) / p(look | present) on the copy's own
  // surface (the observed-absence test's per-look ratio). The prior validity
  // v of a session copy has no measured rate, so it is the indifference
  // value 1/2 and (1 - v) / v = 1.
  calibrated = false;
  measured_looks = 0;
  log_ratio = 0.0;
  for (const auto& look : copy.looks) {
    calibrated = calibrated || look.has_calibrated_absence_source;
    if (look.has_measured_absence_likelihood) {
      log_ratio += look.measured_absence_log_ratio;
      ++measured_looks;
    }
  }
  constexpr double kCopyValidityPrior = 0.5;
  return calibrated ? (1.0 - kCopyValidityPrior) / kCopyValidityPrior * std::exp(log_ratio) : 0.0;
}

bool PersistentObjectState::sessionCopyElsewhere(const PhysicalState& state,
                                                 const Fragment& inherited,
                                                 const size_t session_reliable_samples) const {
  if (!state.b_session || !state.b_session->current) return false;
  if (session_reliable_samples == 0) return false;  // no established surface measurement
  const Fragment& copy = state.b_session->fragments[*state.b_session->current];
  if (copy.geometry.points.empty()) return false;  // no surface correspondence measurement
  // README 1.1, one decision over S (A persists, N is more of A), M (A ended,
  // N is its successor) and U (N is not a valid surface of this identity).
  // Handing over is wrong under S and under U, keeping is wrong under M, all
  // at the same surface loss, so the Bayes action is
  //   hand over  iff  q B_{M:S} > (1 - q) + ((1 - v) / v) L_{U:N},
  // with q the motion prior of the A->N relation, B_{M:S} the M1h geometry
  // factor (<= 2), the U geometry equal to the unrestricted reference model of
  // M1h, and the U odds term from N's own looks (copyInvalidityTerm). Callers
  // without a calibrated look channel (unit fixtures) have no such ratio and
  // keep the previous count contract, as observedEmptySince does.
  bool calibrated = false;
  size_t measured_looks = 0;
  double look_log_ratio = 0.0;
  const double invalid_term = copyInvalidityTerm(copy, calibrated, measured_looks, look_log_ratio);
  const double q = stateChangeProbability(state, inherited);
  bool admissible = calibrated;
  if (!calibrated) {
    const double n = static_cast<double>(session_reliable_samples);
    const double scale = static_cast<double>(kEstablishedSamples);
    const double deviance = n * std::log(n / scale) - n + scale;
    admissible = std::log(q) - std::log1p(-q) + (n >= scale ? deviance : -deviance) > 0.0;
  }
  const double stay = (1.0 - q) + (calibrated ? invalid_term : 0.0);
  // B_{M:S} <= 2: when even the bound cannot make M the Bayes action, the geometry cannot
  // change the decision and is not evaluated.
  const bool bound_allows = admissible && q * kGeometryFactorBound > stay;
  double effective_cells = 0.0, off = 0.0, factor = kGeometryFactorBound;
  if (bound_allows) {
    off = offStateShare(copy.geometry, copy.bbox, inherited.geometry, inherited.bbox,
                        kStateTolerance, effective_cells);
    factor = motionGeometryBayesFactor(off, effective_cells);
  }
  const bool elsewhere = bound_allows && q * factor > stay;
  LOG(INFO) << "SAME_STATE inst=" << inherited.semantic_label << "/" << copy.geometry.numVertices()
            << "v copy_reliable=" << session_reliable_samples << " change_prior=" << q
            << " calibrated=" << calibrated << " measured_looks=" << measured_looks
            << " invalid_term=" << invalid_term << " effective_cells=" << effective_cells
            << " off_share=" << (bound_allows ? std::to_string(off) : "unmeasured")
            << " bayes_factor_or_bound=" << factor << " elsewhere=" << elsewhere;
  return elsewhere;
}

void PersistentObjectState::recordLook(Fragment& fragment,
                                       const SurfaceEvidence& evidence,
                                       const TimeStamp stamp) {
  if (evidence.surface_samples == 0) {
    return;  // nothing of this fragment was measured in this round
  }
  fragment.looks.push_back({stamp, evidence.support_rays, evidence.reliable_in_view,
                            evidence.reliable_seen_through,
                            evidence.measured_absence_log_ratio,
                            evidence.has_measured_absence_likelihood,
                            evidence.has_calibrated_absence_source});
}

bool PersistentObjectState::observedEmptySince(
    const Fragment& fragment, const TimeStamp since, const double change_probability,
    const size_t physical_instance_id) {
  size_t support = 0, judged = 0, seen_through = 0;
  double log_ratio = 0.0;
  bool measured = false;
  bool calibrated_source = false;
  for (const auto& look : fragment.looks) {
    if (look.stamp > since) {
      support += look.support_rays;
      judged += look.reliable_in_view;
      seen_through += look.reliable_seen_through;
      log_ratio += look.measured_absence_log_ratio;
      measured = measured || look.has_measured_absence_likelihood;
      calibrated_source = calibrated_source || look.has_calibrated_absence_source;
    }
  }
  const double log_odds = std::log(change_probability) - std::log1p(-change_probability) + log_ratio;
  // Positive identity support conditions the current discrete measurement model;
  // it is not multiplied into the reliable-surface count as independent data.
  // Count-only callers supply the original hard observation contract. Keep its
  // zero-uncertainty limit; lack of a calibrated channel is not lack of a look.
  // A calibrated round with no fresh evidence must never take that exact path.
  const bool empty = calibrated_source
      ? measured && support == 0 && log_odds > std::log(99.0)
      : seen_through > 0 && seen_through == judged && support == 0;
  LOG(INFO) << "EMPTY_INTERVAL_POSTERIOR inst=" << physical_instance_id
            << " since=" << since << " support=" << support << " judged=" << judged
            << " seen_through=" << seen_through << " measured=" << measured
            << " calibrated_source=" << calibrated_source
            << " log_ratio=" << log_ratio << " change_prior=" << change_probability
            << " log_odds=" << log_odds << " empty=" << empty;
  return empty;
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
  target.track_first_seen = std::min(target.track_first_seen, trackFirstSeen(attrs, first));
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

void PersistentObjectState::absorbObservedThrough(PhysicalState& state) {
  if (!state.current || !state.observed_new) {
    return;
  }
  // Precondition: a real measurement confirmed CURRENT present through its
  // last_confirmed_support. One physical ID cannot be in two places at one instant, so
  // observations that began no later than that support are more views of the same state; a
  // candidate born after it may be the object's new site (README: born no later than the
  // support).
  Fragment& current = state.fragments[*state.current];
  if (state.observed_new->birth_time > current.last_confirmed_support) {
    return;
  }
  ++state.mobility_continuations;
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
  current.track_first_seen =
      std::min(current.track_first_seen, state.observed_new->track_first_seen);
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
  ++state.mobility_changes;
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
    b.fragments[*b.current].death_time =
        std::max(stamp, b.fragments[*b.current].last_support_time);
    b.current.reset();  // archived with the rest of the session state below
  }
  foldSessionState(state, stamp);
}

bool PersistentObjectState::handOverInherited(PhysicalState& state, const TimeStamp stamp) {
  closeCurrent(state, stamp);
  if (!state.b_session || !state.b_session->current) return false;
  PhysicalState& b = *state.b_session;
  state.fragments.push_back(std::move(b.fragments[*b.current]));
  b.fragments.erase(b.fragments.begin() + static_cast<std::ptrdiff_t>(*b.current));
  b.current.reset();
  state.current = state.fragments.size() - 1;
  return true;
}

void PersistentObjectState::foldSessionState(PhysicalState& state, const TimeStamp stamp) {
  if (state.b_session) {
    PhysicalState& b = *state.b_session;
    // Nothing of the session state is deleted: its closed fragments and its leftover candidate
    // (a different site: archived, never united) join the identity's history. Its CURRENT, if
    // still set, was just united with the inherited state (same state at finalization).
    for (size_t i = 0; i < b.fragments.size(); ++i) {
      if (b.current && i == *b.current) continue;
      Fragment& fragment = b.fragments[i];
      if (!fragment.death_time) fragment.death_time = std::max(stamp, fragment.last_support_time);
      state.fragments.push_back(std::move(fragment));
    }
    if (b.observed_new) {
      b.observed_new->death_time = stamp;
      state.fragments.push_back(std::move(*b.observed_new));
    }
    state.mobility_changes += b.mobility_changes;
    state.mobility_continuations += b.mobility_continuations;
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
      ++state.mobility_changes;  // D1 within the first segment; no close is called here.
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
    // Keep inherited and session observations independent until measured
    // evidence resolves their relationship online.
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
  const double change_probability = stateChangeProbability(state, current);
  const double shared_probability = sharedSpaceProbability(
      current.geometry, current.bbox, attrs.mesh, attrs.bounding_box,
      map_resolution, change_probability);
  const bool same_session_overlap = shared_probability > change_probability;
  LOG(INFO) << "OVERLAP_POSTERIOR inst=" << physical_instance_id
            << " shared_probability_lower_bound=" << shared_probability
            << " change_prior=" << change_probability
            << " same_state=" << same_session_overlap;
  // Shared space is not confirmation: an object moved by less than its own size
  // lands in space its old state occupied. If CURRENT was observed empty, with
  // nothing supporting it, while this segment was being observed, one identity
  // cannot be in both places: the segment is not a view of CURRENT.
  const bool contradicted = same_session_overlap &&
      observedEmptySince(current, first, change_probability, physical_instance_id);
  LOG(INFO) << "INGEST_DECIDE inst=" << physical_instance_id
            << " same_session_overlap=" << same_session_overlap
            << " contradicted=" << contradicted;
  if (same_session_overlap && !contradicted) {
    // This refines an already associated state; surface segmentation is not
    // an independent opportunity for a competing state (README M2a).
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
    merged.mesh = current.geometry;
    merged.bounding_box = current.bbox;
    merged.position = current.position;
    size_t frames = current.reconstruction_frames;
    if (current.requires_current_session_support &&
        state.b_session && state.b_session->current) {
      // The rays of the last round: contradiction outvotes support (P12-P14).
      const bool already_absent = state.last_contradiction_rays > state.last_support_rays;
      const Fragment& b_current =
          state.b_session->fragments[*state.b_session->current];
      // M1i: the same geometry likelihood and persistence prior as M1h. Insufficient
      // coverage to commit handover does not license a pose union.
      const auto posterior =
          sameStatePosterior(stateChangeProbability(state, current), b_current, current);
      const bool same_site = posterior.same;
      LOG(INFO) << "MATERIALIZE_POSTERIOR inst=" << *instance_id << posterior
                << " inherited_verts=" << current.geometry.numVertices()
                << " session_verts=" << b_current.geometry.numVertices()
                << " same_site=" << same_site << " already_absent=" << already_absent;
      // Same physical state: the A+B refinement is visible online. Otherwise (old site
      // contradicted, or the B-session state at a different site) the inherited state is shown
      // alone; the next evidence round hands over, or the terminal round archives the session
      // state separately.
      if (!already_absent && same_site) {
        appendMeshUnion(merged.mesh, merged.bounding_box,
                        b_current.geometry, b_current.bbox);
        merged.position = merged.bounding_box.world_P_center.cast<double>();
        frames += b_current.reconstruction_frames;
      }
    }
    merged.details[kReconstructionFramesDetail] = {frames};
    merged.details[kHasDynamicHistoryDetail] = {state.has_dynamic_history ? 1u : 0u};
  } else if (!state.fragments.empty()) {
    merged.mesh = spark_dsg::Mesh(merged.mesh.has_colors,
                                  merged.mesh.has_timestamps,
                                  merged.mesh.has_labels,
                                  merged.mesh.has_first_seen_stamps);
    merged.details[kReconstructionFramesDetail] = {0};
    merged.details[kHasDynamicHistoryDetail] = {state.has_dynamic_history ? 1u : 0u};
    // The identity's last state ended with no successor: it is absent, not present without a
    // surface. It left somewhere in (last support, closure]; its presence ends at the estimate
    // of minimum expected risk under a uniform departure time, the midpoint (the reconciler's
    // rule for an observed disappearance). A later segment of the identity opens a new interval.
    const bool session_state = state.b_session && state.b_session->current;
    const Fragment* last = nullptr;
    for (const auto& fragment : state.fragments) {
      if (fragment.death_time && (!last || *fragment.death_time > *last->death_time)) {
        last = &fragment;
      }
    }
    if (!session_state && last && !merged.last_observed_ns.empty()) {
      // The last evidence of presence is the later of its last observation and the last ray
      // confirmation in this session (the reconciler's max(last_persistent, last_seen)).
      const TimeStamp seen = std::max(last->last_support_time, last->last_confirmed_support);
      const TimeStamp left = seen + (std::max(*last->death_time, seen) - seen) / 2;
      while (merged.first_observed_ns.size() > 1 && merged.first_observed_ns.back() > left) {
        merged.first_observed_ns.pop_back();
        merged.last_observed_ns.pop_back();
      }
      merged.last_observed_ns.back() = std::min(merged.last_observed_ns.back(),
                                                std::max(left, merged.first_observed_ns.back()));
    }
  }
  merged.details[kMobilityChangesDetail] = {state.mobility_changes};
  merged.details[kMobilityContinuationsDetail] = {state.mobility_continuations};
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
  absorbObservedThrough(state);
  return true;
}

size_t PersistentObjectState::finalizePendingAbsences(const TimeStamp stamp) {
  // Finalization is a callback of the current event, not a measurement.
  beginObservationEvent(stamp);
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
    const bool have_b_current =
        state.b_session && state.b_session->current;
    const bool inherited_absent =
        state.last_contradiction_rays > state.last_support_rays ||
        sessionCopyElsewhere(state, current, state.last_session_reliable_samples);

    if (inherited_absent) {
      LOG(INFO) << "INHERITED_CLOSE inst=" << id << " by_observed_absence="
                << (state.last_contradiction_rays > state.last_support_rays)
                << " by_session_elsewhere=" << !(state.last_contradiction_rays > state.last_support_rays)
                << " support=" << state.last_support_rays
                << " contradiction=" << state.last_contradiction_rays << " terminal=1";
      handOverInherited(state, stamp);
      ++closed;
    } else if (have_b_current) {
      PhysicalState& b = *state.b_session;
      const Fragment& b_current = b.fragments[*b.current];
      // M1j: finalization uses the same association posterior as online
      // materialization. A different-site unresolved hypothesis is archived
      // by the existing branch below, without changing absence or handover.
      const auto posterior =
          sameStatePosterior(stateChangeProbability(state, current), b_current, current);
      const bool same_site = posterior.same;
      LOG(INFO) << "FINALIZE inst=" << id << posterior
                << " inherited_verts=" << current.geometry.numVertices()
                << " session_verts=" << b_current.geometry.numVertices()
                << " same_site=" << same_site;
      if (same_site) {
        ++state.mobility_continuations;
        appendMeshUnion(current.geometry, current.bbox,
                        b_current.geometry, b_current.bbox);
        current.position = current.bbox.world_P_center.cast<double>();
        current.reconstruction_frames += b_current.reconstruction_frames;
        current.last_support_time =
            std::max(current.last_support_time, b_current.last_support_time);
        current.last_confirmed_support =
            std::max(current.last_confirmed_support, b_current.last_confirmed_support);
        current.birth_time = std::min(current.birth_time, b_current.birth_time);
        current.track_first_seen =
            std::min(current.track_first_seen, b_current.track_first_seen);
      } else {
        // Different site and not absent: identity conflict or a hidden move.
        // Keep the inherited state CURRENT; archive the B-session hypotheses
        // as closed fragments instead of merging or deleting them.
        archiveSessionState(state, stamp);
      }
    }
    foldSessionState(state, stamp);
    state.pending_absence_stamp = 0;
  }
  return closed;
}

bool PersistentObjectState::resolveCurrentEvidence(
    const size_t physical_instance_id,
    const SurfaceEvidence& inherited_evidence,
    const SurfaceEvidence& session_evidence,
    const TimeStamp stamp) {
  beginObservationEvent(stamp);
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
            << " inherited_absent_flag=" << inherited_evidence.absence_coverage_sufficient
            << " cur_verts=" << (state.current ? state.fragments[*state.current].geometry.numVertices() : 0)
            << " observed_verts=" << (state.observed_new ? state.observed_new->geometry.numVertices() : 0);

  // Resolve the independent B-session mini state first. Its D2 decisions are
  // allowed online because both the old and the new observations belong to B.
  if (state.b_session && state.b_session->current) {
    resolveSupportDominance(*state.b_session, session_evidence, physical_instance_id, stamp,
                            "SESSION");
  }

  // Keep the inherited geometry separate and evaluate its measured evidence
  // on every reconciliation round, including the terminal round.
  if (state.current &&
      state.fragments[*state.current].requires_current_session_support) {
    Fragment& inherited = state.fragments[*state.current];
    if (inherited_evidence.support_rays) {
      inherited.last_confirmed_support = std::max(inherited.last_confirmed_support,
          std::min(inherited_evidence.latest_support_stamp, stamp));
    }
    state.last_support_rays = inherited_evidence.support_rays;
    state.last_contradiction_rays = inherited_evidence.absence_coverage_sufficient
                                        ? inherited_evidence.contradiction_rays : 0;
    state.last_surface_samples = inherited_evidence.surface_samples;
    state.last_session_reliable_samples = session_evidence.reliable_samples;
    // Online D2/D3 transition: as soon as the B-session state exists and A's
    // old surface is seen through, switch CURRENT to the B state. Do not wait
    // until the end of the session.
    const bool inherited_absent =
        state.last_contradiction_rays > inherited_evidence.support_rays ||
        sessionCopyElsewhere(state, inherited, session_evidence.reliable_samples);
    if (inherited_absent) {
      LOG(INFO) << "INHERITED_CLOSE inst=" << physical_instance_id << " by_observed_absence="
                << (state.last_contradiction_rays > inherited_evidence.support_rays)
                << " by_session_elsewhere=" << !(state.last_contradiction_rays > inherited_evidence.support_rays)
                << " support=" << inherited_evidence.support_rays
                << " contradiction=" << state.last_contradiction_rays;
      // Seeing the old site empty closes its state even before the identity
      // is seen elsewhere. A new observation is not a deletion prerequisite.
      // So does this session's own established reconstruction of the identity
      // standing mostly off the inherited surface (one identity, one pose).
      if (handOverInherited(state, stamp)) foldSessionState(state, stamp);
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
        session_evidence.surface_samples == 0 && inherited_evidence.surface_samples > 0;
    return resolveSupportDominance(state, use_inherited_slot ? inherited_evidence : session_evidence,
                                   physical_instance_id, stamp, "TOP");
  }
  return false;
}

bool PersistentObjectState::resolveSupportDominance(PhysicalState& b,
                                                    const SurfaceEvidence& evidence,
                                                    const size_t physical_instance_id,
                                                    const TimeStamp stamp, const char* scope) {
  Fragment& current = b.fragments[*b.current];
  recordLook(current, evidence, stamp);
  const size_t support = evidence.support_rays;
  const size_t contradiction =
      evidence.absence_coverage_sufficient ? evidence.contradiction_rays : 0;
  // M1d: a candidate at a different site. The map follows the real world: the candidate is the
  // object's current place as soon as nothing supports the old site (the camera sees the
  // object elsewhere and nothing confirms it at the old site); the old site is kept as closed
  // history, never deleted by the new position. Without a candidate only the old site seen
  // empty (contradiction outvotes support, P12-P14) closes the state.
  bool different_site = false;
  if (b.observed_new) {
    const double same_site_probability =
        extentSameSiteProbability(current.geometry, current.bbox, b.observed_new->geometry,
                                  b.observed_new->bbox, map_resolution_);
    const double q = stateChangeProbability(b, current);
    different_site = same_site_probability < q;
    LOG(INFO) << "EXTENT_POSTERIOR inst=" << physical_instance_id
              << " same_site_probability=" << same_site_probability << " change_prior=" << q
              << " different_site=" << different_site;
  }
  const bool by_new_site = different_site && support == 0;
  const bool by_observed_absence = contradiction > support;
  if (by_new_site || by_observed_absence) {
    LOG(INFO) << scope << "_CLOSE inst=" << physical_instance_id << " by_new_site=" << by_new_site
              << " by_observed_absence=" << by_observed_absence << " support=" << support
              << " contradiction=" << contradiction;
    closeCurrent(b, stamp);
    promoteObservedNew(b);
    return true;
  }
  if (support == 0) return false;
  // Absorbing a candidate presupposes that CURRENT was confirmed present
  // (absorbObservedThrough); shared space alone is not that confirmation. A decision at t=20
  // may only contain support observed at t=5: advancing to t=20 would hide a real departure at
  // t=15 from the next query.
  current.last_confirmed_support = std::max(current.last_confirmed_support,
                                            std::min(evidence.latest_support_stamp, stamp));
  if (!b.observed_new) return false;  // no candidate to absorb
  // M1k/M1l: CURRENT is the supported measurement; the candidate is the shape hypothesized to
  // explain it, absorbed only when it is the same state (an in-session move stays a separate
  // hypothesis until free-space evidence closes the current site).
  const auto posterior =
      sameStatePosterior(stateChangeProbability(b, current), current, *b.observed_new);
  LOG(INFO) << scope << "_ABSORB inst=" << physical_instance_id << posterior
            << " candidate_vertices=" << b.observed_new->geometry.numVertices()
            << " support=" << support << " absorb=" << posterior.same;
  if (posterior.same) absorbObservedThrough(b);
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
    state.mobility_changes = attrs->details.count(kMobilityChangesDetail)
        ? detailValue(*attrs, kMobilityChangesDetail)
        : (state.has_dynamic_history ? 1u : 0u);
    state.mobility_continuations = detailValue(*attrs, kMobilityContinuationsDetail);
  }
}

void PersistentObjectState::clear() {
  states_.clear();
  event_history_.clear();
  event_open_ = false;
  event_stamp_ = 0;
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
