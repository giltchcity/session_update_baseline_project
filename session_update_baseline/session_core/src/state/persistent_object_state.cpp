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

#include <map>
#include <algorithm>
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

// README M2a: ontology groups the prior population; only resolved historical
// relations update the Bernoulli probability. The target never trains its own prior.
double PersistentObjectState::stateChangeProbability(const PhysicalState& state,
                                                     const Fragment& current) const {
  const PhysicalState* owner = &state;
  size_t instance_id = 0;
  double changes = state.mobility_changes;
  double continuations = state.mobility_continuations;
  for (const auto& [id, root] : states_) {
    if (&root == &state || root.b_session.get() == &state) {
      owner = &root;
      instance_id = id;
      if (&root != &state) {
        changes += root.mobility_changes;
        continuations += root.mobility_continuations;
      }
      break;
    }
  }
  double class_changes = 0.0, class_continuations = 0.0;
  double group_changes = 0.0, group_continuations = 0.0;
  const bool group = high_mobility_semantic_labels_.count(current.semantic_label) > 0;
  for (const auto& [id, other] : states_) {
    (void)id;
    if (&other == owner || other.fragments.empty()) continue;
    const auto& fragment = other.current ? other.fragments[*other.current]
                                        : other.fragments.back();
    if (fragment.semantic_label == current.semantic_label) {
      class_changes += other.mobility_changes;
      class_continuations += other.mobility_continuations;
    } else if ((high_mobility_semantic_labels_.count(fragment.semantic_label) > 0) == group) {
      group_changes += other.mobility_changes;
      group_continuations += other.mobility_continuations;
    }
  }
  // One configured ontology judgment contributes one prior opinion, not a
  // measured movement. Empty ontology contributes no directional information.
  const double ontology_opinions = high_mobility_semantic_labels_.empty() ? 0.0 : 1.0;
  const double prior_mass = 1.0 + ontology_opinions;
  const double group_mean = (group_changes + 0.5 + ontology_opinions * group) /
                            (group_changes + group_continuations + prior_mass);
  const double alpha = prior_mass * group_mean + class_changes;
  const double beta = prior_mass * (1.0 - group_mean) + class_continuations;
  const double probability = (alpha + changes) /
                             (alpha + beta + changes + continuations);
  LOG(INFO) << "MOBILITY_PRIOR inst=" << instance_id
            << " session=" << (owner != &state)
            << " class=" << current.semantic_label
            << " changes=" << changes << " continuations=" << continuations
            << " alpha=" << alpha << " beta=" << beta << " q=" << probability;
  return probability;
}

bool PersistentObjectState::isHighMobility(const PhysicalState& state,
                                           const Fragment& current) const {
  return stateChangeProbability(state, current) > 0.5;
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
  const float tol2 = tolerance * tolerance;  // retain exact legacy diagnostic arithmetic
  const double radius2 = radius * radius;
  const double certain_radius = tolerance - resolution;
  const double certain_radius2 = certain_radius * certain_radius;
  size_t hard_off = 0;
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
    for (int dx = -1; dx <= 1 && !certain; ++dx)
      for (int dy = -1; dy <= 1 && !certain; ++dy)
        for (int dz = -1; dz <= 1 && !certain; ++dz) {
          const auto it = grid.find(Key(std::get<0>(k) + dx, std::get<1>(k) + dy, std::get<2>(k) + dz));
          if (it == grid.end()) continue;
          for (const auto& q : it->second) {
            nearest2 = std::min(nearest2, static_cast<double>((q - p).squaredNorm()));
            if (certain_radius >= 0.0 && nearest2 <= certain_radius2) {
              certain = true;
              break;
            }
          }
        }
    if (nearest2 > tol2) ++hard_off;
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
  const double soft_share = expected_off / vertices;
  LOG(INFO) << "COPY_CORRESPONDENCE copy_vertices=" << copy.points.size()
            << " reference_vertices=" << reference.points.size()
            << " resolution=" << resolution << " tolerance=" << tolerance
            << " hard_off_share=" << static_cast<double>(hard_off) / copy.points.size()
            << " soft_off_share=" << soft_share;
  return soft_share;
}

bool PersistentObjectState::sessionCopyElsewhere(const PhysicalState& state,
                                                 const Fragment& inherited,
                                                 const size_t session_reliable_samples) const {
  if (!state.b_session || !state.b_session->current) return false;
  if (session_reliable_samples == 0) return false;  // no established surface measurement
  const Fragment& copy = state.b_session->fragments[*state.b_session->current];
  if (copy.geometry.points.empty()) return false;  // no surface correspondence measurement
  double effective_cells = 0.0;
  const double off = offStateShare(copy.geometry, copy.bbox, inherited.geometry, inherited.bbox,
                                   kStateTolerance, effective_cells);
  // M1f: reliable spatial-cell count under a Poisson coverage model.
  // Profile the unknown intensity on either side of the existing one-look
  // establishment scale. This is a finite likelihood, not a sample-count veto.
  const double n = static_cast<double>(session_reliable_samples);
  const double scale = static_cast<double>(kEstablishedSamples);
  const double deviance = n * std::log(n / scale) - n + scale;
  const double log_ratio = n >= scale ? deviance : -deviance;
  const double q = stateChangeProbability(state, inherited);
  const double log_odds = std::log(q) - std::log1p(-q) + log_ratio;
  const bool established = log_odds > 0.0;
  // M1h: an unchanged object's unseen face can also be disjoint. The moved
  // hypothesis is a majority-off restriction of that common reference model.
  const double motion_bayes_factor = motionGeometryBayesFactor(off, effective_cells);
  const double majority_probability = motion_bayes_factor / 2.0;
  const bool elsewhere = established && q * motion_bayes_factor > 1.0 - q;
  LOG(INFO) << "SAME_STATE inst=" << inherited.semantic_label << "/" << copy.geometry.numVertices()
            << "v copy_reliable=" << session_reliable_samples << " off_share=" << off
            << " tolerance=" << kStateTolerance << " elsewhere=" << elsewhere;
  LOG(INFO) << "COPY_ESTABLISHED_POSTERIOR semantic=" << inherited.semantic_label
            << " reliable=" << session_reliable_samples << " log_ratio=" << log_ratio
            << " change_prior=" << q << " log_odds=" << log_odds
            << " established=" << established;
  LOG(INFO) << "COPY_MOTION_POSTERIOR semantic=" << inherited.semantic_label
            << " effective_cells=" << effective_cells << " off_share=" << off
            << " majority_probability=" << majority_probability
            << " bayes_factor=" << motion_bayes_factor << " change_prior=" << q
            << " established=" << established << " elsewhere=" << elsewhere;
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
  ++state.mobility_continuations;
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
  if (state.b_session) {
    state.mobility_changes += state.b_session->mobility_changes;
    state.mobility_continuations += state.b_session->mobility_continuations;
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
      // M1i: use the same geometry likelihood and persistence prior as M1h.
      // Insufficient coverage to commit handover does not license a pose union.
      const double q = stateChangeProbability(state, current);
      double geometry_factor_or_bound = 2.0;  // exact upper bound of M1h
      double effective_cells = 0.0;
      double off = 0.0;
      const bool geometry_evaluated = 1.0 - q < q * geometry_factor_or_bound;
      if (geometry_evaluated) {
        off = offStateShare(b_current.geometry, b_current.bbox,
                            current.geometry, current.bbox,
                            kStateTolerance, effective_cells);
        geometry_factor_or_bound = b_current.geometry.points.empty()
            ? 1.0  // no correspondence observation, so no likelihood update
            : motionGeometryBayesFactor(off, effective_cells);
      }
      const bool same_site = 1.0 - q >= q * geometry_factor_or_bound;
      LOG(INFO) << "MATERIALIZE_POSTERIOR inst=" << *instance_id
                << " change_prior=" << q
                << " geometry_factor_or_bound=" << geometry_factor_or_bound
                << " geometry_evaluated=" << geometry_evaluated
                << " effective_cells=" << effective_cells
                << " off_share=" << (geometry_evaluated ? std::to_string(off) : "unmeasured")
                << " same_site=" << same_site
                << " already_absent=" << already_absent;
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
  absorbObservedThrough(state, stamp);
  return true;
}

bool PersistentObjectState::inheritedEvidenceAbsent(const PhysicalState&,
                             const Fragment&,
                             size_t support,
                             size_t contradiction,
                             size_t,
                             size_t) {
  // Shared mesh samples are a correspondence hypothesis, not independent
  // RGB-D measurements. They must not outvote an observed empty old site
  // merely because the mesh was tessellated more densely. This also applies
  // to large, usually static objects (a moved bed can overlap its old footprint).
  return contradiction > support;
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
    const bool have_b_current =
        state.b_session && state.b_session->current;
    const bool inherited_absent =
        inheritedEvidenceAbsent(state, current, support, contradiction,
                                geometric, samples) ||
        state.last_session_copy_elsewhere;

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
      // M1j: finalization uses the same association posterior as online
      // materialization. A different-site unresolved hypothesis is archived
      // by the existing branch below, without changing absence or handover.
      const double q = stateChangeProbability(state, current);
      double geometry_factor_or_bound = 2.0;  // exact upper bound of M1h
      double effective_cells = 0.0;
      double off = 0.0;
      const bool geometry_evaluated = 1.0 - q < q * geometry_factor_or_bound;
      if (geometry_evaluated) {
        off = offStateShare(b_current.geometry, b_current.bbox,
                            current.geometry, current.bbox,
                            kStateTolerance, effective_cells);
        geometry_factor_or_bound = b_current.geometry.points.empty()
            ? 1.0
            : motionGeometryBayesFactor(off, effective_cells);
      }
      const bool same_site = 1.0 - q >= q * geometry_factor_or_bound;
      LOG(INFO) << "FINALIZE_POSTERIOR inst=" << id
                << " change_prior=" << q
                << " geometry_factor_or_bound=" << geometry_factor_or_bound
                << " geometry_evaluated=" << geometry_evaluated
                << " effective_cells=" << effective_cells
                << " off_share=" << (geometry_evaluated ? std::to_string(off) : "unmeasured")
                << " legacy_same_site=" << (q <= 0.5 || shared > 0)
                << " same_site=" << same_site;
      LOG(INFO) << "FINALIZE inst=" << id
                << " shared=" << shared
                << " same_site=" << same_site
                << " inherited_verts=" << current.geometry.numVertices()
                << " session_verts=" << b_current.geometry.numVertices();
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
    if (state.b_session) {
      state.mobility_changes += state.b_session->mobility_changes;
      state.mobility_continuations += state.b_session->mobility_continuations;
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
            << " inherited_absent_flag=" << inherited_evidence.absence_coverage_sufficient
            << " cur_verts=" << (state.current ? state.fragments[*state.current].geometry.numVertices() : 0)
            << " observed_verts=" << (state.observed_new ? state.observed_new->geometry.numVertices() : 0);

  // Score the copy measurement against the fragment and prior that produced
  // it. Resolving B may replace CURRENT or enlarge its geometry; B1's reliable
  // sample count must not become a measurement of that successor B2.
  const bool measured_copy_elsewhere =
      state.current &&
      state.fragments[*state.current].requires_current_session_support &&
      sessionCopyElsewhere(state, state.fragments[*state.current],
                           session_evidence.reliable_samples);

  // Resolve the independent B-session mini state first. Its D2 decisions are
  // allowed online because both the old and the new observations belong to B.
  if (state.b_session) {
    PhysicalState& b = *state.b_session;
    const size_t support = session_evidence.support_rays;
    const size_t contradiction = session_evidence.absence_coverage_sufficient
                                     ? session_evidence.contradiction_rays : 0;
    const size_t samples = session_evidence.surface_samples;

    if (b.current) {
      recordLook(b.fragments[*b.current], session_evidence, stamp);
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

      // The map follows the real world: a candidate at a different site is the
      // object's current place as soon as the old site is no longer actively
      // ray-supported (the camera sees the object elsewhere and nothing
      // confirms it at the old site). The old site is preserved as a closed
      // history fragment -- never deleted by the new position. Contradiction
      // dominance (the old site was seen empty) also closes it; this remains
      // the only path for objects that disappear without a replacement.
      // Preserve V37's D2 handoff: a directly observed different-site
      // candidate can take over when the old site has no active support.
      // Without a candidate, only measured absence can close the state.
      const double extent_probability = b.observed_new
          ? extentSameSiteProbability(b.fragments[*b.current].geometry,
                                      b.fragments[*b.current].bbox,
                                      b.observed_new->geometry,
                                      b.observed_new->bbox, map_resolution_)
          : 1.0;
      const double change_prior = b.observed_new
          ? stateChangeProbability(b, b.fragments[*b.current]) : 0.0;
      const bool different_site = b.observed_new && extent_probability < change_prior;
      if (b.observed_new) {
        LOG(INFO) << "EXTENT_POSTERIOR inst=" << physical_instance_id
                  << " same_site_probability=" << extent_probability
                  << " change_prior=" << change_prior
                  << " different_site=" << different_site;
      }
      if ((different_site && support_rate <= 0.0) ||
          contradiction_rate > support_rate) {
        LOG(INFO) << "SESSION_CLOSE inst=" << physical_instance_id
                  << " by_new_site=" << (different_site && support_rate <= 0.0)
                  << " by_observed_absence=" << (contradiction_rate > support_rate);
        closeCurrent(b, stamp);
        promoteObservedNew(b);
        b.has_dynamic_history = true;
      } else if (support_rate > 0.0) {
        // Absorbing a candidate presupposes that CURRENT was confirmed present
        // (absorbObservedThrough). Shared space alone is not that confirmation.
        Fragment& current_b = b.fragments[*b.current];
        // A decision at t=20 may only contain support observed at t=5.
        // Advancing to t=20 would hide a real departure at t=15 from the next query.
        current_b.last_confirmed_support = std::max(current_b.last_confirmed_support,
            std::min(session_evidence.latest_support_stamp, stamp));
        // Absorb the accumulated candidate only when it is actually the same
        // site. A movable identity's candidate at a different location (an
        // in-session move, cabinet X->Y) must stay a separate hypothesis until
        // free-space evidence closes the current site.
        // M1k: CURRENT is the supported measurement; the candidate is the
        // shape hypothesized to explain it. Additional candidate faces are
        // not observed absence of that supported CURRENT surface.
        const double q = stateChangeProbability(b, current_b);
        double geometry_factor_or_bound = 1.0;  // no candidate: no measurement
        double effective_cells = 0.0;
        double off = 0.0;
        bool geometry_evaluated = false;
        if (b.observed_new && !b.observed_new->geometry.points.empty()) {
          geometry_factor_or_bound = 2.0;  // exact upper bound of M1h
          geometry_evaluated = 1.0 - q < q * geometry_factor_or_bound;
          if (geometry_evaluated) {
            off = offStateShare(current_b.geometry, current_b.bbox,
                                b.observed_new->geometry, b.observed_new->bbox,
                                kStateTolerance, effective_cells);
            geometry_factor_or_bound = motionGeometryBayesFactor(off, effective_cells);
          }
        }
        const bool same_site = 1.0 - q >= q * geometry_factor_or_bound;
        LOG(INFO) << "SESSION_ABSORB_POSTERIOR inst=" << physical_instance_id
                  << " measurement=confirmed_current"
                  << " change_prior=" << q
                  << " geometry_factor_or_bound=" << geometry_factor_or_bound
                  << " geometry_evaluated=" << geometry_evaluated
                  << " candidate_vertices="
                  << (b.observed_new ? b.observed_new->geometry.numVertices() : 0)
                  << " effective_cells=" << effective_cells
                  << " off_share=" << (geometry_evaluated ? std::to_string(off) : "unmeasured")
                  << " legacy_same_site=" << (q <= 0.5 || geom > 0)
                  << " absorb=" << same_site;
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
    state.last_session_copy_elsewhere = measured_copy_elsewhere;
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
        state.last_contradiction_rays,
        state.last_geometric_support,
        inherited_evidence.surface_samples) ||
        measured_copy_elsewhere;
    if (inherited_absent) {
      // Seeing the old site empty closes its state even before the identity
      // is seen elsewhere. A new observation is not a deletion prerequisite.
      // So does this session's own established reconstruction of the identity
      // standing mostly off the inherited surface (one identity, one pose).
      closeCurrent(state, stamp);
      if (!state.b_session || !state.b_session->current) {
        state.pending_absence_stamp = 0;
        return true;
      }
      PhysicalState& b = *state.b_session;
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
      if (state.b_session) {
        state.mobility_changes += state.b_session->mobility_changes;
        state.mobility_continuations += state.b_session->mobility_continuations;
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
    recordLook(b.fragments[*b.current], evidence, stamp);
    const size_t support = evidence.support_rays;
    const size_t contradiction = evidence.absence_coverage_sufficient
                                     ? evidence.contradiction_rays : 0;
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

    const double extent_probability = b.observed_new
        ? extentSameSiteProbability(b.fragments[*b.current].geometry,
                                    b.fragments[*b.current].bbox,
                                    b.observed_new->geometry,
                                    b.observed_new->bbox, map_resolution_)
        : 1.0;
    const double change_prior = b.observed_new
        ? stateChangeProbability(b, b.fragments[*b.current]) : 0.0;
    const bool different_site = b.observed_new && extent_probability < change_prior;
    if (b.observed_new) {
      LOG(INFO) << "EXTENT_POSTERIOR inst=" << physical_instance_id
                << " same_site_probability=" << extent_probability
                << " change_prior=" << change_prior
                << " different_site=" << different_site;
    }
    if (b.observed_new && !different_site) {
      LOG(INFO) << "TOP_SAME_SITE_CANDIDATE inst=" << physical_instance_id
                << " candidate_verts=" << b.observed_new->geometry.numVertices();
    }
    if ((different_site && support_rate <= 0.0) ||
          contradiction_rate > support_rate) {
      LOG(INFO) << "TOP_CLOSE inst=" << physical_instance_id
                << " by_new_site=" << (different_site && support_rate <= 0.0)
                << " by_observed_absence=" << (contradiction_rate > support_rate)
                << " support=" << support << " contradiction=" << contradiction
                << " cur_verts=" << b.fragments[*b.current].geometry.numVertices();
      closeCurrent(b, stamp);
      promoteObservedNew(b);
      return true;
    }
    if (support_rate > 0.0) {
      // Absorbing a candidate presupposes that CURRENT was confirmed present
      // (absorbObservedThrough). Shared space alone is not that confirmation.
      Fragment& current = b.fragments[*b.current];
      current.last_confirmed_support = std::max(current.last_confirmed_support,
          std::min(evidence.latest_support_stamp, stamp));
      // M1l: CURRENT is the supported measurement; the candidate is the
      // shape hypothesized to explain it. Additional candidate faces are
      // not observed absence of that supported CURRENT surface.
      const double q = stateChangeProbability(b, current);
      double geometry_factor_or_bound = 1.0;  // no candidate: no measurement
      double effective_cells = 0.0;
      double off = 0.0;
      bool geometry_evaluated = false;
      if (b.observed_new && !b.observed_new->geometry.points.empty()) {
        geometry_factor_or_bound = 2.0;  // exact upper bound of M1h
        geometry_evaluated = 1.0 - q < q * geometry_factor_or_bound;
        if (geometry_evaluated) {
          off = offStateShare(current.geometry, current.bbox,
                              b.observed_new->geometry, b.observed_new->bbox,
                              kStateTolerance, effective_cells);
          geometry_factor_or_bound = motionGeometryBayesFactor(off, effective_cells);
        }
      }
      const bool same_site = 1.0 - q >= q * geometry_factor_or_bound;
      LOG(INFO) << "TOP_ABSORB_POSTERIOR inst=" << physical_instance_id
                << " measurement=confirmed_current"
                << " change_prior=" << q
                << " geometry_factor_or_bound=" << geometry_factor_or_bound
                << " geometry_evaluated=" << geometry_evaluated
                << " candidate_vertices="
                << (b.observed_new ? b.observed_new->geometry.numVertices() : 0)
                << " effective_cells=" << effective_cells
                << " off_share=" << (geometry_evaluated ? std::to_string(off) : "unmeasured")
                << " legacy_same_site=" << (q <= 0.5 || geom > 0)
                << " absorb=" << same_site;
      LOG(INFO) << "TOP_ABSORB inst=" << physical_instance_id
                << " geom=" << geom
                << " support=" << support
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
    state.mobility_changes = attrs->details.count(kMobilityChangesDetail)
        ? detailValue(*attrs, kMobilityChangesDetail)
        : (state.has_dynamic_history ? 1u : 0u);
    state.mobility_continuations = detailValue(*attrs, kMobilityContinuationsDetail);
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
