#include "session_update_baseline/base1/object_guided_map_reconciler.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <cmath>
#include <memory>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include <hydra/reconstruction/marching_cubes.h>
#include <hydra/utils/nearest_neighbor_utilities.h>
#include <khronos/backend/change_detection/ray_verificator.h>
#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <nlohmann/json.hpp>
#include <spark_dsg/mesh.h>

namespace session_update::base1 {

namespace {

uint64_t firstOrZero(const std::vector<uint64_t>& values) {
  return values.empty() ? 0ul : values.front();
}

uint64_t lastOrZero(const std::vector<uint64_t>& values) {
  return values.empty() ? 0ul : values.back();
}

struct InjectionAppendResult {
  std::size_t vertices = 0;
  std::size_t faces = 0;
};

struct PlaneFillResult {
  std::size_t planes = 0;
  std::size_t vertices = 0;
  std::size_t faces = 0;
  std::size_t graph_cut_cells = 0;
  std::size_t graph_cut_fill_cells = 0;
};

enum class ObjectVertexFilter {
  kMinSeparation,
  kMaxSupportDistance,
};

struct ObjectAlignment {
  khronos::Point translation = khronos::Point(0.0f, 0.0f, 0.0f);
  std::size_t support_vertices = 0;
  double before_median_m = 0.0;
  double after_median_m = 0.0;
  bool applied = false;
};

struct FreeSpaceCullResult {
  std::size_t checked_vertices = 0;
  std::size_t absent_vertices = 0;
  std::size_t present_vertices = 0;
  std::size_t removed_vertices = 0;
};

struct PriorMemoryEvidence {
  std::size_t samples = 0;
  std::size_t absent_votes = 0;
  std::size_t present_votes = 0;
  double alpha = 2.0;
  double beta = 1.0;

  double mean() const {
    return alpha / (alpha + beta);
  }
};

struct VolumeGraphCutResult {
  std::size_t cells = 0;
  std::size_t free_evidence_cells = 0;
  std::size_t full_evidence_cells = 0;
  std::size_t structural_cells = 0;
  std::size_t full_cells = 0;
  std::size_t vertices = 0;
  std::size_t faces = 0;
};

class DinicMaxFlow {
 public:
  explicit DinicMaxFlow(std::size_t nodes) : graph_(nodes) {}

  void addEdge(int from, int to, double capacity) {
    Edge forward{to, static_cast<int>(graph_[to].size()), capacity};
    Edge reverse{from, static_cast<int>(graph_[from].size()), 0.0};
    graph_[from].push_back(forward);
    graph_[to].push_back(reverse);
  }

  double maxFlow(int source, int sink) {
    double flow = 0.0;
    constexpr double kEps = 1.0e-9;
    while (buildLevels(source, sink)) {
      iter_.assign(graph_.size(), 0);
      while (true) {
        const double pushed = sendFlow(source, sink, std::numeric_limits<double>::infinity());
        if (pushed <= kEps) {
          break;
        }
        flow += pushed;
      }
    }
    return flow;
  }

  std::vector<bool> sourceReachable(int source) const {
    constexpr double kEps = 1.0e-9;
    std::vector<bool> visited(graph_.size(), false);
    std::queue<int> queue;
    visited[source] = true;
    queue.push(source);
    while (!queue.empty()) {
      const int node = queue.front();
      queue.pop();
      for (const auto& edge : graph_[node]) {
        if (edge.capacity <= kEps || visited[edge.to]) {
          continue;
        }
        visited[edge.to] = true;
        queue.push(edge.to);
      }
    }
    return visited;
  }

 private:
  struct Edge {
    int to = 0;
    int reverse = 0;
    double capacity = 0.0;
  };

  bool buildLevels(int source, int sink) {
    constexpr double kEps = 1.0e-9;
    level_.assign(graph_.size(), -1);
    std::queue<int> queue;
    level_[source] = 0;
    queue.push(source);
    while (!queue.empty()) {
      const int node = queue.front();
      queue.pop();
      for (const auto& edge : graph_[node]) {
        if (edge.capacity <= kEps || level_[edge.to] >= 0) {
          continue;
        }
        level_[edge.to] = level_[node] + 1;
        queue.push(edge.to);
      }
    }
    return level_[sink] >= 0;
  }

  double sendFlow(int node, int sink, double flow) {
    if (node == sink) {
      return flow;
    }
    constexpr double kEps = 1.0e-9;
    for (int& edge_idx = iter_[node]; edge_idx < static_cast<int>(graph_[node].size());
         ++edge_idx) {
      auto& edge = graph_[node][edge_idx];
      if (edge.capacity <= kEps || level_[edge.to] != level_[node] + 1) {
        continue;
      }
      const double pushed = sendFlow(edge.to, sink, std::min(flow, edge.capacity));
      if (pushed <= kEps) {
        continue;
      }
      edge.capacity -= pushed;
      graph_[edge.to][edge.reverse].capacity += pushed;
      return pushed;
    }
    return 0.0;
  }

  std::vector<std::vector<Edge>> graph_;
  std::vector<int> level_;
  std::vector<int> iter_;
};

std::unique_ptr<hydra::PointNeighborSearch> makeInjectionSearch(
    const spark_dsg::Mesh& global_mesh,
    std::vector<khronos::Point>* points) {
  if (!points) {
    return nullptr;
  }

  points->clear();
  points->reserve(global_mesh.numVertices());
  for (std::size_t i = 0; i < global_mesh.numVertices(); ++i) {
    points->push_back(global_mesh.pos(i));
  }

  if (points->empty()) {
    return nullptr;
  }

  return std::make_unique<hydra::PointNeighborSearch>(*points);
}

bool pointInsideExpandedBox(const khronos::BoundingBox& box,
                            const khronos::Point& point,
                            double margin_m) {
  const auto min_corner = box.world_P_center - box.dimensions * 0.5f;
  const auto max_corner = box.world_P_center + box.dimensions * 0.5f;
  return point.x() >= min_corner.x() - margin_m && point.x() <= max_corner.x() + margin_m &&
         point.y() >= min_corner.y() - margin_m && point.y() <= max_corner.y() + margin_m &&
         point.z() >= min_corner.z() - margin_m && point.z() <= max_corner.z() + margin_m;
}

khronos::Point pointFromDouble(const Eigen::Vector3d& point) {
  return khronos::Point(static_cast<float>(point.x()),
                        static_cast<float>(point.y()),
                        static_cast<float>(point.z()));
}

std::vector<khronos::Point> sampleBoxEvidencePoints(const khronos::BoundingBox& box) {
  const Eigen::Vector3d center = box.world_P_center.cast<double>();
  const Eigen::Vector3d half = 0.5 * box.dimensions.cast<double>();
  std::vector<khronos::Point> points;
  points.reserve(15);
  points.push_back(pointFromDouble(center));
  for (int axis = 0; axis < 3; ++axis) {
    for (const double sign : {-1.0, 1.0}) {
      Eigen::Vector3d point = center;
      point(axis) += sign * half(axis);
      points.push_back(pointFromDouble(point));
    }
  }
  for (const double sx : {-1.0, 1.0}) {
    for (const double sy : {-1.0, 1.0}) {
      for (const double sz : {-1.0, 1.0}) {
        points.push_back(pointFromDouble(center + half.cwiseProduct(Eigen::Vector3d(sx, sy, sz))));
      }
    }
  }
  return points;
}

PriorMemoryEvidence evaluatePriorMemoryEvidence(
    const khronos::BoundingBox& prior_box,
    const khronos::DynamicSceneGraph& current_dsg,
    const ReconcilerConfig& config,
    double prior_alpha,
    double prior_beta) {
  PriorMemoryEvidence evidence;
  evidence.alpha = prior_alpha > 0.0 ? prior_alpha : 2.0;
  evidence.beta = prior_beta > 0.0 ? prior_beta : 1.0;
  auto current_clone = current_dsg.clone();
  if (!current_clone) {
    return evidence;
  }

  khronos::RayVerificator::Config ray_config;
  ray_config.block_size = static_cast<float>(config.free_space_culling_block_size_m);
  ray_config.radial_tolerance =
      static_cast<float>(config.free_space_culling_radial_tolerance_m);
  ray_config.depth_tolerance =
      static_cast<float>(config.free_space_culling_depth_tolerance_m);
  ray_config.active_window_duration =
      static_cast<float>(config.free_space_culling_active_window_duration_s);
  ray_config.ray_policy = khronos::RayVerificator::Config::RayPolicy::kMiddle;

  khronos::RayVerificator verifier(ray_config);
  verifier.setDsg(current_clone);
  for (const auto& point : sampleBoxEvidencePoints(prior_box)) {
    const auto check = verifier.check(point);
    if (!check.absent.empty()) {
      ++evidence.absent_votes;
    }
    if (!check.present.empty()) {
      ++evidence.present_votes;
    }
    ++evidence.samples;
  }

  evidence.alpha += static_cast<double>(evidence.present_votes);
  evidence.beta += static_cast<double>(evidence.absent_votes);
  return evidence;
}

khronos::Point translatedPoint(const khronos::Point& point, const khronos::Point& translation) {
  return khronos::Point(point.x() + translation.x(),
                        point.y() + translation.y(),
                        point.z() + translation.z());
}

double medianScalar(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  const auto mid = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
  std::nth_element(values.begin(), mid, values.end());
  double median = *mid;
  if (values.size() % 2 == 0) {
    const auto lower = std::max_element(values.begin(), mid);
    median = 0.5 * (median + *lower);
  }
  return median;
}

ObjectAlignment estimateObjectTranslationAlignment(
    const khronos::KhronosObjectAttributes& attrs,
    const hydra::PointNeighborSearch* background_search,
    const std::vector<khronos::Point>& background_points,
    double support_distance_m,
    double max_translation_m,
    std::size_t min_support_vertices) {
  ObjectAlignment result;
  if (!background_search || background_points.empty() || attrs.mesh.numVertices() == 0 ||
      support_distance_m <= 0.0 || max_translation_m <= 0.0) {
    return result;
  }

  const double support_threshold_sq = support_distance_m * support_distance_m;
  const std::size_t max_samples = 5000;
  const std::size_t stride =
      std::max<std::size_t>(1, attrs.mesh.numVertices() / max_samples);
  std::vector<double> dx_values;
  std::vector<double> dy_values;
  std::vector<double> dz_values;
  std::vector<double> before_distances;
  std::vector<khronos::Point> residuals;
  dx_values.reserve(max_samples);
  dy_values.reserve(max_samples);
  dz_values.reserve(max_samples);
  before_distances.reserve(max_samples);
  residuals.reserve(max_samples);

  for (std::size_t i = 0; i < attrs.mesh.numVertices(); i += stride) {
    const auto world_point = attrs.bounding_box.pointToWorldFrame(attrs.mesh.pos(i));
    float distance_sq = std::numeric_limits<float>::max();
    std::size_t nearest_idx = 0;
    if (!background_search->search(world_point, distance_sq, nearest_idx) ||
        nearest_idx >= background_points.size() || distance_sq > support_threshold_sq) {
      continue;
    }

    const auto& nearest = background_points[nearest_idx];
    const khronos::Point residual(nearest.x() - world_point.x(),
                                  nearest.y() - world_point.y(),
                                  nearest.z() - world_point.z());
    dx_values.push_back(static_cast<double>(residual.x()));
    dy_values.push_back(static_cast<double>(residual.y()));
    dz_values.push_back(static_cast<double>(residual.z()));
    before_distances.push_back(std::sqrt(static_cast<double>(distance_sq)));
    residuals.push_back(residual);
  }

  result.support_vertices = residuals.size();
  if (result.support_vertices < min_support_vertices) {
    return result;
  }

  khronos::Point translation(static_cast<float>(medianScalar(dx_values)),
                             static_cast<float>(medianScalar(dy_values)),
                             static_cast<float>(medianScalar(dz_values)));
  double norm = std::sqrt(static_cast<double>(translation.x()) * translation.x() +
                          static_cast<double>(translation.y()) * translation.y() +
                          static_cast<double>(translation.z()) * translation.z());
  if (norm > max_translation_m) {
    const double scale = max_translation_m / std::max(norm, 1.0e-9);
    translation = khronos::Point(static_cast<float>(translation.x() * scale),
                                 static_cast<float>(translation.y() * scale),
                                 static_cast<float>(translation.z() * scale));
    norm = max_translation_m;
  }
  if (norm < 0.002) {
    return result;
  }

  std::vector<double> after_distances;
  after_distances.reserve(residuals.size());
  for (const auto& residual : residuals) {
    const double rx = static_cast<double>(residual.x() - translation.x());
    const double ry = static_cast<double>(residual.y() - translation.y());
    const double rz = static_cast<double>(residual.z() - translation.z());
    after_distances.push_back(std::sqrt(rx * rx + ry * ry + rz * rz));
  }

  result.before_median_m = medianScalar(before_distances);
  result.after_median_m = medianScalar(after_distances);
  if (result.after_median_m + 0.005 >= result.before_median_m) {
    return result;
  }

  result.translation = translation;
  result.applied = true;
  return result;
}

FreeSpaceCullResult cullFreeSpaceContradictedAddedVertices(
    const khronos::DynamicSceneGraph::Ptr& reference_dsg,
    spark_dsg::Mesh& mesh,
    std::size_t first_added_vertex,
    const ReconcilerConfig& config,
    std::vector<ReconcileResult::VertexUpdateRow>* update_rows) {
  FreeSpaceCullResult result;
  if (!config.free_space_culling || !reference_dsg || first_added_vertex >= mesh.numVertices()) {
    return result;
  }

  khronos::RayVerificator::Config ray_config;
  ray_config.block_size = static_cast<float>(config.free_space_culling_block_size_m);
  ray_config.radial_tolerance =
      static_cast<float>(config.free_space_culling_radial_tolerance_m);
  ray_config.depth_tolerance =
      static_cast<float>(config.free_space_culling_depth_tolerance_m);
  ray_config.active_window_duration =
      static_cast<float>(config.free_space_culling_active_window_duration_s);
  ray_config.ray_policy = khronos::RayVerificator::Config::RayPolicy::kMiddle;

  const std::size_t min_absent = std::max<std::size_t>(1, config.free_space_culling_min_absent);
  std::unordered_set<std::size_t> vertices_to_delete;
  vertices_to_delete.reserve(mesh.numVertices() - first_added_vertex);

  khronos::RayVerificator verifier(ray_config);
  verifier.setDsg(reference_dsg);

  for (std::size_t vertex_idx = first_added_vertex; vertex_idx < mesh.numVertices();
       ++vertex_idx) {
    ++result.checked_vertices;
    const auto check = verifier.check(mesh.pos(vertex_idx));
    if (!check.absent.empty()) {
      ++result.absent_vertices;
    }
    if (!check.present.empty()) {
      ++result.present_vertices;
    }
    bool remove_vertex = false;
    if (config.free_space_culling_decision == "absence_majority") {
      remove_vertex = check.absent.size() >= min_absent &&
                      check.absent.size() > check.present.size();
    } else {
      remove_vertex = check.absent.size() >= min_absent &&
                      check.present.size() <= config.free_space_culling_max_present;
    }
    if (!remove_vertex) {
      continue;
    }

    vertices_to_delete.insert(vertex_idx);
    if (update_rows) {
      const int vertex_label =
          vertex_idx < mesh.labels.size() ? static_cast<int>(mesh.labels[vertex_idx]) : -1;
      update_rows->push_back(ReconcileResult::VertexUpdateRow{
          vertex_idx,
          0,
          vertex_label,
          -1,
          0.0,
          config.dry_run ? "free_space_candidate_dry_run" : "free_space_remove"});
    }
  }

  if (!config.dry_run && !vertices_to_delete.empty()) {
    mesh.eraseVertices(vertices_to_delete);
    result.removed_vertices = vertices_to_delete.size();
  }
  return result;
}

InjectionAppendResult appendObjectMeshToGlobal(
    const khronos::KhronosObjectAttributes& attrs,
    spark_dsg::Mesh& global_mesh,
    const hydra::PointNeighborSearch* background_search,
    double filter_distance_m,
    ObjectVertexFilter filter,
    const khronos::Point& translation) {
  // Base1 repair/injection action:
  // Khronos stores static object geometry as KhronosObjectAttributes::mesh, whose
  // points are relative to the object bounding-box frame (spark_dsg/node_attributes.h).
  // The optional nearest-neighbor filter supports two evidence directions:
  // kMinSeparation keeps novel repair vertices, while kMaxSupportDistance keeps
  // only vertices close to an already supported surface.
  const bool use_filter = background_search != nullptr && filter_distance_m > 0.0;
  const double threshold_sq = filter_distance_m * filter_distance_m;

  std::vector<std::size_t> kept_vertices;
  kept_vertices.reserve(attrs.mesh.numVertices());
  std::vector<std::size_t> old_to_new(attrs.mesh.numVertices(),
                                      std::numeric_limits<std::size_t>::max());

  for (std::size_t i = 0; i < attrs.mesh.numVertices(); ++i) {
    const auto world_point =
        translatedPoint(attrs.bounding_box.pointToWorldFrame(attrs.mesh.pos(i)), translation);
    if (use_filter) {
      float distance_sq = std::numeric_limits<float>::max();
      std::size_t nearest_idx = 0;
      const bool found = background_search->search(world_point, distance_sq, nearest_idx);
      if (filter == ObjectVertexFilter::kMinSeparation && found &&
          distance_sq <= threshold_sq) {
        continue;
      }
      if (filter == ObjectVertexFilter::kMaxSupportDistance &&
          (!found || distance_sq > threshold_sq)) {
        continue;
      }
    }

    old_to_new[i] = kept_vertices.size();
    kept_vertices.push_back(i);
  }

  if (kept_vertices.empty()) {
    return {};
  }

  spark_dsg::Mesh object_world(global_mesh.has_colors,
                               global_mesh.has_timestamps,
                               global_mesh.has_labels,
                               global_mesh.has_first_seen_stamps);
  object_world.resizeVertices(kept_vertices.size());
  for (std::size_t new_idx = 0; new_idx < kept_vertices.size(); ++new_idx) {
    const auto i = kept_vertices[new_idx];
    object_world.setPos(
        new_idx, translatedPoint(attrs.bounding_box.pointToWorldFrame(attrs.mesh.pos(i)),
                                 translation));
    if (object_world.has_colors) {
      const auto color =
          attrs.mesh.has_colors && i < attrs.mesh.colors.size() ? attrs.mesh.colors[i]
                                                                : spark_dsg::Color(180, 180, 180);
      object_world.setColor(new_idx, color);
    }
    if (object_world.has_timestamps) {
      const auto stamp =
          attrs.mesh.has_timestamps && i < attrs.mesh.stamps.size()
              ? attrs.mesh.stamps[i]
              : lastOrZero(attrs.last_observed_ns);
      object_world.setTimestamp(new_idx, stamp);
    }
    if (object_world.has_first_seen_stamps) {
      const auto stamp =
          attrs.mesh.has_first_seen_stamps && i < attrs.mesh.first_seen_stamps.size()
              ? attrs.mesh.first_seen_stamps[i]
              : firstOrZero(attrs.first_observed_ns);
      object_world.setFirstSeenTimestamp(new_idx, stamp);
    }
    if (object_world.has_labels) {
      object_world.setLabel(new_idx, static_cast<spark_dsg::Mesh::Label>(attrs.semantic_label));
    }
  }

  std::vector<spark_dsg::Mesh::Face> kept_faces;
  kept_faces.reserve(attrs.mesh.numFaces());
  for (std::size_t i = 0; i < attrs.mesh.numFaces(); ++i) {
    const auto face = attrs.mesh.face(i);
    if (face[0] >= old_to_new.size() || face[1] >= old_to_new.size() ||
        face[2] >= old_to_new.size()) {
      continue;
    }

    const auto new_a = old_to_new[face[0]];
    const auto new_b = old_to_new[face[1]];
    const auto new_c = old_to_new[face[2]];
    if (new_a == std::numeric_limits<std::size_t>::max() ||
        new_b == std::numeric_limits<std::size_t>::max() ||
        new_c == std::numeric_limits<std::size_t>::max()) {
      continue;
    }

    kept_faces.push_back({new_a, new_b, new_c});
  }

  object_world.resizeFaces(kept_faces.size());
  for (std::size_t i = 0; i < kept_faces.size(); ++i) {
    object_world.face(i) = kept_faces[i];
  }

  if (!global_mesh.append(object_world)) {
    throw std::runtime_error("Failed to append object private mesh to global mesh.");
  }

  return {kept_vertices.size(), kept_faces.size()};
}

InjectionAppendResult appendBackgroundMeshToGlobal(
    const spark_dsg::Mesh& source_mesh,
    spark_dsg::Mesh& global_mesh,
    const hydra::PointNeighborSearch* background_search,
    double min_separation_m,
    const std::vector<khronos::BoundingBox>& reject_boxes,
    double bbox_margin_m) {
  const bool filter_by_separation = background_search != nullptr && min_separation_m > 0.0;
  const double threshold_sq = min_separation_m * min_separation_m;

  std::vector<std::size_t> kept_vertices;
  kept_vertices.reserve(source_mesh.numVertices());
  std::vector<std::size_t> old_to_new(source_mesh.numVertices(),
                                      std::numeric_limits<std::size_t>::max());

  for (std::size_t i = 0; i < source_mesh.numVertices(); ++i) {
    const auto point = source_mesh.pos(i);
    bool rejected = false;
    for (const auto& box : reject_boxes) {
      if (pointInsideExpandedBox(box, point, bbox_margin_m)) {
        rejected = true;
        break;
      }
    }
    if (rejected) {
      continue;
    }

    if (filter_by_separation) {
      float distance_sq = std::numeric_limits<float>::max();
      std::size_t nearest_idx = 0;
      if (background_search->search(point, distance_sq, nearest_idx) &&
          distance_sq <= threshold_sq) {
        continue;
      }
    }

    old_to_new[i] = kept_vertices.size();
    kept_vertices.push_back(i);
  }

  if (kept_vertices.empty()) {
    return {};
  }

  spark_dsg::Mesh fragment(global_mesh.has_colors,
                           global_mesh.has_timestamps,
                           global_mesh.has_labels,
                           global_mesh.has_first_seen_stamps);
  fragment.resizeVertices(kept_vertices.size());
  for (std::size_t new_idx = 0; new_idx < kept_vertices.size(); ++new_idx) {
    const auto i = kept_vertices[new_idx];
    fragment.setPos(new_idx, source_mesh.pos(i));
    if (fragment.has_colors) {
      const auto color =
          source_mesh.has_colors && i < source_mesh.colors.size()
              ? source_mesh.colors[i]
              : spark_dsg::Color(160, 160, 160);
      fragment.setColor(new_idx, color);
    }
    if (fragment.has_timestamps) {
      const auto stamp =
          source_mesh.has_timestamps && i < source_mesh.stamps.size() ? source_mesh.stamps[i] : 0;
      fragment.setTimestamp(new_idx, stamp);
    }
    if (fragment.has_first_seen_stamps) {
      const auto stamp =
          source_mesh.has_first_seen_stamps && i < source_mesh.first_seen_stamps.size()
              ? source_mesh.first_seen_stamps[i]
              : 0;
      fragment.setFirstSeenTimestamp(new_idx, stamp);
    }
    if (fragment.has_labels) {
      const auto label =
          source_mesh.has_labels && i < source_mesh.labels.size()
              ? source_mesh.labels[i]
              : static_cast<spark_dsg::Mesh::Label>(-1);
      fragment.setLabel(new_idx, label);
    }
  }

  std::vector<spark_dsg::Mesh::Face> kept_faces;
  kept_faces.reserve(source_mesh.numFaces());
  for (std::size_t i = 0; i < source_mesh.numFaces(); ++i) {
    const auto face = source_mesh.face(i);
    if (face[0] >= old_to_new.size() || face[1] >= old_to_new.size() ||
        face[2] >= old_to_new.size()) {
      continue;
    }
    const auto a = old_to_new[face[0]];
    const auto b = old_to_new[face[1]];
    const auto c = old_to_new[face[2]];
    if (a == std::numeric_limits<std::size_t>::max() ||
        b == std::numeric_limits<std::size_t>::max() ||
        c == std::numeric_limits<std::size_t>::max()) {
      continue;
    }
    kept_faces.push_back({a, b, c});
  }

  fragment.resizeFaces(kept_faces.size());
  for (std::size_t i = 0; i < kept_faces.size(); ++i) {
    fragment.face(i) = kept_faces[i];
  }

  if (!global_mesh.append(fragment)) {
    throw std::runtime_error("Failed to append temporal background mesh to global mesh.");
  }

  return {kept_vertices.size(), kept_faces.size()};
}

InjectionAppendResult appendBackgroundMeshWelded(
    const spark_dsg::Mesh& source_mesh,
    spark_dsg::Mesh& global_mesh,
    const hydra::PointNeighborSearch* background_search,
    double weld_distance_m,
    const std::vector<khronos::BoundingBox>& reject_boxes,
    double bbox_margin_m) {
  const bool weld_vertices = background_search != nullptr && weld_distance_m > 0.0;
  const double threshold_sq = weld_distance_m * weld_distance_m;
  const auto initial_vertices = global_mesh.numVertices();
  const auto initial_faces = global_mesh.numFaces();
  const auto invalid_index = std::numeric_limits<std::size_t>::max();

  std::vector<std::size_t> source_to_global(source_mesh.numVertices(), invalid_index);
  std::vector<std::size_t> new_source_vertices;
  new_source_vertices.reserve(source_mesh.numVertices());

  for (std::size_t i = 0; i < source_mesh.numVertices(); ++i) {
    const auto point = source_mesh.pos(i);
    bool rejected = false;
    for (const auto& box : reject_boxes) {
      if (pointInsideExpandedBox(box, point, bbox_margin_m)) {
        rejected = true;
        break;
      }
    }
    if (rejected) {
      continue;
    }

    if (weld_vertices) {
      float distance_sq = std::numeric_limits<float>::max();
      std::size_t nearest_idx = 0;
      if (background_search->search(point, distance_sq, nearest_idx) &&
          distance_sq <= threshold_sq) {
        source_to_global[i] = nearest_idx;
        continue;
      }
    }

    source_to_global[i] = initial_vertices + new_source_vertices.size();
    new_source_vertices.push_back(i);
  }

  global_mesh.resizeVertices(initial_vertices + new_source_vertices.size());
  for (std::size_t offset = 0; offset < new_source_vertices.size(); ++offset) {
    const auto source_idx = new_source_vertices[offset];
    const auto target_idx = initial_vertices + offset;
    global_mesh.setPos(target_idx, source_mesh.pos(source_idx));
    if (global_mesh.has_colors) {
      const auto color =
          source_mesh.has_colors && source_idx < source_mesh.colors.size()
              ? source_mesh.colors[source_idx]
              : spark_dsg::Color(160, 160, 160);
      global_mesh.setColor(target_idx, color);
    }
    if (global_mesh.has_timestamps) {
      const auto stamp =
          source_mesh.has_timestamps && source_idx < source_mesh.stamps.size()
              ? source_mesh.stamps[source_idx]
              : 0;
      global_mesh.setTimestamp(target_idx, stamp);
    }
    if (global_mesh.has_first_seen_stamps) {
      const auto stamp =
          source_mesh.has_first_seen_stamps &&
                  source_idx < source_mesh.first_seen_stamps.size()
              ? source_mesh.first_seen_stamps[source_idx]
              : 0;
      global_mesh.setFirstSeenTimestamp(target_idx, stamp);
    }
    if (global_mesh.has_labels) {
      const auto label =
          source_mesh.has_labels && source_idx < source_mesh.labels.size()
              ? source_mesh.labels[source_idx]
              : static_cast<spark_dsg::Mesh::Label>(-1);
      global_mesh.setLabel(target_idx, label);
    }
  }

  std::set<std::array<std::size_t, 3>> face_keys;
  for (std::size_t i = 0; i < initial_faces; ++i) {
    auto key = global_mesh.face(i);
    std::sort(key.begin(), key.end());
    face_keys.insert(key);
  }

  std::vector<spark_dsg::Mesh::Face> new_faces;
  new_faces.reserve(source_mesh.numFaces());
  for (std::size_t i = 0; i < source_mesh.numFaces(); ++i) {
    const auto source_face = source_mesh.face(i);
    if (source_face[0] >= source_to_global.size() ||
        source_face[1] >= source_to_global.size() ||
        source_face[2] >= source_to_global.size()) {
      continue;
    }

    spark_dsg::Mesh::Face mapped = {source_to_global[source_face[0]],
                                    source_to_global[source_face[1]],
                                    source_to_global[source_face[2]]};
    if (mapped[0] == invalid_index || mapped[1] == invalid_index ||
        mapped[2] == invalid_index || mapped[0] == mapped[1] ||
        mapped[0] == mapped[2] || mapped[1] == mapped[2]) {
      continue;
    }

    auto key = mapped;
    std::sort(key.begin(), key.end());
    if (!face_keys.insert(key).second) {
      continue;
    }
    new_faces.push_back(mapped);
  }

  global_mesh.resizeFaces(initial_faces + new_faces.size());
  for (std::size_t i = 0; i < new_faces.size(); ++i) {
    global_mesh.face(initial_faces + i) = new_faces[i];
  }

  return {new_source_vertices.size(), new_faces.size()};
}

bool meshesHaveIdenticalTopology(const spark_dsg::Mesh& lhs,
                                 const spark_dsg::Mesh& rhs) {
  if (lhs.numVertices() != rhs.numVertices() || lhs.numFaces() != rhs.numFaces()) {
    return false;
  }
  for (std::size_t i = 0; i < lhs.numVertices(); ++i) {
    if (lhs.pos(i) != rhs.pos(i)) {
      return false;
    }
  }
  for (std::size_t i = 0; i < lhs.numFaces(); ++i) {
    if (lhs.face(i) != rhs.face(i)) {
      return false;
    }
  }
  return true;
}

void initializeCrossSessionMesh(
    khronos::DynamicSceneGraph& output_dsg,
    const khronos::DynamicSceneGraph::Ptr& current_evidence_dsg,
    const khronos::DynamicSceneGraph::Ptr& prior_dsg,
    const ReconcilerConfig& config,
    MeshUpdateSummary* summary,
    std::vector<ReconcileResult::VertexUpdateRow>* update_rows) {
  if (!summary || !current_evidence_dsg || !prior_dsg || !prior_dsg->hasMesh()) {
    return;
  }

  const auto current_mesh =
      current_evidence_dsg->hasMesh() ? current_evidence_dsg->mesh()->clone() : nullptr;
  const auto prior_mesh = prior_dsg->mesh()->clone();
  if (!prior_mesh) {
    return;
  }

  summary->cross_session_prior_vertices = prior_mesh->numVertices();
  summary->cross_session_prior_faces = prior_mesh->numFaces();
  if (current_mesh) {
    summary->cross_session_current_vertices = current_mesh->numVertices();
    summary->cross_session_current_faces = current_mesh->numFaces();
  }

  // Panoptic-style initialization: the loaded prior is the working map. Current
  // session geometry is evidence that may confirm, remove, or extend that map.
  output_dsg.setMesh(prior_mesh);
  auto& output_mesh = *output_dsg.mesh();

  std::vector<khronos::Point> current_points;
  auto current_search =
      current_mesh && current_mesh->numVertices() > 0
          ? makeInjectionSearch(*current_mesh, &current_points)
          : nullptr;
  const double surface_support_threshold_sq =
      config.cross_session_mesh_merge_distance_m *
      config.cross_session_mesh_merge_distance_m;

  if (config.cross_session_remove_absent_prior && output_mesh.numVertices() > 0 &&
      current_evidence_dsg->hasMesh()) {
    khronos::RayVerificator::Config ray_config;
    ray_config.block_size = static_cast<float>(config.free_space_culling_block_size_m);
    ray_config.radial_tolerance =
        static_cast<float>(config.free_space_culling_radial_tolerance_m);
    ray_config.depth_tolerance =
        static_cast<float>(config.free_space_culling_depth_tolerance_m);
    ray_config.active_window_duration =
        static_cast<float>(config.free_space_culling_active_window_duration_s);
    ray_config.ray_policy = khronos::RayVerificator::Config::RayPolicy::kMiddle;

    const std::size_t min_absent =
        std::max<std::size_t>(1, config.free_space_culling_min_absent);
    std::unordered_set<std::size_t> vertices_to_delete;
    khronos::RayVerificator verifier(ray_config);
    verifier.setDsg(current_evidence_dsg);

    for (std::size_t vertex_idx = 0; vertex_idx < output_mesh.numVertices(); ++vertex_idx) {
      ++summary->cross_session_prior_checked_vertices;
      if (current_search && config.cross_session_mesh_merge_distance_m > 0.0) {
        float distance_sq = std::numeric_limits<float>::max();
        std::size_t nearest_idx = 0;
        if (current_search->search(output_mesh.pos(vertex_idx), distance_sq, nearest_idx) &&
            distance_sq <= surface_support_threshold_sq) {
          ++summary->cross_session_prior_persistent_vertices;
          continue;
        }
      }

      const auto check = verifier.check(output_mesh.pos(vertex_idx));
      const bool has_absent = check.absent.size() >= min_absent;
      const bool has_present = !check.present.empty();
      bool remove = false;
      if (config.free_space_culling_decision == "absence_majority") {
        remove = has_absent && check.absent.size() > check.present.size();
      } else {
        remove =
            has_absent && check.present.size() <= config.free_space_culling_max_present;
      }

      if (remove) {
        ++summary->cross_session_prior_absent_vertices;
        vertices_to_delete.insert(vertex_idx);
        if (update_rows) {
          const int label =
              vertex_idx < output_mesh.labels.size()
                  ? static_cast<int>(output_mesh.labels[vertex_idx])
                  : -1;
          update_rows->push_back(ReconcileResult::VertexUpdateRow{
              vertex_idx, 0, label, -1, 0.0, "cross_session_prior_absent_remove"});
        }
      } else if (has_present) {
        ++summary->cross_session_prior_persistent_vertices;
      } else {
        ++summary->cross_session_prior_unobserved_vertices;
      }
    }

    if (!config.dry_run && !vertices_to_delete.empty()) {
      output_mesh.eraseVertices(vertices_to_delete);
    }
  } else {
    summary->cross_session_prior_unobserved_vertices = output_mesh.numVertices();
  }

  if (!current_mesh || current_mesh->numVertices() == 0) {
    return;
  }
  if (meshesHaveIdenticalTopology(*current_mesh, output_mesh)) {
    return;
  }

  std::vector<khronos::Point> prior_points;
  auto prior_search =
      config.cross_session_mesh_merge_distance_m > 0.0
          ? makeInjectionSearch(output_mesh, &prior_points)
          : nullptr;
  const auto appended = appendBackgroundMeshWelded(
      *current_mesh,
      output_mesh,
      prior_search.get(),
      config.cross_session_mesh_merge_distance_m,
      {},
      config.bbox_margin_m);
  summary->cross_session_current_injected_vertices = appended.vertices;
  summary->cross_session_current_injected_faces = appended.faces;
}

PlaneFillResult fillHorizontalPlane(spark_dsg::Mesh& global_mesh,
                                    double z,
                                    double min_x,
                                    double max_x,
                                    double min_y,
                                    double max_y,
                                    double resolution_m) {
  if (resolution_m <= 0.0 || min_x >= max_x || min_y >= max_y) {
    return {};
  }

  std::vector<khronos::Point> reference_points;
  auto search = makeInjectionSearch(global_mesh, &reference_points);
  if (!search) {
    return {};
  }
  const double threshold_sq = resolution_m * resolution_m;
  const std::size_t nx = static_cast<std::size_t>(std::floor((max_x - min_x) / resolution_m)) + 1;
  const std::size_t ny = static_cast<std::size_t>(std::floor((max_y - min_y) / resolution_m)) + 1;
  if (nx < 2 || ny < 2 || nx * ny > 2000000) {
    return {};
  }

  std::vector<std::size_t> grid_to_vertex(nx * ny, std::numeric_limits<std::size_t>::max());
  std::vector<khronos::Point> points;
  points.reserve(nx * ny);
  for (std::size_t ix = 0; ix < nx; ++ix) {
    const double x = min_x + static_cast<double>(ix) * resolution_m;
    for (std::size_t iy = 0; iy < ny; ++iy) {
      const double y = min_y + static_cast<double>(iy) * resolution_m;
      const khronos::Point point(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
      float distance_sq = std::numeric_limits<float>::max();
      std::size_t nearest_idx = 0;
      if (search->search(point, distance_sq, nearest_idx) && distance_sq <= threshold_sq) {
        continue;
      }
      grid_to_vertex[ix * ny + iy] = points.size();
      points.push_back(point);
    }
  }

  if (points.empty()) {
    return {};
  }

  std::vector<spark_dsg::Mesh::Face> faces;
  faces.reserve(points.size() * 2);
  for (std::size_t ix = 0; ix + 1 < nx; ++ix) {
    for (std::size_t iy = 0; iy + 1 < ny; ++iy) {
      const auto a = grid_to_vertex[ix * ny + iy];
      const auto b = grid_to_vertex[(ix + 1) * ny + iy];
      const auto c = grid_to_vertex[ix * ny + iy + 1];
      const auto d = grid_to_vertex[(ix + 1) * ny + iy + 1];
      if (a == std::numeric_limits<std::size_t>::max() ||
          b == std::numeric_limits<std::size_t>::max() ||
          c == std::numeric_limits<std::size_t>::max() ||
          d == std::numeric_limits<std::size_t>::max()) {
        continue;
      }
      faces.push_back({a, b, c});
      faces.push_back({b, d, c});
    }
  }

  spark_dsg::Mesh plane(global_mesh.has_colors,
                        global_mesh.has_timestamps,
                        global_mesh.has_labels,
                        global_mesh.has_first_seen_stamps);
  plane.resizeVertices(points.size());
  for (std::size_t i = 0; i < points.size(); ++i) {
    plane.setPos(i, points[i]);
    if (plane.has_colors) {
      plane.setColor(i, spark_dsg::Color(150, 150, 150));
    }
    if (plane.has_timestamps) {
      plane.setTimestamp(i, 0);
    }
    if (plane.has_first_seen_stamps) {
      plane.setFirstSeenTimestamp(i, 0);
    }
    if (plane.has_labels) {
      plane.setLabel(i, static_cast<spark_dsg::Mesh::Label>(-1));
    }
  }
  plane.resizeFaces(faces.size());
  for (std::size_t i = 0; i < faces.size(); ++i) {
    plane.face(i) = faces[i];
  }

  if (!global_mesh.append(plane)) {
    throw std::runtime_error("Failed to append horizontal plane fill.");
  }
  return {1, points.size(), faces.size()};
}

PlaneFillResult fillHorizontalPlaneFromSupport(
    spark_dsg::Mesh& global_mesh,
    double z,
    const std::vector<khronos::Point>& support_points,
    double resolution_m) {
  if (support_points.empty() || resolution_m <= 0.0) {
    return {};
  }

  int min_ix = std::numeric_limits<int>::max();
  int max_ix = std::numeric_limits<int>::min();
  int min_iy = std::numeric_limits<int>::max();
  int max_iy = std::numeric_limits<int>::min();
  std::vector<std::pair<int, int>> support_cells;
  support_cells.reserve(support_points.size());
  for (const auto& point : support_points) {
    const int ix = static_cast<int>(std::llround(point.x() / resolution_m));
    const int iy = static_cast<int>(std::llround(point.y() / resolution_m));
    support_cells.emplace_back(ix, iy);
    min_ix = std::min(min_ix, ix);
    max_ix = std::max(max_ix, ix);
    min_iy = std::min(min_iy, iy);
    max_iy = std::max(max_iy, iy);
  }
  const std::size_t nx = static_cast<std::size_t>(max_ix - min_ix + 1);
  const std::size_t ny = static_cast<std::size_t>(max_iy - min_iy + 1);
  if (nx < 2 || ny < 2 || nx * ny > 2000000) {
    return {};
  }

  const int unset_min = std::numeric_limits<int>::max();
  const int unset_max = std::numeric_limits<int>::min();
  std::vector<int> row_min(ny, unset_min);
  std::vector<int> row_max(ny, unset_max);
  std::vector<int> col_min(nx, unset_min);
  std::vector<int> col_max(nx, unset_max);
  for (const auto& [ix_abs, iy_abs] : support_cells) {
    const int ix = ix_abs - min_ix;
    const int iy = iy_abs - min_iy;
    row_min[iy] = std::min(row_min[iy], ix);
    row_max[iy] = std::max(row_max[iy], ix);
    col_min[ix] = std::min(col_min[ix], iy);
    col_max[ix] = std::max(col_max[ix], iy);
  }

  std::vector<khronos::Point> reference_points;
  auto search = makeInjectionSearch(global_mesh, &reference_points);
  if (!search) {
    return {};
  }
  const double threshold_sq = resolution_m * resolution_m;
  std::vector<std::size_t> grid_to_vertex(nx * ny, std::numeric_limits<std::size_t>::max());
  std::vector<khronos::Point> points;
  points.reserve(nx * ny / 4);
  for (std::size_t ix = 0; ix < nx; ++ix) {
    if (col_min[ix] == unset_min) {
      continue;
    }
    for (std::size_t iy = 0; iy < ny; ++iy) {
      if (row_min[iy] == unset_min) {
        continue;
      }
      if (static_cast<int>(ix) < row_min[iy] || static_cast<int>(ix) > row_max[iy] ||
          static_cast<int>(iy) < col_min[ix] || static_cast<int>(iy) > col_max[ix]) {
        continue;
      }

      const double x = static_cast<double>(min_ix + static_cast<int>(ix)) * resolution_m;
      const double y = static_cast<double>(min_iy + static_cast<int>(iy)) * resolution_m;
      const khronos::Point point(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
      float distance_sq = std::numeric_limits<float>::max();
      std::size_t nearest_idx = 0;
      if (search->search(point, distance_sq, nearest_idx) && distance_sq <= threshold_sq) {
        continue;
      }
      grid_to_vertex[ix * ny + iy] = points.size();
      points.push_back(point);
    }
  }

  if (points.empty()) {
    return {};
  }

  std::vector<spark_dsg::Mesh::Face> faces;
  faces.reserve(points.size() * 2);
  for (std::size_t ix = 0; ix + 1 < nx; ++ix) {
    for (std::size_t iy = 0; iy + 1 < ny; ++iy) {
      const auto a = grid_to_vertex[ix * ny + iy];
      const auto b = grid_to_vertex[(ix + 1) * ny + iy];
      const auto c = grid_to_vertex[ix * ny + iy + 1];
      const auto d = grid_to_vertex[(ix + 1) * ny + iy + 1];
      if (a == std::numeric_limits<std::size_t>::max() ||
          b == std::numeric_limits<std::size_t>::max() ||
          c == std::numeric_limits<std::size_t>::max() ||
          d == std::numeric_limits<std::size_t>::max()) {
        continue;
      }
      faces.push_back({a, b, c});
      faces.push_back({b, d, c});
    }
  }

  spark_dsg::Mesh plane(global_mesh.has_colors,
                        global_mesh.has_timestamps,
                        global_mesh.has_labels,
                        global_mesh.has_first_seen_stamps);
  plane.resizeVertices(points.size());
  for (std::size_t i = 0; i < points.size(); ++i) {
    plane.setPos(i, points[i]);
    if (plane.has_colors) {
      plane.setColor(i, spark_dsg::Color(150, 150, 150));
    }
    if (plane.has_timestamps) {
      plane.setTimestamp(i, 0);
    }
    if (plane.has_first_seen_stamps) {
      plane.setFirstSeenTimestamp(i, 0);
    }
    if (plane.has_labels) {
      plane.setLabel(i, static_cast<spark_dsg::Mesh::Label>(-1));
    }
  }
  plane.resizeFaces(faces.size());
  for (std::size_t i = 0; i < faces.size(); ++i) {
    plane.face(i) = faces[i];
  }
  if (!global_mesh.append(plane)) {
    throw std::runtime_error("Failed to append footprint-guarded horizontal plane fill.");
  }
  return {1, points.size(), faces.size()};
}

double axisCoordinate(const khronos::Point& point, int axis) {
  if (axis == 0) {
    return static_cast<double>(point.x());
  }
  if (axis == 1) {
    return static_cast<double>(point.y());
  }
  return static_cast<double>(point.z());
}

double planeU(const khronos::Point& point, int fixed_axis) {
  if (fixed_axis == 0) {
    return static_cast<double>(point.y());
  }
  return static_cast<double>(point.x());
}

double planeV(const khronos::Point& point, int fixed_axis) {
  if (fixed_axis == 2) {
    return static_cast<double>(point.y());
  }
  return static_cast<double>(point.z());
}

khronos::Point makeAxisAlignedPoint(int fixed_axis, double fixed_coord, double u, double v) {
  if (fixed_axis == 0) {
    return khronos::Point(static_cast<float>(fixed_coord),
                          static_cast<float>(u),
                          static_cast<float>(v));
  }
  if (fixed_axis == 1) {
    return khronos::Point(static_cast<float>(u),
                          static_cast<float>(fixed_coord),
                          static_cast<float>(v));
  }
  return khronos::Point(static_cast<float>(u),
                        static_cast<float>(v),
                        static_cast<float>(fixed_coord));
}

double medianCoordinate(const std::vector<khronos::Point>& points, int axis) {
  std::vector<double> values;
  values.reserve(points.size());
  for (const auto& point : points) {
    values.push_back(axisCoordinate(point, axis));
  }
  if (values.empty()) {
    return 0.0;
  }
  const auto mid = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
  std::nth_element(values.begin(), mid, values.end());
  double median = *mid;
  if (values.size() % 2 == 0) {
    const auto lower = std::max_element(values.begin(), mid);
    median = 0.5 * (median + *lower);
  }
  return median;
}

double medianZ(const std::vector<khronos::Point>& points) {
  return medianCoordinate(points, 2);
}

PlaneFillResult fillAxisAlignedPlaneByGraphCut(
    spark_dsg::Mesh& global_mesh,
    int fixed_axis,
    double fixed_coord,
    const std::vector<khronos::Point>& support_points,
    double resolution_m,
    int max_support_distance_cells,
    const khronos::DynamicSceneGraph::Ptr& reference_dsg = nullptr,
    const std::vector<khronos::BoundingBox>* object_mask_boxes = nullptr,
    const ReconcilerConfig* config = nullptr) {
  if (support_points.empty() || resolution_m <= 0.0) {
    return {};
  }

  int min_ix = std::numeric_limits<int>::max();
  int max_ix = std::numeric_limits<int>::min();
  int min_iy = std::numeric_limits<int>::max();
  int max_iy = std::numeric_limits<int>::min();
  std::vector<std::pair<int, int>> support_cells;
  support_cells.reserve(support_points.size());
  for (const auto& point : support_points) {
    const int ix = static_cast<int>(std::llround(planeU(point, fixed_axis) / resolution_m));
    const int iy = static_cast<int>(std::llround(planeV(point, fixed_axis) / resolution_m));
    support_cells.emplace_back(ix, iy);
    min_ix = std::min(min_ix, ix);
    max_ix = std::max(max_ix, ix);
    min_iy = std::min(min_iy, iy);
    max_iy = std::max(max_iy, iy);
  }
  const std::size_t nx = static_cast<std::size_t>(max_ix - min_ix + 1);
  const std::size_t ny = static_cast<std::size_t>(max_iy - min_iy + 1);
  if (nx < 2 || ny < 2 || nx * ny > 2000000) {
    return {};
  }

  const int unset_min = std::numeric_limits<int>::max();
  const int unset_max = std::numeric_limits<int>::min();
  std::vector<int> row_min(ny, unset_min);
  std::vector<int> row_max(ny, unset_max);
  std::vector<int> col_min(nx, unset_min);
  std::vector<int> col_max(nx, unset_max);
  std::vector<int> support_count(nx * ny, 0);
  for (const auto& [ix_abs, iy_abs] : support_cells) {
    const int ix = ix_abs - min_ix;
    const int iy = iy_abs - min_iy;
    const auto idx = static_cast<std::size_t>(ix) * ny + static_cast<std::size_t>(iy);
    ++support_count[idx];
    row_min[iy] = std::min(row_min[iy], ix);
    row_max[iy] = std::max(row_max[iy], ix);
    col_min[ix] = std::min(col_min[ix], iy);
    col_max[ix] = std::max(col_max[ix], iy);
  }

  std::vector<unsigned char> in_domain(nx * ny, 0);
  std::size_t domain_cells = 0;
  for (std::size_t ix = 0; ix < nx; ++ix) {
    if (col_min[ix] == unset_min) {
      continue;
    }
    for (std::size_t iy = 0; iy < ny; ++iy) {
      if (row_min[iy] == unset_min) {
        continue;
      }
      if (static_cast<int>(ix) < row_min[iy] || static_cast<int>(ix) > row_max[iy] ||
          static_cast<int>(iy) < col_min[ix] || static_cast<int>(iy) > col_max[ix]) {
        continue;
      }
      in_domain[ix * ny + iy] = 1;
      ++domain_cells;
    }
  }
  if (domain_cells == 0) {
    return {};
  }

  const bool use_visibility =
      config && config->structural_plane_visibility_filter && reference_dsg &&
      (config->structural_plane_visibility_scope != "vertical" || fixed_axis != 2);
  std::vector<unsigned char> ray_absent(nx * ny, 0);
  std::vector<unsigned char> ray_present(nx * ny, 0);
  std::vector<unsigned char> inside_object_mask(nx * ny, 0);
  if (use_visibility) {
    khronos::RayVerificator::Config ray_config;
    ray_config.block_size = static_cast<float>(config->free_space_culling_block_size_m);
    ray_config.radial_tolerance =
        static_cast<float>(config->free_space_culling_radial_tolerance_m);
    ray_config.depth_tolerance =
        static_cast<float>(config->free_space_culling_depth_tolerance_m);
    ray_config.active_window_duration =
        static_cast<float>(config->free_space_culling_active_window_duration_s);
    ray_config.ray_policy = khronos::RayVerificator::Config::RayPolicy::kMiddle;
    khronos::RayVerificator verifier(ray_config);
    verifier.setDsg(reference_dsg);

    for (std::size_t ix = 0; ix < nx; ++ix) {
      for (std::size_t iy = 0; iy < ny; ++iy) {
        const auto idx = ix * ny + iy;
        if (!in_domain[idx]) {
          continue;
        }
        const double u = static_cast<double>(min_ix + static_cast<int>(ix)) * resolution_m;
        const double v = static_cast<double>(min_iy + static_cast<int>(iy)) * resolution_m;
        const auto point = makeAxisAlignedPoint(fixed_axis, fixed_coord, u, v);
        const auto check = verifier.check(point);
        ray_absent[idx] = static_cast<unsigned char>(
            std::min<std::size_t>(3, check.absent.size()));
        ray_present[idx] = static_cast<unsigned char>(
            std::min<std::size_t>(3, check.present.size()));
        if (object_mask_boxes) {
          inside_object_mask[idx] = std::any_of(
              object_mask_boxes->begin(),
              object_mask_boxes->end(),
              [&](const auto& box) {
                return pointInsideExpandedBox(box, point, config->bbox_margin_m);
              });
        }
      }
    }
  }

  const int unreachable = std::numeric_limits<int>::max();
  std::vector<int> support_distance(nx * ny, unreachable);
  std::queue<std::size_t> queue;
  for (std::size_t idx = 0; idx < support_count.size(); ++idx) {
    if (!in_domain[idx] || support_count[idx] == 0) {
      continue;
    }
    support_distance[idx] = 0;
    queue.push(idx);
  }
  const auto tryVisit = [&](std::size_t next, int distance, std::queue<std::size_t>* q) {
    if (!in_domain[next] || support_distance[next] != unreachable || !q) {
      return;
    }
    support_distance[next] = distance;
    q->push(next);
  };
  while (!queue.empty()) {
    const auto idx = queue.front();
    queue.pop();
    const std::size_t ix = idx / ny;
    const std::size_t iy = idx % ny;
    const int next_distance = support_distance[idx] + 1;
    if (ix > 0) {
      tryVisit((ix - 1) * ny + iy, next_distance, &queue);
    }
    if (ix + 1 < nx) {
      tryVisit((ix + 1) * ny + iy, next_distance, &queue);
    }
    if (iy > 0) {
      tryVisit(ix * ny + (iy - 1), next_distance, &queue);
    }
    if (iy + 1 < ny) {
      tryVisit(ix * ny + (iy + 1), next_distance, &queue);
    }
  }

  std::vector<int> cell_to_node(nx * ny, -1);
  int node_count = 0;
  for (std::size_t idx = 0; idx < in_domain.size(); ++idx) {
    if (!in_domain[idx]) {
      continue;
    }
    cell_to_node[idx] = node_count++;
  }
  const int source = node_count;
  const int sink = node_count + 1;
  DinicMaxFlow graph(static_cast<std::size_t>(node_count + 2));
  constexpr double kHardCost = 1.0e6;
  constexpr double kPairwiseCost = 0.18;

  for (std::size_t ix = 0; ix < nx; ++ix) {
    for (std::size_t iy = 0; iy < ny; ++iy) {
      const auto idx = ix * ny + iy;
      const int node = cell_to_node[idx];
      if (node < 0) {
        continue;
      }

      double fill_cost = 0.0;
      double empty_cost = 0.0;
      if (support_count[idx] > 0) {
        fill_cost = 0.0;
        empty_cost = kHardCost;
      } else if (max_support_distance_cells >= 0 &&
                 support_distance[idx] > max_support_distance_cells) {
        fill_cost = kHardCost;
        empty_cost = 0.0;
      } else {
        const int distance = support_distance[idx] == unreachable ? 16 : support_distance[idx];
        const int row_margin =
            std::min(static_cast<int>(ix) - row_min[iy], row_max[iy] - static_cast<int>(ix));
        const int col_margin =
            std::min(static_cast<int>(iy) - col_min[ix], col_max[ix] - static_cast<int>(iy));
        const int boundary_margin = std::min(row_margin, col_margin);
        double boundary_penalty = 0.0;
        if (boundary_margin <= 0) {
          boundary_penalty = 0.35;
        } else if (boundary_margin == 1) {
          boundary_penalty = 0.16;
        }

        fill_cost = 0.22 + 0.045 * static_cast<double>(std::min(distance, 12)) +
                    boundary_penalty;
        empty_cost = 0.82;
      }

      if (use_visibility && support_count[idx] == 0) {
        fill_cost += 0.75 * static_cast<double>(ray_absent[idx]);
        empty_cost += 0.45 * static_cast<double>(ray_present[idx]);
        if (inside_object_mask[idx]) {
          fill_cost += 0.20;
        }
      }

      // Source side is FILL. A cut pays node->sink for FILL and source->node for EMPTY.
      graph.addEdge(source, node, empty_cost);
      graph.addEdge(node, sink, fill_cost);

      const auto addPairwise = [&](std::size_t other_idx) {
        const int other_node = cell_to_node[other_idx];
        if (other_node < 0) {
          return;
        }
        graph.addEdge(node, other_node, kPairwiseCost);
        graph.addEdge(other_node, node, kPairwiseCost);
      };
      if (ix + 1 < nx) {
        addPairwise((ix + 1) * ny + iy);
      }
      if (iy + 1 < ny) {
        addPairwise(ix * ny + (iy + 1));
      }
    }
  }

  graph.maxFlow(source, sink);
  const auto reachable = graph.sourceReachable(source);

  std::vector<unsigned char> fill_cell(nx * ny, 0);
  std::size_t fill_cells = 0;
  for (std::size_t idx = 0; idx < cell_to_node.size(); ++idx) {
    const int node = cell_to_node[idx];
    if (node < 0 || !reachable[node]) {
      continue;
    }
    fill_cell[idx] = 1;
    ++fill_cells;
  }
  if (fill_cells == 0) {
    return {0, 0, 0, domain_cells, fill_cells};
  }

  std::vector<khronos::Point> reference_points;
  auto search = makeInjectionSearch(global_mesh, &reference_points);
  if (!search) {
    return {0, 0, 0, domain_cells, fill_cells};
  }
  const double novelty_radius_m = std::min(0.04, resolution_m * 0.5);
  const double threshold_sq = novelty_radius_m * novelty_radius_m;
  std::vector<std::size_t> grid_to_vertex(nx * ny, std::numeric_limits<std::size_t>::max());
  std::vector<khronos::Point> points;
  const bool supersample_output =
      config != nullptr && config->structural_plane_output_supersample &&
      resolution_m > 0.0;
  points.reserve(fill_cells * (supersample_output ? 5 : 1));
  const auto tryAppendNovelPoint = [&](const khronos::Point& point) -> std::size_t {
    float distance_sq = std::numeric_limits<float>::max();
    std::size_t nearest_idx = 0;
    if (search->search(point, distance_sq, nearest_idx) && distance_sq <= threshold_sq) {
      return std::numeric_limits<std::size_t>::max();
    }
    const auto point_idx = points.size();
    points.push_back(point);
    return point_idx;
  };
  for (std::size_t ix = 0; ix < nx; ++ix) {
    for (std::size_t iy = 0; iy < ny; ++iy) {
      const auto idx = ix * ny + iy;
      if (!fill_cell[idx]) {
        continue;
      }
      const double u = static_cast<double>(min_ix + static_cast<int>(ix)) * resolution_m;
      const double v = static_cast<double>(min_iy + static_cast<int>(iy)) * resolution_m;
      const auto point = makeAxisAlignedPoint(fixed_axis, fixed_coord, u, v);
      grid_to_vertex[idx] = tryAppendNovelPoint(point);
      const bool observed_supersample_cell =
          support_count[idx] > 0 || ray_present[idx] > 0;
      const bool support_supersample_cell = support_count[idx] > 0;
      const bool allow_supersample =
          supersample_output &&
          (config->structural_plane_output_supersample_scope == "all" ||
           (config->structural_plane_output_supersample_scope == "observed" &&
            observed_supersample_cell) ||
           (config->structural_plane_output_supersample_scope == "support" &&
            support_supersample_cell));
      if (allow_supersample) {
        const double offset = resolution_m * 0.25;
        for (const double du : {-offset, offset}) {
          for (const double dv : {-offset, offset}) {
            (void)tryAppendNovelPoint(
                makeAxisAlignedPoint(fixed_axis, fixed_coord, u + du, v + dv));
          }
        }
      }
    }
  }

  if (points.empty()) {
    return {0, 0, 0, domain_cells, fill_cells};
  }

  std::vector<spark_dsg::Mesh::Face> faces;
  faces.reserve(points.size() * 2);
  for (std::size_t ix = 0; ix + 1 < nx; ++ix) {
    for (std::size_t iy = 0; iy + 1 < ny; ++iy) {
      const auto a = grid_to_vertex[ix * ny + iy];
      const auto b = grid_to_vertex[(ix + 1) * ny + iy];
      const auto c = grid_to_vertex[ix * ny + iy + 1];
      const auto d = grid_to_vertex[(ix + 1) * ny + iy + 1];
      if (a == std::numeric_limits<std::size_t>::max() ||
          b == std::numeric_limits<std::size_t>::max() ||
          c == std::numeric_limits<std::size_t>::max() ||
          d == std::numeric_limits<std::size_t>::max()) {
        continue;
      }
      faces.push_back({a, b, c});
      faces.push_back({b, d, c});
    }
  }

  spark_dsg::Mesh plane(global_mesh.has_colors,
                        global_mesh.has_timestamps,
                        global_mesh.has_labels,
                        global_mesh.has_first_seen_stamps);
  plane.resizeVertices(points.size());
  for (std::size_t i = 0; i < points.size(); ++i) {
    plane.setPos(i, points[i]);
    if (plane.has_colors) {
      plane.setColor(i, spark_dsg::Color(150, 150, 150));
    }
    if (plane.has_timestamps) {
      plane.setTimestamp(i, 0);
    }
    if (plane.has_first_seen_stamps) {
      plane.setFirstSeenTimestamp(i, 0);
    }
    if (plane.has_labels) {
      plane.setLabel(i, static_cast<spark_dsg::Mesh::Label>(-1));
    }
  }
  plane.resizeFaces(faces.size());
  for (std::size_t i = 0; i < faces.size(); ++i) {
    plane.face(i) = faces[i];
  }
  if (!global_mesh.append(plane)) {
    throw std::runtime_error("Failed to append graph-cut axis-aligned plane fill.");
  }
  return {1, points.size(), faces.size(), domain_cells, fill_cells};
}

PlaneFillResult fillHorizontalPlaneByGraphCut(
    spark_dsg::Mesh& global_mesh,
    double z,
    const std::vector<khronos::Point>& support_points,
    double resolution_m,
    int max_support_distance_cells,
    const khronos::DynamicSceneGraph::Ptr& reference_dsg = nullptr,
    const std::vector<khronos::BoundingBox>* object_mask_boxes = nullptr,
    const ReconcilerConfig* config = nullptr) {
  return fillAxisAlignedPlaneByGraphCut(global_mesh,
                                        2,
                                        z,
                                        support_points,
                                        resolution_m,
                                        max_support_distance_cells,
                                        reference_dsg,
                                        object_mask_boxes,
                                        config);
}

std::vector<int> selectDominantHorizontalPlaneKeys(const spark_dsg::Mesh& mesh,
                                                   double resolution_m,
                                                   std::size_t min_support_vertices) {
  double min_z = std::numeric_limits<double>::infinity();
  double max_z = -std::numeric_limits<double>::infinity();
  std::unordered_map<int, std::size_t> z_bins;
  for (std::size_t i = 0; i < mesh.numVertices(); ++i) {
    const auto point = mesh.pos(i);
    min_z = std::min(min_z, static_cast<double>(point.z()));
    max_z = std::max(max_z, static_cast<double>(point.z()));
    const int key = static_cast<int>(std::llround(point.z() / resolution_m));
    z_bins[key]++;
  }
  if (z_bins.empty()) {
    return {};
  }

  const double mid_z = 0.5 * (min_z + max_z);
  auto chooseBin = [&](bool upper) -> int {
    int best_key = 0;
    std::size_t best_count = 0;
    for (const auto& [key, count] : z_bins) {
      const double z = key * resolution_m;
      if (count < min_support_vertices) {
        continue;
      }
      if (upper && z <= mid_z) {
        continue;
      }
      if (!upper && z >= mid_z) {
        continue;
      }
      if (count > best_count) {
        best_count = count;
        best_key = key;
      }
    }
    return best_count == 0 ? std::numeric_limits<int>::max() : best_key;
  };

  std::vector<int> keys;
  const auto addKey = [&](int key) {
    if (key == std::numeric_limits<int>::max()) {
      return;
    }
    if (std::find(keys.begin(), keys.end(), key) == keys.end()) {
      keys.push_back(key);
    }
  };

  const int lower = chooseBin(false);
  const int upper = chooseBin(true);
  addKey(lower);
  addKey(lower - 1);
  addKey(upper);
  addKey(upper - 1);
  return keys;
}

PlaneFillResult fillAxisAlignedBoundaryPlanes(spark_dsg::Mesh& mesh,
                                              double resolution_m,
                                              std::size_t min_support_vertices,
                                              const std::string& fill_mode,
                                              int max_support_distance_cells,
                                              const khronos::DynamicSceneGraph::Ptr& reference_dsg = nullptr,
                                              const std::vector<khronos::BoundingBox>* object_mask_boxes = nullptr,
                                              const ReconcilerConfig* config = nullptr) {
  if (mesh.numVertices() == 0 || resolution_m <= 0.0 || fill_mode != "graph_cut") {
    return {};
  }

  struct AxisStats {
    std::size_t support = 0;
    double min_z = std::numeric_limits<double>::infinity();
    double max_z = -std::numeric_limits<double>::infinity();
    double min_u = std::numeric_limits<double>::infinity();
    double max_u = -std::numeric_limits<double>::infinity();
  };
  struct SelectedPlane {
    int axis = 0;
    double fixed_coord = 0.0;
    std::vector<khronos::Point> support_points;
  };

  const std::size_t source_vertex_count = mesh.numVertices();
  const std::size_t vertical_support_threshold =
      std::max<std::size_t>(1000, min_support_vertices / 2);
  const double tolerance = resolution_m * 0.5;
  const double min_wall_height_m = 1.4;
  const double min_wall_length_m = 1.0;
  const auto horizontal_keys =
      selectDominantHorizontalPlaneKeys(mesh, resolution_m, min_support_vertices);

  const auto nearDominantHorizontal = [&](const khronos::Point& point) {
    for (const int key : horizontal_keys) {
      const double z = static_cast<double>(key) * resolution_m;
      if (std::abs(static_cast<double>(point.z()) - z) <= resolution_m * 0.75) {
        return true;
      }
    }
    return false;
  };

  std::vector<SelectedPlane> selected_planes;
  for (int axis = 0; axis < 2; ++axis) {
    std::unordered_map<int, AxisStats> bins;
    for (std::size_t i = 0; i < source_vertex_count; ++i) {
      const auto point = mesh.pos(i);
      if (nearDominantHorizontal(point)) {
        continue;
      }
      const int key = static_cast<int>(std::llround(axisCoordinate(point, axis) / resolution_m));
      auto& stats = bins[key];
      ++stats.support;
      const double z = static_cast<double>(point.z());
      const double u = planeU(point, axis);
      stats.min_z = std::min(stats.min_z, z);
      stats.max_z = std::max(stats.max_z, z);
      stats.min_u = std::min(stats.min_u, u);
      stats.max_u = std::max(stats.max_u, u);
    }

    std::vector<std::pair<int, AxisStats>> candidates;
    candidates.reserve(bins.size());
    for (const auto& [key, stats] : bins) {
      if (stats.support < vertical_support_threshold) {
        continue;
      }
      if (stats.max_z - stats.min_z < min_wall_height_m) {
        continue;
      }
      if (stats.max_u - stats.min_u < min_wall_length_m) {
        continue;
      }
      candidates.emplace_back(key, stats);
    }
    if (candidates.empty()) {
      continue;
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.first < rhs.first;
    });
    std::vector<int> selected_keys;
    const auto addBoundaryKey = [&](int key) {
      const bool nearby = std::any_of(
          selected_keys.begin(), selected_keys.end(), [key](int existing) {
            return std::abs(existing - key) <= 2;
          });
      if (!nearby) {
        selected_keys.push_back(key);
      }
    };
    if (config != nullptr &&
        config->axis_aligned_plane_candidate_policy == "strong_all") {
      for (const auto& [key, stats] : candidates) {
        (void)stats;
        addBoundaryKey(key);
      }
    } else {
      addBoundaryKey(candidates.front().first);
      addBoundaryKey(candidates.back().first);
    }

    for (const int key : selected_keys) {
      const double fixed_coord = static_cast<double>(key) * resolution_m;
      std::vector<khronos::Point> support_points;
      support_points.reserve(vertical_support_threshold * 2);
      double min_z = std::numeric_limits<double>::infinity();
      double max_z = -std::numeric_limits<double>::infinity();
      double min_u = std::numeric_limits<double>::infinity();
      double max_u = -std::numeric_limits<double>::infinity();
      for (std::size_t i = 0; i < source_vertex_count; ++i) {
        const auto point = mesh.pos(i);
        if (nearDominantHorizontal(point)) {
          continue;
        }
        if (std::abs(axisCoordinate(point, axis) - fixed_coord) > tolerance) {
          continue;
        }
        support_points.push_back(point);
        const double z = static_cast<double>(point.z());
        const double u = planeU(point, axis);
        min_z = std::min(min_z, z);
        max_z = std::max(max_z, z);
        min_u = std::min(min_u, u);
        max_u = std::max(max_u, u);
      }
      if (support_points.size() < vertical_support_threshold ||
          max_z - min_z < min_wall_height_m || max_u - min_u < min_wall_length_m) {
        continue;
      }
      selected_planes.push_back(
          SelectedPlane{axis, medianCoordinate(support_points, axis), std::move(support_points)});
    }
  }

  PlaneFillResult total;
  for (const auto& plane : selected_planes) {
    const auto filled = fillAxisAlignedPlaneByGraphCut(
        mesh,
        plane.axis,
        plane.fixed_coord,
        plane.support_points,
        resolution_m,
        max_support_distance_cells,
        reference_dsg,
        object_mask_boxes,
        config);
    total.planes += filled.planes;
    total.vertices += filled.vertices;
    total.faces += filled.faces;
    total.graph_cut_cells += filled.graph_cut_cells;
    total.graph_cut_fill_cells += filled.graph_cut_fill_cells;
  }
  return total;
}

PlaneFillResult fillHorizontalPlanes(spark_dsg::Mesh& mesh,
                                     double resolution_m,
                                     std::size_t min_support_vertices,
                                     const std::string& fill_mode,
                                     const khronos::DynamicSceneGraph::Ptr& reference_dsg = nullptr,
                                     const std::vector<khronos::BoundingBox>* object_mask_boxes = nullptr,
                                     const ReconcilerConfig* config = nullptr) {
  if (mesh.numVertices() == 0 || resolution_m <= 0.0) {
    return {};
  }

  double min_z = std::numeric_limits<double>::infinity();
  double max_z = -std::numeric_limits<double>::infinity();
  std::unordered_map<int, std::size_t> z_bins;
  for (std::size_t i = 0; i < mesh.numVertices(); ++i) {
    const auto point = mesh.pos(i);
    min_z = std::min(min_z, static_cast<double>(point.z()));
    max_z = std::max(max_z, static_cast<double>(point.z()));
    const int key = static_cast<int>(std::llround(point.z() / resolution_m));
    z_bins[key]++;
  }
  const double mid_z = 0.5 * (min_z + max_z);

  auto chooseBin = [&](bool upper) -> int {
    int best_key = 0;
    std::size_t best_count = 0;
    for (const auto& [key, count] : z_bins) {
      const double z = key * resolution_m;
      if (count < min_support_vertices) {
        continue;
      }
      if (upper && z <= mid_z) {
        continue;
      }
      if (!upper && z >= mid_z) {
        continue;
      }
      if (count > best_count) {
        best_count = count;
        best_key = key;
      }
    }
    return best_count == 0 ? std::numeric_limits<int>::max() : best_key;
  };

  std::vector<std::pair<int, std::size_t>> plane_bins;
  const auto addPlaneBin = [&](int key, std::size_t support_threshold) {
    if (key == std::numeric_limits<int>::max()) {
      return;
    }
    const auto count_it = z_bins.find(key);
    if (count_it == z_bins.end() || count_it->second < support_threshold) {
      return;
    }
    const auto existing = std::find_if(
        plane_bins.begin(), plane_bins.end(), [key](const auto& item) {
          return item.first == key;
        });
    if (existing != plane_bins.end()) {
      existing->second = std::min(existing->second, support_threshold);
      return;
    }
    plane_bins.emplace_back(key, support_threshold);
  };

  const int lower = chooseBin(false);
  const int upper = chooseBin(true);
  const std::string horizontal_candidate_policy =
      config == nullptr ? "extreme_pair" : config->horizontal_plane_candidate_policy;
  if (horizontal_candidate_policy == "dominant_all") {
    std::vector<std::pair<int, std::size_t>> supported_bins;
    supported_bins.reserve(z_bins.size());
    for (const auto& [key, count] : z_bins) {
      if (count >= min_support_vertices) {
        supported_bins.emplace_back(key, count);
      }
    }
    std::sort(supported_bins.begin(), supported_bins.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.second > rhs.second;
    });
    std::vector<int> selected_keys;
    selected_keys.reserve(supported_bins.size());
    for (const auto& [key, count] : supported_bins) {
      (void)count;
      const bool near_existing = std::any_of(
          selected_keys.begin(), selected_keys.end(), [key](int existing) {
            return std::abs(existing - key) <= 2;
          });
      if (near_existing) {
        continue;
      }
      selected_keys.push_back(key);
      addPlaneBin(key, min_support_vertices);
    }
  } else {
    addPlaneBin(lower, min_support_vertices);
    addPlaneBin(upper, min_support_vertices);
    if (horizontal_candidate_policy == "upper_band" && upper != std::numeric_limits<int>::max()) {
      constexpr double kUpperBandMeters = 0.40;
      const int upper_band_cells =
          std::max<int>(2, static_cast<int>(std::ceil(kUpperBandMeters / resolution_m)));
      for (int key = upper - upper_band_cells; key <= upper - 2; ++key) {
        addPlaneBin(key, min_support_vertices);
      }
    }
  }
  if (fill_mode == "graph_cut" &&
      horizontal_candidate_policy != "dominant_all") {
    const std::size_t secondary_support =
        std::max<std::size_t>(1000, min_support_vertices / 2);
    addPlaneBin(lower - 1, secondary_support);
    addPlaneBin(upper - 1, secondary_support);
  }
  std::sort(plane_bins.begin(), plane_bins.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.first < rhs.first;
  });

  PlaneFillResult total;
  const double tolerance = resolution_m * 0.5;
  for (const auto& [key, support_threshold] : plane_bins) {
    const double z = key * resolution_m;
    std::vector<khronos::Point> support_points;
    support_points.reserve(z_bins[key]);
    for (std::size_t i = 0; i < mesh.numVertices(); ++i) {
      const auto point = mesh.pos(i);
      if (std::abs(static_cast<double>(point.z()) - z) > tolerance) {
        continue;
      }
      support_points.push_back(point);
    }
    if (support_points.size() < support_threshold) {
      continue;
    }
    const double plane_z = fill_mode == "graph_cut" ? medianZ(support_points) : z;
    const auto filled =
        fill_mode == "graph_cut"
            ? fillHorizontalPlaneByGraphCut(mesh,
                                            plane_z,
                                            support_points,
                                            resolution_m,
                                            config == nullptr
                                                ? -1
                                                : config->horizontal_plane_support_band_cells,
                                            reference_dsg,
                                            object_mask_boxes,
                                            config)
            : fillHorizontalPlaneFromSupport(mesh, plane_z, support_points, resolution_m);
    total.planes += filled.planes;
    total.vertices += filled.vertices;
    total.faces += filled.faces;
    total.graph_cut_cells += filled.graph_cut_cells;
    total.graph_cut_fill_cells += filled.graph_cut_fill_cells;
  }

  return total;
}

struct VolumeStructuralPlane {
  int axis = 0;
  double coord = 0.0;
};

std::vector<VolumeStructuralPlane> selectVolumeStructuralPlanes(
    const spark_dsg::Mesh& mesh,
    double resolution_m,
    std::size_t min_support_vertices) {
  std::vector<VolumeStructuralPlane> planes;
  if (mesh.numVertices() == 0 || resolution_m <= 0.0) {
    return planes;
  }

  const auto horizontal_keys =
      selectDominantHorizontalPlaneKeys(mesh, resolution_m, min_support_vertices);
  for (const int key : horizontal_keys) {
    if (key == std::numeric_limits<int>::max()) {
      continue;
    }
    const double z = static_cast<double>(key) * resolution_m;
    const bool exists = std::any_of(planes.begin(), planes.end(), [z](const auto& plane) {
      return plane.axis == 2 && std::abs(plane.coord - z) < 0.02;
    });
    if (!exists) {
      planes.push_back({2, z});
    }
  }

  const auto nearDominantHorizontal = [&](const khronos::Point& point) {
    for (const int key : horizontal_keys) {
      if (key == std::numeric_limits<int>::max()) {
        continue;
      }
      const double z = static_cast<double>(key) * resolution_m;
      if (std::abs(static_cast<double>(point.z()) - z) <= resolution_m * 0.75) {
        return true;
      }
    }
    return false;
  };

  struct AxisStats {
    std::size_t support = 0;
    double min_z = std::numeric_limits<double>::infinity();
    double max_z = -std::numeric_limits<double>::infinity();
    double min_u = std::numeric_limits<double>::infinity();
    double max_u = -std::numeric_limits<double>::infinity();
  };

  const std::size_t vertical_support_threshold =
      std::max<std::size_t>(1000, min_support_vertices / 2);
  const double min_wall_height_m = 1.4;
  const double min_wall_length_m = 1.0;
  for (int axis = 0; axis < 2; ++axis) {
    std::unordered_map<int, AxisStats> bins;
    for (std::size_t i = 0; i < mesh.numVertices(); ++i) {
      const auto point = mesh.pos(i);
      if (nearDominantHorizontal(point)) {
        continue;
      }
      const int key = static_cast<int>(std::llround(axisCoordinate(point, axis) / resolution_m));
      auto& stats = bins[key];
      ++stats.support;
      const double z = static_cast<double>(point.z());
      const double u = planeU(point, axis);
      stats.min_z = std::min(stats.min_z, z);
      stats.max_z = std::max(stats.max_z, z);
      stats.min_u = std::min(stats.min_u, u);
      stats.max_u = std::max(stats.max_u, u);
    }

    std::vector<std::pair<int, AxisStats>> candidates;
    for (const auto& [key, stats] : bins) {
      if (stats.support < vertical_support_threshold) {
        continue;
      }
      if (stats.max_z - stats.min_z < min_wall_height_m ||
          stats.max_u - stats.min_u < min_wall_length_m) {
        continue;
      }
      candidates.emplace_back(key, stats);
    }
    if (candidates.empty()) {
      continue;
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.first < rhs.first;
    });

    const std::array<int, 2> boundary_keys{candidates.front().first, candidates.back().first};
    for (const int key : boundary_keys) {
      const double coord = static_cast<double>(key) * resolution_m;
      std::vector<khronos::Point> support_points;
      support_points.reserve(vertical_support_threshold * 2);
      for (std::size_t i = 0; i < mesh.numVertices(); ++i) {
        const auto point = mesh.pos(i);
        if (nearDominantHorizontal(point)) {
          continue;
        }
        if (std::abs(axisCoordinate(point, axis) - coord) <= resolution_m * 0.5) {
          support_points.push_back(point);
        }
      }
      if (support_points.size() < vertical_support_threshold) {
        continue;
      }
      const double median = medianCoordinate(support_points, axis);
      const bool exists =
          std::any_of(planes.begin(), planes.end(), [axis, median](const auto& plane) {
            return plane.axis == axis && std::abs(plane.coord - median) < 0.02;
          });
      if (!exists) {
        planes.push_back({axis, median});
      }
    }
  }
  return planes;
}

bool pointInsideAnyBox(const std::vector<khronos::BoundingBox>& boxes,
                       const khronos::Point& point,
                       double margin_m) {
  return std::any_of(boxes.begin(), boxes.end(), [&](const auto& box) {
    return pointInsideExpandedBox(box, point, margin_m);
  });
}

bool usesStructuralVolumeBoundaries(const std::string& surface_policy) {
  return surface_policy == "structural_boundaries" ||
         surface_policy == "structural_plane_snap_boundaries" ||
         surface_policy == "structural_marching_cubes" ||
         surface_policy == "structural_binary_marching_cubes";
}

bool usesMarchingCubesVolumeSurface(const std::string& surface_policy) {
  return surface_policy == "marching_cubes" ||
         surface_policy == "structural_marching_cubes" ||
         surface_policy == "structural_binary_marching_cubes";
}

bool usesPlanePulledMarchingCubesSurface(const std::string& surface_policy) {
  return surface_policy == "marching_cubes" ||
         surface_policy == "structural_marching_cubes";
}

bool usesStructuralPlaneSnapSurface(const std::string& surface_policy) {
  return surface_policy == "structural_plane_snap_boundaries";
}

VolumeGraphCutResult fillVolumeByGraphCut(
    spark_dsg::Mesh& mesh,
    const khronos::DynamicSceneGraph::Ptr& reference_dsg,
    const std::vector<khronos::BoundingBox>& object_mask_boxes,
    const std::vector<khronos::BoundingBox>& absent_object_boxes,
    const ReconcilerConfig& config) {
  VolumeGraphCutResult result;
  if (!config.volume_graph_cut_fill || !reference_dsg || mesh.numVertices() == 0 ||
      config.volume_graph_cut_resolution_m <= 0.0) {
    return result;
  }

  const double resolution = config.volume_graph_cut_resolution_m;
  khronos::Point min_corner(std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::infinity());
  khronos::Point max_corner(-std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity());
  for (std::size_t i = 0; i < mesh.numVertices(); ++i) {
    const auto point = mesh.pos(i);
    min_corner.x() = std::min(min_corner.x(), point.x());
    min_corner.y() = std::min(min_corner.y(), point.y());
    min_corner.z() = std::min(min_corner.z(), point.z());
    max_corner.x() = std::max(max_corner.x(), point.x());
    max_corner.y() = std::max(max_corner.y(), point.y());
    max_corner.z() = std::max(max_corner.z(), point.z());
  }

  const float padding = static_cast<float>(std::max(0.20, 2.0 * resolution));
  min_corner -= khronos::Point(padding, padding, padding);
  max_corner += khronos::Point(padding, padding, padding);

  const auto dimFor = [&](float min_value, float max_value) {
    return static_cast<std::size_t>(
               std::floor((static_cast<double>(max_value) - static_cast<double>(min_value)) /
                          resolution)) +
           1;
  };
  const std::size_t nx = dimFor(min_corner.x(), max_corner.x());
  const std::size_t ny = dimFor(min_corner.y(), max_corner.y());
  const std::size_t nz = dimFor(min_corner.z(), max_corner.z());
  const std::size_t total_cells = nx * ny * nz;
  result.cells = total_cells;
  if (nx < 2 || ny < 2 || nz < 2 || total_cells == 0 ||
      total_cells > config.volume_graph_cut_max_cells) {
    return result;
  }

  const auto indexOf = [ny, nz](std::size_t ix, std::size_t iy, std::size_t iz) {
    return (ix * ny + iy) * nz + iz;
  };
  const auto centerOf = [&](std::size_t ix, std::size_t iy, std::size_t iz) {
    return khronos::Point(
        static_cast<float>(static_cast<double>(min_corner.x()) +
                           (static_cast<double>(ix) + 0.5) * resolution),
        static_cast<float>(static_cast<double>(min_corner.y()) +
                           (static_cast<double>(iy) + 0.5) * resolution),
        static_cast<float>(static_cast<double>(min_corner.z()) +
                           (static_cast<double>(iz) + 0.5) * resolution));
  };
  const auto coordToIndex = [&](float value, float min_value, std::size_t dim) {
    const int idx = static_cast<int>(
        std::floor((static_cast<double>(value) - static_cast<double>(min_value)) / resolution));
    return static_cast<std::size_t>(std::clamp(idx, 0, static_cast<int>(dim) - 1));
  };

  std::vector<float> full_score(total_cells, 0.05f);
  std::vector<float> free_score(total_cells, 0.05f);
  std::vector<unsigned char> has_full_evidence(total_cells, 0);
  std::vector<unsigned char> has_free_evidence(total_cells, 0);
  std::vector<unsigned char> has_structural_evidence(total_cells, 0);

  for (std::size_t i = 0; i < mesh.numVertices(); ++i) {
    const auto point = mesh.pos(i);
    const auto ix = coordToIndex(point.x(), min_corner.x(), nx);
    const auto iy = coordToIndex(point.y(), min_corner.y(), ny);
    const auto iz = coordToIndex(point.z(), min_corner.z(), nz);
    const auto idx = indexOf(ix, iy, iz);
    full_score[idx] = std::min(7.0f, full_score[idx] + 1.2f);
    has_full_evidence[idx] = 1;
  }

  const auto structural_planes =
      selectVolumeStructuralPlanes(mesh,
                                   resolution,
                                   config.horizontal_plane_min_support_vertices);
  for (std::size_t ix = 0; ix < nx; ++ix) {
    for (std::size_t iy = 0; iy < ny; ++iy) {
      for (std::size_t iz = 0; iz < nz; ++iz) {
        const auto idx = indexOf(ix, iy, iz);
        const auto center = centerOf(ix, iy, iz);
        if (pointInsideAnyBox(object_mask_boxes, center, config.bbox_margin_m)) {
          continue;
        }
        for (const auto& plane : structural_planes) {
          if (std::abs(axisCoordinate(center, plane.axis) - plane.coord) > resolution * 0.65) {
            continue;
          }
          full_score[idx] = std::min(5.0f, full_score[idx] + 0.55f);
          has_full_evidence[idx] = 1;
          has_structural_evidence[idx] = 1;
          break;
        }
      }
    }
  }

  for (std::size_t ix = 0; ix < nx; ++ix) {
    for (std::size_t iy = 0; iy < ny; ++iy) {
      for (std::size_t iz = 0; iz < nz; ++iz) {
        const auto idx = indexOf(ix, iy, iz);
        const auto center = centerOf(ix, iy, iz);
        if (pointInsideAnyBox(absent_object_boxes, center, config.bbox_margin_m)) {
          free_score[idx] = std::min(7.0f, free_score[idx] + 2.5f);
          has_free_evidence[idx] = 1;
        }
        if (ix == 0 || iy == 0 || iz == 0 || ix + 1 == nx || iy + 1 == ny || iz + 1 == nz) {
          full_score[idx] = std::min(7.0f, full_score[idx] + 1.0f);
          has_full_evidence[idx] = 1;
        }
      }
    }
  }

  khronos::RayVerificator::Config ray_config;
  ray_config.block_size = static_cast<float>(config.free_space_culling_block_size_m);
  ray_config.radial_tolerance =
      static_cast<float>(config.free_space_culling_radial_tolerance_m);
  ray_config.depth_tolerance =
      static_cast<float>(config.free_space_culling_depth_tolerance_m);
  ray_config.active_window_duration =
      static_cast<float>(config.free_space_culling_active_window_duration_s);
  ray_config.ray_policy = khronos::RayVerificator::Config::RayPolicy::kMiddle;
  khronos::RayVerificator verifier(ray_config);
  verifier.setDsg(reference_dsg);

  for (std::size_t ix = 0; ix < nx; ++ix) {
    for (std::size_t iy = 0; iy < ny; ++iy) {
      for (std::size_t iz = 0; iz < nz; ++iz) {
        const auto idx = indexOf(ix, iy, iz);
        const auto check = verifier.check(centerOf(ix, iy, iz));
        if (!check.absent.empty()) {
          const auto evidence = static_cast<float>(std::min<std::size_t>(3, check.absent.size()));
          free_score[idx] = std::min(7.0f, free_score[idx] + 0.85f * evidence);
          has_free_evidence[idx] = 1;
        }
        if (!check.present.empty()) {
          const auto evidence = static_cast<float>(std::min<std::size_t>(3, check.present.size()));
          full_score[idx] = std::min(7.0f, full_score[idx] + 0.85f * evidence);
          has_full_evidence[idx] = 1;
        }
      }
    }
  }

  for (std::size_t idx = 0; idx < total_cells; ++idx) {
    if (has_free_evidence[idx]) {
      ++result.free_evidence_cells;
    }
    if (has_full_evidence[idx]) {
      ++result.full_evidence_cells;
    }
    if (has_structural_evidence[idx]) {
      ++result.structural_cells;
    }
  }

  const int source = static_cast<int>(total_cells);
  const int sink = static_cast<int>(total_cells + 1);
  DinicMaxFlow graph(total_cells + 2);
  constexpr double kPairwiseCost = 0.18;
  for (std::size_t ix = 0; ix < nx; ++ix) {
    for (std::size_t iy = 0; iy < ny; ++iy) {
      for (std::size_t iz = 0; iz < nz; ++iz) {
        const auto idx = indexOf(ix, iy, iz);
        const int node = static_cast<int>(idx);
        graph.addEdge(source, node, static_cast<double>(full_score[idx]));
        graph.addEdge(node, sink, static_cast<double>(free_score[idx]));

        const auto addPairwise = [&](std::size_t other_idx) {
          const int other_node = static_cast<int>(other_idx);
          graph.addEdge(node, other_node, kPairwiseCost);
          graph.addEdge(other_node, node, kPairwiseCost);
        };
        if (ix + 1 < nx) {
          addPairwise(indexOf(ix + 1, iy, iz));
        }
        if (iy + 1 < ny) {
          addPairwise(indexOf(ix, iy + 1, iz));
        }
        if (iz + 1 < nz) {
          addPairwise(indexOf(ix, iy, iz + 1));
        }
      }
    }
  }

  graph.maxFlow(source, sink);
  const auto reachable = graph.sourceReachable(source);
  std::vector<unsigned char> is_full(total_cells, 0);
  for (std::size_t idx = 0; idx < total_cells; ++idx) {
    if (reachable[idx]) {
      is_full[idx] = 1;
      ++result.full_cells;
    }
  }
  if (result.full_cells == 0 || result.full_cells == total_cells) {
    return result;
  }

  std::vector<khronos::Point> reference_points;
  auto novelty_search = makeInjectionSearch(mesh, &reference_points);
  if (!novelty_search) {
    return result;
  }
  const double novelty_threshold_sq =
      std::pow(std::min(0.08, resolution * 0.65), 2.0);

  std::vector<khronos::Point> vertices;
  std::vector<spark_dsg::Mesh::Face> faces;
  const auto addTriangle = [&](const khronos::Point& a,
                               const khronos::Point& b,
                               const khronos::Point& c) {
    khronos::Point center(0.0f, 0.0f, 0.0f);
    center = (a + b + c) / 3.0f;
    if (pointInsideAnyBox(object_mask_boxes, center, config.bbox_margin_m)) {
      return;
    }
    const auto area_vector = (b - a).cross(c - a);
    if (area_vector.squaredNorm() < 1.0e-8f) {
      return;
    }
    float distance_sq = std::numeric_limits<float>::max();
    std::size_t nearest_idx = 0;
    if (novelty_search->search(center, distance_sq, nearest_idx) &&
        distance_sq <= novelty_threshold_sq) {
      return;
    }
    const auto base = vertices.size();
    vertices.push_back(a);
    vertices.push_back(b);
    vertices.push_back(c);
    faces.push_back({base, base + 1, base + 2});
  };

  const auto addQuad = [&](const std::array<khronos::Point, 4>& quad) {
    addTriangle(quad[0], quad[1], quad[2]);
    addTriangle(quad[0], quad[2], quad[3]);
  };

  const bool structural_boundary_policy =
      usesStructuralVolumeBoundaries(config.volume_graph_cut_surface_policy);
  const bool marching_cubes_policy =
      usesMarchingCubesVolumeSurface(config.volume_graph_cut_surface_policy);
  const bool plane_pulled_marching_cubes_policy =
      usesPlanePulledMarchingCubesSurface(config.volume_graph_cut_surface_policy);
  const bool structural_plane_snap_policy =
      usesStructuralPlaneSnapSurface(config.volume_graph_cut_surface_policy);

  const auto structuralDistance = [&](const khronos::Point& point) {
    double best = std::numeric_limits<double>::infinity();
    for (const auto& plane : structural_planes) {
      best = std::min(best, std::abs(axisCoordinate(point, plane.axis) - plane.coord));
    }
    return best;
  };

  const auto signedCellValue = [&](std::size_t ix, std::size_t iy, std::size_t iz) {
    const auto idx = indexOf(ix, iy, iz);
    const auto center = centerOf(ix, iy, iz);
    double magnitude = resolution * 0.5;
    const double plane_distance = structuralDistance(center);
    if (plane_pulled_marching_cubes_policy && plane_distance <= resolution * 1.5) {
      magnitude = std::clamp(plane_distance, resolution * 0.05, resolution * 0.75);
    }
    return static_cast<float>(is_full[idx] ? -magnitude : magnitude);
  };

  const auto corner = [&](double x, double y, double z) {
    return khronos::Point(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
  };

  const auto addBoundaryIfCut = [&](std::size_t ix,
                                    std::size_t iy,
                                    std::size_t iz,
                                    std::size_t jx,
                                    std::size_t jy,
                                    std::size_t jz,
                                    int axis) {
    const auto a = indexOf(ix, iy, iz);
    const auto b = indexOf(jx, jy, jz);
    if (is_full[a] == is_full[b]) {
      return;
    }
    if (structural_boundary_policy && (!has_structural_evidence[a] && !has_structural_evidence[b])) {
      return;
    }
    if (structural_boundary_policy && (!has_free_evidence[a] && !has_free_evidence[b])) {
      return;
    }
    const double x0 = static_cast<double>(min_corner.x()) + static_cast<double>(ix) * resolution;
    const double y0 = static_cast<double>(min_corner.y()) + static_cast<double>(iy) * resolution;
    const double z0 = static_cast<double>(min_corner.z()) + static_cast<double>(iz) * resolution;
    const auto snapToStructuralPlane = [&](double face_coord, double* snapped_coord) {
      double best_distance = std::numeric_limits<double>::infinity();
      double best_coord = face_coord;
      for (const auto& plane : structural_planes) {
        if (plane.axis != axis) {
          continue;
        }
        const double distance = std::abs(face_coord - plane.coord);
        if (distance < best_distance) {
          best_distance = distance;
          best_coord = plane.coord;
        }
      }
      if (best_distance > resolution * 1.0) {
        return false;
      }
      *snapped_coord = best_coord;
      return true;
    };
    if (axis == 0) {
      double x = static_cast<double>(min_corner.x()) +
                 static_cast<double>(std::max(ix, jx)) * resolution;
      if (structural_plane_snap_policy && !snapToStructuralPlane(x, &x)) {
        return;
      }
      addQuad({corner(x, y0, z0),
               corner(x, y0 + resolution, z0),
               corner(x, y0 + resolution, z0 + resolution),
               corner(x, y0, z0 + resolution)});
    } else if (axis == 1) {
      double y = static_cast<double>(min_corner.y()) +
                 static_cast<double>(std::max(iy, jy)) * resolution;
      if (structural_plane_snap_policy && !snapToStructuralPlane(y, &y)) {
        return;
      }
      addQuad({corner(x0, y, z0),
               corner(x0 + resolution, y, z0),
               corner(x0 + resolution, y, z0 + resolution),
               corner(x0, y, z0 + resolution)});
    } else {
      double z = static_cast<double>(min_corner.z()) +
                 static_cast<double>(std::max(iz, jz)) * resolution;
      if (structural_plane_snap_policy && !snapToStructuralPlane(z, &z)) {
        return;
      }
      addQuad({corner(x0, y0, z),
               corner(x0 + resolution, y0, z),
               corner(x0 + resolution, y0 + resolution, z),
               corner(x0, y0 + resolution, z)});
    }
  };

  vertices.reserve(total_cells / 8);
  faces.reserve(total_cells / 8);
  if (marching_cubes_policy) {
    const std::array<std::array<std::size_t, 3>, 8> offsets{{
        {0, 0, 0},
        {1, 0, 0},
        {1, 1, 0},
        {0, 1, 0},
        {0, 0, 1},
        {1, 0, 1},
        {1, 1, 1},
        {0, 1, 1},
    }};
    const hydra::BlockIndex block = hydra::BlockIndex::Zero();
    for (std::size_t ix = 0; ix + 1 < nx; ++ix) {
      for (std::size_t iy = 0; iy + 1 < ny; ++iy) {
        for (std::size_t iz = 0; iz + 1 < nz; ++iz) {
          bool has_full_label = false;
          bool has_free_label = false;
          bool has_structural = false;
          bool has_free_space = false;
          hydra::MarchingCubes::SdfPoints points;
          for (std::size_t corner_idx = 0; corner_idx < offsets.size(); ++corner_idx) {
            const auto cx = ix + offsets[corner_idx][0];
            const auto cy = iy + offsets[corner_idx][1];
            const auto cz = iz + offsets[corner_idx][2];
            const auto cell_idx = indexOf(cx, cy, cz);
            has_full_label = has_full_label || is_full[cell_idx];
            has_free_label = has_free_label || !is_full[cell_idx];
            has_structural = has_structural || has_structural_evidence[cell_idx];
            has_free_space = has_free_space || has_free_evidence[cell_idx];
            auto& point = points[corner_idx];
            point.distance = signedCellValue(cx, cy, cz);
            point.weight = 1.0f;
            point.pos = centerOf(cx, cy, cz);
            point.color = spark_dsg::Color(120, 150, 175);
          }
          if (!has_full_label || !has_free_label) {
            continue;
          }
          if (structural_boundary_policy && (!has_structural || !has_free_space)) {
            continue;
          }

          spark_dsg::Mesh cube_mesh(true, false, false, false);
          hydra::MarchingCubes::meshCube(block, points, cube_mesh, false);
          for (std::size_t face_idx = 0; face_idx < cube_mesh.numFaces(); ++face_idx) {
            const auto face = cube_mesh.face(face_idx);
            addTriangle(cube_mesh.pos(face[0]),
                        cube_mesh.pos(face[1]),
                        cube_mesh.pos(face[2]));
          }
        }
      }
    }
  } else {
    for (std::size_t ix = 0; ix < nx; ++ix) {
      for (std::size_t iy = 0; iy < ny; ++iy) {
        for (std::size_t iz = 0; iz < nz; ++iz) {
          if (ix + 1 < nx) {
            addBoundaryIfCut(ix, iy, iz, ix + 1, iy, iz, 0);
          }
          if (iy + 1 < ny) {
            addBoundaryIfCut(ix, iy, iz, ix, iy + 1, iz, 1);
          }
          if (iz + 1 < nz) {
            addBoundaryIfCut(ix, iy, iz, ix, iy, iz + 1, 2);
          }
        }
      }
    }
  }

  if (vertices.empty() || faces.empty()) {
    return result;
  }

  spark_dsg::Mesh surface(mesh.has_colors,
                          mesh.has_timestamps,
                          mesh.has_labels,
                          mesh.has_first_seen_stamps);
  surface.resizeVertices(vertices.size());
  for (std::size_t i = 0; i < vertices.size(); ++i) {
    surface.setPos(i, vertices[i]);
    if (surface.has_colors) {
      surface.setColor(i, spark_dsg::Color(120, 150, 175));
    }
    if (surface.has_timestamps) {
      surface.setTimestamp(i, 0);
    }
    if (surface.has_first_seen_stamps) {
      surface.setFirstSeenTimestamp(i, 0);
    }
    if (surface.has_labels) {
      surface.setLabel(i, static_cast<spark_dsg::Mesh::Label>(-1));
    }
  }
  surface.resizeFaces(faces.size());
  for (std::size_t i = 0; i < faces.size(); ++i) {
    surface.face(i) = faces[i];
  }

  if (!mesh.append(surface)) {
    throw std::runtime_error("Failed to append volume graph-cut surface.");
  }
  result.vertices = vertices.size();
  result.faces = faces.size();
  return result;
}

}  // namespace

ObjectGuidedMapReconciler::ObjectGuidedMapReconciler(ReconcilerConfig config)
    : config_(std::move(config)) {}

ReconcileResult ObjectGuidedMapReconciler::reconcile(khronos::DynamicSceneGraph& dsg) const {
  ReconcileResult result;
  const auto current_evidence_dsg = dsg.clone();
  const bool initialize_from_prior =
      config_.dynamic_mode == "cross_session" && config_.prior_session_dsg &&
      config_.prior_session_dsg->hasMesh();
  if (initialize_from_prior) {
    initializeCrossSessionMesh(dsg,
                               current_evidence_dsg,
                               config_.prior_session_dsg,
                               config_,
                               &result.mesh_summary,
                               &result.vertex_update_rows);
  }

  if (!dsg.hasMesh()) {
    result.object_rows = auditObjects(dsg);
    return result;
  }

  auto& mesh = *dsg.mesh();
  result.mesh_summary.initial_vertices =
      initialize_from_prior ? result.mesh_summary.cross_session_prior_vertices
                            : mesh.numVertices();
  result.mesh_summary.initial_faces =
      initialize_from_prior ? result.mesh_summary.cross_session_prior_faces
                            : mesh.numFaces();
  const auto evidence_reference_dsg =
      (config_.free_space_culling || config_.volume_graph_cut_fill ||
       config_.structural_plane_visibility_filter || config_.dynamic_residue_cleanup)
          ? (initialize_from_prior ? current_evidence_dsg : dsg.clone())
          : nullptr;
  const std::size_t free_space_added_vertex_start = mesh.numVertices();
  const auto applyFreeSpaceCulling = [&](std::size_t first_added_vertex) {
    const auto culled = cullFreeSpaceContradictedAddedVertices(
        evidence_reference_dsg, mesh, first_added_vertex, config_, &result.vertex_update_rows);
    result.mesh_summary.free_space_checked_vertices += culled.checked_vertices;
    result.mesh_summary.free_space_absent_vertices += culled.absent_vertices;
    result.mesh_summary.free_space_present_vertices += culled.present_vertices;
    result.mesh_summary.free_space_removed_vertices += culled.removed_vertices;
  };
  result.object_rows = auditObjects(dsg);
  const auto prior_objects = loadPriorObjects();
  auto object_change_evidence = loadObjectChangeEvidence();
  const auto synthetic_change = loadSyntheticChangeSpec();
  for (auto& row : result.object_rows) {
    auto evidence_it = object_change_evidence.find(row.node_id);
    if (evidence_it == object_change_evidence.end()) {
      continue;
    }
    auto& evidence = evidence_it->second;
    if (config_.object_move_decision == "expected_utility") {
      const double support_ratio =
          row.object_mesh_vertices == 0
              ? 1.0
              : static_cast<double>(row.global_vertices_in_bbox) /
                    static_cast<double>(row.object_mesh_vertices);
      const double missing_probability = std::clamp(1.0 - support_ratio, 0.0, 1.0);
      const double static_repair_gain =
          (1.0 - evidence.move_probability) * missing_probability;
      const double moved_hallucination_risk = evidence.move_probability;
      evidence.skip_injection = moved_hallucination_risk > static_repair_gain;
    }
    row.change_first_absent_ns = evidence.first_absent_ns;
    row.change_last_absent_ns = evidence.last_absent_ns;
    row.change_first_persistent_ns = evidence.first_persistent_ns;
    row.change_last_persistent_ns = evidence.last_persistent_ns;
    row.object_move_probability = evidence.move_probability;
    row.skipped_by_object_move = evidence.skip_injection;
  }
  applyPriorObjectMemory(initialize_from_prior ? *current_evidence_dsg : dsg,
                         &result.object_rows,
                         &result.mesh_summary);
  result.mesh_summary.objects_total = result.object_rows.size();
  for (const auto& row : result.object_rows) {
    if (row.object_mesh_vertices > 0) {
      ++result.mesh_summary.objects_with_private_mesh;
    }
    if (row.object_mesh_vertices >= config_.min_object_mesh_vertices) {
      const double global_ratio =
          row.object_mesh_vertices == 0
              ? 0.0
              : static_cast<double>(row.global_vertices_in_bbox) /
                    static_cast<double>(row.object_mesh_vertices);
      if (global_ratio < config_.repair_global_vertex_ratio_threshold) {
        ++result.mesh_summary.repair_candidate_objects;
        result.mesh_summary.repair_candidate_vertices += row.object_mesh_vertices;
      }
    }
  }

  for (auto& row : result.object_rows) {
    if (row.object_mesh_vertices < config_.min_object_mesh_vertices) {
      continue;
    }
    const double global_ratio =
        row.object_mesh_vertices == 0
            ? 0.0
            : static_cast<double>(row.global_vertices_in_bbox) /
                  static_cast<double>(row.object_mesh_vertices);
    if (global_ratio < config_.repair_global_vertex_ratio_threshold) {
      row.repair_candidate = true;
      row.repair_candidate_vertices = row.object_mesh_vertices;
    }
  }

  std::unordered_set<khronos::NodeId> repair_candidate_object_ids;
  std::unordered_map<khronos::NodeId, std::size_t> object_row_by_id;
  for (std::size_t row_idx = 0; row_idx < result.object_rows.size(); ++row_idx) {
    const auto& row = result.object_rows[row_idx];
    object_row_by_id[row.node_id] = row_idx;
    if (row.repair_candidate) {
      repair_candidate_object_ids.insert(row.node_id);
    }
  }
  const auto shouldInjectCurrentObject = [&](khronos::NodeId id) {
    return config_.object_injection_policy != "repair_candidates" ||
           repair_candidate_object_ids.count(id) > 0;
  };
  const auto isRepairCandidateObject = [&](khronos::NodeId id) {
    return repair_candidate_object_ids.count(id) > 0;
  };

  std::vector<khronos::BoundingBox> object_mask_boxes;
  std::vector<khronos::BoundingBox> absent_object_boxes;
  if (dsg.hasLayer(khronos::DsgLayers::OBJECTS)) {
    for (const auto& [id, node] : dsg.getLayer(khronos::DsgLayers::OBJECTS).nodes()) {
      const auto& attrs = node->attributes<khronos::KhronosObjectAttributes>();
      object_mask_boxes.push_back(attrs.bounding_box);
      const auto evidence_it = object_change_evidence.find(id);
      if (evidence_it != object_change_evidence.end() &&
          evidence_it->second.skip_injection) {
        absent_object_boxes.push_back(attrs.bounding_box);
      }
    }
  }

  if (config_.axis_aligned_plane_fill) {
    const auto filled = fillAxisAlignedBoundaryPlanes(mesh,
                                                     config_.horizontal_plane_grid_resolution_m,
                                                     config_.horizontal_plane_min_support_vertices,
                                                     config_.horizontal_plane_fill_mode,
                                                     config_.axis_aligned_plane_support_band_cells,
                                                     evidence_reference_dsg,
                                                     &object_mask_boxes,
                                                     &config_);
    result.mesh_summary.axis_aligned_planes_filled += filled.planes;
    result.mesh_summary.axis_aligned_plane_vertices += filled.vertices;
    result.mesh_summary.axis_aligned_plane_faces += filled.faces;
    result.mesh_summary.axis_aligned_plane_graph_cut_cells += filled.graph_cut_cells;
    result.mesh_summary.axis_aligned_plane_graph_cut_fill_cells += filled.graph_cut_fill_cells;
  }

  if (config_.horizontal_plane_fill) {
    const auto filled = fillHorizontalPlanes(mesh,
                                            config_.horizontal_plane_grid_resolution_m,
                                            config_.horizontal_plane_min_support_vertices,
                                            config_.horizontal_plane_fill_mode,
                                            evidence_reference_dsg,
                                            &object_mask_boxes,
                                            &config_);
    result.mesh_summary.horizontal_planes_filled += filled.planes;
    result.mesh_summary.horizontal_plane_vertices += filled.vertices;
    result.mesh_summary.horizontal_plane_faces += filled.faces;
    result.mesh_summary.horizontal_plane_graph_cut_cells += filled.graph_cut_cells;
    result.mesh_summary.horizontal_plane_graph_cut_fill_cells += filled.graph_cut_fill_cells;
  }

  if (config_.volume_graph_cut_fill) {
    const auto filled = fillVolumeByGraphCut(mesh,
                                            evidence_reference_dsg,
                                            object_mask_boxes,
                                            absent_object_boxes,
                                            config_);
    result.mesh_summary.volume_graph_cut_cells += filled.cells;
    result.mesh_summary.volume_graph_cut_free_evidence_cells += filled.free_evidence_cells;
    result.mesh_summary.volume_graph_cut_full_evidence_cells += filled.full_evidence_cells;
    result.mesh_summary.volume_graph_cut_structural_cells += filled.structural_cells;
    result.mesh_summary.volume_graph_cut_full_cells += filled.full_cells;
    result.mesh_summary.volume_graph_cut_vertices += filled.vertices;
    result.mesh_summary.volume_graph_cut_faces += filled.faces;
  }

  if (config_.temporal_background_repair) {
    std::vector<khronos::BoundingBox> absent_object_boxes;
    if (dsg.hasLayer(khronos::DsgLayers::OBJECTS)) {
      for (const auto& [id, node] : dsg.getLayer(khronos::DsgLayers::OBJECTS).nodes()) {
        const auto evidence_it = object_change_evidence.find(id);
        if (evidence_it == object_change_evidence.end() ||
            !evidence_it->second.skip_injection) {
          continue;
        }
        const auto& attrs = node->attributes<khronos::KhronosObjectAttributes>();
        absent_object_boxes.push_back(attrs.bounding_box);
      }
    }

    for (const auto& prior_dsg : config_.temporal_background_dsgs) {
      if (!prior_dsg || !prior_dsg->hasMesh() || prior_dsg->mesh()->numVertices() == 0) {
        continue;
      }
      std::vector<khronos::Point> temporal_reference_points;
      auto temporal_search = makeInjectionSearch(mesh, &temporal_reference_points);
      const auto appended =
          appendBackgroundMeshToGlobal(*prior_dsg->mesh(),
                                       mesh,
                                       temporal_search.get(),
                                       config_.temporal_background_min_separation_m,
                                       absent_object_boxes,
                                       config_.bbox_margin_m);
      if (appended.vertices == 0) {
        continue;
      }
      ++result.mesh_summary.temporal_background_sources;
      result.mesh_summary.temporal_background_injected_vertices += appended.vertices;
      result.mesh_summary.temporal_background_injected_faces += appended.faces;
    }
  }

  const std::size_t free_space_object_vertex_start = mesh.numVertices();
  std::vector<khronos::Point> injection_reference_points;
  const bool needs_supported_object_search =
      config_.object_injection_policy == "repair_or_supported";
  const bool needs_alignment_search = config_.object_alignment_policy == "translation";
  auto injection_search = config_.injection_min_separation_m > 0.0 ||
                                  needs_supported_object_search ||
                                  needs_alignment_search
                              ? makeInjectionSearch(mesh, &injection_reference_points)
                              : nullptr;
  const auto alignObject = [&](khronos::NodeId id,
                               const khronos::KhronosObjectAttributes& attrs) {
    ObjectAlignment alignment;
    if (config_.object_alignment_policy != "translation") {
      return alignment;
    }
    ++result.mesh_summary.object_alignment_candidates;
    alignment = estimateObjectTranslationAlignment(
        attrs,
        injection_search.get(),
        injection_reference_points,
        config_.object_alignment_support_distance_m,
        config_.object_alignment_max_translation_m,
        config_.object_alignment_min_support_vertices);
    result.mesh_summary.object_alignment_support_vertices += alignment.support_vertices;
    if (alignment.applied) {
      ++result.mesh_summary.object_alignment_applied;
    }
    const auto row_it = object_row_by_id.find(id);
    if (row_it != object_row_by_id.end()) {
      auto& row = result.object_rows[row_it->second];
      row.object_alignment_applied = alignment.applied;
      row.object_alignment_support_vertices = alignment.support_vertices;
      row.object_alignment_translation_norm_m =
          std::sqrt(static_cast<double>(alignment.translation.x()) * alignment.translation.x() +
                    static_cast<double>(alignment.translation.y()) * alignment.translation.y() +
                    static_cast<double>(alignment.translation.z()) * alignment.translation.z());
      row.object_alignment_before_median_m = alignment.before_median_m;
      row.object_alignment_after_median_m = alignment.after_median_m;
    }
    return alignment;
  };
  const std::size_t cleanup_vertex_limit = mesh.numVertices();

  if (config_.mode == "no_op" || config_.mode == "audit" || config_.mode == "injection") {
    if (config_.mode == "injection" && dsg.hasLayer(khronos::DsgLayers::OBJECTS)) {
      for (const auto& [id, node] : dsg.getLayer(khronos::DsgLayers::OBJECTS).nodes()) {
        const auto evidence_it = object_change_evidence.find(id);
        if (evidence_it != object_change_evidence.end() &&
            evidence_it->second.skip_injection) {
          continue;
        }
        if (!shouldInjectCurrentObject(id)) {
          continue;
        }
        const auto& attrs = node->attributes<khronos::KhronosObjectAttributes>();
        if (attrs.mesh.numVertices() < config_.min_object_mesh_vertices) {
          continue;
        }
        const bool keep_supported_only =
            config_.object_injection_policy == "repair_or_supported" &&
            !isRepairCandidateObject(id);
        const double filter_distance =
            keep_supported_only ? config_.object_surface_support_distance_m
                                : config_.injection_min_separation_m;
        const auto filter = keep_supported_only ? ObjectVertexFilter::kMaxSupportDistance
                                                : ObjectVertexFilter::kMinSeparation;
        const auto alignment = alignObject(id, attrs);
        const auto appended = appendObjectMeshToGlobal(
            attrs,
            mesh,
            injection_search.get(),
            filter_distance,
            filter,
            alignment.applied ? alignment.translation : khronos::Point(0.0f, 0.0f, 0.0f));
        if (appended.vertices == 0) {
          continue;
        }
        ++result.mesh_summary.injected_objects;
        result.mesh_summary.injected_vertices += appended.vertices;
        result.mesh_summary.injected_faces += appended.faces;
      }

      if (config_.temporal_object_repair) {
        for (const auto& prior_dsg : config_.temporal_background_dsgs) {
          if (!prior_dsg || !prior_dsg->hasLayer(khronos::DsgLayers::OBJECTS)) {
            continue;
          }
          std::vector<khronos::Point> temporal_object_reference_points;
          auto temporal_object_search =
              makeInjectionSearch(mesh, &temporal_object_reference_points);
          for (const auto& [id, node] :
               prior_dsg->getLayer(khronos::DsgLayers::OBJECTS).nodes()) {
            const auto evidence_it = object_change_evidence.find(id);
            if (evidence_it != object_change_evidence.end() &&
                evidence_it->second.skip_injection) {
              continue;
            }
            const auto& attrs = node->attributes<khronos::KhronosObjectAttributes>();
            if (attrs.mesh.numVertices() < config_.min_object_mesh_vertices) {
              continue;
            }
            const auto appended =
                appendObjectMeshToGlobal(attrs,
                                         mesh,
                                         temporal_object_search.get(),
                                         config_.temporal_object_min_separation_m,
                                         ObjectVertexFilter::kMinSeparation,
                                         khronos::Point(0.0f, 0.0f, 0.0f));
            if (appended.vertices == 0) {
              continue;
            }
            ++result.mesh_summary.temporal_object_sources;
            result.mesh_summary.temporal_object_injected_vertices += appended.vertices;
            result.mesh_summary.temporal_object_injected_faces += appended.faces;
          }
        }
      }
    }
    const std::size_t free_space_start =
        config_.free_space_culling_scope == "objects" ? free_space_object_vertex_start
                                                      : free_space_added_vertex_start;
    applyFreeSpaceCulling(free_space_start);
    restorePriorObjectNodes(dsg, &result.object_rows, &result.mesh_summary);
    result.mesh_summary.final_vertices = mesh.numVertices();
    result.mesh_summary.final_faces = mesh.numFaces();
    return result;
  }

  if (config_.mode != "cleanup" && config_.mode != "dynamic_cleanup" &&
      config_.mode != "full") {
    throw std::runtime_error("Unsupported Base1 mode: " + config_.mode);
  }

  std::vector<khronos::Point> source_points;
  std::vector<std::size_t> point_to_source;
  std::vector<CleanupSource> cleanup_sources;
  if (config_.mode != "dynamic_cleanup") {
    cleanup_sources =
        collectCleanupSources(dsg, &source_points, &point_to_source, &result.object_rows);
  }
  result.mesh_summary.objects_used_for_cleanup = cleanup_sources.size();
  result.mesh_summary.cleanup_source_points = source_points.size();

  if (config_.mode == "full" && dsg.hasLayer(khronos::DsgLayers::OBJECTS)) {
    for (const auto& [id, node] : dsg.getLayer(khronos::DsgLayers::OBJECTS).nodes()) {
      const auto evidence_it = object_change_evidence.find(id);
      if (evidence_it != object_change_evidence.end() &&
          evidence_it->second.skip_injection) {
        continue;
      }
      if (!shouldInjectCurrentObject(id)) {
        continue;
      }
      const auto& attrs = node->attributes<khronos::KhronosObjectAttributes>();
      if (attrs.mesh.numVertices() < config_.min_object_mesh_vertices) {
        continue;
      }
      const bool keep_supported_only =
          config_.object_injection_policy == "repair_or_supported" &&
          !isRepairCandidateObject(id);
      const double filter_distance =
          keep_supported_only ? config_.object_surface_support_distance_m
                              : config_.injection_min_separation_m;
      const auto filter = keep_supported_only ? ObjectVertexFilter::kMaxSupportDistance
                                              : ObjectVertexFilter::kMinSeparation;
      const auto alignment = alignObject(id, attrs);
      const auto appended = appendObjectMeshToGlobal(
          attrs,
          mesh,
          injection_search.get(),
          filter_distance,
          filter,
          alignment.applied ? alignment.translation : khronos::Point(0.0f, 0.0f, 0.0f));
      if (appended.vertices == 0) {
        continue;
      }
      ++result.mesh_summary.injected_objects;
      result.mesh_summary.injected_vertices += appended.vertices;
      result.mesh_summary.injected_faces += appended.faces;
    }
  }

  const double threshold_sq = config_.object_distance_m * config_.object_distance_m;
  std::unordered_set<std::size_t> vertices_to_delete;
  std::unordered_map<khronos::NodeId, std::size_t> deleted_by_object;
  std::unordered_map<khronos::NodeId, std::size_t> candidate_by_object;
  const bool use_cleanup_reobservation_gate =
      config_.object_cleanup_reobservation_gate &&
      config_.dynamic_mode == "cross_session" && current_evidence_dsg &&
      current_evidence_dsg->hasMesh();
  std::vector<khronos::Point> cleanup_current_points;
  auto cleanup_current_search =
      use_cleanup_reobservation_gate
          ? makeInjectionSearch(*current_evidence_dsg->mesh(), &cleanup_current_points)
          : nullptr;
  std::unique_ptr<khronos::RayVerificator> cleanup_verifier;
  if (use_cleanup_reobservation_gate) {
    khronos::RayVerificator::Config ray_config;
    ray_config.block_size =
        static_cast<float>(config_.free_space_culling_block_size_m);
    ray_config.radial_tolerance =
        static_cast<float>(config_.free_space_culling_radial_tolerance_m);
    ray_config.depth_tolerance =
        static_cast<float>(config_.free_space_culling_depth_tolerance_m);
    ray_config.active_window_duration =
        static_cast<float>(config_.free_space_culling_active_window_duration_s);
    ray_config.ray_policy = khronos::RayVerificator::Config::RayPolicy::kMiddle;
    cleanup_verifier = std::make_unique<khronos::RayVerificator>(ray_config);
    cleanup_verifier->setDsg(current_evidence_dsg);
  }
  const double cleanup_support_threshold_sq =
      config_.object_cleanup_support_distance_m *
      config_.object_cleanup_support_distance_m;
  const std::size_t cleanup_min_absent =
      std::max<std::size_t>(1, config_.free_space_culling_min_absent);

  if (!source_points.empty()) {
    const hydra::PointNeighborSearch search(source_points);

    const std::size_t vertex_limit = std::min(cleanup_vertex_limit, mesh.numVertices());
    for (std::size_t vertex_idx = 0; vertex_idx < vertex_limit; ++vertex_idx) {
      const auto point = mesh.pos(vertex_idx);
      const int vertex_label =
          vertex_idx < mesh.labels.size() ? static_cast<int>(mesh.labels[vertex_idx]) : -1;

      float distance_sq = std::numeric_limits<float>::max();
      std::size_t source_point_idx = 0;
      if (!search.search(point, distance_sq, source_point_idx)) {
        continue;
      }
      if (distance_sq > threshold_sq || source_point_idx >= point_to_source.size()) {
        continue;
      }

      const std::size_t source_idx = point_to_source[source_point_idx];
      if (source_idx >= cleanup_sources.size()) {
        continue;
      }
      const auto& source = cleanup_sources[source_idx];
      if (config_.require_bbox_containment &&
          !isInsideExpandedBox(source.bounding_box, point, config_.bbox_margin_m)) {
        continue;
      }
      if (config_.require_same_label && vertex_label != source.semantic_label) {
        continue;
      }

      candidate_by_object[source.node_id]++;
      if (use_cleanup_reobservation_gate) {
        float support_distance_sq = std::numeric_limits<float>::max();
        std::size_t support_idx = 0;
        const bool has_surface_support =
            cleanup_current_search &&
            cleanup_current_search->search(point, support_distance_sq, support_idx) &&
            support_distance_sq <= cleanup_support_threshold_sq;
        if (has_surface_support) {
          ++result.mesh_summary.object_cleanup_present_protected_vertices;
          result.vertex_update_rows.push_back(ReconcileResult::VertexUpdateRow{
              vertex_idx,
              source.node_id,
              vertex_label,
              source.semantic_label,
              std::sqrt(static_cast<double>(distance_sq)),
              "object_cleanup_protected_current_surface",
          });
          continue;
        }

        const auto check = cleanup_verifier->check(point);
        bool absent_confirmed = false;
        if (config_.free_space_culling_decision == "absence_majority") {
          absent_confirmed =
              check.absent.size() >= cleanup_min_absent &&
              check.absent.size() > check.present.size();
        } else {
          absent_confirmed =
              check.absent.size() >= cleanup_min_absent &&
              check.present.size() <= config_.free_space_culling_max_present;
        }
        if (!absent_confirmed) {
          const bool has_present_evidence = !check.present.empty();
          if (has_present_evidence) {
            ++result.mesh_summary.object_cleanup_present_protected_vertices;
          } else {
            ++result.mesh_summary.object_cleanup_unobserved_protected_vertices;
          }
          result.vertex_update_rows.push_back(ReconcileResult::VertexUpdateRow{
              vertex_idx,
              source.node_id,
              vertex_label,
              source.semantic_label,
              std::sqrt(static_cast<double>(distance_sq)),
              has_present_evidence ? "object_cleanup_protected_ray_present"
                                   : "object_cleanup_protected_unobserved",
          });
          continue;
        }
        ++result.mesh_summary.object_cleanup_absent_confirmed_vertices;
      }

      vertices_to_delete.insert(vertex_idx);
      deleted_by_object[source.node_id]++;
      result.vertex_update_rows.push_back(
          ReconcileResult::VertexUpdateRow{vertex_idx,
                                           source.node_id,
                                           vertex_label,
                                           source.semantic_label,
                                           std::sqrt(static_cast<double>(distance_sq)),
                                           config_.dry_run ? "candidate_dry_run" : "remove"});
    }
  }

  if (config_.dynamic_residue_cleanup && evidence_reference_dsg &&
      dsg.hasLayer(khronos::DsgLayers::OBJECTS)) {
    struct DynamicTrackSupport {
      khronos::NodeId node_id = 0;
      int semantic_label = -1;
      uint64_t first_stamp = 0;
      uint64_t last_stamp = 0;
    };

    std::vector<DynamicTrackSupport> dynamic_tracks;
    std::vector<khronos::Point> dynamic_support_points;
    std::vector<std::size_t> point_to_track;
    for (const auto& [id, node] : dsg.getLayer(khronos::DsgLayers::OBJECTS).nodes()) {
      const auto& attrs = node->attributes<khronos::KhronosObjectAttributes>();
      if (attrs.dynamic_object_points.size() < config_.dynamic_residue_min_track_frames ||
          attrs.trajectory_timestamps.empty()) {
        continue;
      }

      std::size_t num_points = 0;
      for (const auto& frame : attrs.dynamic_object_points) {
        num_points += frame.size();
      }
      if (num_points == 0) {
        continue;
      }

      const std::size_t track_idx = dynamic_tracks.size();
      dynamic_tracks.push_back(DynamicTrackSupport{
          id,
          static_cast<int>(attrs.semantic_label),
          attrs.trajectory_timestamps.front(),
          attrs.trajectory_timestamps.back(),
      });
      dynamic_support_points.reserve(dynamic_support_points.size() + num_points);
      point_to_track.reserve(point_to_track.size() + num_points);
      for (const auto& frame : attrs.dynamic_object_points) {
        dynamic_support_points.insert(
            dynamic_support_points.end(), frame.begin(), frame.end());
        point_to_track.insert(point_to_track.end(), frame.size(), track_idx);
      }
    }

    result.mesh_summary.dynamic_residue_tracks = dynamic_tracks.size();
    result.mesh_summary.dynamic_residue_support_points =
        dynamic_support_points.size();
    if (!dynamic_support_points.empty()) {
      const hydra::PointNeighborSearch dynamic_search(dynamic_support_points);
      const double support_threshold_sq =
          config_.dynamic_residue_support_distance_m *
          config_.dynamic_residue_support_distance_m;

      khronos::RayVerificator::Config ray_config;
      ray_config.block_size =
          static_cast<float>(config_.free_space_culling_block_size_m);
      ray_config.radial_tolerance =
          static_cast<float>(config_.free_space_culling_radial_tolerance_m);
      ray_config.depth_tolerance =
          static_cast<float>(config_.free_space_culling_depth_tolerance_m);
      ray_config.active_window_duration =
          static_cast<float>(config_.free_space_culling_active_window_duration_s);
      ray_config.ray_policy =
          khronos::RayVerificator::Config::RayPolicy::kMiddle;
      khronos::RayVerificator verifier(ray_config);
      verifier.setDsg(evidence_reference_dsg);

      const std::size_t min_absent =
          std::max<std::size_t>(1, config_.dynamic_residue_min_absent);
      const std::size_t vertex_limit =
          std::min(cleanup_vertex_limit, mesh.numVertices());
      std::vector<khronos::Point> vertex_normal_sums;
      if (config_.dynamic_residue_protect_horizontal_surfaces) {
        vertex_normal_sums.assign(
            vertex_limit, khronos::Point(0.0f, 0.0f, 0.0f));
        for (std::size_t face_idx = 0; face_idx < mesh.numFaces(); ++face_idx) {
          const auto& face = mesh.face(face_idx);
          if (face[0] >= vertex_limit || face[1] >= vertex_limit ||
              face[2] >= vertex_limit) {
            continue;
          }
          const auto normal =
              (mesh.pos(face[1]) - mesh.pos(face[0]))
                  .cross(mesh.pos(face[2]) - mesh.pos(face[0]));
          if (normal.squaredNorm() <= 1.0e-12f) {
            continue;
          }
          vertex_normal_sums[face[0]] += normal;
          vertex_normal_sums[face[1]] += normal;
          vertex_normal_sums[face[2]] += normal;
        }
      }
      for (std::size_t vertex_idx = 0; vertex_idx < vertex_limit; ++vertex_idx) {
        float distance_sq = std::numeric_limits<float>::max();
        std::size_t support_idx = 0;
        if (!dynamic_search.search(mesh.pos(vertex_idx), distance_sq, support_idx) ||
            distance_sq > support_threshold_sq ||
            support_idx >= point_to_track.size()) {
          continue;
        }

        const std::size_t track_idx = point_to_track[support_idx];
        if (track_idx >= dynamic_tracks.size()) {
          continue;
        }
        const auto& track = dynamic_tracks[track_idx];
        const auto object_row_it = object_row_by_id.find(track.node_id);
        ObjectAuditRow* object_row =
            object_row_it == object_row_by_id.end()
                ? nullptr
                : &result.object_rows[object_row_it->second];

        if (config_.dynamic_residue_require_time_overlap) {
          if (!mesh.has_timestamps) {
            continue;
          }
          const auto vertex_stamp = mesh.timestamp(vertex_idx);
          if (vertex_stamp < track.first_stamp || vertex_stamp > track.last_stamp) {
            continue;
          }
        }

        ++result.mesh_summary.dynamic_residue_candidates;
        ++candidate_by_object[track.node_id];
        if (object_row) {
          ++object_row->dynamic_residue_candidates;
        }
        const int vertex_label =
            vertex_idx < mesh.labels.size()
                ? static_cast<int>(mesh.labels[vertex_idx])
                : -1;
        const double distance_m = std::sqrt(static_cast<double>(distance_sq));

        bool horizontal_surface = false;
        if (!vertex_normal_sums.empty()) {
          const auto& normal_sum = vertex_normal_sums[vertex_idx];
          if (normal_sum.squaredNorm() > 1.0e-12f) {
            horizontal_surface =
                std::abs(normal_sum.normalized().z()) >=
                config_.dynamic_residue_horizontal_normal_z_min;
          }
        }
        const bool protected_label =
            config_.dynamic_residue_protected_labels.count(vertex_label) > 0;
        if (protected_label || horizontal_surface) {
          ++result.mesh_summary.dynamic_residue_protected_vertices;
          if (object_row) {
            ++object_row->dynamic_residue_protected;
          }
          result.vertex_update_rows.push_back(ReconcileResult::VertexUpdateRow{
              vertex_idx,
              track.node_id,
              vertex_label,
              track.semantic_label,
              distance_m,
              protected_label ? "dynamic_residue_protected_label"
                              : "dynamic_residue_protected_horizontal_surface",
          });
          continue;
        }

        const uint64_t earliest_later_stamp =
            track.last_stamp == std::numeric_limits<uint64_t>::max()
                ? track.last_stamp
                : track.last_stamp + 1;
        const auto check =
            verifier.check(mesh.pos(vertex_idx), earliest_later_stamp);
        bool remove_vertex = false;
        if (config_.dynamic_residue_decision == "absence_majority") {
          remove_vertex =
              check.absent.size() >= min_absent &&
              check.absent.size() > check.present.size();
        } else {
          remove_vertex =
              check.absent.size() >= min_absent &&
              check.present.size() <= config_.dynamic_residue_max_present;
        }

        if (remove_vertex) {
          ++result.mesh_summary.dynamic_residue_absent_confirmed_vertices;
          if (object_row) {
            ++object_row->dynamic_residue_absent_confirmed;
          }
          const bool inserted = vertices_to_delete.insert(vertex_idx).second;
          if (inserted) {
            ++deleted_by_object[track.node_id];
            if (!config_.dry_run) {
              ++result.mesh_summary.dynamic_residue_removed_vertices;
              if (object_row) {
                ++object_row->dynamic_residue_removed;
              }
            }
          }
          result.vertex_update_rows.push_back(ReconcileResult::VertexUpdateRow{
              vertex_idx,
              track.node_id,
              vertex_label,
              track.semantic_label,
              distance_m,
              config_.dry_run ? "dynamic_residue_confirmed_dry_run"
                              : "remove_dynamic_residue",
          });
          continue;
        }

        if (!check.present.empty()) {
          ++result.mesh_summary.dynamic_residue_present_rejected_vertices;
          if (object_row) {
            ++object_row->dynamic_residue_present_rejected;
          }
          result.vertex_update_rows.push_back(ReconcileResult::VertexUpdateRow{
              vertex_idx,
              track.node_id,
              vertex_label,
              track.semantic_label,
              distance_m,
              "dynamic_residue_rejected_later_present",
          });
        } else {
          ++result.mesh_summary.dynamic_residue_unobserved_vertices;
          if (object_row) {
            ++object_row->dynamic_residue_unobserved;
          }
          if (config_.dynamic_residue_unobserved_policy == "quarantine") {
            const bool inserted = vertices_to_delete.insert(vertex_idx).second;
            if (inserted) {
              ++result.mesh_summary.dynamic_residue_quarantined_vertices;
              ++deleted_by_object[track.node_id];
              if (!config_.dry_run) {
                ++result.mesh_summary.dynamic_residue_removed_vertices;
                if (object_row) {
                  ++object_row->dynamic_residue_quarantined;
                  ++object_row->dynamic_residue_removed;
                }
              }
            }
            result.vertex_update_rows.push_back(ReconcileResult::VertexUpdateRow{
                vertex_idx,
                track.node_id,
                vertex_label,
                track.semantic_label,
                distance_m,
                config_.dry_run ? "dynamic_residue_quarantine_dry_run"
                                : "quarantine_dynamic_residue",
            });
            continue;
          }
          result.vertex_update_rows.push_back(ReconcileResult::VertexUpdateRow{
              vertex_idx,
              track.node_id,
              vertex_label,
              track.semantic_label,
              distance_m,
              "dynamic_residue_unobserved_keep",
          });
        }
      }
    }
  }

  applyAbsentPriorMemoryObjects(mesh,
                                &vertices_to_delete,
                                &result.vertex_update_rows,
                                &result.object_rows,
                                &result.mesh_summary);

  applyForcedAbsentPriorObjects(prior_objects,
                                synthetic_change,
                                mesh,
                                &vertices_to_delete,
                                &result.vertex_update_rows,
                                &result.object_rows,
                                &result.mesh_summary);
  result.mesh_summary.objects_total = result.object_rows.size();

  result.mesh_summary.candidate_vertices = vertices_to_delete.size();
  result.mesh_summary.removed_vertices = config_.dry_run ? 0 : vertices_to_delete.size();

  for (auto& row : result.object_rows) {
    if (row.session_state == "forced_absent_prior") {
      continue;
    }
    auto candidate_it = candidate_by_object.find(row.node_id);
    if (candidate_it != candidate_by_object.end()) {
      row.vertices_candidate = candidate_it->second;
    }
    auto deleted_it = deleted_by_object.find(row.node_id);
    if (deleted_it != deleted_by_object.end()) {
      row.vertices_removed = config_.dry_run ? 0 : deleted_it->second;
    }
  }

  const std::size_t free_space_requested_start =
      config_.free_space_culling_scope == "objects" ? free_space_object_vertex_start
                                                    : free_space_added_vertex_start;
  std::size_t cleanup_deleted_before_culling_start = 0;
  if (!config_.dry_run) {
    for (const auto vertex_idx : vertices_to_delete) {
      if (vertex_idx < free_space_requested_start) {
        ++cleanup_deleted_before_culling_start;
      }
    }
  }

  if (!config_.dry_run && !vertices_to_delete.empty()) {
    mesh.eraseVertices(vertices_to_delete);
  }
  const std::size_t adjusted_free_space_start =
      free_space_requested_start -
      std::min(free_space_requested_start, cleanup_deleted_before_culling_start);
  applyFreeSpaceCulling(adjusted_free_space_start);

  restorePriorObjectNodes(dsg, &result.object_rows, &result.mesh_summary);

  result.mesh_summary.final_vertices = mesh.numVertices();
  result.mesh_summary.final_faces = mesh.numFaces();
  return result;
}

bool ObjectGuidedMapReconciler::isInsideExpandedBox(const khronos::BoundingBox& box,
                                                    const khronos::Point& point,
                                                    double margin_m) {
  const auto min_corner = box.world_P_center - box.dimensions * 0.5f;
  const auto max_corner = box.world_P_center + box.dimensions * 0.5f;
  return point.x() >= min_corner.x() - margin_m && point.x() <= max_corner.x() + margin_m &&
         point.y() >= min_corner.y() - margin_m && point.y() <= max_corner.y() + margin_m &&
         point.z() >= min_corner.z() - margin_m && point.z() <= max_corner.z() + margin_m;
}

std::vector<ObjectAuditRow> ObjectGuidedMapReconciler::auditObjects(
    const khronos::DynamicSceneGraph& dsg) const {
  std::vector<ObjectAuditRow> rows;
  if (!dsg.hasLayer(khronos::DsgLayers::OBJECTS)) {
    return rows;
  }

  const auto* mesh = dsg.hasMesh() ? dsg.mesh().get() : nullptr;
  const auto& layer = dsg.getLayer(khronos::DsgLayers::OBJECTS);
  rows.reserve(layer.numNodes());

  for (const auto& [id, node] : layer.nodes()) {
    const auto& attrs = node->attributes<khronos::KhronosObjectAttributes>();
    ObjectAuditRow row;
    row.node_id = id;
    row.semantic_label = attrs.semantic_label;
    row.first_observed_ns = firstOrZero(attrs.first_observed_ns);
    row.last_observed_ns = lastOrZero(attrs.last_observed_ns);
    row.bbox_cx = attrs.bounding_box.world_P_center.x();
    row.bbox_cy = attrs.bounding_box.world_P_center.y();
    row.bbox_cz = attrs.bounding_box.world_P_center.z();
    row.bbox_dx = attrs.bounding_box.dimensions.x();
    row.bbox_dy = attrs.bounding_box.dimensions.y();
    row.bbox_dz = attrs.bounding_box.dimensions.z();
    row.bbox_volume = attrs.bounding_box.volume();
    row.object_mesh_vertices = attrs.mesh.numVertices();
    row.object_mesh_faces = attrs.mesh.numFaces();
    row.dynamic_trajectory_points = attrs.trajectory_positions.size();
    row.dynamic_point_frames = attrs.dynamic_object_points.size();
    row.session_state = "current";

    if (mesh) {
      std::vector<khronos::Point> dynamic_points;
      for (const auto& frame : attrs.dynamic_object_points) {
        dynamic_points.insert(dynamic_points.end(), frame.begin(), frame.end());
      }
      const std::unique_ptr<hydra::PointNeighborSearch> dynamic_point_search =
          dynamic_points.empty()
              ? nullptr
              : std::make_unique<hydra::PointNeighborSearch>(dynamic_points);
      const double dynamic_point_threshold_sq =
          config_.object_distance_m * config_.object_distance_m;

      for (std::size_t i = 0; i < mesh->numVertices(); ++i) {
        const auto point = mesh->pos(i);
        if (isInsideExpandedBox(attrs.bounding_box, point, config_.bbox_margin_m)) {
          ++row.global_vertices_in_bbox;
          if (i < mesh->labels.size() &&
              static_cast<int>(mesh->labels[i]) == attrs.semantic_label) {
            ++row.global_vertices_in_bbox_same_label;
          }
        }

        bool inside_dynamic_swept_bbox = false;
        if (!attrs.trajectory_positions.empty()) {
          auto moving_box = attrs.bounding_box;
          for (const auto& position : attrs.trajectory_positions) {
            moving_box.world_P_center = position;
            if (isInsideExpandedBox(moving_box, point, config_.bbox_margin_m)) {
              inside_dynamic_swept_bbox = true;
              break;
            }
          }
        }
        if (!inside_dynamic_swept_bbox) {
          continue;
        }

        ++row.global_vertices_in_dynamic_swept_bbox;
        if (i < mesh->labels.size() &&
            static_cast<int>(mesh->labels[i]) == attrs.semantic_label) {
          ++row.global_vertices_in_dynamic_swept_bbox_same_label;
        }
        if (mesh->has_timestamps && !attrs.trajectory_timestamps.empty()) {
          const auto stamp = mesh->timestamp(i);
          if (stamp >= attrs.trajectory_timestamps.front() &&
              stamp <= attrs.trajectory_timestamps.back()) {
            ++row.global_vertices_in_dynamic_swept_bbox_time_overlap;
          }
        }

        if (dynamic_point_search) {
          float distance_sq = std::numeric_limits<float>::max();
          std::size_t nearest_idx = 0;
          if (dynamic_point_search->search(point, distance_sq, nearest_idx) &&
              nearest_idx < dynamic_points.size() &&
              distance_sq <= dynamic_point_threshold_sq) {
            ++row.global_vertices_near_dynamic_points;
            if (i < mesh->labels.size() &&
                static_cast<int>(mesh->labels[i]) == attrs.semantic_label) {
              ++row.global_vertices_near_dynamic_points_same_label;
            }
            if (mesh->has_timestamps && !attrs.trajectory_timestamps.empty()) {
              const auto stamp = mesh->timestamp(i);
              if (stamp >= attrs.trajectory_timestamps.front() &&
                  stamp <= attrs.trajectory_timestamps.back()) {
                ++row.global_vertices_near_dynamic_points_time_overlap;
              }
            }
          }
        }
      }
    }

    rows.push_back(row);
  }

  return rows;
}

std::vector<ObjectGuidedMapReconciler::PriorObject>
ObjectGuidedMapReconciler::loadPriorObjects() const {
  std::vector<PriorObject> objects;
  if (config_.prior_object_memory.empty()) {
    return objects;
  }

  std::ifstream in(config_.prior_object_memory);
  if (!in) {
    return objects;
  }

  const auto root = nlohmann::json::parse(in, nullptr, false);
  if (root.is_discarded() || !root.contains("objects") || !root["objects"].is_array()) {
    return objects;
  }

  for (const auto& item : root["objects"]) {
    PriorObject obj;
    obj.object_id = item.value("object_id", 0ul);
    obj.semantic_label = item.value("semantic_label", -1);
    obj.object_mesh_vertices = item.value("object_mesh_vertices", 0ul);
    obj.stationarity_alpha = item.value("prior_stationarity_alpha", 2.0);
    obj.stationarity_beta = item.value("prior_stationarity_beta", 1.0);
    obj.session_state = item.value("session_state", std::string());

    if (item.contains("bbox_center") && item["bbox_center"].is_array() &&
        item["bbox_center"].size() >= 3) {
      obj.bbox_cx = item["bbox_center"][0].get<float>();
      obj.bbox_cy = item["bbox_center"][1].get<float>();
      obj.bbox_cz = item["bbox_center"][2].get<float>();
    }
    if (item.contains("bbox_dimensions") && item["bbox_dimensions"].is_array() &&
        item["bbox_dimensions"].size() >= 3) {
      obj.bbox_dx = item["bbox_dimensions"][0].get<float>();
      obj.bbox_dy = item["bbox_dimensions"][1].get<float>();
      obj.bbox_dz = item["bbox_dimensions"][2].get<float>();
    }

    objects.push_back(obj);
  }

  return objects;
}

std::unordered_map<khronos::NodeId, ObjectGuidedMapReconciler::ObjectChangeEvidence>
ObjectGuidedMapReconciler::loadObjectChangeEvidence() const {
  std::unordered_map<khronos::NodeId, ObjectChangeEvidence> evidence_by_id;
  if (config_.object_changes_csv.empty()) {
    return evidence_by_id;
  }

  std::ifstream in(config_.object_changes_csv);
  if (!in) {
    return evidence_by_id;
  }

  std::string line;
  std::getline(in, line);
  while (std::getline(in, line)) {
    std::vector<std::string> values;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) {
      values.push_back(item);
    }
    if (values.size() < 6) {
      continue;
    }

    try {
      const auto node_id = static_cast<khronos::NodeId>(std::stoull(values[0]));
      ObjectChangeEvidence evidence;
      evidence.first_absent_ns = std::stoull(values[2]);
      evidence.last_absent_ns = std::stoull(values[3]);
      evidence.first_persistent_ns = std::stoull(values[4]);
      evidence.last_persistent_ns = std::stoull(values[5]);

      const bool has_absent =
          evidence.first_absent_ns > 0 || evidence.last_absent_ns > 0;
      const bool has_persistent =
          evidence.first_persistent_ns > 0 || evidence.last_persistent_ns > 0;
      const auto absent_time_ns =
          evidence.last_absent_ns > 0 ? evidence.last_absent_ns : evidence.first_absent_ns;
      const auto persistent_time_ns = evidence.last_persistent_ns > 0
                                          ? evidence.last_persistent_ns
                                          : evidence.first_persistent_ns;

      double logit = -2.0;
      if (has_absent && has_persistent) {
        const double delta_s =
            (static_cast<double>(absent_time_ns) -
             static_cast<double>(persistent_time_ns)) /
            1.0e9;
        const double scale_s = std::max(config_.object_move_time_scale_s, 1.0);
        const double normalized = std::clamp(delta_s / scale_s, -2.0, 2.0);
        logit += 3.0 * normalized;
      } else if (has_absent) {
        logit += 4.0;
      } else if (has_persistent) {
        logit -= 2.0;
      }

      if (evidence.last_absent_ns > evidence.last_persistent_ns &&
          evidence.last_absent_ns > 0) {
        logit += 1.0;
      }
      if (evidence.last_persistent_ns >= evidence.last_absent_ns &&
          evidence.last_persistent_ns > 0) {
        logit -= 1.0;
      }

      if (logit >= 0.0) {
        evidence.move_probability = 1.0 / (1.0 + std::exp(-logit));
      } else {
        const double e = std::exp(logit);
        evidence.move_probability = e / (1.0 + e);
      }

      if (config_.object_move_decision == "probability") {
        evidence.skip_injection =
            evidence.move_probability >= config_.object_move_skip_probability;
      } else if (config_.object_move_decision == "hard") {
        evidence.skip_injection =
            evidence.last_absent_ns > evidence.last_persistent_ns;
      } else {
        evidence.skip_injection = false;
      }

      evidence_by_id[node_id] = evidence;
    } catch (const std::exception&) {
      continue;
    }
  }

  return evidence_by_id;
}

ObjectGuidedMapReconciler::SyntheticChangeSpec
ObjectGuidedMapReconciler::loadSyntheticChangeSpec() const {
  SyntheticChangeSpec spec;
  if (config_.synthetic_change_file.empty()) {
    return spec;
  }

  std::ifstream in(config_.synthetic_change_file);
  if (!in) {
    return spec;
  }

  const auto root = nlohmann::json::parse(in, nullptr, false);
  if (root.is_discarded()) {
    return spec;
  }

  spec.delete_global_vertices_in_forced_absent_bbox =
      root.value("delete_global_vertices_in_forced_absent_bbox", true);
  if (root.contains("force_absent_prior_object_ids") &&
      root["force_absent_prior_object_ids"].is_array()) {
    for (const auto& item : root["force_absent_prior_object_ids"]) {
      if (item.is_number_unsigned() || item.is_number_integer()) {
        spec.force_absent_prior_object_ids.insert(item.get<khronos::NodeId>());
      }
    }
  }

  return spec;
}

khronos::BoundingBox ObjectGuidedMapReconciler::makeBox(const PriorObject& object) {
  khronos::BoundingBox box;
  box.world_P_center =
      khronos::Point(object.bbox_cx, object.bbox_cy, object.bbox_cz);
  box.dimensions = khronos::Point(object.bbox_dx, object.bbox_dy, object.bbox_dz);
  return box;
}

void ObjectGuidedMapReconciler::applyForcedAbsentPriorObjects(
    const std::vector<PriorObject>& prior_objects,
    const SyntheticChangeSpec& synthetic_change,
    spark_dsg::Mesh& mesh,
    std::unordered_set<std::size_t>* vertices_to_delete,
    std::vector<ReconcileResult::VertexUpdateRow>* update_rows,
    std::vector<ObjectAuditRow>* object_rows,
    MeshUpdateSummary* summary) const {
  if (!vertices_to_delete || !update_rows || !object_rows || !summary ||
      !synthetic_change.delete_global_vertices_in_forced_absent_bbox ||
      synthetic_change.force_absent_prior_object_ids.empty()) {
    return;
  }

  for (const auto& prior : prior_objects) {
    if (!synthetic_change.force_absent_prior_object_ids.count(prior.object_id)) {
      continue;
    }

    const auto box = makeBox(prior);
    std::size_t object_candidates = 0;
    std::size_t object_new_deletions = 0;
    for (std::size_t vertex_idx = 0; vertex_idx < mesh.numVertices(); ++vertex_idx) {
      const auto point = mesh.pos(vertex_idx);
      if (!isInsideExpandedBox(box, point, config_.bbox_margin_m)) {
        continue;
      }

      ++object_candidates;
      const bool inserted = vertices_to_delete->insert(vertex_idx).second;
      if (inserted) {
        ++object_new_deletions;
      }

      const int vertex_label =
          vertex_idx < mesh.labels.size() ? static_cast<int>(mesh.labels[vertex_idx]) : -1;
      update_rows->push_back(ReconcileResult::VertexUpdateRow{
          vertex_idx,
          prior.object_id,
          vertex_label,
          prior.semantic_label,
          0.0,
          config_.dry_run ? "forced_absent_candidate_dry_run" : "forced_absent_remove"});
    }

    ObjectAuditRow row;
    row.node_id = prior.object_id;
    row.semantic_label = prior.semantic_label;
    row.bbox_cx = prior.bbox_cx;
    row.bbox_cy = prior.bbox_cy;
    row.bbox_cz = prior.bbox_cz;
    row.bbox_dx = prior.bbox_dx;
    row.bbox_dy = prior.bbox_dy;
    row.bbox_dz = prior.bbox_dz;
    row.bbox_volume = prior.bbox_dx * prior.bbox_dy * prior.bbox_dz;
    row.object_mesh_vertices = prior.object_mesh_vertices;
    row.global_vertices_in_bbox = object_candidates;
    row.vertices_candidate = object_candidates;
    row.vertices_removed = config_.dry_run ? 0 : object_new_deletions;
    row.prior_matched = false;
    row.prior_match_object_id = prior.object_id;
    row.session_state = "forced_absent_prior";
    object_rows->push_back(row);

    ++summary->forced_absent_prior_objects;
    summary->forced_absent_vertices_removed += config_.dry_run ? 0 : object_new_deletions;
  }
}

void ObjectGuidedMapReconciler::applyAbsentPriorMemoryObjects(
    spark_dsg::Mesh& mesh,
    std::unordered_set<std::size_t>* vertices_to_delete,
    std::vector<ReconcileResult::VertexUpdateRow>* update_rows,
    std::vector<ObjectAuditRow>* object_rows,
    MeshUpdateSummary* summary) const {
  if (!vertices_to_delete || !update_rows || !object_rows || !summary) {
    return;
  }

  for (auto& row : *object_rows) {
    if (row.session_state != "absent_prior_conflict") {
      continue;
    }

    khronos::BoundingBox box;
    box.world_P_center = khronos::Point(row.bbox_cx, row.bbox_cy, row.bbox_cz);
    box.dimensions = khronos::Point(row.bbox_dx, row.bbox_dy, row.bbox_dz);

    std::size_t object_candidates = 0;
    std::size_t object_new_deletions = 0;
    for (std::size_t vertex_idx = 0; vertex_idx < mesh.numVertices(); ++vertex_idx) {
      const auto point = mesh.pos(vertex_idx);
      if (!isInsideExpandedBox(box, point, config_.bbox_margin_m)) {
        continue;
      }

      const int vertex_label =
          vertex_idx < mesh.labels.size() ? static_cast<int>(mesh.labels[vertex_idx]) : -1;
      if (row.semantic_label >= 0 && vertex_label >= 0 && vertex_label != row.semantic_label) {
        continue;
      }

      ++object_candidates;
      if (vertices_to_delete->insert(vertex_idx).second) {
        ++object_new_deletions;
      }

      update_rows->push_back(ReconcileResult::VertexUpdateRow{
          vertex_idx,
          row.node_id,
          vertex_label,
          row.semantic_label,
          0.0,
          config_.dry_run ? "prior_absent_candidate_dry_run" : "prior_absent_remove"});
    }

    row.global_vertices_in_bbox = object_candidates;
    row.global_vertices_in_bbox_same_label = object_candidates;
    row.vertices_candidate = object_candidates;
    row.vertices_removed = config_.dry_run ? 0 : object_new_deletions;
    summary->prior_absent_vertices_removed += config_.dry_run ? 0 : object_new_deletions;
  }
}

void ObjectGuidedMapReconciler::applyPriorObjectMemory(
    const khronos::DynamicSceneGraph& dsg,
    std::vector<ObjectAuditRow>* object_rows,
    MeshUpdateSummary* summary) const {
  if (!object_rows || !summary || config_.prior_object_memory.empty()) {
    return;
  }

  const auto prior_objects = loadPriorObjects();
  summary->prior_memory_objects = prior_objects.size();
  if (prior_objects.empty()) {
    for (auto& row : *object_rows) {
      row.session_state = "current_no_prior";
    }
    summary->prior_unmatched_current_objects = object_rows->size();
    return;
  }

  const double threshold_sq = config_.prior_match_distance_m * config_.prior_match_distance_m;
  std::vector<std::tuple<double, std::size_t, std::size_t>> candidates;
  for (std::size_t row_idx = 0; row_idx < object_rows->size(); ++row_idx) {
    const auto& row = object_rows->at(row_idx);
    for (std::size_t prior_idx = 0; prior_idx < prior_objects.size(); ++prior_idx) {
      const auto& prior = prior_objects[prior_idx];
      if (prior.semantic_label != row.semantic_label) {
        continue;
      }
      const double dx = static_cast<double>(prior.bbox_cx) - row.bbox_cx;
      const double dy = static_cast<double>(prior.bbox_cy) - row.bbox_cy;
      const double dz = static_cast<double>(prior.bbox_cz) - row.bbox_cz;
      const double distance_sq = dx * dx + dy * dy + dz * dz;
      if (distance_sq <= threshold_sq) {
        candidates.emplace_back(distance_sq, row_idx, prior_idx);
      }
    }
  }

  std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
    return std::get<0>(lhs) < std::get<0>(rhs);
  });

  std::vector<bool> row_used(object_rows->size(), false);
  std::vector<bool> prior_used(prior_objects.size(), false);
  for (const auto& [distance_sq, row_idx, prior_idx] : candidates) {
    if (row_used[row_idx] || prior_used[prior_idx]) {
      continue;
    }
    auto& row = object_rows->at(row_idx);
    const auto& prior = prior_objects[prior_idx];
    row.prior_matched = true;
    row.prior_match_object_id = prior.object_id;
    row.prior_match_distance_m = std::sqrt(distance_sq);
    row.prior_present_evidence = 1;
    row.prior_stationarity_alpha =
        (prior.stationarity_alpha > 0.0 ? prior.stationarity_alpha : 2.0) + 1.0;
    row.prior_stationarity_beta =
        prior.stationarity_beta > 0.0 ? prior.stationarity_beta : 1.0;
    row.prior_stationarity_mean =
        row.prior_stationarity_alpha /
        (row.prior_stationarity_alpha + row.prior_stationarity_beta);
    row.session_state = "persistent_prior_matched";
    row_used[row_idx] = true;
    prior_used[prior_idx] = true;
    ++summary->prior_matched_objects;
  }

  for (auto& row : *object_rows) {
    if (row.prior_matched) {
      row.session_state = "persistent_prior_matched";
    } else {
      row.session_state = "new_or_moved_no_prior_match";
      ++summary->prior_unmatched_current_objects;
    }
  }

  for (std::size_t prior_idx = 0; prior_idx < prior_objects.size(); ++prior_idx) {
    if (prior_used[prior_idx]) {
      continue;
    }
    const auto& prior = prior_objects[prior_idx];
    const auto prior_box = makeBox(prior);
    const auto evidence = evaluatePriorMemoryEvidence(
        prior_box, dsg, config_, prior.stationarity_alpha, prior.stationarity_beta);

    ObjectAuditRow row;
    row.node_id = prior.object_id;
    row.semantic_label = prior.semantic_label;
    row.bbox_cx = prior.bbox_cx;
    row.bbox_cy = prior.bbox_cy;
    row.bbox_cz = prior.bbox_cz;
    row.bbox_dx = prior.bbox_dx;
    row.bbox_dy = prior.bbox_dy;
    row.bbox_dz = prior.bbox_dz;
    row.bbox_volume = prior.bbox_dx * prior.bbox_dy * prior.bbox_dz;
    row.object_mesh_vertices = prior.object_mesh_vertices;
    row.prior_matched = false;
    row.prior_match_object_id = prior.object_id;
    row.prior_absent_evidence = evidence.absent_votes;
    row.prior_present_evidence = evidence.present_votes;
    row.prior_stationarity_alpha = evidence.alpha;
    row.prior_stationarity_beta = evidence.beta;
    row.prior_stationarity_mean = evidence.mean();

    constexpr double kStationarityAbsentTheta = 0.4;
    if (evidence.absent_votes > 0 && evidence.mean() < kStationarityAbsentTheta) {
      row.session_state = "absent_prior_conflict";
      ++summary->prior_absent_objects;
    } else if (evidence.present_votes > 0) {
      row.session_state = "prior_supported_no_object_match";
      ++summary->prior_unobserved_objects;
    } else {
      row.session_state = "unobserved_prior_memory";
      ++summary->prior_unobserved_objects;
    }
    object_rows->push_back(row);
  }
}

void ObjectGuidedMapReconciler::restorePriorObjectNodes(
    khronos::DynamicSceneGraph& dsg,
    std::vector<ObjectAuditRow>* object_rows,
    MeshUpdateSummary* summary) const {
  if (!object_rows || !summary || config_.dynamic_mode != "cross_session" ||
      !config_.prior_session_dsg ||
      !config_.prior_session_dsg->hasLayer(khronos::DsgLayers::OBJECTS)) {
    return;
  }

  std::size_t next_object_id = 0;
  if (dsg.hasLayer(khronos::DsgLayers::OBJECTS)) {
    for (const auto& [id, unused] : dsg.getLayer(khronos::DsgLayers::OBJECTS).nodes()) {
      (void)unused;
      next_object_id =
          std::max(next_object_id, spark_dsg::NodeSymbol(id).categoryId());
    }
  }

  for (auto& row : *object_rows) {
    if (row.session_state != "prior_supported_no_object_match" &&
        row.session_state != "unobserved_prior_memory") {
      continue;
    }

    const auto* prior_node =
        config_.prior_session_dsg->findNode(row.prior_match_object_id);
    if (!prior_node) {
      continue;
    }
    const auto* attrs = prior_node->tryAttributes<khronos::KhronosObjectAttributes>();
    if (!attrs) {
      continue;
    }

    khronos::NodeId target_id = row.prior_match_object_id;
    if (dsg.hasNode(target_id)) {
      do {
        target_id = spark_dsg::NodeSymbol('O', ++next_object_id);
      } while (dsg.hasNode(target_id));
    } else {
      next_object_id = std::max(
          next_object_id, spark_dsg::NodeSymbol(target_id).categoryId());
    }

    if (!dsg.emplaceNode(khronos::DsgLayers::OBJECTS,
                         target_id,
                         std::make_unique<khronos::KhronosObjectAttributes>(*attrs))) {
      continue;
    }

    row.node_id = target_id;
    ++summary->prior_restored_objects;
  }
}

std::vector<ObjectGuidedMapReconciler::CleanupSource>
ObjectGuidedMapReconciler::collectCleanupSources(
    const khronos::DynamicSceneGraph& dsg,
    std::vector<khronos::Point>* source_points,
    std::vector<std::size_t>* point_to_source,
    std::vector<ObjectAuditRow>* object_rows) const {
  std::vector<CleanupSource> sources;
  if (!dsg.hasLayer(khronos::DsgLayers::OBJECTS)) {
    return sources;
  }

  const auto& layer = dsg.getLayer(khronos::DsgLayers::OBJECTS);
  for (const auto& [id, node] : layer.nodes()) {
    const auto& attrs = node->attributes<khronos::KhronosObjectAttributes>();
    if (attrs.mesh.numVertices() < config_.min_object_mesh_vertices) {
      continue;
    }
    if (object_rows) {
      const auto row_it =
          std::find_if(object_rows->begin(), object_rows->end(), [id](const auto& row) {
            return row.node_id == id;
          });
      if (row_it != object_rows->end()) {
        const double global_ratio =
            row_it->object_mesh_vertices == 0
                ? 0.0
                : static_cast<double>(row_it->global_vertices_in_bbox) /
                      static_cast<double>(row_it->object_mesh_vertices);
        if (global_ratio < config_.repair_global_vertex_ratio_threshold) {
          continue;
        }
      }
    }

    CleanupSource source;
    source.node_id = id;
    source.semantic_label = attrs.semantic_label;
    source.bounding_box = attrs.bounding_box;
    source.first_point_index = source_points ? source_points->size() : 0;
    for (const auto& point_box : attrs.mesh.points) {
      if (source_points) {
        source_points->push_back(attrs.bounding_box.pointToWorldFrame(point_box));
      }
      if (point_to_source) {
        point_to_source->push_back(sources.size());
      }
    }
    source.num_points = attrs.mesh.numVertices();

    if (source.num_points > 0) {
      sources.push_back(std::move(source));
    }
  }

  if (object_rows) {
    for (auto& row : *object_rows) {
      const bool used =
          std::any_of(sources.begin(), sources.end(), [&row](const CleanupSource& source) {
            return source.node_id == row.node_id;
          });
      if (!used) {
        row.vertices_removed = 0;
      }
    }
  }

  return sources;
}

bool saveMapWithSingleDsg(const khronos::DynamicSceneGraph::Ptr& dsg,
                          khronos::TimeStamp stamp,
                          const std::string& path) {
  khronos::SpatioTemporalMap::Config config;
  config.finalize_incrementally = true;
  khronos::SpatioTemporalMap map(config);
  map.update(dsg, stamp);
  return map.save(path);
}

void writeObjectAuditCsv(const std::string& path, const std::vector<ObjectAuditRow>& rows) {
  std::ofstream out(path);
  out << "object_id,semantic_label,first_observed_ns,last_observed_ns,bbox_cx,bbox_cy,bbox_cz,"
         "bbox_dx,bbox_dy,bbox_dz,bbox_volume,object_mesh_vertices,object_mesh_faces,"
         "dynamic_trajectory_points,dynamic_point_frames,global_vertices_in_bbox,"
         "global_vertices_in_bbox_same_label,global_vertices_in_dynamic_swept_bbox,"
         "global_vertices_in_dynamic_swept_bbox_same_label,"
         "global_vertices_in_dynamic_swept_bbox_time_overlap,"
         "global_vertices_near_dynamic_points,"
         "global_vertices_near_dynamic_points_same_label,"
         "global_vertices_near_dynamic_points_time_overlap,"
         "dynamic_residue_candidates,dynamic_residue_protected,"
         "dynamic_residue_absent_confirmed,dynamic_residue_present_rejected,"
         "dynamic_residue_unobserved,dynamic_residue_quarantined,"
         "dynamic_residue_removed,"
         "vertices_candidate,vertices_removed,"
         "repair_candidate,repair_candidate_vertices,change_first_absent_ns,"
         "change_last_absent_ns,change_first_persistent_ns,change_last_persistent_ns,"
         "object_move_probability,skipped_by_object_move,object_alignment_applied,"
         "object_alignment_support_vertices,object_alignment_translation_norm_m,"
         "object_alignment_before_median_m,object_alignment_after_median_m,"
         "prior_matched,prior_match_object_id,prior_match_distance_m,"
         "prior_absent_evidence,prior_present_evidence,prior_stationarity_alpha,"
         "prior_stationarity_beta,prior_stationarity_mean,session_state\n";
  for (const auto& row : rows) {
    out << row.node_id << "," << row.semantic_label << "," << row.first_observed_ns << ","
        << row.last_observed_ns << "," << row.bbox_cx << "," << row.bbox_cy << ","
        << row.bbox_cz << "," << row.bbox_dx << "," << row.bbox_dy << "," << row.bbox_dz
        << "," << row.bbox_volume << "," << row.object_mesh_vertices << ","
        << row.object_mesh_faces << "," << row.dynamic_trajectory_points << ","
        << row.dynamic_point_frames << "," << row.global_vertices_in_bbox << ","
        << row.global_vertices_in_bbox_same_label << ","
        << row.global_vertices_in_dynamic_swept_bbox << ","
        << row.global_vertices_in_dynamic_swept_bbox_same_label << ","
        << row.global_vertices_in_dynamic_swept_bbox_time_overlap << ","
        << row.global_vertices_near_dynamic_points << ","
        << row.global_vertices_near_dynamic_points_same_label << ","
        << row.global_vertices_near_dynamic_points_time_overlap << ","
        << row.dynamic_residue_candidates << ","
        << row.dynamic_residue_protected << ","
        << row.dynamic_residue_absent_confirmed << ","
        << row.dynamic_residue_present_rejected << ","
        << row.dynamic_residue_unobserved << ","
        << row.dynamic_residue_quarantined << ","
        << row.dynamic_residue_removed << ","
        << row.vertices_candidate << ","
        << row.vertices_removed << "," << (row.repair_candidate ? "true" : "false") << ","
        << row.repair_candidate_vertices << "," << row.change_first_absent_ns << ","
        << row.change_last_absent_ns << "," << row.change_first_persistent_ns << ","
        << row.change_last_persistent_ns << "," << row.object_move_probability << ","
        << (row.skipped_by_object_move ? "true" : "false") << ","
        << (row.object_alignment_applied ? "true" : "false") << ","
        << row.object_alignment_support_vertices << ","
        << row.object_alignment_translation_norm_m << ","
        << row.object_alignment_before_median_m << ","
        << row.object_alignment_after_median_m << ","
        << (row.prior_matched ? "true" : "false") << "," << row.prior_match_object_id
        << "," << row.prior_match_distance_m << "," << row.prior_absent_evidence
        << "," << row.prior_present_evidence << "," << row.prior_stationarity_alpha
        << "," << row.prior_stationarity_beta << "," << row.prior_stationarity_mean
        << "," << row.session_state << "\n";
  }
}

void writeMeshUpdateCsv(const std::string& path, const MeshUpdateSummary& summary) {
  std::ofstream out(path);
  out << "initial_vertices,initial_faces,final_vertices,final_faces,candidate_vertices,"
         "removed_vertices,objects_total,objects_with_private_mesh,objects_used_for_cleanup,"
         "cleanup_source_points,object_cleanup_present_protected_vertices,"
         "object_cleanup_unobserved_protected_vertices,"
         "object_cleanup_absent_confirmed_vertices,"
         "repair_candidate_objects,repair_candidate_vertices,"
         "injected_objects,injected_vertices,injected_faces,object_alignment_candidates,"
         "object_alignment_applied,object_alignment_support_vertices,prior_memory_objects,"
         "temporal_background_sources,temporal_background_injected_vertices,"
         "temporal_background_injected_faces,temporal_object_sources,"
         "temporal_object_injected_vertices,temporal_object_injected_faces,"
         "horizontal_planes_filled,horizontal_plane_vertices,horizontal_plane_faces,"
         "horizontal_plane_graph_cut_cells,horizontal_plane_graph_cut_fill_cells,"
         "axis_aligned_planes_filled,axis_aligned_plane_vertices,axis_aligned_plane_faces,"
         "axis_aligned_plane_graph_cut_cells,axis_aligned_plane_graph_cut_fill_cells,"
         "free_space_checked_vertices,free_space_absent_vertices,"
         "free_space_present_vertices,free_space_removed_vertices,"
         "dynamic_residue_tracks,dynamic_residue_support_points,"
         "dynamic_residue_candidates,dynamic_residue_protected_vertices,"
         "dynamic_residue_absent_confirmed_vertices,"
         "dynamic_residue_present_rejected_vertices,"
         "dynamic_residue_unobserved_vertices,dynamic_residue_quarantined_vertices,"
         "dynamic_residue_removed_vertices,"
         "volume_graph_cut_cells,volume_graph_cut_free_evidence_cells,"
         "volume_graph_cut_full_evidence_cells,volume_graph_cut_structural_cells,"
         "volume_graph_cut_full_cells,volume_graph_cut_vertices,volume_graph_cut_faces,"
         "prior_matched_objects,prior_absent_objects,prior_unobserved_objects,"
         "prior_unmatched_current_objects,prior_restored_objects,"
         "prior_absent_vertices_removed,"
         "forced_absent_prior_objects,"
         "forced_absent_vertices_removed,"
         "cross_session_prior_vertices,cross_session_prior_faces,"
         "cross_session_current_vertices,cross_session_current_faces,"
         "cross_session_prior_checked_vertices,cross_session_prior_absent_vertices,"
         "cross_session_prior_persistent_vertices,cross_session_prior_unobserved_vertices,"
         "cross_session_current_injected_vertices,cross_session_current_injected_faces\n";
  out << summary.initial_vertices << "," << summary.initial_faces << "," << summary.final_vertices
      << "," << summary.final_faces << "," << summary.candidate_vertices << ","
      << summary.removed_vertices << "," << summary.objects_total << ","
      << summary.objects_with_private_mesh << "," << summary.objects_used_for_cleanup << ","
      << summary.cleanup_source_points << ","
      << summary.object_cleanup_present_protected_vertices << ","
      << summary.object_cleanup_unobserved_protected_vertices << ","
      << summary.object_cleanup_absent_confirmed_vertices << ","
      << summary.repair_candidate_objects << ","
      << summary.repair_candidate_vertices << "," << summary.injected_objects << ","
      << summary.injected_vertices << "," << summary.injected_faces << ","
      << summary.object_alignment_candidates << ","
      << summary.object_alignment_applied << ","
      << summary.object_alignment_support_vertices << ","
      << summary.prior_memory_objects << "," << summary.temporal_background_sources << ","
      << summary.temporal_background_injected_vertices << ","
      << summary.temporal_background_injected_faces << ","
      << summary.temporal_object_sources << ","
      << summary.temporal_object_injected_vertices << ","
      << summary.temporal_object_injected_faces << ","
      << summary.horizontal_planes_filled << ","
      << summary.horizontal_plane_vertices << ","
      << summary.horizontal_plane_faces << ","
      << summary.horizontal_plane_graph_cut_cells << ","
      << summary.horizontal_plane_graph_cut_fill_cells << ","
      << summary.axis_aligned_planes_filled << ","
      << summary.axis_aligned_plane_vertices << ","
      << summary.axis_aligned_plane_faces << ","
      << summary.axis_aligned_plane_graph_cut_cells << ","
      << summary.axis_aligned_plane_graph_cut_fill_cells << ","
      << summary.free_space_checked_vertices << ","
      << summary.free_space_absent_vertices << ","
      << summary.free_space_present_vertices << ","
      << summary.free_space_removed_vertices << ","
      << summary.dynamic_residue_tracks << ","
      << summary.dynamic_residue_support_points << ","
      << summary.dynamic_residue_candidates << ","
      << summary.dynamic_residue_protected_vertices << ","
      << summary.dynamic_residue_absent_confirmed_vertices << ","
      << summary.dynamic_residue_present_rejected_vertices << ","
      << summary.dynamic_residue_unobserved_vertices << ","
      << summary.dynamic_residue_quarantined_vertices << ","
      << summary.dynamic_residue_removed_vertices << ","
      << summary.volume_graph_cut_cells << ","
      << summary.volume_graph_cut_free_evidence_cells << ","
      << summary.volume_graph_cut_full_evidence_cells << ","
      << summary.volume_graph_cut_structural_cells << ","
      << summary.volume_graph_cut_full_cells << ","
      << summary.volume_graph_cut_vertices << ","
      << summary.volume_graph_cut_faces << ","
      << summary.prior_matched_objects << ","
      << summary.prior_absent_objects << "," << summary.prior_unobserved_objects
      << "," << summary.prior_unmatched_current_objects << ","
      << summary.prior_restored_objects << ","
      << summary.prior_absent_vertices_removed << ","
      << summary.forced_absent_prior_objects << ","
      << summary.forced_absent_vertices_removed << ","
      << summary.cross_session_prior_vertices << ","
      << summary.cross_session_prior_faces << ","
      << summary.cross_session_current_vertices << ","
      << summary.cross_session_current_faces << ","
      << summary.cross_session_prior_checked_vertices << ","
      << summary.cross_session_prior_absent_vertices << ","
      << summary.cross_session_prior_persistent_vertices << ","
      << summary.cross_session_prior_unobserved_vertices << ","
      << summary.cross_session_current_injected_vertices << ","
      << summary.cross_session_current_injected_faces << "\n";
}

void writeObjectUpdateCsv(const std::string& path, const std::vector<ObjectAuditRow>& rows) {
  writeObjectAuditCsv(path, rows);
}

void writeMeshVertexUpdateCsv(const std::string& path,
                              const std::vector<ReconcileResult::VertexUpdateRow>& rows) {
  std::ofstream out(path);
  out << "vertex_index,object_id,vertex_label,object_label,distance_m,decision\n";
  for (const auto& row : rows) {
    out << row.vertex_index << "," << row.object_id << "," << row.vertex_label << ","
        << row.object_label << "," << row.distance_m << "," << row.decision << "\n";
  }
}

void writeEvidenceSummaryJson(const std::string& path,
                              const ReconcilerConfig& config,
                              const MeshUpdateSummary& summary,
                              bool prior_map_loaded,
                              std::size_t prior_memory_objects) {
  std::ofstream out(path);
  out << "{\n";
  out << "  \"mode\": \"" << config.mode << "\",\n";
  out << "  \"dynamic_mode\": \"" << config.dynamic_mode << "\",\n";
  out << "  \"prior_map\": \"" << config.prior_map << "\",\n";
  out << "  \"prior_object_memory\": \"" << config.prior_object_memory << "\",\n";
  out << "  \"object_changes_csv\": \"" << config.object_changes_csv << "\",\n";
  out << "  \"object_move_decision\": \"" << config.object_move_decision << "\",\n";
  out << "  \"object_injection_policy\": \"" << config.object_injection_policy << "\",\n";
  out << "  \"object_surface_support_distance_m\": "
      << config.object_surface_support_distance_m << ",\n";
  out << "  \"object_alignment_policy\": \"" << config.object_alignment_policy << "\",\n";
  out << "  \"object_alignment_support_distance_m\": "
      << config.object_alignment_support_distance_m << ",\n";
  out << "  \"object_alignment_max_translation_m\": "
      << config.object_alignment_max_translation_m << ",\n";
  out << "  \"object_alignment_min_support_vertices\": "
      << config.object_alignment_min_support_vertices << ",\n";
  out << "  \"synthetic_change_file\": \"" << config.synthetic_change_file << "\",\n";
  out << "  \"prior_map_loaded\": " << (prior_map_loaded ? "true" : "false") << ",\n";
  out << "  \"prior_memory_objects\": " << prior_memory_objects << ",\n";
  out << "  \"output_scope\": \"final_current_map_single_timestep\",\n";
  out << "  \"object_distance_m\": " << config.object_distance_m << ",\n";
  out << "  \"bbox_margin_m\": " << config.bbox_margin_m << ",\n";
  out << "  \"injection_min_separation_m\": " << config.injection_min_separation_m << ",\n";
  out << "  \"injection_hole_radius_m\": " << config.injection_min_separation_m << ",\n";
  out << "  \"temporal_background_repair\": "
      << (config.temporal_background_repair ? "true" : "false") << ",\n";
  out << "  \"temporal_background_min_separation_m\": "
      << config.temporal_background_min_separation_m << ",\n";
  out << "  \"temporal_object_repair\": "
      << (config.temporal_object_repair ? "true" : "false") << ",\n";
  out << "  \"temporal_object_min_separation_m\": "
      << config.temporal_object_min_separation_m << ",\n";
  out << "  \"horizontal_plane_fill\": "
      << (config.horizontal_plane_fill ? "true" : "false") << ",\n";
  out << "  \"horizontal_plane_fill_mode\": \""
      << config.horizontal_plane_fill_mode << "\",\n";
  out << "  \"horizontal_plane_candidate_policy\": \""
      << config.horizontal_plane_candidate_policy << "\",\n";
  out << "  \"horizontal_plane_support_band_cells\": "
      << config.horizontal_plane_support_band_cells << ",\n";
  out << "  \"axis_aligned_plane_fill\": "
      << (config.axis_aligned_plane_fill ? "true" : "false") << ",\n";
  out << "  \"axis_aligned_plane_support_band_cells\": "
      << config.axis_aligned_plane_support_band_cells << ",\n";
  out << "  \"axis_aligned_plane_candidate_policy\": \""
      << config.axis_aligned_plane_candidate_policy << "\",\n";
  out << "  \"structural_plane_visibility_filter\": "
      << (config.structural_plane_visibility_filter ? "true" : "false") << ",\n";
  out << "  \"structural_plane_visibility_scope\": \""
      << config.structural_plane_visibility_scope << "\",\n";
  out << "  \"structural_plane_output_supersample\": "
      << (config.structural_plane_output_supersample ? "true" : "false") << ",\n";
  out << "  \"structural_plane_output_supersample_scope\": \""
      << config.structural_plane_output_supersample_scope << "\",\n";
  out << "  \"horizontal_plane_grid_resolution_m\": "
      << config.horizontal_plane_grid_resolution_m << ",\n";
  out << "  \"horizontal_plane_min_support_vertices\": "
      << config.horizontal_plane_min_support_vertices << ",\n";
  out << "  \"free_space_culling\": "
      << (config.free_space_culling ? "true" : "false") << ",\n";
  out << "  \"free_space_culling_scope\": \""
      << config.free_space_culling_scope << "\",\n";
  out << "  \"free_space_culling_decision\": \""
      << config.free_space_culling_decision << "\",\n";
  out << "  \"free_space_culling_block_size_m\": "
      << config.free_space_culling_block_size_m << ",\n";
  out << "  \"free_space_culling_radial_tolerance_m\": "
      << config.free_space_culling_radial_tolerance_m << ",\n";
  out << "  \"free_space_culling_depth_tolerance_m\": "
      << config.free_space_culling_depth_tolerance_m << ",\n";
  out << "  \"free_space_culling_active_window_duration_s\": "
      << config.free_space_culling_active_window_duration_s << ",\n";
  out << "  \"free_space_culling_min_absent\": "
      << config.free_space_culling_min_absent << ",\n";
  out << "  \"free_space_culling_max_present\": "
      << config.free_space_culling_max_present << ",\n";
  out << "  \"dynamic_residue_cleanup\": "
      << (config.dynamic_residue_cleanup ? "true" : "false") << ",\n";
  out << "  \"dynamic_residue_support_distance_m\": "
      << config.dynamic_residue_support_distance_m << ",\n";
  out << "  \"dynamic_residue_min_track_frames\": "
      << config.dynamic_residue_min_track_frames << ",\n";
  out << "  \"dynamic_residue_require_time_overlap\": "
      << (config.dynamic_residue_require_time_overlap ? "true" : "false") << ",\n";
  out << "  \"dynamic_residue_decision\": \""
      << config.dynamic_residue_decision << "\",\n";
  out << "  \"dynamic_residue_unobserved_policy\": \""
      << config.dynamic_residue_unobserved_policy << "\",\n";
  out << "  \"dynamic_residue_min_absent\": "
      << config.dynamic_residue_min_absent << ",\n";
  out << "  \"dynamic_residue_max_present\": "
      << config.dynamic_residue_max_present << ",\n";
  out << "  \"dynamic_residue_protected_labels\": [";
  std::size_t protected_label_index = 0;
  for (const int label : config.dynamic_residue_protected_labels) {
    out << (protected_label_index++ == 0 ? "" : ", ") << label;
  }
  out << "],\n";
  out << "  \"dynamic_residue_protect_horizontal_surfaces\": "
      << (config.dynamic_residue_protect_horizontal_surfaces ? "true" : "false")
      << ",\n";
  out << "  \"dynamic_residue_horizontal_normal_z_min\": "
      << config.dynamic_residue_horizontal_normal_z_min << ",\n";
  out << "  \"volume_graph_cut_fill\": "
      << (config.volume_graph_cut_fill ? "true" : "false") << ",\n";
  out << "  \"volume_graph_cut_surface_policy\": \""
      << config.volume_graph_cut_surface_policy << "\",\n";
  out << "  \"volume_graph_cut_resolution_m\": "
      << config.volume_graph_cut_resolution_m << ",\n";
  out << "  \"volume_graph_cut_max_cells\": "
      << config.volume_graph_cut_max_cells << ",\n";
  out << "  \"cross_session_remove_absent_prior\": "
      << (config.cross_session_remove_absent_prior ? "true" : "false") << ",\n";
  out << "  \"cross_session_mesh_merge_distance_m\": "
      << config.cross_session_mesh_merge_distance_m << ",\n";
  out << "  \"object_cleanup_reobservation_gate\": "
      << (config.object_cleanup_reobservation_gate ? "true" : "false") << ",\n";
  out << "  \"object_cleanup_support_distance_m\": "
      << config.object_cleanup_support_distance_m << ",\n";
  out << "  \"temporal_background_dsgs\": " << config.temporal_background_dsgs.size()
      << ",\n";
  out << "  \"object_move_skip_probability\": "
      << config.object_move_skip_probability << ",\n";
  out << "  \"object_move_time_scale_s\": " << config.object_move_time_scale_s << ",\n";
  out << "  \"prior_match_distance_m\": " << config.prior_match_distance_m << ",\n";
  out << "  \"min_object_mesh_vertices\": " << config.min_object_mesh_vertices << ",\n";
  out << "  \"repair_global_vertex_ratio_threshold\": "
      << config.repair_global_vertex_ratio_threshold << ",\n";
  out << "  \"require_same_label\": " << (config.require_same_label ? "true" : "false") << ",\n";
  out << "  \"require_bbox_containment\": "
      << (config.require_bbox_containment ? "true" : "false") << ",\n";
  out << "  \"dry_run\": " << (config.dry_run ? "true" : "false") << ",\n";
  out << "  \"initial_vertices\": " << summary.initial_vertices << ",\n";
  out << "  \"initial_faces\": " << summary.initial_faces << ",\n";
  out << "  \"final_vertices\": " << summary.final_vertices << ",\n";
  out << "  \"final_faces\": " << summary.final_faces << ",\n";
  out << "  \"candidate_vertices\": " << summary.candidate_vertices << ",\n";
  out << "  \"removed_vertices\": " << summary.removed_vertices << ",\n";
  out << "  \"objects_total\": " << summary.objects_total << ",\n";
  out << "  \"objects_with_private_mesh\": " << summary.objects_with_private_mesh << ",\n";
  out << "  \"objects_used_for_cleanup\": " << summary.objects_used_for_cleanup << ",\n";
  out << "  \"cleanup_source_points\": " << summary.cleanup_source_points << ",\n";
  out << "  \"object_cleanup_present_protected_vertices\": "
      << summary.object_cleanup_present_protected_vertices << ",\n";
  out << "  \"object_cleanup_unobserved_protected_vertices\": "
      << summary.object_cleanup_unobserved_protected_vertices << ",\n";
  out << "  \"object_cleanup_absent_confirmed_vertices\": "
      << summary.object_cleanup_absent_confirmed_vertices << ",\n";
  out << "  \"repair_candidate_objects\": " << summary.repair_candidate_objects << ",\n";
  out << "  \"repair_candidate_vertices\": " << summary.repair_candidate_vertices << ",\n";
  out << "  \"injected_objects\": " << summary.injected_objects << ",\n";
  out << "  \"injected_vertices\": " << summary.injected_vertices << ",\n";
  out << "  \"injected_faces\": " << summary.injected_faces << ",\n";
  out << "  \"object_alignment_candidates\": "
      << summary.object_alignment_candidates << ",\n";
  out << "  \"object_alignment_applied\": "
      << summary.object_alignment_applied << ",\n";
  out << "  \"object_alignment_support_vertices\": "
      << summary.object_alignment_support_vertices << ",\n";
  out << "  \"temporal_background_sources\": "
      << summary.temporal_background_sources << ",\n";
  out << "  \"temporal_background_injected_vertices\": "
      << summary.temporal_background_injected_vertices << ",\n";
  out << "  \"temporal_background_injected_faces\": "
      << summary.temporal_background_injected_faces << ",\n";
  out << "  \"temporal_object_sources\": " << summary.temporal_object_sources << ",\n";
  out << "  \"temporal_object_injected_vertices\": "
      << summary.temporal_object_injected_vertices << ",\n";
  out << "  \"temporal_object_injected_faces\": "
      << summary.temporal_object_injected_faces << ",\n";
  out << "  \"horizontal_planes_filled\": "
      << summary.horizontal_planes_filled << ",\n";
  out << "  \"horizontal_plane_vertices\": "
      << summary.horizontal_plane_vertices << ",\n";
  out << "  \"horizontal_plane_faces\": " << summary.horizontal_plane_faces << ",\n";
  out << "  \"horizontal_plane_graph_cut_cells\": "
      << summary.horizontal_plane_graph_cut_cells << ",\n";
  out << "  \"horizontal_plane_graph_cut_fill_cells\": "
      << summary.horizontal_plane_graph_cut_fill_cells << ",\n";
  out << "  \"axis_aligned_planes_filled\": "
      << summary.axis_aligned_planes_filled << ",\n";
  out << "  \"axis_aligned_plane_vertices\": "
      << summary.axis_aligned_plane_vertices << ",\n";
  out << "  \"axis_aligned_plane_faces\": "
      << summary.axis_aligned_plane_faces << ",\n";
  out << "  \"axis_aligned_plane_graph_cut_cells\": "
      << summary.axis_aligned_plane_graph_cut_cells << ",\n";
  out << "  \"axis_aligned_plane_graph_cut_fill_cells\": "
      << summary.axis_aligned_plane_graph_cut_fill_cells << ",\n";
  out << "  \"free_space_checked_vertices\": "
      << summary.free_space_checked_vertices << ",\n";
  out << "  \"free_space_absent_vertices\": "
      << summary.free_space_absent_vertices << ",\n";
  out << "  \"free_space_present_vertices\": "
      << summary.free_space_present_vertices << ",\n";
  out << "  \"free_space_removed_vertices\": "
      << summary.free_space_removed_vertices << ",\n";
  out << "  \"dynamic_residue_tracks\": "
      << summary.dynamic_residue_tracks << ",\n";
  out << "  \"dynamic_residue_support_points\": "
      << summary.dynamic_residue_support_points << ",\n";
  out << "  \"dynamic_residue_candidates\": "
      << summary.dynamic_residue_candidates << ",\n";
  out << "  \"dynamic_residue_protected_vertices\": "
      << summary.dynamic_residue_protected_vertices << ",\n";
  out << "  \"dynamic_residue_absent_confirmed_vertices\": "
      << summary.dynamic_residue_absent_confirmed_vertices << ",\n";
  out << "  \"dynamic_residue_present_rejected_vertices\": "
      << summary.dynamic_residue_present_rejected_vertices << ",\n";
  out << "  \"dynamic_residue_unobserved_vertices\": "
      << summary.dynamic_residue_unobserved_vertices << ",\n";
  out << "  \"dynamic_residue_quarantined_vertices\": "
      << summary.dynamic_residue_quarantined_vertices << ",\n";
  out << "  \"dynamic_residue_removed_vertices\": "
      << summary.dynamic_residue_removed_vertices << ",\n";
  out << "  \"volume_graph_cut_cells\": "
      << summary.volume_graph_cut_cells << ",\n";
  out << "  \"volume_graph_cut_free_evidence_cells\": "
      << summary.volume_graph_cut_free_evidence_cells << ",\n";
  out << "  \"volume_graph_cut_full_evidence_cells\": "
      << summary.volume_graph_cut_full_evidence_cells << ",\n";
  out << "  \"volume_graph_cut_structural_cells\": "
      << summary.volume_graph_cut_structural_cells << ",\n";
  out << "  \"volume_graph_cut_full_cells\": "
      << summary.volume_graph_cut_full_cells << ",\n";
  out << "  \"volume_graph_cut_vertices\": "
      << summary.volume_graph_cut_vertices << ",\n";
  out << "  \"volume_graph_cut_faces\": "
      << summary.volume_graph_cut_faces << ",\n";
  out << "  \"prior_memory_objects_used_for_matching\": " << summary.prior_memory_objects
      << ",\n";
  out << "  \"prior_matched_objects\": " << summary.prior_matched_objects << ",\n";
  out << "  \"prior_absent_objects\": " << summary.prior_absent_objects << ",\n";
  out << "  \"prior_unobserved_objects\": " << summary.prior_unobserved_objects << ",\n";
  out << "  \"prior_unmatched_current_objects\": "
      << summary.prior_unmatched_current_objects << ",\n";
  out << "  \"prior_restored_objects\": " << summary.prior_restored_objects << ",\n";
  out << "  \"prior_absent_vertices_removed\": "
      << summary.prior_absent_vertices_removed << ",\n";
  out << "  \"forced_absent_prior_objects\": "
      << summary.forced_absent_prior_objects << ",\n";
  out << "  \"forced_absent_vertices_removed\": "
      << summary.forced_absent_vertices_removed << ",\n";
  out << "  \"cross_session_prior_vertices\": "
      << summary.cross_session_prior_vertices << ",\n";
  out << "  \"cross_session_prior_faces\": "
      << summary.cross_session_prior_faces << ",\n";
  out << "  \"cross_session_current_vertices\": "
      << summary.cross_session_current_vertices << ",\n";
  out << "  \"cross_session_current_faces\": "
      << summary.cross_session_current_faces << ",\n";
  out << "  \"cross_session_prior_checked_vertices\": "
      << summary.cross_session_prior_checked_vertices << ",\n";
  out << "  \"cross_session_prior_absent_vertices\": "
      << summary.cross_session_prior_absent_vertices << ",\n";
  out << "  \"cross_session_prior_persistent_vertices\": "
      << summary.cross_session_prior_persistent_vertices << ",\n";
  out << "  \"cross_session_prior_unobserved_vertices\": "
      << summary.cross_session_prior_unobserved_vertices << ",\n";
  out << "  \"cross_session_current_injected_vertices\": "
      << summary.cross_session_current_injected_vertices << ",\n";
  out << "  \"cross_session_current_injected_faces\": "
      << summary.cross_session_current_injected_faces << "\n";
  out << "}\n";
}

void writeObjectMemoryJson(const std::string& path, const std::vector<ObjectAuditRow>& rows) {
  std::ofstream out(path);
  out << "{\n";
  out << "  \"version\": 1,\n";
  out << "  \"objects\": [\n";
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto& row = rows[i];
    out << "    {\n";
    out << "      \"object_id\": " << row.node_id << ",\n";
    out << "      \"semantic_label\": " << row.semantic_label << ",\n";
    out << "      \"first_observed_ns\": " << row.first_observed_ns << ",\n";
    out << "      \"last_observed_ns\": " << row.last_observed_ns << ",\n";
    out << "      \"bbox_center\": [" << row.bbox_cx << ", " << row.bbox_cy << ", " << row.bbox_cz
        << "],\n";
    out << "      \"bbox_dimensions\": [" << row.bbox_dx << ", " << row.bbox_dy << ", "
        << row.bbox_dz << "],\n";
    out << "      \"object_mesh_vertices\": " << row.object_mesh_vertices << ",\n";
    out << "      \"object_mesh_faces\": " << row.object_mesh_faces << ",\n";
    out << "      \"global_vertices_in_bbox\": " << row.global_vertices_in_bbox << ",\n";
    out << "      \"dynamic_residue_candidates\": "
        << row.dynamic_residue_candidates << ",\n";
    out << "      \"dynamic_residue_protected\": "
        << row.dynamic_residue_protected << ",\n";
    out << "      \"dynamic_residue_absent_confirmed\": "
        << row.dynamic_residue_absent_confirmed << ",\n";
    out << "      \"dynamic_residue_present_rejected\": "
        << row.dynamic_residue_present_rejected << ",\n";
    out << "      \"dynamic_residue_unobserved\": "
        << row.dynamic_residue_unobserved << ",\n";
    out << "      \"dynamic_residue_quarantined\": "
        << row.dynamic_residue_quarantined << ",\n";
    out << "      \"dynamic_residue_removed\": "
        << row.dynamic_residue_removed << ",\n";
    out << "      \"vertices_candidate\": " << row.vertices_candidate << ",\n";
    out << "      \"vertices_removed\": " << row.vertices_removed << ",\n";
    out << "      \"repair_candidate\": " << (row.repair_candidate ? "true" : "false") << ",\n";
    out << "      \"repair_candidate_vertices\": " << row.repair_candidate_vertices << ",\n";
    out << "      \"change_first_absent_ns\": " << row.change_first_absent_ns << ",\n";
    out << "      \"change_last_absent_ns\": " << row.change_last_absent_ns << ",\n";
    out << "      \"change_first_persistent_ns\": " << row.change_first_persistent_ns << ",\n";
    out << "      \"change_last_persistent_ns\": " << row.change_last_persistent_ns << ",\n";
    out << "      \"object_move_probability\": " << row.object_move_probability << ",\n";
    out << "      \"skipped_by_object_move\": "
        << (row.skipped_by_object_move ? "true" : "false") << ",\n";
    out << "      \"prior_matched\": " << (row.prior_matched ? "true" : "false") << ",\n";
    out << "      \"prior_match_object_id\": " << row.prior_match_object_id << ",\n";
    out << "      \"prior_match_distance_m\": ";
    if (std::isfinite(row.prior_match_distance_m)) {
      out << row.prior_match_distance_m;
    } else {
      out << "null";
    }
    out << ",\n";
    out << "      \"prior_absent_evidence\": " << row.prior_absent_evidence << ",\n";
    out << "      \"prior_present_evidence\": " << row.prior_present_evidence << ",\n";
    out << "      \"prior_stationarity_alpha\": " << row.prior_stationarity_alpha << ",\n";
    out << "      \"prior_stationarity_beta\": " << row.prior_stationarity_beta << ",\n";
    out << "      \"prior_stationarity_mean\": " << row.prior_stationarity_mean << ",\n";
    out << "      \"session_state\": \"" << row.session_state << "\"\n";
    out << "    }" << (i + 1 == rows.size() ? "\n" : ",\n");
  }
  out << "  ]\n";
  out << "}\n";
}

std::size_t countPriorMemoryObjects(const std::string& path) {
  if (path.empty()) {
    return 0;
  }

  std::ifstream in(path);
  if (!in) {
    return 0;
  }

  std::size_t count = 0;
  std::string line;
  while (std::getline(in, line)) {
    if (line.find("\"object_id\"") != std::string::npos) {
      ++count;
    }
  }
  return count;
}

}  // namespace session_update::base1
