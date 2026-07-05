#include "session_update_baseline/base1/object_guided_map_reconciler.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <cmath>
#include <memory>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include <hydra/utils/nearest_neighbor_utilities.h>
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

InjectionAppendResult appendObjectMeshToGlobal(
    const khronos::KhronosObjectAttributes& attrs,
    spark_dsg::Mesh& global_mesh,
    const hydra::PointNeighborSearch* background_search,
    double injection_min_separation_m) {
  // Base1 repair/injection action:
  // Khronos stores static object geometry as KhronosObjectAttributes::mesh, whose
  // points are relative to the object bounding-box frame (spark_dsg/node_attributes.h).
  // We inject only vertices that are novel relative to the current global mesh:
  // an object vertex is kept iff its nearest global/background vertex is farther
  // than injection_min_separation_m. Setting the radius <= 0 restores the original
  // full-object append.
  const bool filter_by_separation =
      background_search != nullptr && injection_min_separation_m > 0.0;
  const double threshold_sq = injection_min_separation_m * injection_min_separation_m;

  std::vector<std::size_t> kept_vertices;
  kept_vertices.reserve(attrs.mesh.numVertices());
  std::vector<std::size_t> old_to_new(attrs.mesh.numVertices(),
                                      std::numeric_limits<std::size_t>::max());

  for (std::size_t i = 0; i < attrs.mesh.numVertices(); ++i) {
    const auto world_point = attrs.bounding_box.pointToWorldFrame(attrs.mesh.pos(i));
    if (filter_by_separation) {
      float distance_sq = std::numeric_limits<float>::max();
      std::size_t nearest_idx = 0;
      if (background_search->search(world_point, distance_sq, nearest_idx) &&
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

  spark_dsg::Mesh object_world(global_mesh.has_colors,
                               global_mesh.has_timestamps,
                               global_mesh.has_labels,
                               global_mesh.has_first_seen_stamps);
  object_world.resizeVertices(kept_vertices.size());
  for (std::size_t new_idx = 0; new_idx < kept_vertices.size(); ++new_idx) {
    const auto i = kept_vertices[new_idx];
    object_world.setPos(new_idx, attrs.bounding_box.pointToWorldFrame(attrs.mesh.pos(i)));
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

double medianZ(const std::vector<khronos::Point>& points) {
  std::vector<double> z_values;
  z_values.reserve(points.size());
  for (const auto& point : points) {
    z_values.push_back(static_cast<double>(point.z()));
  }
  if (z_values.empty()) {
    return 0.0;
  }
  const auto mid = z_values.begin() + static_cast<std::ptrdiff_t>(z_values.size() / 2);
  std::nth_element(z_values.begin(), mid, z_values.end());
  double median = *mid;
  if (z_values.size() % 2 == 0) {
    const auto lower = std::max_element(z_values.begin(), mid);
    median = 0.5 * (median + *lower);
  }
  return median;
}

PlaneFillResult fillHorizontalPlaneByGraphCut(
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
  points.reserve(fill_cells);
  for (std::size_t ix = 0; ix < nx; ++ix) {
    for (std::size_t iy = 0; iy < ny; ++iy) {
      const auto idx = ix * ny + iy;
      if (!fill_cell[idx]) {
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
      grid_to_vertex[idx] = points.size();
      points.push_back(point);
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
    throw std::runtime_error("Failed to append graph-cut horizontal plane fill.");
  }
  return {1, points.size(), faces.size(), domain_cells, fill_cells};
}

PlaneFillResult fillHorizontalPlanes(spark_dsg::Mesh& mesh,
                                     double resolution_m,
                                     std::size_t min_support_vertices,
                                     const std::string& fill_mode) {
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
  addPlaneBin(lower, min_support_vertices);
  addPlaneBin(upper, min_support_vertices);
  if (fill_mode == "graph_cut") {
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
            ? fillHorizontalPlaneByGraphCut(mesh, plane_z, support_points, resolution_m)
            : fillHorizontalPlaneFromSupport(mesh, plane_z, support_points, resolution_m);
    total.planes += filled.planes;
    total.vertices += filled.vertices;
    total.faces += filled.faces;
    total.graph_cut_cells += filled.graph_cut_cells;
    total.graph_cut_fill_cells += filled.graph_cut_fill_cells;
  }

  return total;
}

}  // namespace

ObjectGuidedMapReconciler::ObjectGuidedMapReconciler(ReconcilerConfig config)
    : config_(std::move(config)) {}

ReconcileResult ObjectGuidedMapReconciler::reconcile(khronos::DynamicSceneGraph& dsg) const {
  ReconcileResult result;
  if (!dsg.hasMesh()) {
    result.object_rows = auditObjects(dsg);
    return result;
  }

  auto& mesh = *dsg.mesh();
  result.mesh_summary.initial_vertices = mesh.numVertices();
  result.mesh_summary.initial_faces = mesh.numFaces();
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
  applyPriorObjectMemory(&result.object_rows, &result.mesh_summary);
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

  if (config_.horizontal_plane_fill) {
    const auto filled = fillHorizontalPlanes(mesh,
                                            config_.horizontal_plane_grid_resolution_m,
                                            config_.horizontal_plane_min_support_vertices,
                                            config_.horizontal_plane_fill_mode);
    result.mesh_summary.horizontal_planes_filled += filled.planes;
    result.mesh_summary.horizontal_plane_vertices += filled.vertices;
    result.mesh_summary.horizontal_plane_faces += filled.faces;
    result.mesh_summary.horizontal_plane_graph_cut_cells += filled.graph_cut_cells;
    result.mesh_summary.horizontal_plane_graph_cut_fill_cells += filled.graph_cut_fill_cells;
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

  std::vector<khronos::Point> injection_reference_points;
  auto injection_search = config_.injection_min_separation_m > 0.0
                              ? makeInjectionSearch(mesh, &injection_reference_points)
                              : nullptr;
  const std::size_t cleanup_vertex_limit = mesh.numVertices();

  if (config_.mode == "no_op" || config_.mode == "audit" || config_.mode == "injection") {
    if (config_.mode == "injection" && dsg.hasLayer(khronos::DsgLayers::OBJECTS)) {
      for (const auto& [id, node] : dsg.getLayer(khronos::DsgLayers::OBJECTS).nodes()) {
        const auto evidence_it = object_change_evidence.find(id);
        if (evidence_it != object_change_evidence.end() &&
            evidence_it->second.skip_injection) {
          continue;
        }
        const auto& attrs = node->attributes<khronos::KhronosObjectAttributes>();
        if (attrs.mesh.numVertices() < config_.min_object_mesh_vertices) {
          continue;
        }
        const auto appended = appendObjectMeshToGlobal(
            attrs, mesh, injection_search.get(), config_.injection_min_separation_m);
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
                                         config_.temporal_object_min_separation_m);
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
    result.mesh_summary.final_vertices = mesh.numVertices();
    result.mesh_summary.final_faces = mesh.numFaces();
    return result;
  }

  if (config_.mode != "cleanup" && config_.mode != "full") {
    throw std::runtime_error("Unsupported Base1 mode: " + config_.mode);
  }

  std::vector<khronos::Point> source_points;
  std::vector<std::size_t> point_to_source;
  auto cleanup_sources = collectCleanupSources(dsg, &source_points, &point_to_source, &result.object_rows);
  result.mesh_summary.objects_used_for_cleanup = cleanup_sources.size();
  result.mesh_summary.cleanup_source_points = source_points.size();

  if (config_.mode == "full" && dsg.hasLayer(khronos::DsgLayers::OBJECTS)) {
    for (const auto& [id, node] : dsg.getLayer(khronos::DsgLayers::OBJECTS).nodes()) {
      const auto evidence_it = object_change_evidence.find(id);
      if (evidence_it != object_change_evidence.end() &&
          evidence_it->second.skip_injection) {
        continue;
      }
      const auto& attrs = node->attributes<khronos::KhronosObjectAttributes>();
      if (attrs.mesh.numVertices() < config_.min_object_mesh_vertices) {
        continue;
      }
      const auto appended = appendObjectMeshToGlobal(
          attrs, mesh, injection_search.get(), config_.injection_min_separation_m);
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

      vertices_to_delete.insert(vertex_idx);
      candidate_by_object[source.node_id]++;
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

  if (!config_.dry_run && !vertices_to_delete.empty()) {
    mesh.eraseVertices(vertices_to_delete);
  }

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
      for (std::size_t i = 0; i < mesh->numVertices(); ++i) {
        const auto point = mesh->pos(i);
        if (!isInsideExpandedBox(attrs.bounding_box, point, config_.bbox_margin_m)) {
          continue;
        }

        ++row.global_vertices_in_bbox;
        if (i < mesh->labels.size() && static_cast<int>(mesh->labels[i]) == attrs.semantic_label) {
          ++row.global_vertices_in_bbox_same_label;
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

void ObjectGuidedMapReconciler::applyPriorObjectMemory(
    std::vector<ObjectAuditRow>* object_rows,
    MeshUpdateSummary* summary) const {
  if (!object_rows || !summary || config_.dynamic_mode != "cross_session") {
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
         "global_vertices_in_bbox_same_label,vertices_candidate,vertices_removed,"
         "repair_candidate,repair_candidate_vertices,change_first_absent_ns,"
         "change_last_absent_ns,change_first_persistent_ns,change_last_persistent_ns,"
         "object_move_probability,skipped_by_object_move,prior_matched,prior_match_object_id,"
         "prior_match_distance_m,session_state\n";
  for (const auto& row : rows) {
    out << row.node_id << "," << row.semantic_label << "," << row.first_observed_ns << ","
        << row.last_observed_ns << "," << row.bbox_cx << "," << row.bbox_cy << ","
        << row.bbox_cz << "," << row.bbox_dx << "," << row.bbox_dy << "," << row.bbox_dz
        << "," << row.bbox_volume << "," << row.object_mesh_vertices << ","
        << row.object_mesh_faces << "," << row.dynamic_trajectory_points << ","
        << row.dynamic_point_frames << "," << row.global_vertices_in_bbox << ","
        << row.global_vertices_in_bbox_same_label << "," << row.vertices_candidate << ","
        << row.vertices_removed << "," << (row.repair_candidate ? "true" : "false") << ","
        << row.repair_candidate_vertices << "," << row.change_first_absent_ns << ","
        << row.change_last_absent_ns << "," << row.change_first_persistent_ns << ","
        << row.change_last_persistent_ns << "," << row.object_move_probability << ","
        << (row.skipped_by_object_move ? "true" : "false") << ","
        << (row.prior_matched ? "true" : "false") << "," << row.prior_match_object_id
        << "," << row.prior_match_distance_m << "," << row.session_state << "\n";
  }
}

void writeMeshUpdateCsv(const std::string& path, const MeshUpdateSummary& summary) {
  std::ofstream out(path);
  out << "initial_vertices,initial_faces,final_vertices,final_faces,candidate_vertices,"
         "removed_vertices,objects_total,objects_with_private_mesh,objects_used_for_cleanup,"
         "cleanup_source_points,repair_candidate_objects,repair_candidate_vertices,"
         "injected_objects,injected_vertices,injected_faces,prior_memory_objects,"
         "temporal_background_sources,temporal_background_injected_vertices,"
         "temporal_background_injected_faces,temporal_object_sources,"
         "temporal_object_injected_vertices,temporal_object_injected_faces,"
         "horizontal_planes_filled,horizontal_plane_vertices,horizontal_plane_faces,"
         "horizontal_plane_graph_cut_cells,horizontal_plane_graph_cut_fill_cells,"
         "prior_matched_objects,"
         "prior_unmatched_current_objects,forced_absent_prior_objects,"
         "forced_absent_vertices_removed\n";
  out << summary.initial_vertices << "," << summary.initial_faces << "," << summary.final_vertices
      << "," << summary.final_faces << "," << summary.candidate_vertices << ","
      << summary.removed_vertices << "," << summary.objects_total << ","
      << summary.objects_with_private_mesh << "," << summary.objects_used_for_cleanup << ","
      << summary.cleanup_source_points << "," << summary.repair_candidate_objects << ","
      << summary.repair_candidate_vertices << "," << summary.injected_objects << ","
      << summary.injected_vertices << "," << summary.injected_faces << ","
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
      << summary.prior_matched_objects
      << "," << summary.prior_unmatched_current_objects << ","
      << summary.forced_absent_prior_objects << ","
      << summary.forced_absent_vertices_removed << "\n";
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
  out << "  \"horizontal_plane_grid_resolution_m\": "
      << config.horizontal_plane_grid_resolution_m << ",\n";
  out << "  \"horizontal_plane_min_support_vertices\": "
      << config.horizontal_plane_min_support_vertices << ",\n";
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
  out << "  \"repair_candidate_objects\": " << summary.repair_candidate_objects << ",\n";
  out << "  \"repair_candidate_vertices\": " << summary.repair_candidate_vertices << ",\n";
  out << "  \"injected_objects\": " << summary.injected_objects << ",\n";
  out << "  \"injected_vertices\": " << summary.injected_vertices << ",\n";
  out << "  \"injected_faces\": " << summary.injected_faces << ",\n";
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
  out << "  \"prior_memory_objects_used_for_matching\": " << summary.prior_memory_objects
      << ",\n";
  out << "  \"prior_matched_objects\": " << summary.prior_matched_objects << ",\n";
  out << "  \"prior_unmatched_current_objects\": "
      << summary.prior_unmatched_current_objects << ",\n";
  out << "  \"forced_absent_prior_objects\": "
      << summary.forced_absent_prior_objects << ",\n";
  out << "  \"forced_absent_vertices_removed\": "
      << summary.forced_absent_vertices_removed << "\n";
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
