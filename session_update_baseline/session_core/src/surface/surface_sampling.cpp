#include "session_core/surface/surface_sampling.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>

#include "session_core/surface/triangle_grid.h"

namespace khronos {
namespace {

void requirePositive(float value, const char* message) {
  if (!std::isfinite(value) || value <= 0.f) throw std::invalid_argument(message);
}

std::vector<Point> worldPoints(const spark_dsg::Mesh& mesh, const BoundingBox& bbox) {
  std::vector<Point> points;
  points.reserve(mesh.numVertices());
  for (const auto& point : mesh.points) {
    if (!point.allFinite()) throw std::invalid_argument("Non-finite surface vertex");
    const Point world = bbox.pointToWorldFrame(point);
    if (!world.allFinite()) throw std::invalid_argument("Non-finite world surface vertex");
    points.push_back(world);
  }
  for (const auto& face : mesh.faces) {
    for (const auto vertex : face) {
      if (vertex >= points.size()) throw std::invalid_argument("Invalid surface face index");
    }
  }
  return points;
}

std::array<int64_t, 3> sampleCell(const Point& point, float spacing) {
  std::array<int64_t, 3> cell;
  const long double limit = std::ldexp(1.L, 63);
  for (size_t axis = 0; axis < 3; ++axis) {
    const auto value = std::floor(static_cast<long double>(point[axis]) / spacing);
    if (value < -limit || value >= limit)
      throw std::out_of_range("Surface sample cell exceeds signed 64-bit indexing");
    cell[axis] = static_cast<int64_t>(value);
  }
  return cell;
}

// Canonical cyclic order preserves orientation while making arithmetic independent
// of which triangle corner is listed first or how coincident vertices are indexed.
template <typename Face>
std::array<Point, 3> orderedCorners(const std::vector<Point>& points, const Face& face) {
  const auto rank = [&](size_t start) {
    std::array<float, 9> result;
    for (size_t i = 0; i < 3; ++i) {
      const auto& point = points[face[(start + i) % 3]];
      for (size_t axis = 0; axis < 3; ++axis) result[3 * i + axis] = point[axis];
    }
    return result;
  };
  size_t first = 0;
  for (size_t i = 1; i < 3; ++i) if (rank(i) < rank(first)) first = i;
  return {points[face[first]], points[face[(first + 1) % 3]],
          points[face[(first + 2) % 3]]};
}

}  // namespace

std::vector<SurfaceSample> sampleSurface(const spark_dsg::Mesh& mesh,
                                         const BoundingBox& bbox,
                                         float spacing,
                                         size_t budget) {
  requirePositive(spacing, "Surface sampling spacing must be finite and positive");
  const auto points = worldPoints(mesh, bbox);
  if (budget == 0) return {};
  std::map<std::array<int64_t, 3>, SurfaceSample> cells;
  const auto add = [&](const Point& point, const Eigen::Vector3f& normal, bool has_normal) {
    const auto cell = sampleCell(point, spacing);
    const SurfaceSample candidate{cell, point, normal, has_normal};
    auto [it, inserted] = cells.try_emplace(cell, candidate);
    if (inserted) return;
    const auto rank = [&](const SurfaceSample& sample) {
      long double distance = 0;
      for (size_t axis = 0; axis < 3; ++axis) {
        const auto centre = (static_cast<long double>(cell[axis]) + 0.5L) * spacing;
        const auto delta = static_cast<long double>(sample.point[axis]) - centre;
        distance += delta * delta;
      }
      return std::make_tuple(distance, sample.point.x(), sample.point.y(), sample.point.z(),
                             !sample.has_normal, sample.normal.x(), sample.normal.y(),
                             sample.normal.z());
    };
    if (rank(candidate) < rank(it->second)) it->second = candidate;
  };
  if (mesh.faces.empty()) {
    for (const auto& point : points) add(point, Eigen::Vector3f::Zero(), false);
  } else {
    for (const auto& face : mesh.faces) {
      const auto corners = orderedCorners(points, face);
      const Eigen::Vector3d a = corners[0].cast<double>();
      const Eigen::Vector3d b = corners[1].cast<double>();
      const Eigen::Vector3d c = corners[2].cast<double>();
      Eigen::Vector3d normal = (b - a).cross(c - a);
      const double length = normal.norm();
      if (!std::isfinite(length)) throw std::out_of_range("Surface normal exceeds numeric range");
      const bool has_normal = length > 0.;
      if (has_normal) normal /= length;
      else normal.setZero();
      const Point centroid = ((a + b + c) / 3.).cast<Point::Scalar>();
      add(centroid, normal.cast<float>(), has_normal);
    }
  }

  const size_t selected = std::min(cells.size(), budget);
  std::vector<SurfaceSample> result;
  result.reserve(selected);
  if (selected == 0) return result;
  // Incremental quotient/remainder form avoids overflow in k * cell_count.
  const size_t step = cells.size() / selected, remainder = cells.size() % selected;
  size_t index = 0, next = 0, error = 0;
  for (const auto& [cell, sample] : cells) {
    (void)cell;
    if (result.size() < selected && index == next) {
      result.push_back(sample);
      next += step;
      if (error >= selected - remainder) {
        ++next;
        error -= selected - remainder;
      } else {
        error += remainder;
      }
    }
    ++index;
  }
  return result;
}

SurfaceAgreement surfaceAgreement(const std::vector<Point>& samples,
                                  const spark_dsg::Mesh& reference,
                                  const BoundingBox& bbox,
                                  float tolerance) {
  requirePositive(tolerance, "Surface agreement tolerance must be finite and positive");
  for (const auto& point : samples) {
    if (!point.allFinite()) throw std::invalid_argument("Non-finite surface query");
  }
  const auto world = worldPoints(reference, bbox);
  SurfaceAgreement result{samples.size(), 0};
  if (world.empty() || samples.empty()) return result;
  const size_t face_count = reference.faces.empty() ? world.size() : reference.faces.size();
  if (face_count > std::numeric_limits<uint32_t>::max())
    throw std::length_error("Surface exceeds TriangleGrid face indexing");

  // Translation and tolerance units preserve the distance predicate while avoiding
  // overflow in TriangleGrid's float distance products for large world coordinates.
  const auto coordinate_rank = [&](size_t index) {
    const auto& point = world[index];
    return std::make_tuple(point.x(), point.y(), point.z());
  };
  size_t origin_index = reference.faces.empty() ? 0 : reference.faces.front()[0];
  const auto consider_origin = [&](size_t index) {
    if (coordinate_rank(index) < coordinate_rank(origin_index)) origin_index = index;
  };
  if (reference.faces.empty()) {
    for (size_t i = 0; i < world.size(); ++i) consider_origin(i);
  } else {
    for (const auto& face : reference.faces) for (const auto index : face) consider_origin(index);
  }
  const Eigen::Vector3d origin = world[origin_index].cast<double>();
  std::vector<Eigen::Vector3f> vertices;
  std::vector<TriangleGrid::Face> faces;
  faces.reserve(face_count);
  const auto unused = std::numeric_limits<uint32_t>::max();
  std::vector<uint32_t> indices(world.size(), unused);
  const auto vertex = [&](size_t source) {
    auto& index = indices[source];
    if (index != unused) return index;
    if (vertices.size() >= unused)
      throw std::length_error("Surface exceeds TriangleGrid vertex indexing");
    const Eigen::Vector3d normalized = (world[source].cast<double>() - origin) / tolerance;
    const Eigen::Vector3f point = normalized.cast<float>();
    if (!point.allFinite()) throw std::out_of_range("Surface exceeds normalized float coordinates");
    index = static_cast<uint32_t>(vertices.size());
    vertices.push_back(point);
    return index;
  };
  if (reference.faces.empty()) {
    for (size_t i = 0; i < world.size(); ++i) {
      const auto index = vertex(i);
      // All-coincident corners take closestPointOnTriangle's first vertex branch.
      faces.push_back({index, index, index});
    }
  } else {
    for (const auto& source : reference.faces) {
      std::array<size_t, 3> order{source[0], source[1], source[2]};
      std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return coordinate_rank(a) < coordinate_rank(b);
      });
      TriangleGrid::Face face{vertex(order[0]), vertex(order[1]), vertex(order[2])};
      const Eigen::Vector3d a = vertices[face[0]].cast<double>();
      const Eigen::Vector3d b = vertices[face[1]].cast<double>();
      const Eigen::Vector3d c = vertices[face[2]].cast<double>();
      if ((b - a).cross(c - a).squaredNorm() == 0.) {
        // Ericson's general branch can divide by zero for {a,a,c}. A repeated
        // final corner represents the same segment and uses its safe edge branch.
        const double ab = (b - a).squaredNorm(), ac = (c - a).squaredNorm();
        const double bc = (c - b).squaredNorm();
        if (ac > ab && ac >= bc) face = {face[0], face[2], face[2]};
        else if (bc > ab) face = {face[1], face[2], face[2]};
        else face = {face[0], face[1], face[1]};
      }
      faces.push_back(face);
    }
  }

  // The cell width affects only the index, not the distance threshold. Bounding
  // each face's span by one cell keeps registration bounded for large triangles.
  double width = 1.;
  for (const auto& face : faces) {
    Eigen::AlignedBox3f box(vertices[face[0]]);
    box.extend(vertices[face[1]]);
    box.extend(vertices[face[2]]);
    width = std::max(width, (box.max().cast<double>() - box.min().cast<double>()).maxCoeff());
  }
  float cell = static_cast<float>(width);
  if (cell < width) cell = std::nextafter(cell, std::numeric_limits<float>::infinity());
  if (!std::isfinite(cell)) throw std::out_of_range("Surface exceeds TriangleGrid cell size");
  constexpr double grid_limit = double(int64_t{1} << 20);
  for (const auto& point : vertices) {
    for (size_t axis = 0; axis < 3; ++axis) {
      const auto coordinate = std::floor(point[axis] / cell);
      if (coordinate < -grid_limit || coordinate >= grid_limit)
        throw std::out_of_range("Surface exceeds TriangleGrid packed cell indexing");
    }
  }
  // Bound the float dot/product/sum intermediates used by closestPointOnTriangle.
  // A query inside the expanded bounds differs on each axis by at most delta.
  Eigen::AlignedBox3f bounds;
  bounds.setEmpty();
  for (const auto& point : vertices) bounds.extend(point);
  const float delta = (bounds.max() - bounds.min()).maxCoeff() + 2.f;
  const float dot_bound = 3.f * delta * delta;
  const float determinant_bound = 2.f * dot_bound * dot_bound;
  if (!std::isfinite(3.f * determinant_bound))
    throw std::out_of_range("Surface exceeds TriangleGrid distance arithmetic");
  const TriangleGrid grid(vertices, faces, nullptr, cell);
  const Eigen::Vector3d low = (grid.bounds().min().cast<double>().array() - 1.).matrix();
  const Eigen::Vector3d high = (grid.bounds().max().cast<double>().array() + 1.).matrix();
  for (const auto& point : samples) {
    const Eigen::Vector3d query = (point.cast<double>() - origin) / tolerance;
    if ((query.array() < low.array()).any() || (query.array() > high.array()).any()) continue;
    float distance = 0.f;
    Eigen::Vector3f closest;
    uint32_t face = 0;
    if (grid.closest(query.cast<float>(), 1.f, distance, closest, face)) ++result.shared;
  }
  return result;
}

}  // namespace khronos
