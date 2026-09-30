#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "khronos/common/common_types.h"

namespace khronos {

struct SurfaceSample {
  std::array<int64_t, 3> cell{};
  Point point = Point::Zero();
  Eigen::Vector3f normal = Eigen::Vector3f::Zero();
  bool has_normal = false;
};

// One representative per world-space cell: triangle centroid, or point for a
// point cloud. Rank by distance to the cell centre, then coordinates and normal.
// Select floor(k * cell_count / selected_count) in lexicographic cell order.
// Invalid geometry, indices or nonpositive/nonfinite spacing throw. Budget zero
// returns no samples after validating the source. Zero-area faces have no normal.
std::vector<SurfaceSample> sampleSurface(const spark_dsg::Mesh& mesh,
                                         const BoundingBox& bbox,
                                         float spacing,
                                         size_t budget = 1500);

struct SurfaceAgreement {
  size_t total = 0;
  size_t shared = 0;
};

// Samples are world coordinates. Count samples within tolerance of any reference
// triangle (or any reference point when it has no faces). Empty reference means
// zero shared samples. Tolerance must be finite and positive. Invalid geometry,
// indices or coordinates outside TriangleGrid's representable range throw.
SurfaceAgreement surfaceAgreement(const std::vector<Point>& samples,
                                  const spark_dsg::Mesh& reference,
                                  const BoundingBox& bbox,
                                  float tolerance);

}  // namespace khronos
