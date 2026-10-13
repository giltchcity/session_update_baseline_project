#pragma once

#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <Eigen/Geometry>

namespace khronos {

/** Closest point of triangle (a, b, c) to p (Ericson, Real-Time Collision Detection, 5.1.5). */
Eigen::Vector3f closestPointOnTriangle(const Eigen::Vector3f& p,
                                       const Eigen::Vector3f& a,
                                       const Eigen::Vector3f& b,
                                       const Eigen::Vector3f& c);

/**
 * @brief Uniform hash grid over a triangle set with exact queries.
 *
 * Triangles are registered in every cell their axis-aligned bounding box
 * overlaps. closest() visits Chebyshev rings of cells around the query until
 * no unvisited triangle can be nearer (exact point-triangle distance);
 * firstHit() walks the cells along the ray (3D-DDA) and returns the nearest
 * two-sided Moeller-Trumbore intersection with t > 0.
 */
class TriangleGrid {
 public:
  using Face = std::array<uint32_t, 3>;

  /**
   * @param vertices,faces The mesh (referenced, must outlive the grid).
   * @param subset Face indices to register (nullptr: all faces).
   * @param cell Cell edge length [m].
   */
  TriangleGrid(const std::vector<Eigen::Vector3f>& vertices,
               const std::vector<Face>& faces,
               const std::vector<uint32_t>* subset,
               float cell);

  bool empty() const { return num_registered_ == 0; }
  const Eigen::AlignedBox3f& bounds() const { return bounds_; }

  /**
   * @brief Exact nearest point of the registered triangles within r_max.
   * @returns false if no triangle lies within r_max.
   */
  bool closest(const Eigen::Vector3f& p,
               float r_max,
               float& distance,
               Eigen::Vector3f& point,
               uint32_t& face) const;

  /** @brief Nearest intersection p = o + t d (t > 0, d unit) of the ray with a registered triangle. */
  bool firstHit(const Eigen::Vector3f& origin,
                const Eigen::Vector3f& direction,
                float& t,
                uint32_t& face) const;

 private:
  using Key = uint64_t;
  Key key(int x, int y, int z) const;
  Eigen::Vector3i cellOf(const Eigen::Vector3f& p) const;
  const std::vector<uint32_t>* cellFaces(int x, int y, int z) const;

  const std::vector<Eigen::Vector3f>& vertices_;
  const std::vector<Face>& faces_;
  const float cell_;
  size_t num_registered_ = 0;
  Eigen::AlignedBox3f bounds_;
  Eigen::Vector3i min_cell_ = Eigen::Vector3i::Zero(), max_cell_ = Eigen::Vector3i::Zero();
  std::unordered_map<Key, std::vector<uint32_t>> cells_;
};

}  // namespace khronos
