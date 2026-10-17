#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Geometry>

namespace khronos {

/**
 * @brief Sparse TSDF of the session's own frames (README section 5, eq. 9), with
 * the semantics of Open3D 0.18 pipelines.integration.ScalableTSDFVolume
 * (NoColor):
 *  - volume units of 16^3 voxels; a frame updates only the units within +-T of
 *    one of its stride-4 pixels (u % 4 == 0 and v % 4 == 0) back-projected;
 *  - inside an updated unit every voxel centre (k + 0.5) v is projected to the
 *    rounded pixel int(u_f + 0.5) and updated where the projective signed
 *    distance (depth - z) * |ray| exceeds -T: f = min(1, sdf / T), constant
 *    weight 1, no weight cap;
 *  - marching cubes at 0 over cubes whose 8 corner weights are all > 0, edge
 *    vertices interpolated from the lower corner, triangle normals
 *    (b - a) x (c - a) pointing to the observed free space.
 */
class PresentTsdf {
 public:
  struct Camera {
    uint32_t width = 0, height = 0;
    float fx = 0.f, fy = 0.f, cx = 0.f, cy = 0.f;
  };
  using Face = std::array<uint32_t, 3>;

  /**
   * @brief One frame's legal contribution domain (README section 5, eq. 8).
   * depth_z: z-depth [m] per pixel (row-major, width * height), <= 0 contributes
   * nothing. free_limit (optional): range [m] per pixel, +inf = none; the
   * pixel's free-space updates (sdf > T) stop one truncation before it, its
   * surface band is legal as usual. Without depth_z the domain is empty.
   */
  struct Domain {
    const std::vector<float>* depth_z = nullptr;
    const std::vector<float>* free_limit = nullptr;
  };

  PresentTsdf(double voxel, double truncation, int num_threads);
  ~PresentTsdf();

  /**
   * @brief Move one frame's contribution from domain `from` to domain `to`:
   * every voxel update legal in only one of them (or with another value) is
   * removed / added, the equal-weight mean (eq. 9) or its exact inverse. From
   * an empty domain this integrates the frame, to an empty one it removes it.
   * Only the volume units where the two domains can differ are visited.
   * @param ray_norm |((u-cx)/fx, (v-cy)/fy, 1)| per pixel (the depth to camera
   * distance multiplier).
   */
  void update(const Camera& camera,
              const Eigen::Isometry3d& world_T_sensor,
              const std::vector<float>& ray_norm,
              const Domain& from,
              const Domain& to);

  void extractMesh(std::vector<Eigen::Vector3f>& vertices, std::vector<Face>& faces) const;

  /** Depth to camera distance multiplier per pixel (Open3D CreateDepthToCameraDistanceMultiplierFloatImage). */
  static std::vector<float> rayNorm(const Camera& camera);

  /**
   * z-depth [m] from a range image [mm] (0 = invalid), in the float32 numerics
   * of the measured reference: z = (R * 0.001f) / |((u - cx) / fx, (v - cy) / fy, 1)|.
   */
  static void depthFromRange(const Camera& camera,
                             const std::vector<uint16_t>& range_mm,
                             std::vector<float>& depth_z);

  size_t numUnits() const;
  size_t numBytes() const;

  static constexpr int kUnitResolution = 16;

 private:
  struct Unit;
  struct KeyHash {
    size_t operator()(const Eigen::Vector3i& k) const;
  };
  using UnitSet = std::unordered_set<Eigen::Vector3i, KeyHash>;
  Unit* openUnit(const Eigen::Vector3i& index);
  /** Units a frame of this depth updates: within +-T of its stride-4 pixels back-projected. */
  UnitSet activeUnits(const Camera& camera,
                      const Eigen::Isometry3d& world_T_sensor,
                      const std::vector<float>& depth_z) const;

  const double voxel_;
  const double truncation_;
  const double unit_length_;
  const int num_threads_;
  std::unordered_map<Eigen::Vector3i, std::unique_ptr<Unit>, KeyHash> units_;
};

}  // namespace khronos
