#pragma once

#include <array>
#include <functional>
#include <memory>
#include <limits>
#include <vector>
#include <Eigen/Geometry>
#include <hydra/reconstruction/mesh_integrator.h>
#include <hydra/reconstruction/projective_integrator.h>

namespace khronos {

/** Persistent geometry using Khronos/Hydra's native integration and meshing.
 * Tracking stays in the active window. This volume stores no tracking history.
 * Changed evidence rebuilds affected blocks in observation order, preserving
 * native weight saturation instead of subtracting an uncapped weighted mean.
 */
class PresentTsdf {
 public:
  struct Camera {
    uint32_t width = 0, height = 0;
    float fx = 0.f, fy = 0.f, cx = 0.f, cy = 0.f;
    float min_range = 0.f, max_range = std::numeric_limits<float>::infinity();
  };
  using Face = std::array<uint32_t, 3>;
  struct Domain {
    const std::vector<float>* depth_z = nullptr;
    const std::vector<float>* free_limit = nullptr;
  };
  using ReadFrame = std::function<void(size_t, Eigen::Isometry3d&,
                                       std::vector<float>&, std::vector<float>&)>;

  PresentTsdf(const hydra::VolumetricMap::Config& map,
              const hydra::ProjectiveIntegrator::Config& integrator,
              const hydra::MeshIntegratorConfig& mesh);
  ~PresentTsdf();

  void integrate(const Camera& camera, const Eigen::Isometry3d& pose,
                 const std::vector<float>& ray_norm, const Domain& domain);
  // Mark blocks touched by a changed contribution. Rebuild once after all
  // changes in a round, using the existing compressed frame archive.
  void invalidate(const Camera& camera, const Eigen::Isometry3d& pose,
                  const std::vector<float>& ray_norm, const Domain& before,
                  const Domain& after);
  void rebuild(const Camera& camera, const std::vector<float>& ray_norm,
               size_t frame_count, const ReadFrame& read);
  void extractMesh(std::vector<Eigen::Vector3f>& vertices, std::vector<Face>& faces);

  static std::vector<float> rayNorm(const Camera& camera);
  static void depthFromRange(const Camera& camera, const std::vector<uint16_t>& range_mm,
                             std::vector<float>& depth_z);
  size_t numUnits() const;

  // Read-only access for native-integrator equivalence tests.
  const hydra::VolumetricMap& map() const { return map_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  hydra::VolumetricMap map_;
  hydra::ProjectiveIntegrator integrator_;
  hydra::MeshIntegrator mesh_integrator_;
  void integrateBlocks(const hydra::InputData& data, const Domain& domain,
                       const hydra::BlockIndices& blocks);
};

}  // namespace khronos
