#pragma once

#include <algorithm>
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
  /** Pixel columns [u0, u1] by rows [v0, v1] of the image. */
  struct Window {
    int u0 = 0, u1 = -1, v0 = 0, v1 = -1;
    bool empty() const { return u0 > u1 || v0 > v1; }
    void extend(const Window& o) {
      if (o.empty()) return;
      if (empty()) {
        *this = o;
        return;
      }
      u0 = std::min(u0, o.u0), u1 = std::max(u1, o.u1), v0 = std::min(v0, o.v0), v1 = std::max(v1, o.v1);
    }
  };
  struct Domain {
    const std::vector<float>* depth_z = nullptr;
    const std::vector<float>* free_limit = nullptr;
  };
  using FramePose = std::function<Eigen::Isometry3d(size_t)>;
  // (reader, frame, depth, free limit) -> whether the frame is read; readers below the
  // integrator's thread count run at once.
  using ReadFrame = std::function<bool(int, size_t, std::vector<float>&, std::vector<float>&)>;

  PresentTsdf(const hydra::VolumetricMap::Config& map,
              const hydra::ProjectiveIntegrator::Config& integrator,
              const hydra::MeshIntegratorConfig& mesh);
  ~PresentTsdf();

  // Integrate `frames` (ascending), each in the domain `read` gives it, in observation order.
  void integrate(const Camera& camera, const std::vector<float>& ray_norm, const std::vector<size_t>& frames,
                 const FramePose& pose, const ReadFrame& read);
  // One bit per voxel of a block (linear voxel index).
  using VoxelMask = std::vector<uint64_t>;
  using Changes = std::vector<std::pair<hydra::BlockIndex, VoxelMask>>;
  // The voxels a frame's contribution updates differently under `after` than under `before`
  // (which read the same outside `differs`). Mark them; rebuild once after all changes in a
  // round, from the frames that reach a marked voxel (compressed frame archive).
  Changes changedVoxels(const Camera& camera, const Eigen::Isometry3d& pose,
                        const std::vector<float>& ray_norm, const Domain& before,
                        const Domain& after, const Window& differs) const;
  void invalidate(const Changes& changes);
  // Bounds of the marked voxels (whole voxels), one per block holding any.
  std::vector<Eigen::AlignedBox3f> marked() const;
  // How far behind a reading the native integrator updates: the truncation and its extra band.
  float behind() const;
  // What a rebuild read and integrated: frames, seconds reading them (concurrently), seconds integrating.
  struct Rebuilt {
    size_t frames = 0;
    double read_s = 0.0, integrate_s = 0.0;
  };
  // Rebuild the marked voxels from `frames` (ascending): every frame that can update one.
  Rebuilt rebuild(const Camera& camera, const std::vector<float>& ray_norm,
                  const std::vector<size_t>& frames, const FramePose& pose, const ReadFrame& read);
  void extractMesh(std::vector<Eigen::Vector3f>& vertices, std::vector<Face>& faces);

  static std::vector<float> rayNorm(const Camera& camera);
  static void depthFromRange(const std::vector<float>& ray_norm, const std::vector<uint16_t>& range_mm,
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
  // Fold `frames` (ascending) into the blocks each sees, in observation order: every voxel, or
  // (`marked_only`) the marked voxels of the marked blocks. Batch by batch, the frames are read
  // concurrently; then each block, in parallel, folds the batch's frames that see it in order.
  Rebuilt fold(const Camera& camera, const std::vector<float>& ray_norm, const std::vector<size_t>& frames,
               const FramePose& pose, const ReadFrame& read, bool marked_only);
  // Native integration of voxel `i` of `block` with `data` (its domain `domain`); whether it updated it.
  bool foldVoxel(const hydra::BlockTuple& block, size_t i, const hydra::InputData& data,
                 const Eigen::Isometry3f& sensor_T_world, const Domain& domain) const;
};

}  // namespace khronos
