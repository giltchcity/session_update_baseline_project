#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "session_core/evidence/error_model.h"

namespace khronos {

// A frozen directed correspondence from the unscaled archived range images.
// Positions and ranges are in metres. Positions and direction must be finite,
// ranges strictly positive and finite; direction is a world-frame unit ray.
struct RangePair {
  Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  Eigen::Vector3f direction = Eigen::Vector3f::Zero();
  Eigen::Vector3f other = Eigen::Vector3f::Zero();
  float range = 0.f, other_range = 0.f;
};

// Computational sampling of the correspondences: how many frames and which pixels of each are
// paired. It bounds the cost of collecting fixed correspondences; it is not a model quantity.
struct ScaleProtocol {
  static constexpr size_t kMaxFrames = 64;
  static constexpr int kPixelStride = 16;
};

class RangeCalibration {
 public:
  // README (9b): the range scale zeta of frozen correspondences by maximum likelihood. The residual
  // e_j(zeta) = (1+zeta) r_b - |o_a - o_b + (1+zeta) r_a d_a| is Gaussian with variance
  // Sigma_e = J Sigma_input J^T, from the effective standard deviations of psi at both ranges.
  // Gauss-Newton on the weighted normal equation, iterated until the step vanishes. When the
  // correspondences do not identify the scale (no information), the scale of psi is kept.
  // Every pair is validated: positions/direction finite, both ranges positive and finite; invalid
  // input throws std::invalid_argument. A non-finite residual throws std::overflow_error.
  static float fitScale(const std::vector<RangePair>& samples,
                        const measurement::ErrorModel& psi);

  // Gaussian-scaled, within-cell interpolated median absolute residual. Only
  // originally qualified bins seed nearest-bin filling; equal distances choose
  // the lower bin. No qualified bins yields zeros, leaving the voxel floor to
  // the consumer. Histogram counts must be nonnegative and fit in int64_t;
  // min_samples and resolution must be positive, and resolution finite.
  // Invalid inputs throw std::invalid_argument. A qualified scale outside the
  // finite float output domain throws std::overflow_error before filling bins.
  // Optional counts are replaced only after the whole result is valid.
  static std::vector<float> finishResidualScale(
      const std::vector<std::vector<int64_t>>& hist,
      size_t min_samples,
      double resolution,
      std::vector<int64_t>* counts = nullptr);
};

}  // namespace khronos
