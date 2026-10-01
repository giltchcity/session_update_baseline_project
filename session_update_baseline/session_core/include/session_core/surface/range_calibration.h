#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "session_core/model/range_model.h"

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

class RangeCalibration {
 public:
  // README (9b), principle 8: the range scale 1 + zeta of frozen correspondences, the value that
  // minimises the median of the re-measurement residuals (the maximum likelihood of Laplace
  // residuals),
  //   e_j(zeta) = (1+zeta) r_b - |o_a - o_b + (1+zeta) r_a d_a|.
  // The search interval is centred on the current estimate `centre`, initially +-10% in steps of
  // 0.2%; an optimum on the boundary of the interval doubles it. The fine search then runs around
  // the coarse optimum in steps of `fine_step` = u / R (a finer scale does not change the
  // quantised readings). Ties go to the value nearest the centre. Every pair is validated:
  // positions/direction finite, both ranges positive and finite; invalid input throws
  // std::invalid_argument; an empty set throws std::invalid_argument too.
  static constexpr double kInitialHalfWidth = 0.10, kCoarseStep = 0.002;
  static float fitScale(const std::vector<RangePair>& samples, double centre, double fine_step);

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
