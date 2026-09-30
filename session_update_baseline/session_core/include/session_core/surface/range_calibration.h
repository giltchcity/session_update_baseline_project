#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

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

// Fixed sampling and numerical protocol inherited from local baseline 48a3033.
// These are estimator settings, not scene-dependent decision thresholds.
struct ScaleProtocol {
  static constexpr size_t kMaxFrames = 64;
  static constexpr int kPixelStride = 16;
  static constexpr size_t kMinPairs = 1000;
  static constexpr int kCoarseRadius = 50;
  static constexpr float kCoarseStep = 0.002f;
  static constexpr int kFineRadius = 10;
  static constexpr float kFineStep = 0.0002f;
};

class RangeCalibration {
 public:
  // Minimize the upper median radial disagreement on frozen correspondences.
  // Evaluate zero first, then the ordered coarse/fine grids; strict improvement
  // preserves the incumbent on ties. The fine grid is not clamped to +/-10%.
  // Validate every pair, including below kMinPairs: positions/direction finite and
  // both ranges positive and finite. Invalid input throws std::invalid_argument;
  // directions are never renormalized. Fewer than kMinPairs pairs returns neutral
  // scale 0. A residual that is not finite throws std::overflow_error.
  // Collection owns observation-domain selection and fixed frame/pixel sampling.
  static float fitScale(const std::vector<RangePair>& samples);

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
