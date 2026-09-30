#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#include "session_core/model/range_model.h"
#include "session_core/surface/range_calibration.h"

namespace khronos::model {

/**
 * README principle 8, eqs. (9b), (9v), (9c): the online estimator of psi from the static
 * re-measurements of the session itself. A re-measurement is a point seen in a frame and looked up
 * in a second frame of the same session; its residual z = r_b - ||x - c_b|| is, for the static
 * majority, a hit of scale sigma_pair(dt) = sqrt(gamma(dt)),
 *
 *   gamma(dt) = (1.4826 median_j |e_j|)^2,  sigma_reg^2(dt) = gamma(dt) - gamma(0+)   (9v),
 *
 * made non-decreasing by isotonic regression; sigma_s(rho, theta) is the same robust scale at the
 * smallest time difference binned by range and incidence; w_pm by expectation maximisation of the
 * mixture of (6e) on the residual histograms; sigma_x from the residual of memory elements against
 * the session's readings. Only histograms are kept, so the cost does not grow with the session.
 */
class SensorCalibrator {
 public:
  // Binning and budgets (README table 5.1, computation budgets): range bins of 0.5 m over 0..8 m,
  // six incidence bins, 12 geometric time-difference bins, residual cells of u/2 = 0.5 mm over
  // +-0.5 m, >= 1000 samples per cell (MAD relative error about 3.7%), reservoir of scale pairs.
  static constexpr size_t kRangeBins = 16;
  static constexpr double kRangeBin = 0.5;
  static constexpr size_t kIncidenceBins = 6;
  static constexpr size_t kTimeBins = 12;
  static constexpr size_t kHalfCells = 1000;
  static constexpr double kCell = 5.0e-4;
  static constexpr size_t kMinSamples = 1000;
  static constexpr size_t kMaxScalePairs = 65536;  // 64 frames of about 1000 stride-16 pixels

  SensorCalibrator();

  /** One static re-measurement: bin of the time difference (pair offset 2^bin frames), the time
   * difference itself, the predicted range, the incidence angle of the reading's ray, and the
   * residual z of (6). */
  void addPair(size_t time_bin, double dt_seconds, double range, double incidence, double z);
  /** A reading at an element of a previous session: residual of (6s) without sigma_x and the
   * sigma_eff^2 that explains it without sigma_x (for the subtraction of (6s)). */
  void addCrossSession(double z, double variance_without_x);
  /** A correspondence for the range scale of (9b); kept in a fixed-size reservoir. */
  void addScalePair(const RangePair& pair);

  /** psi from the statistics, with `previous` (the model the session started from or the one of
   * the last call) supplying what the data do not yet determine. `max_range` is R. */
  RangeModel estimate(const RangeModel& previous, double max_range) const;
  /** (9b): the range scale of this session from the kept correspondences. A correspondence is
   * associated when its raw residual lies within `association_gate` (the background truncation of
   * the map, README principle 8); the scale is the value minimising the median residual. False
   * while fewer than kMinSamples correspondences are associated. */
  bool estimateScale(double association_gate, double& zeta) const;

  size_t numPairs() const;
  size_t numScalePairs() const;

 private:
  struct Histogram {
    std::vector<int64_t> cells = std::vector<int64_t>(2 * kHalfCells, 0);  // signed residual
    int64_t far = 0, near = 0;                                            // beyond +-0.5 m
    void add(double z);
    int64_t total() const;
  };
  static bool medianAbs(const Histogram& h, double& median);

  mutable std::mutex mutex_;
  std::vector<std::vector<Histogram>> by_time_;       // [time bin][range bin]
  std::vector<double> time_sum_;
  std::vector<int64_t> time_count_;
  std::vector<std::vector<Histogram>> by_angle_;      // [range bin][incidence bin], smallest dt
  Histogram cross_;
  double cross_variance_sum_ = 0.0;
  int64_t cross_count_ = 0;
  std::vector<RangePair> scale_pairs_;
  uint64_t scale_seen_ = 0;
  size_t num_pairs_ = 0;
};

}  // namespace khronos::model
