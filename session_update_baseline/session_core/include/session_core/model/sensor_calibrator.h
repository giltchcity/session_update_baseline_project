#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#include "session_core/model/duplicate_mixture.h"
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
 * mixture of (6e) on the residual histograms; Delta_s, sigma_x and pi_dup by the two-component
 * mixture EM of (12d) on the band pairs of memory elements against the session's readings. Only
 * histograms and bounded reservoirs are kept, so the cost does not grow with the session.
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
  static constexpr size_t kMaxBandPairs = 65536;

  SensorCalibrator();

  /** One static re-measurement: bin of the time difference (pair offset 2^bin frames), the time
   * difference itself, the predicted range, the incidence angle of the reading's ray, and the
   * residual z of (6). */
  void addPair(size_t time_bin, double dt_seconds, double range, double incidence, double z);
  /** A pair of the comparison band of (12d): the offset of a previous session's element from the
   * session's reading; kept in a fixed-size reservoir. */
  void addBandPair(const BandPair& pair);
  /** A correspondence for the range scale of (9b) and the time difference of its frames; kept in a
   * fixed-size reservoir. */
  void addScalePair(const RangePair& pair, double dt_seconds);

  /** psi from the statistics, with `previous` (the model the session started from or the one of
   * the last call) supplying what the data do not yet determine. `max_range` is R. */
  RangeModel estimate(const RangeModel& previous, double max_range) const;
  /** (9b): the range scale of this session. The correspondences are the pairs that are mutual hits
   * at the current scale, |e_j(zeta)| <= delta_+*(dt_j) of `psi` (R = `max_range`); the scale is the
   * median-residual minimum over them (RangeCalibration::fitScale), and the two steps alternate to a
   * fixed point from the scale of `psi` (the previous session's, 0 at cold start). False while `psi`
   * is invalid or fewer than kMinSamples correspondences are mutual hits. */
  bool estimateScale(const RangeModel& psi, double max_range, double& zeta) const;

  size_t numPairs() const;
  size_t numScalePairs() const;
  size_t numBandPairs() const;

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
  std::vector<BandPair> band_pairs_;
  uint64_t band_seen_ = 0;
  std::vector<RangePair> scale_pairs_;
  std::vector<float> scale_dt_;
  uint64_t scale_seen_ = 0;
  size_t num_pairs_ = 0;
};

}  // namespace khronos::model
