#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#include "session_core/model/range_model.h"
#include "session_core/surface/range_calibration.h"

namespace khronos::model {

/** One reading of a previous session's surface inside the comparison band, README principle 10:
 * its offset from the element after the predicted scale displacement b has been removed, and the
 * variance of the pair other than sigma_x^2 (the part of (6s) that is known). */
struct BandPair {
  double residual = 0.0;       // d - b [m]
  double base_variance = 0.0;  // sigma_table^2 + (h tan(theta))^2 / 12 (+ the element's h^2/12)
};

/**
 * README principle 8, eqs. (9b), (9v), (9c): the online estimator of psi from the session's own
 * data.
 *
 *  - sigma_table(rho) = 1.4826 median_j |r_j - rho_{x_j}|, the robust scale of the readings against
 *    the fused surface of the session (only surface points inside the truncation band, not
 *    occluded, facing the reading), per range bin; a bin with too few samples takes the nearest
 *    bin with samples, and the previous session's value where none exists (9v).
 *  - w_pm by expectation maximisation of the mixture of (6e) on the residual histograms, first
 *    pooled over all bins and then, in the bins with enough samples, shrunk to the pooled value.
 *  - zeta by the alternation of mutual-hit correspondences and the median-residual minimum (9b).
 *  - sigma_x from the offsets of the readings of a previous session's surface inside the
 *    comparison band, after the predicted scale displacement is removed (principle 10).
 * Only histograms and bounded reservoirs are kept, so the cost does not grow with the session.
 */
class SensorCalibrator {
 public:
  // Binning and budgets (README table 5.1, computation budgets): range bins of 0.5 m over 0..8 m,
  // residual cells of u/2 = 0.5 mm over +-0.5 m, >= 1000 samples per bin (MAD relative error about
  // 3.7%), reservoirs of correspondences.
  static constexpr size_t kRangeBins = 16;
  static constexpr double kRangeBin = 0.5;
  static constexpr size_t kHalfCells = 1000;
  static constexpr double kCell = 5.0e-4;
  static constexpr size_t kMinSamples = 1000;
  static constexpr size_t kMaxScalePairs = 65536;  // 64 frames of about 1000 stride-16 pixels
  static constexpr size_t kMaxBandPairs = 65536;

  SensorCalibrator();

  /** One residual of a reading against the fused surface of the session at predicted range
   * `range`: z = r - rho (9v). */
  void addResidual(double range, double z);
  /** A pair of the comparison band against a previous session's element (principle 10). */
  void addBandPair(const BandPair& pair);
  /** A correspondence for the range scale of (9b); kept in a fixed-size reservoir. */
  void addScalePair(const RangePair& pair);

  /** psi from the statistics, with `previous` (the model the session started from or the one of
   * the last call) supplying what the data do not yet determine. `max_range` is R, `truncation`
   * the truncation band of the fused surface within which sigma_table is measured. */
  RangeModel estimate(const RangeModel& previous, double max_range, double truncation) const;
  /** (9b): the range scale of this session. The correspondences are the pairs that are mutual hits
   * at the current scale, |e_j(zeta)| <= delta_+*; the scale is the median-residual minimum over
   * them (RangeCalibration::fitScale), and the two steps alternate to a fixed point from the scale
   * of `psi` (the previous session's, 0 at cold start). False while `psi` is invalid or fewer than
   * kMinSamples correspondences are mutual hits. */
  bool estimateScale(const RangeModel& psi, double max_range, double& zeta) const;

  /** sigma_x of principle 10: (1.4826 median |d - b|)^2 less the known variance of the pair, from
   * the offsets recorded so far. False while fewer than kMinSamples pairs exist. */
  bool estimateSigmaX(double& sigma_x) const;

  size_t numResiduals() const;
  size_t numScalePairs() const;
  size_t numBandPairs() const;

 private:
  struct Histogram {
    std::vector<int64_t> cells = std::vector<int64_t>(2 * kHalfCells, 0);  // signed residual
    int64_t far = 0, near = 0;                                            // beyond +-0.5 m
    void add(double z);
    int64_t total() const;
  };
  /** Median of |z| over the residuals inside `band` (cells beyond are ignored); false when empty. */
  bool estimateSigmaXLocked(double& sigma_x) const;
  static bool medianAbs(const Histogram& h, double band, int64_t& count, double& median);

  mutable std::mutex mutex_;
  std::vector<Histogram> by_range_;  // [range bin]
  std::vector<BandPair> band_pairs_;
  uint64_t band_seen_ = 0;
  std::vector<RangePair> scale_pairs_;
  uint64_t scale_seen_ = 0;
  size_t num_residuals_ = 0;
};

}  // namespace khronos::model
