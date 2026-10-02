#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#include "session_core/model/range_model.h"
#include "session_core/surface/range_calibration.h"

namespace khronos::model {

/** One reading of a previous session's surface inside the window of eq. (12d), README principle 10:
 * its offset from the element after the predicted scale displacement b has been removed, the
 * variance of the pair other than sigma_x^2 (the part of (6s) that is known), and the half width W
 * of the window it was taken in. */
struct BandPair {
  double residual = 0.0;       // d - b [m]
  double base_variance = 0.0;  // sigma_table^2 + (h tan(theta))^2 / 12
  double window = 0.0;         // W = T + (|zeta_now| + |zeta_e|) rho [m]
};

/**
 * README principle 8, eqs. (9b), (9v), (9c): the online estimator of psi from the session's own
 * data. Every quantity is the posterior of "prior + the data so far", the prior counting with the
 * information of one sample; none switches on at a sample count and none falls back to a fixed value.
 *
 *  - sigma_table(rho) = 1.4826 median_j |r_j - rho_{x_j}|, the robust scale of the readings against
 *    the fused surface of the session (only surface points inside the truncation band, not
 *    occluded, facing the reading), per range bin; the prior of a bin is the previous session's
 *    posterior of that bin, else the estimate of the nearest bin that has data, else the sensor
 *    calibration curve (9v). The prior enters as one residual of its own magnitude.
 *  - w_pm by expectation maximisation of the mixture of (6e) on the residual histograms, first
 *    pooled over all bins and then, in the bins with enough samples, shrunk to the pooled value.
 *  - zeta by the alternation of mutual-hit correspondences and the median-residual minimum (9b).
 *  - sigma_x from the offsets of the readings of a previous session's surface inside the window of
 *    (12d), after the predicted scale displacement is removed (principle 10).
 * Only histograms and bounded reservoirs are kept, so the cost does not grow with the session.
 */
class SensorCalibrator {
 public:
  // Binning and budgets (README table 5.1, computation budgets): 16 range bins of 0.5 m (0..8 m),
  // residual cells of u/2 = 0.5 mm over +-0.5 m, 1000 samples per bin (MAD relative error about
  // 3.7%), the scale estimate from 64 frames of 1000 samples each, a cache of 4096 band pairs.
  static constexpr size_t kRangeBins = 16;
  static constexpr double kRangeBin = 0.5;
  static constexpr size_t kHalfCells = 1000;
  static constexpr double kCell = 5.0e-4;
  static constexpr size_t kMinSamples = 1000;
  static constexpr size_t kScaleFrames = 64;
  static constexpr size_t kMaxScalePairs = kScaleFrames * kMinSamples;
  static constexpr size_t kMaxBandPairs = 4096;

  SensorCalibrator();

  /** One residual of a reading against the fused surface of the session at predicted range
   * `range`: z = r - rho (9v). */
  void addResidual(double range, double z);
  /** The residuals of one range bin in bulk: `counts[c]` residuals fell in the signed cell
   * c + first_cell (cell c covers z in [(c + first_cell) kCell, (c + first_cell + 1) kCell)), `far`
   * and `near` residuals lie beyond the cells on either side. Equal to the same addResidual calls. */
  void addCells(size_t range_bin, long first_cell, const std::vector<int64_t>& counts, int64_t far,
                int64_t near);
  /** A pair of the window of (12d) against a previous session's element (principle 10). */
  void addBandPair(const BandPair& pair);
  /** A correspondence for the range scale of (9b); kept in a fixed-size reservoir. */
  void addScalePair(const RangePair& pair);

  /** The posterior psi. `prior` is the previous session's psi (or the model the session started
   * from), `sensor_curve` the sensor calibration curve of sigma_table (class 1; may be empty),
   * `max_range` is R and `truncation` the truncation band of the fused surface within which
   * sigma_table is measured. */
  RangeModel estimate(const RangeModel& prior, const std::vector<double>& sensor_curve,
                      double max_range, double truncation) const;
  /** (9b): the range scale of this session. The correspondences are the pairs that are mutual hits
   * at the current scale, |e_j(zeta)| <= delta_+*; the scale is the median-residual minimum over
   * them (RangeCalibration::fitScale), and the two steps alternate to a fixed point from the scale
   * of `psi` (the previous session's, 0 at cold start); the result is the posterior of that prior
   * (one sample) and the n mutual pairs. False while `psi` is invalid or no pair is a mutual hit. */
  bool estimateScale(const RangeModel& psi, double max_range, double& zeta) const;

  /** sigma_x of principle 10: (1.4826 median |d - b|)^2 less the known variance of the pair, from
   * the offsets recorded so far, with the prior sigma_x (`prior_known`) or the standard deviation
   * W/sqrt(3) of the uniform distribution on the window as one pair. False while no pair exists. */
  bool estimateSigmaX(double prior_sigma_x, bool prior_known, double& sigma_x) const;

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
  /** Median of |z| over the residuals inside `band` (cells beyond are ignored) and, if
   * `prior_magnitude` > 0, one residual of that magnitude; false when empty. */
  static bool medianAbs(const Histogram& h, double band, double prior_magnitude, int64_t& count,
                        double& median);
  bool estimateSigmaXLocked(double prior_sigma_x, bool prior_known, double& sigma_x) const;

  mutable std::mutex mutex_;
  std::vector<Histogram> by_range_;  // [range bin]
  std::vector<BandPair> band_pairs_;
  uint64_t band_seen_ = 0;
  std::vector<RangePair> scale_pairs_;
  uint64_t scale_seen_ = 0;
  size_t num_residuals_ = 0;
};

}  // namespace khronos::model
