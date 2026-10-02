#include "session_core/model/sensor_calibrator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "session_core/evidence/range_encoding.h"
#include "session_core/model/model_math.h"

namespace khronos::model {
namespace {

constexpr double kNormalConsistency = 1.4826;  // 1 / Phi^{-1}(3/4), README table 5.1
constexpr double kSqrt2Pi = 2.50662827463100050242;
// Variance of the uniform distribution on [-W, W] is W^2 / 3 (README table 5.1, mathematical constants).
constexpr double kUniformVarianceDenominator = 3.0;
constexpr int kEmIterations = 100;  // computation budget of the EM of w_pm
constexpr double kEmTolerance = 1.0e-10;

// Deterministic reservoir slot (splitmix64 of the arrival count): the index to replace, or
// `capacity` when the arrival is not kept.
size_t reservoirSlot(uint64_t seen, size_t capacity) {
  uint64_t x = seen + 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  x ^= x >> 31;
  const uint64_t slot = x % seen;
  return slot < capacity ? static_cast<size_t>(slot) : capacity;
}

}  // namespace

void SensorCalibrator::Histogram::add(double z) {
  const double half = static_cast<double>(kHalfCells);
  const double position = z / kCell;
  if (position >= half) {
    ++far;
  } else if (position < -half) {
    ++near;
  } else {
    ++cells[static_cast<size_t>(std::floor(position + half))];
  }
}

int64_t SensorCalibrator::Histogram::total() const {
  int64_t sum = far + near;
  for (const auto c : cells) sum += c;
  return sum;
}

// Median of |z| over the residuals inside `band`, with interpolation inside the residual cell, and
// one residual of magnitude `prior_magnitude` where that is positive (the prior of a bin).
bool SensorCalibrator::medianAbs(const Histogram& h, double band, double prior_magnitude,
                                 int64_t& count, double& median) {
  const size_t limit = std::min(kHalfCells, static_cast<size_t>(std::ceil(band / kCell)));
  count = 0;
  for (size_t c = 0; c < limit; ++c) count += h.cells[kHalfCells + c] + h.cells[kHalfCells - 1 - c];
  const bool prior = prior_magnitude > 0.0;
  const double total = static_cast<double>(count) + (prior ? 1.0 : 0.0);
  if (!(total > 0.0)) return false;
  const size_t prior_cell = prior ? std::min(limit - 1, static_cast<size_t>(prior_magnitude / kCell)) : limit;
  const double half = 0.5 * total;
  double cumulative = 0.0;
  for (size_t c = 0; c < limit; ++c) {
    // cell c of |z| collects the signed cells kHalfCells + c and kHalfCells - 1 - c
    const double in_cell = static_cast<double>(h.cells[kHalfCells + c] + h.cells[kHalfCells - 1 - c]) +
                           (c == prior_cell ? 1.0 : 0.0);
    if (cumulative + in_cell >= half) {
      median = (static_cast<double>(c) + (half - cumulative) / std::max(in_cell, 1.0)) * kCell;
      return true;
    }
    cumulative += in_cell;
  }
  return false;
}

SensorCalibrator::SensorCalibrator() : by_range_(kRangeBins) {}

void SensorCalibrator::addResidual(double range, double z) {
  if (!std::isfinite(z) || !(range > 0.0)) return;
  const size_t r = std::min(kRangeBins - 1, static_cast<size_t>(range / kRangeBin));
  std::lock_guard<std::mutex> lock(mutex_);
  by_range_[r].add(z);
  ++num_residuals_;
}

void SensorCalibrator::addCells(size_t range_bin, long first_cell, const std::vector<int64_t>& counts,
                                int64_t far, int64_t near) {
  if (range_bin >= kRangeBins) return;
  std::lock_guard<std::mutex> lock(mutex_);
  auto& histogram = by_range_[range_bin];
  const long half = static_cast<long>(kHalfCells);
  for (size_t c = 0; c < counts.size(); ++c) {
    const long cell = static_cast<long>(c) + first_cell + half;
    if (counts[c] <= 0) continue;
    if (cell < 0) {
      histogram.near += counts[c];
    } else if (cell >= 2 * half) {
      histogram.far += counts[c];
    } else {
      histogram.cells[static_cast<size_t>(cell)] += counts[c];
    }
    num_residuals_ += static_cast<size_t>(counts[c]);
  }
  histogram.far += far;
  histogram.near += near;
  num_residuals_ += static_cast<size_t>(std::max<int64_t>(0, far) + std::max<int64_t>(0, near));
}

void SensorCalibrator::addBandPair(const BandPair& pair) {
  if (!std::isfinite(pair.residual) || !(pair.base_variance >= 0.0) || !(pair.window > 0.0)) return;
  std::lock_guard<std::mutex> lock(mutex_);
  ++band_seen_;
  if (band_pairs_.size() < kMaxBandPairs) {
    band_pairs_.push_back(pair);
    return;
  }
  const size_t slot = reservoirSlot(band_seen_, kMaxBandPairs);
  if (slot < kMaxBandPairs) band_pairs_[slot] = pair;
}

void SensorCalibrator::addScalePair(const RangePair& pair) {
  std::lock_guard<std::mutex> lock(mutex_);
  ++scale_seen_;
  if (scale_pairs_.size() < kMaxScalePairs) {
    scale_pairs_.push_back(pair);
    return;
  }
  const size_t slot = reservoirSlot(scale_seen_, kMaxScalePairs);
  if (slot < kMaxScalePairs) scale_pairs_[slot] = pair;
}

size_t SensorCalibrator::numResiduals() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return num_residuals_;
}

size_t SensorCalibrator::numScalePairs() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return scale_pairs_.size();
}

size_t SensorCalibrator::numBandPairs() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return band_pairs_.size();
}

bool SensorCalibrator::estimateScale(const RangeModel& psi, double max_range, double& zeta) const {
  if (!psi.valid() || !(max_range > 0.0)) return false;
  std::vector<RangePair> all;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    all = scale_pairs_;
  }
  if (all.empty()) return false;
  // u / R: a finer scale does not change the quantised readings.
  const double fine_step = static_cast<double>(measurement::kRangeUnit) / max_range;
  double current = psi.zeta;
  std::vector<RangePair> mutual;
  // The alternation of correspondences and scale reaches its fixed point in a few steps: it stops
  // when the scale moves by less than the fine step, and at the latest after one step per pair set
  // size doubling of the search interval (computation budget: the number of scale frames).
  for (size_t iteration = 0; iteration < kScaleFrames; ++iteration) {
    mutual.clear();
    const double factor = 1.0 + current;
    for (const auto& x : all) {
      const Eigen::Vector3d v = x.origin.cast<double>() - x.other.cast<double>() +
                                factor * x.direction.cast<double>() * static_cast<double>(x.range);
      const double predicted = v.norm(), residual = factor * static_cast<double>(x.other_range) - predicted;
      if (!(predicted > 0.0) || predicted >= max_range) continue;
      const double sigma = psi.sigmaEff(predicted, 0.0, 0.0, false);
      if (std::abs(residual) <= psi.bounds(predicted, sigma, max_range).plus) mutual.push_back(x);
    }
    if (mutual.empty()) return false;
    const double next = static_cast<double>(RangeCalibration::fitScale(mutual, current, fine_step));
    const bool fixed = std::abs(next - current) <= fine_step;
    current = next;
    if (fixed) break;
  }
  // The posterior of the prior scale (one sample) and the scale of the n mutual-hit pairs.
  const double n = static_cast<double>(mutual.size());
  zeta = (psi.zeta + n * current) / (1.0 + n);
  return true;
}

bool SensorCalibrator::estimateSigmaXLocked(double prior_sigma_x, bool prior_known,
                                            double& sigma_x) const {
  if (band_pairs_.empty()) return false;
  std::vector<double> magnitude;
  magnitude.reserve(band_pairs_.size() + 1);
  double base = 0.0, window = 0.0;
  for (const auto& pair : band_pairs_) {
    magnitude.push_back(std::abs(pair.residual));
    base += pair.base_variance;
    window += pair.window;
  }
  const double n = static_cast<double>(band_pairs_.size());
  base /= n;
  window /= n;
  // The prior counts as one pair: the previous session's sigma_x, or the standard deviation W / sqrt(3)
  // of the uniform distribution on the window. A pair of that prior has the residual magnitude whose
  // estimate (below) returns the prior.
  const double prior = prior_known ? prior_sigma_x : window / std::sqrt(kUniformVarianceDenominator);
  magnitude.push_back(std::sqrt(prior * prior + base) / kNormalConsistency);
  std::nth_element(magnitude.begin(), magnitude.begin() + magnitude.size() / 2, magnitude.end());
  const double spread = kNormalConsistency * magnitude[magnitude.size() / 2];
  sigma_x = std::sqrt(std::max(0.0, spread * spread - base));
  return true;
}

bool SensorCalibrator::estimateSigmaX(double prior_sigma_x, bool prior_known, double& sigma_x) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return estimateSigmaXLocked(prior_sigma_x, prior_known, sigma_x);
}

RangeModel SensorCalibrator::estimate(const RangeModel& prior, const std::vector<double>& sensor_curve,
                                      double max_range, double truncation) const {
  std::lock_guard<std::mutex> lock(mutex_);
  RangeModel psi = prior;
  psi.range_bin = kRangeBin;
  psi.num_range_bins = kRangeBins;
  psi.zeta = prior.zeta;
  psi.sigma_x = 0.0;
  psi.sigma_x_known = false;

  // sigma_table(rho) of (9v): 1.4826 median |r - rho_x| per range bin, the posterior of the prior
  // centre of the bin and the residuals in it.
  const double quantisation = static_cast<double>(measurement::kRangeUnit) / std::sqrt(12.0);
  // Without a configured truncation the whole residual histogram (+-0.5 m) is the band.
  const double band = truncation > 0.0 ? truncation : static_cast<double>(kHalfCells) * kCell;
  std::vector<double> data_only(kRangeBins, 0.0);  // 1.4826 median of the residuals alone
  for (size_t r = 0; r < kRangeBins; ++r) {
    int64_t count = 0;
    double median = 0.0;
    if (medianAbs(by_range_[r], band, 0.0, count, median) && count > 0) {
      data_only[r] = std::max(kNormalConsistency * median, quantisation);
    }
  }
  const bool prior_sized = prior.sigma_table.size() == kRangeBins;
  psi.sigma_table.assign(kRangeBins, 0.0);
  for (size_t r = 0; r < kRangeBins; ++r) {
    // The prior centre: the previous session's posterior of the bin, else the estimate of the
    // nearest bin of this session that has residuals, else the sensor calibration curve.
    double centre = prior_sized ? prior.sigma_table[r] : 0.0;
    if (!(centre > 0.0)) {
      size_t best_distance = kRangeBins;
      for (size_t r2 = 0; r2 < kRangeBins; ++r2) {
        if (!(data_only[r2] > 0.0)) continue;
        const size_t distance = r2 > r ? r2 - r : r - r2;
        if (distance < best_distance) {
          best_distance = distance;
          centre = data_only[r2];
        }
      }
    }
    if (!(centre > 0.0) && r < sensor_curve.size()) centre = sensor_curve[r];
    if (!(centre > 0.0)) centre = quantisation;
    int64_t count = 0;
    double median = 0.0;
    if (medianAbs(by_range_[r], band, centre / kNormalConsistency, count, median)) {
      psi.sigma_table[r] = std::max(kNormalConsistency * median, quantisation);
    } else {
      psi.sigma_table[r] = centre;
    }
  }

  // w_pm by EM on the residual histograms of the bins that have residuals, pooled over all bins
  // first (hit, farther, nearer), then per bin, shrunk to the pooled value.
  struct Cell {
    const Histogram* h;
    double sigma, range;
    size_t bin;
  };
  std::vector<Cell> cells;
  for (size_t r = 0; r < kRangeBins; ++r) {
    if (by_range_[r].total() > 0) {
      cells.push_back({&by_range_[r], psi.sigma_table[r], (static_cast<double>(r) + 0.5) * kRangeBin, r});
    }
  }
  const double uniform_weight = 1.0 / 3.0;  // three components with no information
  const double prior_plus = prior.w_plus > 0.0 ? prior.w_plus : uniform_weight;
  const double prior_minus = prior.w_minus > 0.0 ? prior.w_minus : uniform_weight;
  if (!cells.empty()) {
    // One EM step over a set of cells from (w_plus, w_minus); the Dirichlet prior has the total
    // weight of one sample (the unit-information prior) centred on (`prior_*`).
    const auto step = [&](const std::vector<const Cell*>& set, double w_plus, double w_minus,
                          double centre_plus, double centre_minus) {
      double far = 0.0, near = 0.0, total = 0.0;
      const double w_hit = 1.0 - w_plus - w_minus;
      for (const Cell* c : set) {
        total += static_cast<double>(c->h->total());
        const double length_far = max_range - c->range, length_near = c->range;
        const double density_far = length_far > 0.0 ? w_plus / length_far : 0.0;
        const double density_near = length_near > 0.0 ? w_minus / length_near : 0.0;
        for (size_t i = 0; i < c->h->cells.size(); ++i) {
          const int64_t count = c->h->cells[i];
          if (count == 0) continue;
          const double z = (static_cast<double>(i) - static_cast<double>(kHalfCells) + 0.5) * kCell;
          const double hit = w_hit * std::exp(-0.5 * (z / c->sigma) * (z / c->sigma)) / (c->sigma * kSqrt2Pi);
          const double out_far = (z > 0.0 && z < length_far) ? density_far : 0.0;
          const double out_near = (z < 0.0 && -z < length_near) ? density_near : 0.0;
          const double p = hit + out_far + out_near;
          if (!(p > 0.0)) continue;
          far += static_cast<double>(count) * out_far / p;
          near += static_cast<double>(count) * out_near / p;
        }
        // beyond the histogram the hit component is negligible: the outlier explains the reading
        far += static_cast<double>(c->h->far);
        near += static_cast<double>(c->h->near);
      }
      return std::pair<double, double>{(far + centre_plus) / (total + 1.0),
                                       (near + centre_minus) / (total + 1.0)};
    };
    const auto fit = [&](const std::vector<const Cell*>& set, double start_plus, double start_minus,
                         double centre_plus, double centre_minus) {
      double w_plus = start_plus, w_minus = start_minus;
      for (int iteration = 0; iteration < kEmIterations; ++iteration) {
        const auto next = step(set, w_plus, w_minus, centre_plus, centre_minus);
        const bool done = std::abs(next.first - w_plus) + std::abs(next.second - w_minus) < kEmTolerance;
        w_plus = next.first;
        w_minus = next.second;
        if (done) break;
      }
      return std::pair<double, double>{w_plus, w_minus};
    };
    std::vector<const Cell*> all;
    for (const auto& c : cells) all.push_back(&c);
    const auto pooled = fit(all, prior_plus, prior_minus, prior_plus, prior_minus);
    psi.w_plus = pooled.first;
    psi.w_minus = pooled.second;
    psi.w_plus_bin.assign(kRangeBins, pooled.first);
    psi.w_minus_bin.assign(kRangeBins, pooled.second);
    for (const auto& c : cells) {
      if (c.h->total() < static_cast<int64_t>(kMinSamples)) continue;
      const auto own_fit = fit({&c}, pooled.first, pooled.second, pooled.first, pooled.second);
      psi.w_plus_bin[c.bin] = own_fit.first;
      psi.w_minus_bin[c.bin] = own_fit.second;
    }
  } else {
    // No residual yet: the posterior is the prior (the previous session's weights; none for the
    // first session, which has no weights until its first residuals).
    psi.w_plus = prior.w_plus;
    psi.w_minus = prior.w_minus;
    psi.w_plus_bin = prior.w_plus_bin;
    psi.w_minus_bin = prior.w_minus_bin;
  }

  // sigma_x of principle 10, from the first readings of a previous session's surface on.
  double sigma_x = 0.0;
  if (estimateSigmaXLocked(prior.sigma_x, prior.sigma_x_known, sigma_x)) {
    psi.sigma_x = sigma_x;
    psi.sigma_x_known = true;
  }
  return psi;
}

}  // namespace khronos::model
