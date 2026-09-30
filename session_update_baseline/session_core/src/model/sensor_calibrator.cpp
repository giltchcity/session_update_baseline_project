#include "session_core/model/sensor_calibrator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "session_core/model/model_math.h"

namespace khronos::model {
namespace {

constexpr double kNormalConsistency = 1.4826;  // 1 / Phi^{-1}(3/4), README table 5.1
constexpr double kSqrt2Pi = 2.50662827463100050242;
constexpr int kEmIterations = 100;  // computation budget of the EM of w_pm
constexpr double kEmTolerance = 1.0e-10;

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

// Median of |z| with interpolation inside the residual cell; false when the median lies beyond
// the histogram range.
bool SensorCalibrator::medianAbs(const Histogram& h, double& median) {
  const int64_t total = h.total();
  if (total <= 0) return false;
  const double half = 0.5 * static_cast<double>(total);
  double cumulative = 0.0;
  for (size_t c = 0; c < kHalfCells; ++c) {
    // cell c of |z| collects the signed cells kHalfCells + c and kHalfCells - 1 - c
    const double count = static_cast<double>(h.cells[kHalfCells + c] + h.cells[kHalfCells - 1 - c]);
    if (cumulative + count >= half) {
      median = (static_cast<double>(c) + (half - cumulative) / std::max(count, 1.0)) * kCell;
      return true;
    }
    cumulative += count;
  }
  return false;
}

SensorCalibrator::SensorCalibrator()
    : by_time_(kTimeBins, std::vector<Histogram>(kRangeBins)),
      time_sum_(kTimeBins, 0.0),
      time_count_(kTimeBins, 0),
      by_angle_(kRangeBins, std::vector<Histogram>(kIncidenceBins)) {}

void SensorCalibrator::addPair(size_t time_bin, double dt_seconds, double range, double incidence,
                               double z) {
  if (time_bin >= kTimeBins || !(dt_seconds > 0.0) || !std::isfinite(z) || !(range > 0.0)) return;
  const size_t r = std::min(kRangeBins - 1, static_cast<size_t>(range / kRangeBin));
  const double incidence_bin = 0.5 * 3.14159265358979323846 / static_cast<double>(kIncidenceBins);
  const size_t t = std::min(kIncidenceBins - 1, static_cast<size_t>(std::max(0.0, incidence) / incidence_bin));
  std::lock_guard<std::mutex> lock(mutex_);
  by_time_[time_bin][r].add(z);
  time_sum_[time_bin] += dt_seconds;
  ++time_count_[time_bin];
  if (time_bin == 0) by_angle_[r][t].add(z);
  ++num_pairs_;
}

void SensorCalibrator::addCrossSession(double z, double variance_without_x) {
  if (!std::isfinite(z) || !(variance_without_x >= 0.0)) return;
  std::lock_guard<std::mutex> lock(mutex_);
  cross_.add(z);
  cross_variance_sum_ += variance_without_x;
  ++cross_count_;
}

void SensorCalibrator::addScalePair(const RangePair& pair) {
  std::lock_guard<std::mutex> lock(mutex_);
  ++scale_seen_;
  if (scale_pairs_.size() < kMaxScalePairs) {
    scale_pairs_.push_back(pair);
    return;
  }
  // Reservoir sampling with a deterministic index sequence (splitmix64 of the arrival count).
  uint64_t x = scale_seen_ + 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  x ^= x >> 31;
  const uint64_t slot = x % scale_seen_;
  if (slot < kMaxScalePairs) {
    scale_pairs_[slot] = pair;
  }
}

size_t SensorCalibrator::numPairs() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return num_pairs_;
}

size_t SensorCalibrator::numScalePairs() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return scale_pairs_.size();
}

bool SensorCalibrator::estimateScale(double association_gate, double& zeta) const {
  if (!(association_gate > 0.0)) return false;
  std::vector<RangePair> pairs;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& x : scale_pairs_) {
      const Eigen::Vector3d v = x.origin.cast<double>() - x.other.cast<double>() +
                                x.direction.cast<double>() * static_cast<double>(x.range);
      const double predicted = v.norm(), z = static_cast<double>(x.other_range) - predicted;
      if (predicted > 0.0 && std::abs(z) <= association_gate) pairs.push_back(x);
    }
  }
  if (pairs.size() < kMinSamples) return false;
  zeta = static_cast<double>(RangeCalibration::fitScale(pairs));
  return true;
}

RangeModel SensorCalibrator::estimate(const RangeModel& previous, double max_range) const {
  std::lock_guard<std::mutex> lock(mutex_);
  RangeModel psi = previous;
  psi.range_bin = kRangeBin;
  psi.num_range_bins = kRangeBins;
  psi.incidence_bin = 0.5 * 3.14159265358979323846 / static_cast<double>(kIncidenceBins);
  psi.num_incidence_bins = kIncidenceBins;
  // A model from another binning has no cell correspondence: start empty.
  if (previous.sigma_s.size() != kRangeBins * kIncidenceBins) {
    psi.sigma_s.assign(kRangeBins * kIncidenceBins, 0.0);
  }

  // sigma_s(rho, theta): the pair scale at the smallest time difference, per cell; a cell without
  // enough samples keeps the previous value, else takes the nearest estimated cell.
  std::vector<double> estimated(kRangeBins * kIncidenceBins, 0.0);
  for (size_t r = 0; r < kRangeBins; ++r) {
    for (size_t t = 0; t < kIncidenceBins; ++t) {
      const auto& h = by_angle_[r][t];
      double median;
      if (h.total() >= static_cast<int64_t>(kMinSamples) && medianAbs(h, median)) {
        estimated[r * kIncidenceBins + t] = kNormalConsistency * median;
      }
    }
  }
  const bool any_estimated = std::any_of(estimated.begin(), estimated.end(), [](double s) { return s > 0.0; });
  for (size_t r = 0; r < kRangeBins; ++r) {
    for (size_t t = 0; t < kIncidenceBins; ++t) {
      double& cell = psi.sigma_s[r * kIncidenceBins + t];
      const double own = estimated[r * kIncidenceBins + t];
      if (own > 0.0) {
        cell = own;
      } else if (!(cell > 0.0) && any_estimated) {
        size_t best = std::numeric_limits<size_t>::max();
        for (size_t r2 = 0; r2 < kRangeBins; ++r2) {
          for (size_t t2 = 0; t2 < kIncidenceBins; ++t2) {
            const double value = estimated[r2 * kIncidenceBins + t2];
            if (!(value > 0.0)) continue;
            const size_t distance =
                (r2 > r ? r2 - r : r - r2) + (t2 > t ? t2 - t : t - t2);
            if (distance < best) {
              best = distance;
              cell = value;
            }
          }
        }
      }
    }
  }

  // gamma(dt) per time bin from the pooled residuals of all range bins, (9v).
  std::vector<double> dt, gamma, weight;
  for (size_t k = 0; k < kTimeBins; ++k) {
    Histogram pooled;
    for (size_t r = 0; r < kRangeBins; ++r) {
      const auto& h = by_time_[k][r];
      for (size_t c = 0; c < pooled.cells.size(); ++c) pooled.cells[c] += h.cells[c];
      pooled.far += h.far;
      pooled.near += h.near;
    }
    double median;
    if (pooled.total() < static_cast<int64_t>(kMinSamples) || !medianAbs(pooled, median)) continue;
    dt.push_back(time_sum_[k] / static_cast<double>(time_count_[k]));
    gamma.push_back((kNormalConsistency * median) * (kNormalConsistency * median));
    weight.push_back(static_cast<double>(pooled.total()));
  }
  if (!gamma.empty()) {
    std::vector<double> excess(gamma.size());
    for (size_t i = 0; i < gamma.size(); ++i) excess[i] = gamma[i] - gamma.front();
    const auto monotone = isotonicNonDecreasing(excess, weight);
    psi.reg_dt = dt;
    psi.reg_variance.resize(monotone.size());
    for (size_t i = 0; i < monotone.size(); ++i) {
      psi.reg_variance[i] = i == 0 ? 0.0 : std::max(0.0, monotone[i]);
    }
    for (size_t i = 1; i < psi.reg_variance.size(); ++i) {
      psi.reg_variance[i] = std::max(psi.reg_variance[i], psi.reg_variance[i - 1]);
    }
  }

  // w_pm by EM on the residual histograms of every (time bin, range bin) cell with an own scale.
  struct Cell {
    const Histogram* h;
    double sigma, range;
  };
  std::vector<Cell> cells;
  for (size_t k = 0; k < kTimeBins; ++k) {
    for (size_t r = 0; r < kRangeBins; ++r) {
      const auto& h = by_time_[k][r];
      double median;
      double sigma = 0.0;
      if (h.total() >= static_cast<int64_t>(kMinSamples) && medianAbs(h, median)) {
        sigma = kNormalConsistency * median;
      } else if (h.total() > 0) {
        // the pooled scale of the time bin
        Histogram pooled;
        for (size_t r2 = 0; r2 < kRangeBins; ++r2) {
          const auto& g = by_time_[k][r2];
          for (size_t c = 0; c < pooled.cells.size(); ++c) pooled.cells[c] += g.cells[c];
          pooled.far += g.far;
          pooled.near += g.near;
        }
        if (pooled.total() >= static_cast<int64_t>(kMinSamples) && medianAbs(pooled, median)) {
          sigma = kNormalConsistency * median;
        }
      }
      if (sigma > 0.0 && h.total() > 0) cells.push_back({&h, sigma, (static_cast<double>(r) + 0.5) * kRangeBin});
    }
  }
  if (!cells.empty()) {
    // EM starts from the uniform weights of the three components (hit, farther, nearer).
    double w_plus = 1.0 / 3.0, w_minus = 1.0 / 3.0;
    double total = 0.0;
    for (const auto& c : cells) total += static_cast<double>(c.h->total());
    for (int iteration = 0; iteration < kEmIterations; ++iteration) {
      double far = 0.0, near = 0.0;
      const double w_hit = 1.0 - w_plus - w_minus;
      for (const auto& c : cells) {
        const double length_far = max_range - c.range, length_near = c.range;
        const double density_far = length_far > 0.0 ? w_plus / length_far : 0.0;
        const double density_near = length_near > 0.0 ? w_minus / length_near : 0.0;
        for (size_t i = 0; i < c.h->cells.size(); ++i) {
          const int64_t count = c.h->cells[i];
          if (count == 0) continue;
          const double z = (static_cast<double>(i) - static_cast<double>(kHalfCells) + 0.5) * kCell;
          const double hit = w_hit * std::exp(-0.5 * (z / c.sigma) * (z / c.sigma)) / (c.sigma * kSqrt2Pi);
          const double out_far = (z > 0.0 && z < length_far) ? density_far : 0.0;
          const double out_near = (z < 0.0 && -z < length_near) ? density_near : 0.0;
          const double p = hit + out_far + out_near;
          if (!(p > 0.0)) continue;
          far += static_cast<double>(count) * out_far / p;
          near += static_cast<double>(count) * out_near / p;
        }
        // beyond the histogram the hit component is negligible: the outlier explains the reading
        far += static_cast<double>(c.h->far);
        near += static_cast<double>(c.h->near);
      }
      // Jeffreys smoothing keeps the weights positive.
      const double next_plus = (far + 0.5) / (total + 1.0), next_minus = (near + 0.5) / (total + 1.0);
      const bool done = std::abs(next_plus - w_plus) + std::abs(next_minus - w_minus) < kEmTolerance;
      w_plus = next_plus;
      w_minus = next_minus;
      if (done) break;
    }
    psi.w_plus = w_plus;
    psi.w_minus = w_minus;
  }

  // sigma_x: the excess of the cross-session residual scale over what the in-session terms explain.
  if (cross_count_ >= static_cast<int64_t>(kMinSamples)) {
    double median;
    if (medianAbs(cross_, median)) {
      const double scale = kNormalConsistency * median;
      const double explained = cross_variance_sum_ / static_cast<double>(cross_count_);
      psi.sigma_x = std::sqrt(std::max(0.0, scale * scale - explained));
    }
  }
  return psi;
}

}  // namespace khronos::model
