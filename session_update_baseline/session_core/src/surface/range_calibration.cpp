#include "session_core/surface/range_calibration.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace khronos {

namespace {

// The pair geometry of (9b) in doubles, so that e_j(zeta) costs one norm per pair.
struct PairGeometry {
  Eigen::Vector3d base;  // o_a - o_b
  Eigen::Vector3d ray;   // r_a d_a
  double other_range;    // r_b
};

// Median of |e_j(zeta)|; `work` is scratch of the size of the sample set.
double medianResidual(const std::vector<PairGeometry>& pairs, double zeta, std::vector<float>& work) {
  const double factor = 1.0 + zeta;
  work.clear();
  for (const auto& x : pairs) {
    const double norm = (x.base + factor * x.ray).norm();
    work.push_back(static_cast<float>(std::abs(factor * x.other_range - norm)));
  }
  const size_t middle = work.size() / 2;
  std::nth_element(work.begin(), work.begin() + middle, work.end());
  double median = work[middle];
  if (work.size() % 2 == 0) {
    median = 0.5 * (median + *std::max_element(work.begin(), work.begin() + middle));
  }
  return median;
}

}  // namespace

float RangeCalibration::fitScale(const std::vector<RangePair>& samples, double centre,
                                 double fine_step) {
  // README (9b): coordinates and ranges must lie in their representation domains.
  if (samples.empty()) throw std::invalid_argument("Range calibration needs correspondences");
  if (!std::isfinite(centre) || centre <= -1.0 || !(fine_step > 0.0) || !std::isfinite(fine_step)) {
    throw std::invalid_argument("Invalid range calibration search");
  }
  std::vector<PairGeometry> pairs;
  pairs.reserve(samples.size());
  for (const auto& x : samples) {
    if (!x.origin.allFinite() || !x.other.allFinite() || !x.direction.allFinite() ||
        !std::isfinite(x.range) || x.range <= 0.f ||
        !std::isfinite(x.other_range) || x.other_range <= 0.f) {
      throw std::invalid_argument("Range calibration needs finite coordinates and positive finite ranges");
    }
    pairs.push_back({x.origin.cast<double>() - x.other.cast<double>(),
                     x.direction.cast<double>() * static_cast<double>(x.range),
                     static_cast<double>(x.other_range)});
  }
  std::vector<float> work;
  work.reserve(pairs.size());
  double best = centre, best_value = std::numeric_limits<double>::infinity();
  // Nodes are visited in order of distance to the centre, so a flat objective keeps the smallest
  // correction; the scale must stay above -1 (a positive range factor).
  const auto consider = [&](double zeta) {
    if (!(zeta > -1.0)) return;
    const double value = medianResidual(pairs, zeta, work);
    if (value < best_value) {
      best_value = value;
      best = zeta;
    }
  };
  double half_width = kInitialHalfWidth;
  for (int doublings = 0;; ++doublings) {
    best_value = std::numeric_limits<double>::infinity();
    const int nodes = static_cast<int>(std::lround(half_width / kCoarseStep));
    for (int i = 0; i <= nodes; ++i) {
      consider(centre + i * kCoarseStep);
      if (i > 0) consider(centre - i * kCoarseStep);
    }
    // An optimum on the boundary of the interval: the interval doubles (computation budget: it
    // cannot grow beyond the whole admissible range of the scale).
    const bool on_boundary = std::abs(std::abs(best - centre) - nodes * kCoarseStep) < 0.5 * kCoarseStep;
    if (!on_boundary || centre - 2.0 * half_width <= -1.0 || doublings >= 8) break;
    half_width *= 2.0;
  }
  const double coarse = best;
  const int fine = static_cast<int>(std::lround(kCoarseStep / fine_step));
  for (int i = -fine; i <= fine; ++i) consider(coarse + i * fine_step);
  return static_cast<float>(best);
}

std::vector<float> RangeCalibration::finishResidualScale(
    const std::vector<std::vector<int64_t>>& hist,
    size_t min_samples,
    double resolution,
    std::vector<int64_t>* counts) {
  if (min_samples == 0 ||
      min_samples > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
      !std::isfinite(resolution) || resolution <= 0.0) {
    throw std::invalid_argument("Invalid residual calibration configuration");
  }
  const size_t nb = hist.size();
  std::vector<double> sig(nb, std::numeric_limits<double>::quiet_NaN());
  std::vector<int64_t> measured_counts;
  if (counts) measured_counts.assign(nb, 0);
  for (size_t b = 0; b < nb; ++b) {
    int64_t cnt = 0;
    for (const auto c : hist[b]) {
      if (c < 0 || cnt > std::numeric_limits<int64_t>::max() - c) {
        throw std::invalid_argument("Invalid residual histogram counts");
      }
      cnt += c;
    }
    if (counts) measured_counts[b] = cnt;
    if (cnt < static_cast<int64_t>(min_samples)) continue;
    const double half = 0.5 * static_cast<double>(cnt);
    int64_t cum = 0;
    for (size_t k = 0; k < hist[b].size(); ++k) {
      const int64_t prev = cum;
      cum += hist[b][k];
      if (static_cast<double>(cum) >= half) {  // numpy searchsorted(side='left')
        sig[b] = 1.4826 * (static_cast<double>(k) +
                           (half - static_cast<double>(prev)) /
                               static_cast<double>(std::max<int64_t>(hist[b][k], 1))) *
                 resolution;
        // An overflow is a failed estimate, not an unpopulated bin. Reject it
        // before nearest-bin filling can silently replace it or narrowing casts
        // can produce a non-finite float tolerance.
        if (!std::isfinite(sig[b]) || sig[b] > std::numeric_limits<float>::max()) {
          throw std::overflow_error("Residual calibration scale exceeds the finite float domain");
        }
        break;
      }
    }
  }
  std::vector<double> filled = sig;
  for (size_t b = 0; b < nb; ++b) {  // nearest populated bin, lower side first
    if (std::isfinite(filled[b])) continue;
    for (size_t off = 1; off < nb; ++off) {
      if (b >= off && std::isfinite(sig[b - off])) {
        filled[b] = sig[b - off];
        break;
      }
      if (b + off < nb && std::isfinite(sig[b + off])) {
        filled[b] = sig[b + off];
        break;
      }
    }
  }
  std::vector<float> out(nb, 0.f);
  for (size_t b = 0; b < nb; ++b) {
    out[b] = std::isfinite(filled[b]) ? static_cast<float>(filled[b]) : 0.f;
  }
  if (counts) *counts = std::move(measured_counts);
  return out;
}

}  // namespace khronos
