#include "session_core/surface/range_calibration.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace khronos {

float RangeCalibration::fitScale(const std::vector<RangePair>& samples,
                                 const measurement::ErrorModel& psi) {
  // README (9b): coordinates and ranges must lie in their representation domains.
  for (const auto& x : samples) {
    if (!x.origin.allFinite() || !x.other.allFinite() || !x.direction.allFinite() ||
        !std::isfinite(x.range) || x.range <= 0.f ||
        !std::isfinite(x.other_range) || x.other_range <= 0.f) {
      throw std::invalid_argument("Range calibration needs finite coordinates and positive finite ranges");
    }
  }
  if (!psi.valid()) throw std::logic_error("The effective range error model is not available");
  double zeta = 0.0;
  constexpr int kMaxIterations = 100;  // a bound on the solver, not a model quantity
  for (int iteration = 0; iteration < kMaxIterations; ++iteration) {
    const double factor = 1.0 + zeta;
    double numerator = 0.0, information = 0.0;
    for (const auto& x : samples) {
      const Eigen::Vector3d ray = x.direction.cast<double>() * static_cast<double>(x.range);
      const Eigen::Vector3d v = x.origin.cast<double>() - x.other.cast<double>() + factor * ray;
      const double norm = v.norm();
      if (!(norm > 0)) continue;
      const double residual = factor * x.other_range - norm;
      const double slope = x.other_range - ray.dot(v) / norm;  // d e / d zeta
      const double along = v.dot(x.direction.cast<double>()) / norm;
      const double sigma_a = psi.sigmaAt(x.range), sigma_b = psi.sigmaAt(x.other_range);
      const double variance = factor * factor * (sigma_b * sigma_b + along * along * sigma_a * sigma_a);
      if (!std::isfinite(residual) || !std::isfinite(slope) || !(variance > 0)) {
        throw std::overflow_error("Range calibration residual is not finite");
      }
      numerator += slope * residual / variance;
      information += slope * slope / variance;
    }
    if (!(information > 0) || !std::isfinite(information)) return static_cast<float>(psi.zeta);
    const double step = -numerator / information;
    zeta += step;
    if (std::abs(step) <= std::numeric_limits<float>::epsilon() * (1.0 + std::abs(zeta))) break;
  }
  return static_cast<float>(zeta);
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
