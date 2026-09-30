#include "session_core/surface/range_calibration.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace khronos {

float RangeCalibration::fitScale(const std::vector<RangePair>& samples) {
  // Unit rays arrive through float normalization and a world-frame rotation.
  // Allow their arithmetic roundoff, without renormalizing and changing data.
  constexpr double unit_roundoff = 64.0 * std::numeric_limits<float>::epsilon();
  for (const auto& x : samples) {
    if (!x.origin.allFinite() || !x.other.allFinite() || !x.direction.allFinite() ||
        !std::isfinite(x.range) || x.range <= 0.f ||
        !std::isfinite(x.other_range) || x.other_range <= 0.f) {
      throw std::invalid_argument("Range calibration needs finite coordinates and positive finite ranges");
    }
    const double squared_norm = x.direction.cast<double>().squaredNorm();
    if (std::abs(squared_norm - 1.0) > unit_roundoff) {
      throw std::invalid_argument("Range calibration direction must be a unit ray");
    }
  }
  if (samples.size() < ScaleProtocol::kMinPairs) return 0.f;
  std::vector<double> residual(samples.size());
  auto disagreement = [&](float scale) {
    for (size_t i = 0; i < samples.size(); ++i) {
      const auto& x = samples[i];
      const float factor = 1.f + scale;
      const Eigen::Vector3f point = x.origin + x.direction * (x.range * factor);
      const float value = std::abs(x.other_range * factor - (point - x.other).norm());
      if (std::isfinite(value)) {
        // Preserve the original float objective wherever it is representable.
        residual[i] = value;
      } else {
        // Finite float coordinates may overflow the float squared norm or the
        // scaled endpoint. Double covers the full validated float input domain.
        const double wide_factor = factor;
        const Eigen::Vector3d wide_point = x.origin.cast<double>() +
            x.direction.cast<double>() * (static_cast<double>(x.range) * wide_factor);
        residual[i] = std::abs(static_cast<double>(x.other_range) * wide_factor -
                              (wide_point - x.other.cast<double>()).norm());
        if (!std::isfinite(residual[i])) {
          throw std::overflow_error("Range calibration residual is not finite");
        }
      }
    }
    auto mid = residual.begin() + residual.size() / 2;
    std::nth_element(residual.begin(), mid, residual.end());
    return *mid;
  };
  float best_s = 0.f;
  double best = disagreement(0.f);
  // Coarse: +/-10% in 0.2% steps.
  for (int k = -ScaleProtocol::kCoarseRadius; k <= ScaleProtocol::kCoarseRadius; ++k) {
    const float scale = ScaleProtocol::kCoarseStep * k;
    const double m = disagreement(scale);
    if (m < best) best = m, best_s = scale;
  }
  const float coarse = best_s;
  // Fine: 0.02% steps around the fixed coarse optimum.
  for (int k = -ScaleProtocol::kFineRadius; k <= ScaleProtocol::kFineRadius; ++k) {
    const float scale = coarse + ScaleProtocol::kFineStep * k;
    const double m = disagreement(scale);
    if (m < best) best = m, best_s = scale;
  }
  return best_s;
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
