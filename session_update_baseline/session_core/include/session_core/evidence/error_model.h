#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "session_core/evidence/range_encoding.h"

namespace khronos::measurement {

/**
 * README (6e), (9c): the effective range error model psi. The standard deviation is stored per
 * range bin (the existing effective residual curve); `zeta` is the depth scale of README (9b).
 */
struct ErrorModel {
  std::vector<double> sigma;  // metres, one entry per range bin
  double range_bin = 0;       // metres per bin
  double zeta = 0;            // README (9b): (1+zeta) is the range scale

  bool valid() const { return !sigma.empty() && range_bin > 0; }

  // Effective standard deviation at a range. A bin without an estimate carries only the
  // quantization variance u^2/12 of README (9c).
  double sigmaAt(double range) const {
    const size_t bin = std::min(sigma.size() - 1,
                                static_cast<size_t>(std::max(0.0, range) / range_bin));
    const double s = sigma[bin];
    return s > 0 ? s : static_cast<double>(kRangeUnit) / std::sqrt(12.0);
  }
};

}  // namespace khronos::measurement
