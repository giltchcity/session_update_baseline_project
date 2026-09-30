#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace khronos::measurement {
// README appendix (encoding contract), (9c): one wire representation for evidence and terminal integration.
inline constexpr float kRangeUnit = 1e-3f;
inline constexpr float kMaxRange = std::numeric_limits<uint16_t>::max() * kRangeUnit;

inline uint16_t encodeRange(float range, float minimum, float maximum) {
  if (!std::isfinite(range) || range <= 0 || range < minimum || range > maximum ||
      range > kMaxRange) return 0;
  const long value = std::lround(static_cast<double>(range) * 1000.0);
  return value > 0 && value <= std::numeric_limits<uint16_t>::max()
      ? static_cast<uint16_t>(value) : 0;
}

inline uint16_t encodeIdentity(int id) {
  if (id > std::numeric_limits<uint16_t>::max()) {
    throw std::invalid_argument("Physical identity exceeds the 16-bit measurement protocol");
  }
  return id > 0 ? static_cast<uint16_t>(id) : 0;
}
}  // namespace khronos::measurement
