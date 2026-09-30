#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "khronos/common/common_types.h"

namespace khronos {

/**
 * README principle 6: the latest first-return verdict of one surface element in one round. A round
 * keeps, per element, the verdict of its newest frame among the new frames; occluded and invalid
 * readings carry likelihood ratio one and are not recorded.
 */
struct ElementVerdict {
  enum Label : uint8_t { kNoLabel = 0, kOwn = 1, kForeign = 2 };
  uint64_t key = 0;          // the element: its cell of the map resolution, packed
  bool through = false;      // T, the first return lies behind the element; otherwise H (hit)
  uint8_t label = kNoLabel;  // physical label of a hit (README (6s), assumption 5)
  TimeStamp stamp = 0;       // the frame of the verdict
};

struct ElementRound {
  std::vector<ElementVerdict> verdicts;
  size_t num_elements = 0;       // elements of the placement that were queried
  TimeStamp latest_hit = 0;      // newest frame with a hit verdict
  TimeStamp first_through = 0;   // earliest frame with a see-through verdict
  size_t occluded = 0, invalid = 0;  // diagnostics: readings classified O or I
};

/** Pack the cell of an element (cells of the map resolution) into one key. */
inline uint64_t packElementCell(const std::array<int64_t, 3>& cell) {
  constexpr int64_t kOffset = int64_t(1) << 20;
  for (const auto c : cell) {
    if (c < -kOffset || c >= kOffset) throw std::out_of_range("Element cell exceeds key range");
  }
  return (static_cast<uint64_t>(cell[0] + kOffset) << 42) |
         (static_cast<uint64_t>(cell[1] + kOffset) << 21) | static_cast<uint64_t>(cell[2] + kOffset);
}

}  // namespace khronos
