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
  float predicted = 0.f;     // m_0 of the reading: the see-through rate the measurement model predicts
};

/** What a placement's element showed in the frames of the round that directly saw the placement in
 * place (README principle 6 (2)): own-identity hits h_e, see-throughs v_e and the sum of the ranges
 * of the hits (the recorded distance rho_e of principle 12). */
struct ElementLearning {
  uint64_t key = 0;
  uint32_t hits = 0, through = 0;
  float range_sum = 0.f;
};

struct ElementRound {
  std::vector<ElementVerdict> verdicts;
  size_t num_elements = 0;       // elements of the placement that were queried
  TimeStamp latest_hit = 0;      // newest frame with a hit verdict
  TimeStamp first_through = 0;   // earliest frame with a see-through verdict
  size_t occluded = 0, invalid = 0;  // diagnostics: readings classified O or I
  size_t cross_session_skipped = 0;  // elements of a previous session not judged: sigma_x unknown

  // README principle 6 (2): a frame in which the number of samples carrying the placement's own
  // identity reaches k_min and exceeds the number of see-through samples directly saw the placement
  // in place. The round did if any of its frames did; `latest_recognized` is the newest such frame.
  bool recognized = false;
  TimeStamp latest_recognized = 0;
  std::vector<ElementLearning> learning;  // per element, in the recognized frames
  double own_hits = 0.0, foreign_hits = 0.0;  // labelled hits in the recognized frames

  // README principle 3: the rays of the round that landed on the placement's own identity (support)
  // and the rays that passed it (contradiction); one per (frame, element).
  size_t support_rays = 0, contradict_rays = 0;
  TimeStamp latest_support = 0;  // newest frame with a supporting ray
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
