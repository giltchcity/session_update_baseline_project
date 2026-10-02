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

/** What a placement's sample showed in a look that directly saw the placement in place (README
 * principle 6 (7s)): whether its identity landed on the placement's surface (h_e, one per look),
 * whether it was seen through or labelled as another identity (v_e, one per look), and the range
 * of that reading (the recorded distance rho_e of principle 12). */
struct ElementLearning {
  uint64_t key = 0;
  bool own = false, vetoed = false;
  float range = 0.f;
};

/**
 * One look of a placement (README principles 4 and 6): the verdicts of its samples, the counts that
 * decide whether the look directly saw the placement in place, and the learning records. These are
 * the intermediate quantities of the decision of block 1; the registry keeps the last of them.
 */
struct ElementRound {
  std::vector<ElementVerdict> verdicts;
  size_t num_elements = 0;       // elements of the placement that were queried
  TimeStamp latest_hit = 0;      // newest frame with a hit verdict
  TimeStamp first_through = 0;   // earliest frame with a see-through verdict
  size_t occluded = 0, invalid = 0;  // diagnostics: readings classified O or I
  size_t entity_passed = 0;      // readings beyond a sample that landed on the entity (O, principle 4)
  size_t cross_session_skipped = 0;  // elements of a previous session not judged: sigma_x unknown
  double delta_plus = 0.0;       // mean delta_+* (6e) of the classified readings of the look

  // README principle 6 (2): a look in which the number of samples whose identity landed on the
  // placement's surface (a hit with its own label, or a reading inside its extent with its own
  // identity) reaches k_min and exceeds the number of samples seen through directly saw the
  // placement in place. `own_samples` is also the support the placement gets from this look.
  size_t own_samples = 0, through_samples = 0, foreign_samples = 0;
  // Samples with a reading of the entity's own identity inside its extent in at least one frame of
  // the look (principle 13 (f): the placement has no support of its own identity).
  size_t extent_own = 0;
  bool recognized = false;
  TimeStamp first_own = 0, latest_own = 0;  // the frames of the first and the newest own-identity outcome
  std::vector<ElementLearning> learning;    // per sample, only for a look that directly saw it
  double labelled_samples = 0.0;            // samples hit by a physical identity (own or foreign)
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
