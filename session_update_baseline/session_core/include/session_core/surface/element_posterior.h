#pragma once

#include <cstdint>
#include <vector>

namespace khronos::surface {

/**
 * README principle 9: the per-frame first-return verdicts of one surface element over the frames of
 * a session (frames are the rounds of eq. (7)). A frame contributes at most one verdict: a hit if
 * a pixel of the element's footprint is explained by the element, else see-through if the
 * footprint's non-occluded pixels all pass it.
 */
struct ElementTally {
  uint32_t hits = 0, through = 0;
  uint8_t last = 0;                          // 0 none, 1 hit, 2 see-through
  uint32_t hh = 0, ht = 0, th = 0, tt = 0;   // adjacent verdict pairs (first, second)
  double range_sum = 0.0;                    // mean range of the viewing frames (bias of (6s))
  uint32_t range_count = 0;
  double precision = 0.0;                    // (9): sum of 1/sigma_eff^2 of the supporting echoes
  // Principle 11: the readings that reach the element and, of them, those cut off by a surface
  // within the truncation band in front of it (the element then shares that surface's zero crossing).
  uint32_t arrived = 0, blocked_in_band = 0;

  uint32_t frames() const { return hits + through; }
  void addFrame(bool see_through, double range);
  void addReading(bool blocked_within_band) {
    ++arrived;
    if (blocked_within_band) ++blocked_in_band;
  }
  /** Principle 11 (representation resolution): the majority of the readings that reach the element
   * are cut off by a surface within the band -- the maximum a posteriori choice. */
  bool blockedMajority() const { return 2 * blocked_in_band > arrived; }
  double meanRange() const { return range_count ? range_sum / range_count : 0.0; }
};

/** The pooled first-order correlation of the verdict sequences of all elements (the AR(1)
 * parameter of (7) at the frame scale). 0 where fewer than two adjacent pairs exist. */
double pooledCorrelation(const std::vector<ElementTally>& tallies);

}  // namespace khronos::surface
