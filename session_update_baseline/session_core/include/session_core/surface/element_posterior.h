#pragma once

#include <cstdint>
#include <limits>

namespace khronos::surface {

/**
 * README principles 9 and 11: the per-frame first-return verdicts of one surface element over the
 * frames of a session. A frame contributes at most one verdict: a hit if a pixel of the element's
 * footprint is explained by the element, else see-through if the footprint's non-occluded pixels
 * all pass it. Frames are the rounds of one look (7) for the deletion of a record, and the votes
 * of the majority (10b) for its display.
 */
struct ElementTally {
  uint32_t hits = 0, through = 0;
  double range_sum = 0.0;                   // mean range of the viewing frames
  uint32_t range_count = 0;
  double min_range = std::numeric_limits<double>::infinity();  // the nearest reading that reaches it
  double predicted_sum = 0.0;               // sum of the see-through rates the measurement model predicts
  double precision = 0.0;                   // (9): sum of 1/sigma_eff^2 of the supporting echoes
  // Principle 11: the readings that reach the element and, of them, those cut off by a surface
  // within the truncation band in front of it (the element then shares that surface's zero crossing).
  uint32_t arrived = 0, blocked_in_band = 0;
  // Principle 11 (14f): a valid reading put the surface within tau of the element.
  bool near_reading = false;

  uint32_t frames() const { return hits + through; }
  void addFrame(bool see_through, double range, double predicted) {
    if (see_through) ++through;
    else ++hits;
    range_sum += range;
    ++range_count;
    if (range < min_range) min_range = range;
    predicted_sum += predicted;
  }
  void addReading(bool blocked_within_band) {
    ++arrived;
    if (blocked_within_band) ++blocked_in_band;
  }
  /** Principle 11 (representation resolution): the majority of the readings that reach the element
   * are cut off by a surface within the band -- the maximum a posteriori choice. */
  bool blockedMajority() const { return 2 * blocked_in_band > arrived; }
  double meanRange() const { return range_count ? range_sum / range_count : 0.0; }
  /** m_0 of the frames: the mean see-through rate the measurement model predicts for them. */
  double meanPredicted() const { return range_count ? predicted_sum / range_count : 0.0; }
  /** The distance of the nearest reading (0 when none reached the element). */
  double nearestRange() const {
    return min_range < std::numeric_limits<double>::infinity() ? min_range : 0.0;
  }
};

}  // namespace khronos::surface
