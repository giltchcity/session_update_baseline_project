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

  uint32_t frames() const { return hits + through; }
  void addFrame(bool see_through, double range);
  double meanRange() const { return range_count ? range_sum / range_count : 0.0; }
};

/** The pooled first-order correlation of the verdict sequences of all elements (the AR(1)
 * parameter of (7) at the frame scale). 0 where fewer than two adjacent pairs exist. */
double pooledCorrelation(const std::vector<ElementTally>& tallies);

/**
 * README principle 10, eqs. (11), (12): ln p(d | H_dup) - ln p(d | H_sep) of the normal offset d
 * between two elements of the same identity. H_dup: d ~ N(b, sigma^2) with the unknown
 * cross-session bias b uniform on [-B, B]; H_sep: d is unconstrained inside the observable band
 * (-band, band), i.e. uniform with density 1 / (2 band).
 */
double duplicateLogRatio(double offset, double sigma, double bias_bound, double band);

}  // namespace khronos::surface
