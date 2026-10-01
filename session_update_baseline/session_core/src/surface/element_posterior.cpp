#include "session_core/surface/element_posterior.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace khronos::surface {

void ElementTally::addFrame(bool see_through, double range) {
  const uint8_t verdict = see_through ? 2 : 1;
  if (last == 1 && verdict == 1) ++hh;
  else if (last == 1 && verdict == 2) ++ht;
  else if (last == 2 && verdict == 1) ++th;
  else if (last == 2 && verdict == 2) ++tt;
  last = verdict;
  if (see_through) ++through;
  else ++hits;
  range_sum += range;
  ++range_count;
}

double pooledCorrelation(const std::vector<ElementTally>& tallies) {
  double numerator = 0.0, denominator = 0.0;
  double pairs_total = 0.0;
  for (const auto& t : tallies) {
    const double n = t.frames();
    const double pairs = static_cast<double>(t.hh) + t.ht + t.th + t.tt;
    if (n < 3.0 || t.through == 0 || t.hits == 0 || pairs < 1.0) continue;
    // The verdicts as x in {0, 1} (1 = see-through); x_t x_{t+1} = tt.
    const double mean = static_cast<double>(t.through) / n;
    numerator += t.tt - mean * (2.0 * t.tt + t.th + t.ht) + mean * mean * pairs;
    denominator += static_cast<double>(t.through) * static_cast<double>(t.hits) / n;
    pairs_total += pairs;
  }
  if (pairs_total < 2.0 || !(denominator > 0.0)) return 0.0;
  return numerator / denominator;
}

}  // namespace khronos::surface
