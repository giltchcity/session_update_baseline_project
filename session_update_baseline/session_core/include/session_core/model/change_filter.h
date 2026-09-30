#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include <nlohmann/json.hpp>

#include "session_core/model/model_math.h"

namespace khronos::model {

/**
 * README principle 3, eqs. (5r), (5t), (5e): the posterior of "this placement has changed" by the
 * Persistence Filter / Shiryaev forward recursion in odds form,
 *
 *   Lambda_n = (Lambda_{n-1} + q_n) / (1 - q_n) * LR_n,
 *   Lambda_0 = q^gap / (1 - q^gap)  (inherited placement),  0  (placement born in this session),
 *
 * with q_n the change probability of the round (Pi) and LR_n = L(changed) / L(in place) the round
 * likelihood ratio of principle 6. The closing and confirming thresholds are those of (5e) with the
 * global alpha. Every round is kept from the last confirmation, so that the change time has the
 * smoothing posterior (5t) and its (1 - alpha) shortest credible interval.
 */
class ChangeFilter {
 public:
  struct Round {
    uint64_t begin = 0, end = 0;  // the actual sensor-time interval (t_{k-1}, t_k] of the round
    double q = 0.0;               // change probability inside the round
    double log_lr = 0.0;          // ln LR
  };
  struct Interval {
    uint64_t left = 0, right = 0;
  };

  ChangeFilter() = default;
  /** `gap_odds` is q^gap / (1 - q^gap) for an inherited placement; 0 for a new one. The gap
   * interval ends at `start`; it begins at `gap_begin` (the previous session's boundary). */
  ChangeFilter(double gap_odds, uint64_t gap_begin, uint64_t start);

  /** The gap of an inherited placement ends where this session's first round begins. */
  void setGapEnd(uint64_t end) { gap_end_ = std::max(gap_end_, end); }

  /** One round. q in [0, 1), log_lr finite. Returns the odds after the round. */
  double update(uint64_t begin, uint64_t end, double q, double log_lr);

  double odds() const { return odds_; }
  /** (5e): kCommitH = changed (close), kCommitNotH = confirmed in place. */
  Commitment commitment() const { return decide(odds_); }

  /** Forget the rounds before now: a confirmation means the change, if any, came later. */
  void anchorAtConfirmation();

  /** (5t): the (1 - alpha) shortest credible interval of the change time given "changed". An
   * empty filter (no round) returns the gap interval. */
  Interval changeInterval() const;

  nlohmann::json toJson() const;
  static ChangeFilter fromJson(const nlohmann::json& value);

 private:
  double odds_ = 0.0;
  // The gap cell of an inherited placement (prior mass q^gap, no round evidence of its own).
  bool has_gap_ = false;
  double gap_q_ = 0.0;
  uint64_t gap_begin_ = 0, gap_end_ = 0;
  std::vector<Round> rounds_;
};

}  // namespace khronos::model
