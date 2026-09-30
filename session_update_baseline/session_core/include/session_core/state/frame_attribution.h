#pragma once

#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <utility>

#include "khronos/common/common_types.h"
#include "session_core/model/motion_model.h"
#include "session_core/model/persistence_prior.h"
#include "session_core/model/range_model.h"
#include "session_core/model/round_model.h"

namespace khronos {

/**
 * README principle 5, (6b), (8): the bridge from the registry (backend) to the active window. The
 * registry publishes after each round what the window's decisions need of the model:
 *  - per identity the right end of (5t) of its closed placements (frames acquired at or before it
 *    belong to a closed placement and are not reconstructed into a successor),
 *  - per identity the predictive hazard of the persistence prior (5c),
 *  - psi and the round model of principle 6 (frame-pair attribution and the visible-motion
 *    recursion use the same likelihood),
 *  - the first frame of the session,
 *  - the interval between the last two evidence rounds: the horizon "until the next round of
 *    evidence" of the write commitment (principle 5, assumption 4).
 * The visible-motion statistics are learned by the active window's tracker and live here too.
 */
class FrameAttribution {
 public:
  using Ptr = std::shared_ptr<FrameAttribution>;
  using ConstPtr = std::shared_ptr<const FrameAttribution>;

  struct Snapshot {
    std::map<size_t, TimeStamp> closed_through;
    std::map<size_t, model::PersistencePrior::Hazard> hazards;
    model::RangeModel psi;
    std::shared_ptr<const model::RoundModel> rounds;
    TimeStamp session_start = std::numeric_limits<TimeStamp>::max();
    double round_seconds = 0.0;  // Delta_round; 0 until two rounds have run
  };

  void publish(Snapshot snapshot) {
    auto next = std::make_shared<const Snapshot>(std::move(snapshot));
    std::lock_guard<std::mutex> lock(mutex_);
    snapshot_ = std::move(next);
  }

  /** The newest published snapshot; null before the first round. */
  std::shared_ptr<const Snapshot> snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return snapshot_;
  }

  // Right end of (5t) of a closed placement of this identity; 0 when it has none.
  TimeStamp closedThrough(size_t physical_id) const {
    const auto current = snapshot();
    if (!current) return 0;
    const auto found = current->closed_through.find(physical_id);
    return found == current->closed_through.end() ? 0 : found->second;
  }

  /** q = 1 - S(seconds) of the identity's predictive hazard (5c). While the registry has no
   * hazard for the identity, the Jeffreys posterior Gamma(1/2, E) of the time E it has been
   * observed in place (`provisional_exposure`) stands in; 0 while that is unknown too. */
  static double changeProbability(const Snapshot& snapshot, size_t physical_id, double seconds,
                                  double provisional_exposure = 0.0) {
    if (!(seconds >= 0.0)) return 0.0;
    const auto found = snapshot.hazards.find(physical_id);
    if (found != snapshot.hazards.end() && found->second.valid()) {
      return -std::expm1(-found->second.shape * std::log1p(seconds / found->second.rate));
    }
    if (!(provisional_exposure > 0.0)) return 0.0;
    return -std::expm1(-0.5 * std::log1p(seconds / provisional_exposure));
  }

  model::MotionModel& motion() { return motion_; }
  const model::MotionModel& motion() const { return motion_; }

 private:
  mutable std::mutex mutex_;
  std::shared_ptr<const Snapshot> snapshot_;
  model::MotionModel motion_;
};

}  // namespace khronos
