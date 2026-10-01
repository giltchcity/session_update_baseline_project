#pragma once

#include <atomic>
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
    // Per semantic class with data, the predictive hazard S_c of a member (principle 5: a reading
    // without identity is judged by the survival of its semantic class).
    std::map<int, model::PersistencePrior::Hazard> class_hazards;
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

  /** q = 1 - S_c(seconds) of a semantic class (5c); 0 while the class has no data. */
  static double classChangeProbability(const Snapshot& snapshot, int cls, double seconds) {
    const auto found = snapshot.class_hazards.find(cls);
    if (found == snapshot.class_hazards.end() || !found->second.valid() || !(seconds >= 0.0)) return 0.0;
    return -std::expm1(-found->second.shape * std::log1p(seconds / found->second.rate));
  }

  /** dt_f of principle 7: the interval between adjacent frames of the stream the mapper fuses. The
   * active window notes every frame it processes; the interval is that of the last two frames. */
  void noteFrame(TimeStamp stamp) {
    const TimeStamp previous = last_frame_.exchange(stamp);
    if (previous > 0 && stamp > previous) {
      frame_interval_.store(static_cast<double>(stamp - previous) * 1e-9);
    }
  }
  double frameInterval() const { return frame_interval_.load(); }

  model::MotionModel& motion() { return motion_; }
  const model::MotionModel& motion() const { return motion_; }

 private:
  mutable std::mutex mutex_;
  std::shared_ptr<const Snapshot> snapshot_;
  std::atomic<TimeStamp> last_frame_{0};
  std::atomic<double> frame_interval_{0.0};
  model::MotionModel motion_;
};

}  // namespace khronos
