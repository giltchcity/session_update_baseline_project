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
#include "session_core/model/range_model.h"
#include "session_core/model/round_model.h"

namespace khronos {

/**
 * README principle 5, (6b), (8): the bridge from the registry (backend) to the active window. The
 * registry publishes after each round what the window's decisions need of the model:
 *  - per identity the right end of (5t) of its closed placements (frames acquired at or before it
 *    belong to a closed placement and are not reconstructed into a successor),
 *  - psi and the look statistics of principle 6 (the frame-pair attribution and the visible-motion
 *    recursion use the same likelihood),
 *  - the first frame of the session.
 * The visible-motion statistics are learned by the active window's tracker and live here too.
 */
class FrameAttribution {
 public:
  using Ptr = std::shared_ptr<FrameAttribution>;
  using ConstPtr = std::shared_ptr<const FrameAttribution>;

  struct Snapshot {
    std::map<size_t, TimeStamp> closed_through;
    model::RangeModel psi;
    std::shared_ptr<const model::RoundModel> rounds;
    TimeStamp session_start = std::numeric_limits<TimeStamp>::max();
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

  /** The active window notes every frame it processes. */
  void noteFrame(TimeStamp stamp) { last_frame_.store(stamp); }

  model::MotionModel& motion() { return motion_; }
  const model::MotionModel& motion() const { return motion_; }

 private:
  mutable std::mutex mutex_;
  std::shared_ptr<const Snapshot> snapshot_;
  std::atomic<TimeStamp> last_frame_{0};
  model::MotionModel motion_;
};

}  // namespace khronos
