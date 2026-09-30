#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <utility>

#include "khronos/common/common_types.h"

namespace khronos {

/**
 * README (4.0) P5, (6b), (5b): which observations belong to which placement. The registry
 * publishes, per physical identity, the latest actual support of its closed placements
 * (w_closed of (5b)); an observation of that identity acquired at or before this time belongs to
 * a closed placement and is not reconstructed into a successor. The extractor reads this table
 * when it selects the frames of a static reconstruction.
 */
class FrameAttribution {
 public:
  using Ptr = std::shared_ptr<FrameAttribution>;
  using ConstPtr = std::shared_ptr<const FrameAttribution>;

  void publish(std::map<size_t, TimeStamp> closed_through) {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_through_ = std::move(closed_through);
  }

  // Latest actual support of a closed placement of this identity; 0 when it has none.
  TimeStamp closedThrough(size_t physical_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = closed_through_.find(physical_id);
    return found == closed_through_.end() ? 0 : found->second;
  }

 private:
  mutable std::mutex mutex_;
  std::map<size_t, TimeStamp> closed_through_;
};

}  // namespace khronos
