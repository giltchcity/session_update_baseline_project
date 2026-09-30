#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "khronos/common/common_types.h"

namespace khronos {

/**
 * README principle 12, eq. (15b): V_free, the free space each session actually observed, as sparse
 * voxel records at the map resolution with the session key and the acquisition times. A voxel
 * record says that rays passed through the voxel, at least one voxel-margin in front of the first
 * return, between `first` and `last` of that session. It is the cross-session counterpart of the
 * stored frames in the exclusion evidence E_{h->o} of principle 13 (c).
 */
class FreeSpaceRecords {
 public:
  struct Record {
    uint32_t session = 0;
    TimeStamp first = 0, last = 0;
  };

  FreeSpaceRecords() = default;
  explicit FreeSpaceRecords(float voxel) : voxel_(voxel) {}

  float voxel() const { return voxel_; }
  size_t size() const { return voxels_.size(); }
  uint32_t numSessions() const { return num_sessions_; }

  /** Key of the voxel containing `point`. */
  uint64_t keyOf(const Eigen::Vector3f& point) const;

  /** Start the records of a new session; returns its key. */
  uint32_t beginSession();
  /** The voxel was observed free in the current session at `stamp`. */
  void markFree(uint64_t key, TimeStamp stamp);
  /** Merge the records of a worker into this store (same session). */
  void merge(const FreeSpaceRecords& other);

  /** Whether the voxel of `point` was observed free in a stored session at some time of
   * [from, to], the span in which the placement under test was in place. */
  bool freeDuring(const Eigen::Vector3f& point, TimeStamp from, TimeStamp to) const;

  nlohmann::json toJson() const;
  static FreeSpaceRecords fromJson(const nlohmann::json& value);

 private:
  float voxel_ = 0.f;
  uint32_t num_sessions_ = 0;
  // Per voxel the records of the sessions that saw it free (session keys ascend).
  std::unordered_map<uint64_t, std::vector<Record>> voxels_;
};

}  // namespace khronos
