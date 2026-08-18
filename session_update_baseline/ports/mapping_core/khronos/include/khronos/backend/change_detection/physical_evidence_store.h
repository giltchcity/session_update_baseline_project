/** -----------------------------------------------------------------------------
 * Copyright (c) 2024 Massachusetts Institute of Technology.
 * All Rights Reserved.
 * -------------------------------------------------------------------------- */

#pragma once

#include <cstddef>
#include <memory>
#include <limits>
#include <mutex>

#include "khronos/common/common_types.h"

namespace khronos {

struct FrameData;

/** The identity carried by the measured endpoint at an exact image pixel. */
enum class EndpointClass {
  kUnavailable,
  kInvalid,
  kBackground,
  kUnidentifiedObject,
  kPhysical,
};

struct EndpointEvidence {
  EndpointClass type = EndpointClass::kUnavailable;
  int physical_id = 0;
  // Measured depth of the endpoint in metres. NaN means unavailable.
  float measured_depth_m = std::numeric_limits<float>::quiet_NaN();
};

/**
 * @brief Session-local, copy-on-write store for endpoint identity evidence.
 *
 * A frame is reduced to an RLE identity mask plus the sensor projection and
 * pose needed to query it. Raw input images are not retained. The store is
 * intentionally not serialized: a new session gets a fresh store while its
 * loaded scene memory continues through the ordinary DSG/map path.
 */
class PhysicalEvidenceStore {
  struct Storage;

 public:
  using Ptr = std::shared_ptr<PhysicalEvidenceStore>;
  using ConstPtr = std::shared_ptr<const PhysicalEvidenceStore>;

  /** Immutable view used for one complete change-detection update. */
  class Snapshot {
   public:
    Snapshot() = default;

    /**
     * @brief Classify the measured endpoint at the pixel onto which an old
     * queried world-space surface projects.
     *
     * The point is the queried physical surface, not a background-mesh ray
     * target. Geometry still decides near/through/occluded; this lookup only
     * restores the endpoint identity at that exact timestamp and pixel.
     * Timestamps are exact: no later or nearest frame is substituted.
     */
    EndpointEvidence classify(TimeStamp stamp, const Point& world_point) const;

    size_t numFrames() const;
    size_t numRuns() const;
    explicit operator bool() const { return static_cast<bool>(storage_); }

   private:
    friend class PhysicalEvidenceStore;
    explicit Snapshot(std::shared_ptr<const Storage> storage);

    std::shared_ptr<const Storage> storage_;
  };

  PhysicalEvidenceStore();

  /**
   * @brief Reduce one full ActiveWindow output frame into typed endpoint runs.
   *
   * Hydra's GraphBuilder calls its default PoseGraphFromOdom tracker once for
   * every ActiveWindowOutput. Every emitted AGENT timestamp is therefore one
   * of these full-output timestamps (possibly the preceding output when an
   * odometry edge is formed), so exact Snapshot lookup is the required
   * contract. The terminal duplicate timestamp replaces the same map entry.
   * @return True if projection and image dimensions were valid and the frame
   * was stored. A repeated timestamp atomically replaces the previous frame.
   */
  bool ingest(const FrameData& data);

  Snapshot snapshot() const;
  size_t numFrames() const;
  size_t numRuns() const;

 private:
  mutable std::mutex mutex_;
  std::shared_ptr<const Storage> storage_;
};

}  // namespace khronos
