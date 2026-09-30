/** -----------------------------------------------------------------------------
 * Copyright (c) 2024 Massachusetts Institute of Technology.
 * All Rights Reserved.
 * -------------------------------------------------------------------------- */

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <limits>
#include <mutex>
#include <vector>

#include <hydra/input/sensor.h>

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

struct ProjectedEndpointEvidence {
  EndpointEvidence endpoint;
  float query_range_m = std::numeric_limits<float>::quiet_NaN();
  uint32_t pixel_index = std::numeric_limits<uint32_t>::max();
  // Valid device range of the sensor that made this measurement (README (6e) normalisation).
  float sensor_min_range = 0.f;
  float sensor_max_range = 0.f;
  // Unit vector from the sensor to the queried point, world frame.
  Eigen::Vector3f view_direction_world = Eigen::Vector3f::Zero();
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

    // Direct sensor visibility, independent of whether meshing produced a ray
    // endpoint in the queried surface's spatial hash block.
    ProjectedEndpointEvidence project(TimeStamp stamp, const Point& world_point) const;
    std::vector<TimeStamp> timestamps(TimeStamp earliest, TimeStamp latest) const;

    /**
     * @brief Expand one stored frame's measured ranges into a dense row-major
     * image in millimetres (0 = invalid), plus the projection needed to query
     * it. Used by the session-end consolidation, which needs every pixel of a
     * surface element's footprint rather than the single pixel of project().
     * @return False if the timestamp is not stored.
     */
    bool denseRange(TimeStamp stamp,
                    uint32_t& width,
                    uint32_t& height,
                    Eigen::Isometry3f& sensor_T_world,
                    hydra::Sensor::ConstPtr& sensor,
                    std::vector<uint16_t>& range_mm) const;

    size_t numFrames() const;
    size_t numRuns() const;
    explicit operator bool() const { return static_cast<bool>(storage_); }

   private:
    friend class PhysicalEvidenceStore;
    explicit Snapshot(std::shared_ptr<const Storage> storage, TimeStamp latest);

    std::shared_ptr<const Storage> storage_;
    TimeStamp latest_ = std::numeric_limits<TimeStamp>::max();
  };

  PhysicalEvidenceStore();

  /**
   * @brief Reduce one full ActiveWindow output frame into typed endpoint runs.
   *
   * Hydra's GraphBuilder calls its default PoseGraphFromOdom tracker once for
   * every ActiveWindowOutput. Every emitted AGENT timestamp is therefore one
   * of these full-output timestamps (possibly the preceding output when an
   * odometry edge is formed), so exact Snapshot lookup is the required
   * contract. Repeated timestamps must have identical protocol measurements:
   * dimensions, original sensor pose, sensor configuration, millimetre ranges
   * and typed endpoint identities. Identical replay returns true and preserves
   * the old immutable record; conflicting replay throws before publication.
   * The terminal extraction reads the same latest frame, so its replay is valid.
   * Pixel count must fit uint32_t; UINT32_MAX remains unavailable as a pixel ID.
   * @return True after storing a new valid frame or accepting identical replay.
   * Invalid measurement representation or missing projection data throws.
   */
  bool ingest(const FrameData& data);

  /** README principle 8: called with a snapshot through the new frame after each newly stored
   * frame (not after an identical replay), outside the store's lock. The online calibration of
   * the range model is fed from here. */
  using IngestObserver = std::function<void(const Snapshot&, TimeStamp)>;
  void setIngestObserver(IngestObserver observer);

  Snapshot snapshot(TimeStamp latest = std::numeric_limits<TimeStamp>::max()) const;
  /** Drop every stored frame (session end, once nothing queries the store any more). */
  void clear();
  size_t numFrames() const;
  size_t numRuns() const;

 private:
  mutable std::mutex mutex_;
  std::shared_ptr<Storage> storage_;
  IngestObserver observer_;
};

}  // namespace khronos
