/** -----------------------------------------------------------------------------
 * Copyright (c) 2024 Massachusetts Institute of Technology.
 * All Rights Reserved.
 * -------------------------------------------------------------------------- */

#pragma once

#include <cstddef>
#include <memory>
#include <limits>
#include <mutex>
#include <string>
#include <optional>
#include <unordered_set>
#include <vector>

#include <hydra/input/sensor.h>

#include "khronos/common/common_types.h"

namespace khronos {

struct FrameData;

// Finite-count view of the existing calibrated absence measurement (README M1e).
struct PhysicalAbsenceLookLikelihood {
  double log_ratio = 0.0;  // fresh-weighted log p(counts|empty)/p(counts|present)
  bool has_measurement = false;
  bool calibrated_source = false;
};
namespace session { class SurfaceEvidence; }
// The latest look of a tested state as the session's surface evidence (`owner`) measured it.
// The population statistics of the observed-absence test across sessions (the sensor's own
// in-place look statistics): saved with the session state, loaded by the next session.
bool saveAbsenceSensorStatistics(const std::string& path);
bool loadAbsenceSensorStatistics(const std::string& path);

PhysicalAbsenceLookLikelihood physicalAbsenceLookLikelihood(
    const session::SurfaceEvidence* owner, size_t physical_id, int state_slot, TimeStamp stamp,
    uint64_t state_id = 0);

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
  // Unit vector from the sensor to the queried point, world frame.
  Eigen::Vector3f view_direction_world = Eigen::Vector3f::Zero();
  // DIAG (Stage 0 of derivation_joint_sensor_20261006; no decision reads it): the segmentation
  // label of a pixel without valid depth, whose endpoint above is kInvalid. kUnavailable when the
  // pixel has depth or is outside the image.
  EndpointEvidence no_depth_label;
};

// M1m: conditional on stored millimetre range and supplied identity;
// these are not whole-object presence probabilities.
struct ProjectedRelationProbabilities {
  double unavailable = 0.0;
  double invalid = 0.0;
  double occluded = 0.0;
  double supported = 0.0;
  double free = 0.0;
  double background = 0.0;
  double other = 0.0;
  double unidentified = 0.0;
};
ProjectedRelationProbabilities projectedRelationProbabilities(
    const ProjectedEndpointEvidence& measurement, size_t physical_id,
    float matching_tolerance);

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
    // The earliest stored frame that measured a pixel of the identity (0: none).
    TimeStamp firstMeasured(size_t physical_id) const;
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
  /** Drop every stored frame (session end, once nothing queries the store any more). */
  void clear();
  size_t numFrames() const;
  size_t numRuns() const;

 private:
  mutable std::mutex mutex_;
  std::shared_ptr<const Storage> storage_;
};

// An immutable view of the store for one round (none: no store).
using PhysicalEvidenceSnapshot = std::optional<PhysicalEvidenceStore::Snapshot>;

// Support and contradiction on a state's surface from the stored pixels, and the observed-absence
// test's look on it (SurfaceEvidence::countCurrent).
struct SurfaceEvidenceCounts {
  size_t support_rays = 0;
  size_t contradiction_rays = 0;
  size_t surface_samples = 0;
  size_t contradicted_surface_samples = 0;
  bool absence_coverage_sufficient = false;  // set only by the observed-absence test
  std::unordered_set<size_t> support_indices;
  std::unordered_set<size_t> contradiction_indices;

  // Per-(sample, pixel) six-class evidence votes for the verification ledger. These are NOT
  // unique-pixel counts: one pixel reading several samples votes several times, so ratios between
  // vote classes are comparable, but the sums differ from support_rays/contradiction_rays by design.
  size_t supported_votes = 0;
  size_t free_space_votes = 0;
  size_t replaced_by_other_votes = 0;
  size_t replaced_by_background_votes = 0;
  size_t occluded_votes = 0;
  size_t unobserved_samples = 0;
  TimeStamp latest_support_stamp = 0;  // Actual sensor time, never reducer/check time.

  // Observed-absence test over reliable surface samples.
  size_t reliable_samples = 0;
  size_t reliable_in_view = 0;
  size_t reliable_seen_through = 0;
  float absence_llr = 0.f;
};

}  // namespace khronos
