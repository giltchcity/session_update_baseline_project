#pragma once
// Evidence about a physical object's surface from the frames the session stored (README M1m):
// support and contradiction on the state's own surface, and the observed-absence test (eq. 7)
// on the same frames. The session's own measurement; one instance per mapping session, which
// owns the absence records it keeps per tested state.
#include <cstdint>
#include <limits>

#include <spark_dsg/bounding_box.h>
#include <spark_dsg/mesh.h>

#include "khronos/backend/change_detection/ray_verificator.h"
#include "khronos/common/common_types.h"

namespace khronos::session {

class SurfaceEvidence {
 public:
  using Snapshot = RayVerificator::PhysicalEvidenceSnapshot;
  using Counts = RayVerificator::SurfaceEvidenceCounts;
  using CheckResult = RayVerificator::CheckResult;

  struct Config {
    // Depth tolerance of a measurement on the surface [m] (the change detection's).
    float depth_tolerance = 0.3f;
    // Without an observed-absence test: the share of the surface contradicted that ends a state.
    float min_absent_surface_fraction = 0.2f;
    // Spacing of the surface samples and their match distance [m].
    float surface_match_tolerance = 0.05f;
    // Steepest incidence a sample is judged at [deg] (with real sensor noise).
    float max_absence_incidence_deg = 60.f;
    // Judged samples for which a look's share stands for its finite counts.
    int min_judged_samples = 30;
    bool real_sensor_noise = true;
    // The values the change detection is configured with (the same keys).
    static Config of(const RayVerificator::Config& rays);
  };

  explicit SurfaceEvidence(const Config& config);

  const Config config;

  // Only measurements after the state's latest support can establish its subsequent
  // disappearance; earlier free space belongs to an earlier world. prior_log_odds: logit of the
  // measured state's change prior where the round tests one unresolved relation (an inherited
  // state); 0 otherwise. Without stored frames (offline tools, unit fixtures) the change
  // detection's mesh rays (`rays`) stand in, decided by their majority.
  Counts countCurrent(size_t physical_id, const spark_dsg::Mesh& mesh,
                      const spark_dsg::BoundingBox& bbox, const Snapshot& snapshot,
                      float map_resolution, uint64_t last_support, uint64_t latest,
                      const RayVerificator* rays = nullptr, bool* projected = nullptr,
                      int state_slot = 0, uint64_t state_birth = 0, double prior_log_odds = 0.0,
                      uint64_t state_id = 0) const;

  // Support and contradiction on the state's surface from the stored pixels. Surface sampling
  // uses the map voxel size; sensor pixels are counted once.
  Counts countProjected(size_t physical_id, const spark_dsg::Mesh& mesh,
                        const spark_dsg::BoundingBox& bbox, const Snapshot& snapshot,
                        float map_resolution, uint64_t earliest, uint64_t latest) const;

  // The observed-absence test (eq. 7) on the same frames; stores its counts in `counts`.
  void applyObservedAbsence(size_t physical_id, const spark_dsg::Mesh& mesh,
                            const spark_dsg::BoundingBox& bbox, const Snapshot& snapshot,
                            uint64_t earliest, uint64_t latest, Counts& counts,
                            int state_slot = 0, uint64_t state_birth = 0,
                            double prior_log_odds = 0.0, uint64_t state_id = 0) const;

  // One point's measurements in the stored pixels.
  CheckResult checkProjected(const Point& point, size_t physical_id, const Snapshot& snapshot,
                             uint64_t earliest, uint64_t latest) const;

  float minAbsenceCos() const;

 private:
  // The absence records of this session (a token, not an address: an allocator may reuse one).
  const uint64_t absence_owner_;
};

}  // namespace khronos::session
