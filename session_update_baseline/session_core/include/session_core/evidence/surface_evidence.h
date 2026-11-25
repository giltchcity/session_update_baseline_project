#pragma once
// Evidence about a physical object's surface from the frames the session stored (README M1m):
// support and contradiction on the state's own surface, and the observed-absence test (eq. 7)
// on the same frames. The session's own measurement; one instance per mapping session, which
// owns the absence records it keeps per tested state.
#include <cstdint>
#include <limits>
#include <vector>

#include <spark_dsg/bounding_box.h>
#include <spark_dsg/mesh.h>

#include "khronos/common/common_types.h"
#include "session_core/evidence/physical_evidence_store.h"

namespace khronos::session {

class SurfaceEvidence {
 public:
  using Snapshot = PhysicalEvidenceSnapshot;
  using Counts = SurfaceEvidenceCounts;

  struct Config {
    // Depth tolerance of a measurement on the surface [m].
    float depth_tolerance = 0.1f;
    // Without an observed-absence test: the share of the surface contradicted that ends a state.
    float min_absent_surface_fraction = 0.2f;
    // Spacing of the surface samples and their match distance [m].
    float surface_match_tolerance = 0.05f;
    // Steepest incidence a sample is judged at [deg] (with real sensor noise).
    float max_absence_incidence_deg = 60.f;
    // Judged samples for which a look's share stands for its finite counts.
    int min_judged_samples = 30;
    bool real_sensor_noise = true;
  };

  explicit SurfaceEvidence(const Config& config);

  const Config config;

  // Only measurements after the state's latest support can establish its subsequent
  // disappearance; earlier free space belongs to an earlier world. prior_log_odds: logit of the
  // measured state's change prior where the round tests one unresolved relation (an inherited
  // state); 0 otherwise. Without stored frames there is no measurement: no support, no contradiction.
  Counts countCurrent(size_t physical_id, const spark_dsg::Mesh& mesh,
                      const spark_dsg::BoundingBox& bbox, const Snapshot& snapshot,
                      float map_resolution, uint64_t last_support, uint64_t latest,
                      bool* projected = nullptr,
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

  // When an identity began to be present: the observed-absence test (eq. 7) run backwards over the
  // rounds before its first observation. The same samples of its surface, the same measurement (a
  // sample on the surface whatever the label, seen through at a judged incidence, otherwise no
  // verdict), one look per round (the last one ends at the first observation, so the frames of that
  // round before it count), Page's statistic with each sample speaking once, committed at 99:1.
  // What the identity's own frames from its first observation on teach, as the absence test learns it
  // for a state, decides what counts: a sample those frames see through while they identify the
  // object in place is no evidence, and the present model is the object's own in-place looks shrunk
  // towards the population. `absent`: the looks up to some round show the site empty; the change lies
  // between that run's latest measurement (`last_absent`) and the first observation (`next_present`).
  // Without such a run the site was never seen empty: the object stands from the start. Called once
  // per round (the frames up to `latest`); round_ends: the session's round stamps in time order.
  struct PresenceStart {
    bool absent = false;
    uint64_t last_absent = 0;
    uint64_t next_present = 0;
  };
  PresenceStart presenceStart(size_t physical_id, const spark_dsg::Mesh& mesh, const spark_dsg::BoundingBox& bbox,
                              const Snapshot& snapshot, const std::vector<uint64_t>& round_ends,
                              uint64_t first_seen, uint64_t latest) const;

  float minAbsenceCos() const;

 private:
  // The absence records of this session (a token, not an address: an allocator may reuse one).
  const uint64_t absence_owner_;
};

}  // namespace khronos::session
