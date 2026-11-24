#pragma once
// A physical object node's presence interval from the session's own stored pixels: the evidence
// before its first and after its last real observation (SurfaceEvidence::checkProjected on its
// sampled surface, through the round's stamp, with the change detection's temporal estimator in
// physical coverage), and the rule that turns it into the interval (the reconciler's: an observed
// absence starts or ends presence halfway between it and the nearest evidence of presence, the
// estimate of minimum expected risk; without one, an unobserved site is not an absent object).
#include <cstddef>

#include "khronos/backend/change_detection/ray_change_detector.h"
#include "khronos/common/common_types.h"
#include "session_core/evidence/surface_evidence.h"

namespace khronos::session {

struct PresenceEvidence {
  TimeStamp first_absent = 0, first_persistent = 0;  // before the first observation
  TimeStamp last_absent = 0, last_persistent = 0;    // after the last observation
};

struct PresenceQuery {
  TimeStamp margin_ns = 0;  // only frames at least this far from the observation (time_filtering_threshold)
  size_t subsampling = 100;  // every n-th surface vertex (query_subsampling)
};

PresenceEvidence observePresence(const KhronosObjectAttributes& attrs, size_t physical_id,
                                 const SurfaceEvidence& evidence,
                                 const SurfaceEvidence::Snapshot& snapshot,
                                 const RayChangeDetector& estimator, const PresenceQuery& query,
                                 TimeStamp through);

// Writes the interval into first/last_observed_ns, keeping the real observation bounds apart
// (persistObservationBounds). conservative: without an absence, the evidence itself.
void estimatePresence(const PresenceEvidence& evidence, bool conservative, KhronosObjectAttributes& attrs);

}  // namespace khronos::session
