#pragma once
// When a physical object begins to be present, from the session's own stored pixels: the evidence
// before its first real observation (SurfaceEvidence::checkProjected on its sampled surface, with
// the change detection's temporal estimator in physical coverage) and the rule that dates it (the
// reconciler's: an observed absence starts presence halfway between it and the nearest evidence of
// presence, the estimate of minimum expected risk; without one, an unobserved site is not an absent
// object). Its end is the state layer's alone (PersistentObjectState::applyPhysicalGeometry): an
// identity stays present while it has a current state and leaves when its last state closes. No
// segment of an identity ends on its own, so a moved object is never absent between its segments.
#include <cstddef>

#include "khronos/backend/change_detection/ray_change_detector.h"
#include "khronos/common/common_types.h"
#include "session_core/evidence/surface_evidence.h"

namespace khronos::session {

struct PresenceEvidence {
  TimeStamp first_absent = 0, first_persistent = 0;  // before the first observation
};

struct PresenceQuery {
  TimeStamp margin_ns = 0;  // only frames at least this far from the observation (time_filtering_threshold)
  size_t subsampling = 100;  // every n-th surface vertex (query_subsampling)
};

PresenceEvidence observePresence(const KhronosObjectAttributes& attrs, size_t physical_id,
                                 const SurfaceEvidence& evidence,
                                 const SurfaceEvidence::Snapshot& snapshot,
                                 const RayChangeDetector& estimator, const PresenceQuery& query);

// Writes the presence into first/last_observed_ns, keeping the real observation bounds apart
// (persistObservationBounds): from the dated start, open-ended (the state layer ends it).
// conservative: without an absence, the first observation itself.
void estimatePresence(const PresenceEvidence& evidence, bool conservative, KhronosObjectAttributes& attrs);

}  // namespace khronos::session
