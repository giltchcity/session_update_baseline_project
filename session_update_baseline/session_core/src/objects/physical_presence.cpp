// A physical object node's presence interval from the session's own stored pixels.
#include "session_core/objects/physical_presence.h"

#include <algorithm>
#include <limits>

#include "khronos/utils/khronos_attribute_utils.h"

namespace khronos::session {

PresenceEvidence observePresence(const KhronosObjectAttributes& attrs, const size_t physical_id,
                                 const SurfaceEvidence& evidence,
                                 const SurfaceEvidence::Snapshot& snapshot,
                                 const RayChangeDetector& estimator, const PresenceQuery& query) {
  PresenceEvidence result;
  if (!hasCurrentObjectMesh(attrs) || attrs.mesh.points.empty() || attrs.first_observed_ns.empty() ||
      attrs.last_observed_ns.empty()) {
    return result;
  }
  const TimeStamp first_seen = observationFirstStamp(attrs);
  const TimeStamp before_latest = first_seen > query.margin_ns ? first_seen - query.margin_ns : 0;
  RayVerificator::CheckResult before;
  const size_t step = std::max<size_t>(1, query.subsampling);
  for (size_t i = 0; i < attrs.mesh.numVertices(); i += step) {
    const Point point = attrs.mesh.pos(i) + attrs.bounding_box.world_P_center;
    before.merge(evidence.checkProjected(point, physical_id, snapshot, 0, before_latest));
  }
  const auto before_result = estimator.detectChanges(before, false, RayChangeDetector::CoverageMode::kPhysical);
  result.first_absent = before_result.closest_absent.value_or(0);
  result.first_persistent = before_result.furthest_persistent.value_or(0);
  return result;
}

void estimatePresence(const PresenceEvidence& e, const bool conservative, KhronosObjectAttributes& attrs) {
  if (attrs.first_observed_ns.empty() || attrs.last_observed_ns.empty()) return;
  persistObservationBounds(attrs);
  const TimeStamp first_seen = observationFirstStamp(attrs);
  // Only the closest evidence on either side of the appearance counts.
  const bool first_absent_valid = e.first_absent > 0 && e.first_absent < first_seen;
  bool first_persistent_valid = e.first_persistent > 0 && e.first_persistent < first_seen;
  if (first_absent_valid && first_persistent_valid && e.first_absent >= e.first_persistent) {
    first_persistent_valid = false;
  }
  const TimeStamp first_evidence = first_persistent_valid ? std::min(e.first_persistent, first_seen) : first_seen;
  attrs.first_observed_ns = {first_absent_valid ? e.first_absent + (first_evidence - e.first_absent) / 2
                                                : (conservative ? first_evidence : 0)};
  attrs.last_observed_ns = {std::numeric_limits<TimeStamp>::max()};
}

}  // namespace khronos::session
