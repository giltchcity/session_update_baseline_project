// A physical object node's presence interval from the session's own stored pixels.
#include "session_core/objects/physical_presence.h"

#include <limits>

#include "khronos/utils/khronos_attribute_utils.h"

namespace khronos::session {

TimeStamp openPresence(KhronosObjectAttributes& attrs) {
  persistObservationBounds(attrs);
  attrs.last_observed_ns = {std::numeric_limits<TimeStamp>::max()};
  return observationFirstStamp(attrs);
}

std::optional<TimeStamp> presenceStart(const SurfaceEvidence& evidence, const SurfaceEvidence::Snapshot& snapshot,
                                       const std::vector<uint64_t>& round_ends, const size_t physical_id,
                                       const KhronosObjectAttributes& attrs, const TimeStamp first_seen,
                                       const TimeStamp latest) {
  if (!snapshot) return std::nullopt;
  const auto stored = snapshot->timestamps(0, std::numeric_limits<TimeStamp>::max());
  if (stored.empty() || first_seen < stored.front()) return std::nullopt;
  if (!hasCurrentObjectMesh(attrs) || attrs.mesh.points.empty()) return std::nullopt;
  const auto start = evidence.presenceStart(physical_id, attrs.mesh, attrs.bounding_box, snapshot, round_ends,
                                            first_seen, latest);
  return start.absent ? start.last_absent + (start.next_present - start.last_absent) / 2 : TimeStamp{0};
}

}  // namespace khronos::session
