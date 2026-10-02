#include "session_core/evidence/observed_absence.h"
#include "session_core/model/range_model.h"
#include "khronos/backend/change_detection/ray_verificator.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace khronos {

// README (6), (6s), (6e): the point is a mesh element sampled at the map resolution, without a
// normal (head-on incidence); every stored frame in the window classifies its reading.
RayVerificator::CheckResult RayVerificator::checkProjectedPhysical(
    const Point& point, const size_t physical_id,
    const PhysicalEvidenceSnapshot& evidence_snapshot,
    const uint64_t earliest, const uint64_t latest, const uint64_t element_time,
    CheckDetails* details) const {
  (void)element_time;
  CheckResult result;
  if (!point.allFinite()) { ++result.reasons.invalid; return result; }
  if (!evidence_snapshot) return result;
  const auto psi = absence_model_->rangeModel();
  if (!psi.valid()) return result;
  for (const auto stamp : evidence_snapshot->timestamps(earliest, latest)) {
    const auto p = evidence_snapshot->project(stamp, point);
    if (p.endpoint.type == EndpointClass::kUnavailable) continue;
    const double rho = p.query_range_m, reading = p.endpoint.measured_depth_m;
    // sigma_eff of (6s) does not depend on the time difference of the element and the reading.
    const double sigma = psi.sigmaEff(rho, 0.0, 0.0, false);
    const auto kind = model::classifyRange(psi, reading, rho, sigma, p.sensor_min_range,
                                           p.sensor_max_range);
    auto decision = CheckDetails::Result::kOccludded;
    switch (kind) {
      case model::RangeClass::kHit: {
        // Geometric presence; the identity of the coincident echo is the label factor of principle 6.
        result.present.push_back(stamp);
        decision = CheckDetails::Result::kMatch;
        const auto& e = p.endpoint;
        if (e.type == EndpointClass::kPhysical && static_cast<size_t>(e.physical_id) == physical_id)
          ++result.reasons.same_id;
        else if (e.type == EndpointClass::kPhysical) ++result.reasons.different_id;
        else if (e.type == EndpointClass::kBackground) ++result.reasons.background_replacement;
        else ++result.reasons.unidentified_object;
        break;
      }
      case model::RangeClass::kThrough:
        result.absent.push_back(stamp);
        ++result.reasons.free_space;
        decision = CheckDetails::Result::kAbsent;
        break;
      case model::RangeClass::kOccluded:
        result.inconclusive.push_back(stamp);
        ++result.reasons.geometric_occlusion;
        break;
      case model::RangeClass::kInvalid:
        result.inconclusive.push_back(stamp);
        ++result.reasons.invalid;
        break;
    }
    if (details) {
      const float q = std::isfinite(p.query_range_m) && p.query_range_m > 0 ? p.query_range_m : 0.f;
      const float depth = std::isfinite(p.endpoint.measured_depth_m) &&
          p.endpoint.measured_depth_m > 0 ? p.endpoint.measured_depth_m : 0.f;
      const Point source = point - q * p.view_direction_world;
      details->start.push_back(source);
      details->end.push_back(source + depth * p.view_direction_world);
      details->range.push_back(depth);
      details->result.push_back(decision);
    }
  }
  return result;
}

}  // namespace khronos
