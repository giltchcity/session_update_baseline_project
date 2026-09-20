#include "khronos/backend/reconciliation/closed_object_background.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <glog/logging.h>
#include <hydra/utils/nearest_neighbor_utilities.h>

namespace khronos {

size_t markClosedObjectBackground(
    const spark_dsg::Mesh& background,
    const PersistentObjectState& objects,
    const RayVerificator& verificator,
    const RayChangeDetector& change_detector,
    const float map_resolution,
    const TimeStamp latest,
    BackgroundChanges& changes) {
  if (!std::isfinite(map_resolution) || map_resolution <= 0.f ||
      !background.has_timestamps || background.numVertices() == 0) {
    return 0;
  }
  const auto evidence = verificator.physicalEvidenceSnapshot();
  if (!evidence) {
    return 0;
  }
  changes.resize(background.numVertices(), ChangeState::kUnobserved);

  // Candidate association and measured absence have different scales. Two
  // independently meshed representations can disagree across a coarse cell;
  // half a cell missed leaf duplicates 5--11 cm from a 10 cm background mesh.
  // A cell diagonal only selects candidates, never authorizes deletion.
  // Keep the actual depth support/absence tolerance at half a voxel below so
  // present walls, new surfaces, occluders and missing coverage stay protected.
  const float radius = std::sqrt(3.f) * map_resolution;
  const float radius_squared = radius * radius;
  const float agree_squared = 0.25f * map_resolution * map_resolution;
  size_t removed = 0;
  for (const size_t id : objects.trackedIds()) {
    for (const auto& fragment : objects.historyFragments(id)) {
      if (!fragment.death_time || *fragment.death_time > latest ||
          !fragment.geometry || !fragment.bbox ||
          fragment.geometry->numVertices() == 0 ||
          fragment.last_support_time >= latest) {
        continue;
      }
      Points surface;
      surface.reserve(fragment.geometry->numVertices());
      Point low = Point::Constant(std::numeric_limits<float>::max());
      Point high = -low;
      for (const auto& vertex : fragment.geometry->points) {
        const Point world = fragment.bbox->pointToWorldFrame(vertex);
        if (!world.array().isFinite().all()) continue;
        surface.push_back(world);
        low = low.cwiseMin(world);
        high = high.cwiseMax(world);
      }
      if (surface.empty()) continue;
      low.array() -= radius;
      high.array() += radius;
      const hydra::PointNeighborSearch search(surface);
      size_t matched = 0;
      size_t closed = 0;
      size_t rigid_completed = 0;
      for (size_t i = 0; i < background.numVertices(); ++i) {
        if (changes[i] == ChangeState::kAbsent ||
            background.timestamp(i) > *fragment.death_time) {
          continue;  // Geometry reconstructed after this state closed is not owned by it.
        }
        const Point point = background.pos(i);
        if (!point.array().isFinite().all() ||
            (point.array() < low.array()).any() ||
            (point.array() > high.array()).any()) continue;
        float distance_squared = std::numeric_limits<float>::max();
        size_t index = 0;
        if (!search.search(point, distance_squared, index) ||
            distance_squared > radius_squared) continue;
        ++matched;

        // A closed object alone does not authorize erasing its whole box.
        // Each corresponding old surface still needs later measured absence.
        // Same-ID observations, occlusion and missing evidence preserve it.
        // The object layer may be frozen at the previous session while the
        // same old surface is re-observed in this session's background TSDF.
        // Its own observation time, not the inherited object's A timestamp,
        // bounds the evidence. Earlier free space cannot erase newer geometry.
        const TimeStamp supported_through =
            std::max({fragment.last_support_time, fragment.last_confirmed_support, background.timestamp(i)});
        if (supported_through >= latest) continue;
        // A closed identity only associates the old surface. Background
        // removal itself is geometric: any measured surface at this position
        // supports it, irrespective of semantic/instance identity. Evaluate
        // disappearance strictly AFTER the latest such support so that an
        // earlier empty interval cannot delete a later reconstructed surface.
        RayVerificator::CheckResult check;
        TimeStamp last_geometric_support = supported_through;
        const float tolerance = 0.5f * map_resolution + 1e-3f;
        for (const TimeStamp t : evidence->timestamps(supported_through + 1, latest)) {
          const auto p = evidence->project(t, point);
          const auto& endpoint = p.endpoint;
          if (endpoint.type == EndpointClass::kUnavailable ||
              endpoint.type == EndpointClass::kInvalid ||
              !std::isfinite(endpoint.measured_depth_m) || endpoint.measured_depth_m <= 0.f ||
              !std::isfinite(p.query_range_m)) continue;
          const float delta = endpoint.measured_depth_m - p.query_range_m;
          if (std::abs(delta) <= tolerance) {
            last_geometric_support = t;
          } else if (delta > tolerance) {
            check.absent.push_back(t);
          } else {
            check.inconclusive.push_back(t); // A nearer surface occludes this point.
          }
        }
        const auto remove_past = [&](std::vector<TimeStamp>& stamps) {
          stamps.erase(std::remove_if(stamps.begin(), stamps.end(), [&](TimeStamp t) {
            return t <= last_geometric_support;
          }), stamps.end());
        };
        remove_past(check.absent);
        remove_past(check.inconclusive);
        const auto change = change_detector.detectChanges(
            check, true, RayChangeDetector::CoverageMode::kPhysical);
        if (!change.closest_absent) {
          // Rigid-state completion. The registry closed this state from
          // spatial absence evidence for the object as a whole, and its private
          // mesh left CURRENT entirely. A background vertex that duplicates
          // that very surface (within half a voxel, the reconstruction
          // agreement scale) belongs to the same closed state, so the
          // per-vertex temporal confidence is not required a second time: one
          // measurement that saw this point empty is enough while nothing
          // supported it. Evidence itself is still required. Candidates farther than half a
          // voxel (an adjacent wall or floor), any vertex with later geometric
          // support, and any vertex that was occluded rather than exposed keep
          // requiring their own measured absence: an occluder means this
          // session could not look, which is never evidence of disappearance.
          if (distance_squared <= agree_squared && !check.absent.empty() &&
              check.inconclusive.empty() && last_geometric_support == supported_through) {
            changes[i] = ChangeState::kAbsent;
            ++closed;
            ++rigid_completed;
          }
          continue;
        }
        changes[i] = ChangeState::kAbsent;
        ++closed;
      }
      removed += closed;
      if (closed > 0) {
        LOG(INFO) << "BACKGROUND_STATE_CLOSURE inst=" << id
                  << " stamp=" << latest
                  << " last_support=" << fragment.last_support_time
                  << " matched_vertices=" << matched
                  << " absent_vertices=" << closed
                  << " rigid_completed=" << rigid_completed;
      }
    }
  }
  return removed;
}

}  // namespace khronos
