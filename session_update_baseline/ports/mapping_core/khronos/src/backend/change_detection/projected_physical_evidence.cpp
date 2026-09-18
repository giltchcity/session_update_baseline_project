#include "khronos/backend/change_detection/ray_verificator.h"

#include <cmath>
#include <set>
#include <tuple>

namespace khronos {
namespace {

enum class Vote { Unavailable, Invalid, Occluded, Supported, Free,
                  Background, Other, Unidentified };

Vote classifyMeasurement(const ProjectedEndpointEvidence& p, size_t id, float tolerance) {
  const auto& e = p.endpoint;
  if (e.type == EndpointClass::kUnavailable) return Vote::Unavailable;
  if (e.type == EndpointClass::kInvalid || !std::isfinite(e.measured_depth_m) ||
      !std::isfinite(p.query_range_m) || e.measured_depth_m <= 0 || p.query_range_m <= 0) {
    return Vote::Invalid;
  }
  const float delta = e.measured_depth_m - p.query_range_m;
  if (delta < -tolerance) return Vote::Occluded;
  const bool same_identity = e.type == EndpointClass::kPhysical &&
      e.physical_id > 0 && static_cast<size_t>(e.physical_id) == id;
  // The matching tolerance permits shape/pose error for the same object;
  // it never permits a different, nearer surface to see through an occluder.
  // Synthetic A I49: measured 5.279 m versus queried 5.5785 m was previously
  // misclassified as background replacement inside the 0.3 m matching band.
  // PhysicalEvidenceStore quantizes range to millimetres; exact same-depth
  // replacement must survive that rounding, without a 30 cm occlusion band.
  if (!same_identity && delta < -1e-3f) return Vote::Occluded;
  if (delta > tolerance) return Vote::Free;
  if (e.type == EndpointClass::kBackground) return Vote::Background;
  if (e.type == EndpointClass::kUnidentifiedObject) return Vote::Unidentified;
  if (e.type == EndpointClass::kPhysical) {
    return e.physical_id > 0 && static_cast<size_t>(e.physical_id) == id
               ? Vote::Supported : Vote::Other;
  }
  return Vote::Invalid;
}

}  // namespace

RayVerificator::CheckResult RayVerificator::checkProjectedPhysical(
    const Point& point, const size_t physical_id,
    const PhysicalEvidenceSnapshot& evidence_snapshot,
    const uint64_t earliest, const uint64_t latest) const {
  CheckResult result;
  if (!evidence_snapshot) return result;
  for (const auto stamp : evidence_snapshot->timestamps(earliest, latest)) {
    const auto p = evidence_snapshot->project(stamp, point);
    switch (classifyMeasurement(p, physical_id, config.depth_tolerance)) {
      case Vote::Supported:
        result.present.push_back(stamp); ++result.reasons.same_id; break;
      case Vote::Free:
        result.absent.push_back(stamp); ++result.reasons.free_space; break;
      case Vote::Background:
        result.absent.push_back(stamp); ++result.reasons.background_replacement; break;
      case Vote::Other:
        result.absent.push_back(stamp); ++result.reasons.different_id; break;
      case Vote::Occluded:
        result.inconclusive.push_back(stamp); ++result.reasons.geometric_occlusion; break;
      case Vote::Unidentified:
        result.inconclusive.push_back(stamp); ++result.reasons.unidentified_object; break;
      case Vote::Invalid:
        result.inconclusive.push_back(stamp); ++result.reasons.invalid; break;
      case Vote::Unavailable:
        break;  // Outside FOV / absent frame is not a coverage measurement.
    }
  }
  return result;
}

RayVerificator::SurfaceEvidenceCounts RayVerificator::countProjectedPhysicalSurface(
    const size_t physical_id, const spark_dsg::Mesh& mesh, const BoundingBox& bbox,
    const PhysicalEvidenceSnapshot& evidence_snapshot, const float map_resolution,
    const uint64_t earliest, const uint64_t latest) const {
  SurfaceEvidenceCounts result;
  if (!evidence_snapshot || !std::isfinite(map_resolution) || map_resolution <= 0) return result;

  // One actual surface sample per map cell keeps duplicate triangle storage
  // from multiplying evidence and bounds the cost of the fallback.
  Points samples;
  std::set<std::tuple<int64_t, int64_t, int64_t>> cells;
  const auto add_sample = [&](const Point& p) {
    if (!p.array().isFinite().all()) return;
    const auto cell = (p / map_resolution).array().floor().cast<int64_t>().eval();
    if (cells.emplace(cell.x(), cell.y(), cell.z()).second) samples.push_back(p);
  };
  if (mesh.faces.empty()) {
    for (const auto& p : mesh.points) add_sample(bbox.pointToWorldFrame(p));
  } else {
    for (const auto& f : mesh.faces) {
      if (f[0] >= mesh.numVertices() || f[1] >= mesh.numVertices() || f[2] >= mesh.numVertices()) continue;
      add_sample(bbox.pointToWorldFrame((mesh.pos(f[0])+mesh.pos(f[1])+mesh.pos(f[2]))/3.f));
    }
  }
  result.surface_samples = samples.size();
  const auto stamps = evidence_snapshot->timestamps(earliest, latest);
  for (const auto& point : samples) {
    bool coverage = false;
    TimeStamp sample_support = 0, sample_absence = 0;
    for (size_t frame = 0; frame < stamps.size(); ++frame) {
      const auto p = evidence_snapshot->project(stamps[frame], point);
      const auto vote = classifyMeasurement(p, physical_id, config.depth_tolerance);
      if (vote == Vote::Unavailable) continue;
      coverage = true;
      // Exact (frame, pixel), not (sample, frame): a pixel is one measurement
      // even when several surface triangles project onto it.
      const size_t key = (frame << 32) | static_cast<size_t>(p.pixel_index);
      switch (vote) {
        case Vote::Supported:
          sample_support = stamps[frame];
          result.latest_support_stamp = std::max(result.latest_support_stamp, stamps[frame]);
          result.support_indices.insert(key); ++result.supported_votes; break;
        case Vote::Free:
          sample_absence = stamps[frame];
          result.contradiction_indices.insert(key); ++result.free_space_votes; break;
        case Vote::Background:
          sample_absence = stamps[frame];
          result.contradiction_indices.insert(key); ++result.replaced_by_background_votes; break;
        case Vote::Other:
          sample_absence = stamps[frame];
          result.contradiction_indices.insert(key); ++result.replaced_by_other_votes; break;
        case Vote::Occluded:
          ++result.occluded_votes; break;
        default:
          break;
      }
    }
    if (!coverage) ++result.unobserved_samples;
    if (sample_absence > sample_support) ++result.contradicted_surface_samples;
  }
  result.absence_coverage_sufficient = result.surface_samples > 0 &&
      result.contradicted_surface_samples > 0 &&
      static_cast<double>(result.contradicted_surface_samples) / result.surface_samples >=
          config.min_absent_surface_fraction;
  result.support_rays = result.support_indices.size();
  result.contradiction_rays = result.contradiction_indices.size();
  return result;
}

RayVerificator::SurfaceEvidenceCounts RayVerificator::countCurrentPhysicalSurface(
    size_t physical_id, const spark_dsg::Mesh& mesh, const BoundingBox& bbox,
    const PhysicalEvidenceSnapshot& snapshot, float map_resolution,
    uint64_t last_support, uint64_t latest, bool* projected) const {
  if (projected) *projected = false;
  // Also prevents unsigned overflow and invalid inclusive intervals.
  if (last_support >= latest) return {};
  const uint64_t earliest = last_support + 1;
  auto counts = countPhysicalSurface(physical_id, mesh, bbox, snapshot, earliest, latest);
  // A sparse mesh-ray subset can reverse the decision (Synthetic I108:
  // indexed support 3 / absence 4, measured pixels support 86 / absence 36).
  // Any proposed deletion must therefore be checked against the actual
  // sensor evidence; a mesh proxy alone cannot authorize disappearance.
  const bool proposed_absence = counts.contradiction_rays > counts.support_rays;
  if (proposed_absence || (counts.support_rays == 0 && counts.contradiction_rays == 0)) {
    auto measured = countProjectedPhysicalSurface(
        physical_id, mesh, bbox, snapshot, map_resolution, earliest, latest);
    if (proposed_absence || measured.support_rays || measured.contradiction_rays) {
      if (projected) *projected = true;
      return measured;
    }
  }
  return counts;
}

}  // namespace khronos
