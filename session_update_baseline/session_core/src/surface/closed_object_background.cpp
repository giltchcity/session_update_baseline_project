#include "session_core/surface/closed_object_background.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <hydra/utils/nearest_neighbor_utilities.h>
#include "session_core/evidence/physical_evidence_store.h"
#include "session_core/model/model_math.h"
#include "session_core/model/range_model.h"
#include "session_core/surface/element_posterior.h"

namespace khronos {

size_t markClosedObjectBackground(
    const spark_dsg::Mesh& background, const PersistentObjectState& objects,
    const RayVerificator& verificator, float map_resolution, TimeStamp latest,
    BackgroundChanges& changes) {
  if (!std::isfinite(map_resolution) || map_resolution <= 0 ||
      background.stamps.size() != background.numVertices()) return 0;
  const auto evidence = verificator.physicalEvidenceSnapshot();
  auto& absence = verificator.observedAbsenceModel();
  const auto psi = absence.rangeModel();
  if (!evidence || !psi.valid()) return 0;
  const TimeStamp session_start = absence.sessionStart();
  const double previous_zeta = absence.previousZeta();

  // Retrieval: the background elements within sqrt(3) map voxels (the voxel diagonal) of the surface
  // of a placement committed ended, with the class of that placement.
  const float radius = std::sqrt(3.f) * map_resolution;
  std::map<size_t, int> candidates;  // vertex -> class of the nearest ended placement
  std::map<size_t, float> nearest_distance;
  for (const size_t id : objects.trackedIds()) {
    for (const auto& fragment : objects.historyFragments(id)) {
      if (!fragment.death_time || *fragment.death_time > latest || !fragment.geometry ||
          !fragment.bbox || fragment.geometry->numVertices() == 0) continue;
      Points points;
      for (const auto& local : fragment.geometry->points) {
        const Point world = fragment.bbox->pointToWorldFrame(local);
        if (world.allFinite()) points.push_back(world);
      }
      if (points.empty()) continue;
      const hydra::PointNeighborSearch search(points);
      for (size_t i = 0; i < background.numVertices(); ++i) {
        const auto& point = background.pos(i);
        const auto stamp = background.stamps[i];
        if (!point.allFinite() || !stamp || stamp > *fragment.death_time) continue;
        float distance_squared;
        size_t index;
        if (!search.search(point, distance_squared, index) || distance_squared > radius * radius) continue;
        const float distance = std::sqrt(distance_squared);
        const auto found = nearest_distance.find(i);
        if (found == nearest_distance.end() || distance < found->second) {
          nearest_distance[i] = distance;
          candidates[i] = fragment.semantic_label;
        }
      }
    }
  }
  if (candidates.empty()) return 0;
  changes.resize(background.numVertices(), ChangeState::kUnobserved);

  // README principle 9, eq. (7) with frames as one look: the verdicts of the frames after the
  // element's own last observation (T see-through, H hit; O and I carry likelihood ratio one).
  size_t removed = 0;
  double device_range = 0.0;
  for (const auto& [index, cls] : candidates) {
    const TimeStamp supported = background.stamps[index];
    if (supported >= latest) continue;
    const auto& point = background.pos(index);
    // Across sessions the alignment residual sigma_x and the scale displacement apply (principle 4);
    // before sigma_x is estimated such an element is not judged.
    const bool cross = supported < session_start;
    if (cross && !psi.sigma_x_known) continue;
    surface::ElementTally tally;
    for (const auto t : evidence->timestamps(supported + 1, latest)) {
      const auto p = evidence->project(t, point);
      if (p.endpoint.type == EndpointClass::kUnavailable) continue;
      device_range = std::max<double>(device_range, p.sensor_max_range);
      const double rho = p.query_range_m;
      const double sigma = psi.sigmaEff(rho, 0.0, map_resolution, cross);
      const double bias = cross ? psi.bias(rho, rho, previous_zeta) : 0.0;
      const auto kind = model::classifyRange(psi, p.endpoint.measured_depth_m, rho, sigma,
                                             p.sensor_min_range, p.sensor_max_range, bias);
      const double predicted = psi.predictedSeeThrough(rho, sigma, p.sensor_max_range);
      if (kind == model::RangeClass::kThrough) tally.addFrame(true, rho, predicted);
      else if (kind == model::RangeClass::kHit) tally.addFrame(false, rho, predicted);
    }
    if (!(tally.through > tally.hits)) continue;
    const auto in_place = objects.roundModel().classInPlace(cls, tally.meanPredicted());
    const double log_lr = model::RoundModel::logLikelihoodRatio(in_place, tally.frames(), tally.through);
    if (model::decideLog(log_lr) != model::Commitment::kCommitH) continue;
    if (changes[index] != ChangeState::kAbsent) {
      changes[index] = ChangeState::kAbsent;
      ++removed;
    }
  }
  return removed;
}
}  // namespace khronos
