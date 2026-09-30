#include "session_core/surface/closed_object_background.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <tuple>
#include <hydra/utils/nearest_neighbor_utilities.h>
#include "session_core/evidence/physical_evidence_store.h"
#include "session_core/model/model_math.h"
#include "session_core/model/range_model.h"
#include "session_core/surface/element_posterior.h"

namespace khronos {
namespace {
using Key = std::tuple<float,float,float,TimeStamp>;
Key keyOf(const Point& point, TimeStamp stamp) {
  return {point.x(),point.y(),point.z(),stamp};
}
}

std::vector<PersistentObjectState::BackgroundObligation> closedObjectBackgroundObligations(
    const spark_dsg::Mesh& background, const PersistentObjectState& objects,
    float map_resolution, TimeStamp latest) {
  using Obligation = PersistentObjectState::BackgroundObligation;
  if (!std::isfinite(map_resolution) || map_resolution <= 0 ||
      background.stamps.size() != background.numVertices()) return {};
  // One key denotes one measured background location. New reconstruction times
  // create new keys, even when the world position is identical.
  struct Entry {
    TimeStamp supported = 0;
    double odds = 0.0;
    float distance = 0.f;
  };
  std::map<Key,Entry> support;
  for (const auto& previous : objects.backgroundObligations())
    support[keyOf(previous.point,previous.reconstructed)] =
        {previous.supported, previous.odds, previous.distance};
  const float radius = std::sqrt(3.f) * map_resolution;
  for (const size_t id : objects.trackedIds()) {
    for (const auto& fragment : objects.historyFragments(id)) {
      if (!fragment.death_time || *fragment.death_time > latest || !(fragment.closure_odds > 0.0) ||
          !fragment.geometry || !fragment.bbox || fragment.geometry->numVertices() == 0) continue;
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
        if (!search.search(point,distance_squared,index) || distance_squared > radius*radius) continue;
        auto& entry = support[keyOf(point,stamp)];
        entry.supported = std::max({entry.supported,fragment.last_support_time,
                                    fragment.last_confirmed_support,stamp});
        // The closest closed placement decides the prior of the element.
        const float distance = std::sqrt(distance_squared);
        if (entry.odds == 0.0 || distance < entry.distance) {
          entry.odds = fragment.closure_odds;
          entry.distance = distance;
        }
      }
    }
  }
  std::vector<Obligation> result;
  result.reserve(std::min(support.size(),background.numVertices()));
  // Rebinding intersects obligations with the actual chain, releasing retired references.
  for (size_t i = 0; i < background.numVertices(); ++i) {
    const auto& point = background.pos(i);
    if (!point.allFinite()) continue;
    const auto found = support.find(keyOf(point,background.stamps[i]));
    if (found == support.end()) continue;
    result.push_back({point,background.stamps[i],found->second.supported,found->second.odds,
                      found->second.distance});
    support.erase(found);
  }
  return result;
}

size_t markClosedObjectBackground(
    const spark_dsg::Mesh& background, const PersistentObjectState& objects,
    const RayVerificator& verificator, float map_resolution, TimeStamp latest,
    BackgroundChanges& changes) {
  const auto obligations = closedObjectBackgroundObligations(background,objects,map_resolution,latest);
  const auto evidence = verificator.physicalEvidenceSnapshot();
  const auto psi = verificator.observedAbsenceModel().rangeModel();
  if (!evidence || obligations.empty() || !psi.valid()) return 0;
  const TimeStamp session_start = verificator.observedAbsenceModel().sessionStart();
  std::map<Key,std::vector<size_t>> vertices;
  for (size_t i=0;i<background.numVertices();++i)
    if (background.pos(i).allFinite()) vertices[keyOf(background.pos(i),background.stamps[i])].push_back(i);
  changes.resize(background.numVertices(),ChangeState::kUnobserved);

  // README principle 9, eq. (7) with frames as rounds: the verdicts of the frames after the last
  // support of each element (T see-through, H hit; O and I carry likelihood ratio one).
  std::vector<surface::ElementTally> tallies(obligations.size());
  double device_range = 0.0;
  for (size_t k=0;k<obligations.size();++k) {
    const auto& obligation = obligations[k];
    if (obligation.supported >= latest) continue;
    // Across sessions the registration drift grows from the alignment at the session start (6s).
    const bool cross = obligation.reconstructed < session_start;
    const TimeStamp t_e = cross ? session_start : obligation.reconstructed;
    for (const auto t:evidence->timestamps(obligation.supported+1,latest)) {
      const auto p=evidence->project(t,obligation.point);
      if (p.endpoint.type == EndpointClass::kUnavailable) continue;
      device_range = std::max<double>(device_range, p.sensor_max_range);
      const double rho = p.query_range_m;
      const double dt = std::abs(static_cast<double>(t)-static_cast<double>(t_e))*1e-9;
      const double sigma = psi.sigmaEff(rho,0.0,dt,map_resolution,cross);
      const auto kind = model::classifyRange(psi,p.endpoint.measured_depth_m,rho,sigma,
                                             p.sensor_min_range,p.sensor_max_range,cross);
      if (kind == model::RangeClass::kThrough) tallies[k].addFrame(true,rho);
      else if (kind == model::RangeClass::kHit) tallies[k].addFrame(false,rho);
    }
  }
  const double rho_frame = surface::pooledCorrelation(tallies);
  const double w = model::RoundModel::autocorrelationExponent(rho_frame);
  size_t removed=0;
  for (size_t k=0;k<obligations.size();++k) {
    const auto& obligation = obligations[k];
    const auto& tally = tallies[k];
    if (tally.frames() == 0 || !(obligation.odds > 0.0)) continue;
    // The prior applies where the element coincides with the closed surface within the hit band.
    const double rho = tally.meanRange();
    const double band = psi.bounds(rho,psi.sigmaEff(rho,0.0,0.0,map_resolution,true),
                                   device_range).plus;
    if (std::isfinite(band) && obligation.distance > band) continue;
    const double ln_lr = objects.roundModel().elementLogRatio(0,tally.frames(),tally.through,
                                                               objects.constructionHits(),0.0,0.0,w);
    const double log_odds = std::log(obligation.odds) + ln_lr;
    if (model::decideLog(log_odds) != model::Commitment::kCommitH) continue;
    for (const auto index:vertices.at(keyOf(obligation.point,obligation.reconstructed))) {
      if (changes[index]!=ChangeState::kAbsent) {changes[index]=ChangeState::kAbsent; ++removed;}
    }
  }
  return removed;
}
}  // namespace khronos
