#include "session_core/surface/closed_object_background.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <tuple>
#include <hydra/utils/nearest_neighbor_utilities.h>

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
  std::map<Key,TimeStamp> support;
  for (const auto& previous : objects.backgroundObligations())
    support[keyOf(previous.point,previous.reconstructed)] = previous.supported;
  const float radius = std::sqrt(3.f) * map_resolution;
  for (const size_t id : objects.trackedIds()) {
    for (const auto& fragment : objects.historyFragments(id)) {
      if (!fragment.death_time || *fragment.death_time > latest ||
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
        auto& last = support[keyOf(point,stamp)];
        last = std::max({last,fragment.last_support_time,fragment.last_confirmed_support,stamp});
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
    result.push_back({point,background.stamps[i],found->second});
    support.erase(found);
  }
  return result;
}

size_t markClosedObjectBackground(
    const spark_dsg::Mesh& background, const PersistentObjectState& objects,
    const RayVerificator& verificator, const RayChangeDetector& change_detector,
    float map_resolution, TimeStamp latest, BackgroundChanges& changes) {
  const auto obligations = closedObjectBackgroundObligations(background,objects,map_resolution,latest);
  const auto evidence = verificator.physicalEvidenceSnapshot();
  if (!evidence || obligations.empty()) return 0;
  std::map<Key,std::vector<size_t>> vertices;
  for (size_t i=0;i<background.numVertices();++i)
    if (background.pos(i).allFinite()) vertices[keyOf(background.pos(i),background.stamps[i])].push_back(i);
  changes.resize(background.numVertices(),ChangeState::kUnobserved);
  size_t removed=0;
  // README (6e): a point query uses the existing surface matching tolerance; geometric presence
  // accepts a same-position measurement of any identity.
  const float tolerance=verificator.config.surface_match_tolerance;
  for (const auto& obligation:obligations) {
    if (obligation.supported >= latest) continue;
    RayVerificator::CheckResult check;
    TimeStamp supported=obligation.supported;
    for (const auto t:evidence->timestamps(supported+1,latest)) {
      const auto p=evidence->project(t,obligation.point);
      const auto& e=p.endpoint;
      if (e.type==EndpointClass::kUnavailable || e.type==EndpointClass::kInvalid ||
          !std::isfinite(e.measured_depth_m) || e.measured_depth_m<=0 ||
          !std::isfinite(p.query_range_m)) continue;
      const float delta=e.measured_depth_m-p.query_range_m;
      if (std::abs(delta)<=tolerance) supported=t;
      else if (delta>tolerance) check.absent.push_back(t);
      else check.inconclusive.push_back(t);
    }
    const auto trim=[supported](auto& stamps) {
      stamps.erase(std::remove_if(stamps.begin(),stamps.end(),
          [supported](TimeStamp t){return t<=supported;}),stamps.end());
    };
    trim(check.absent); trim(check.inconclusive);
    if (!change_detector.detectChanges(check,true,RayChangeDetector::CoverageMode::kPhysical).closest_absent)
      continue;
    for (const auto index:vertices.at(keyOf(obligation.point,obligation.reconstructed))) {
      if (changes[index]!=ChangeState::kAbsent) {changes[index]=ChangeState::kAbsent; ++removed;}
    }
  }
  return removed;
}
}  // namespace khronos
