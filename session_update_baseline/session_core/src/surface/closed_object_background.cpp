#include "session_core/surface/closed_object_background.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <glog/logging.h>
#include <hydra/utils/nearest_neighbor_utilities.h>

namespace khronos {

size_t markClosedObjectBackground(
    const spark_dsg::Mesh& background,
    const PersistentObjectState& objects,
    const float map_resolution,
    const TimeStamp latest,
    BackgroundChanges& changes) {
  if (!std::isfinite(map_resolution) || map_resolution <= 0.f ||
      !background.has_timestamps || background.numVertices() == 0) {
    return 0;
  }
  changes.resize(background.numVertices(), ChangeState::kUnobserved);

  // A closed state owns the background surface that was measured while it stood:
  // the duplicate layer left by integrating its pixels as static background.
  // Ownership is meshing coincidence, not the loose association radius: the wall
  // behind a thin object is further than one cell and stays. The vertex must also
  // predate the state's last support, so the replacement surface measured after
  // the move is never owned by the state that left.
  const float radius = map_resolution;
  const float radius_squared = radius * radius;
  size_t removed = 0;
  for (const size_t id : objects.trackedIds()) {
    for (const auto& fragment : objects.historyFragments(id)) {
      if (!fragment.death_time || *fragment.death_time > latest ||
          !fragment.geometry || !fragment.bbox ||
          fragment.geometry->numVertices() == 0 ||
          fragment.last_support_time >= latest) {
        continue;
      }
      const TimeStamp support_end =
          std::max(fragment.last_support_time, fragment.last_confirmed_support);
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
      for (size_t i = 0; i < background.numVertices(); ++i) {
        if (changes[i] == ChangeState::kAbsent || background.timestamp(i) > support_end) {
          continue;
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
        // The state is closed; its duplicate layer follows atomically. No later
        // per-vertex absence is required: the old site may never be observed
        // again, and a half-cleaned duplicate is worse than none.
        changes[i] = ChangeState::kAbsent;
        ++closed;
      }
      removed += closed;
      if (closed > 0) {
        LOG(INFO) << "BACKGROUND_STATE_CLOSURE inst=" << id
                  << " stamp=" << latest
                  << " last_support=" << fragment.last_support_time
                  << " matched_vertices=" << matched
                  << " absent_vertices=" << closed;
      }
    }
  }
  return removed;
}

}  // namespace khronos
