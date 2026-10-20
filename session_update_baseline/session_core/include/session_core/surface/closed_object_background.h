#pragma once

#include <vector>

#include "session_core/state/persistent_object_state.h"
#include "khronos/backend/change_state.h"

namespace khronos {

// The world-space surface of a physical state that closed, with the last time a
// background measurement still belonged to it. The state layer owns this surface;
// every surface layer that may still show it reads the same list.
struct ClosedStateSurface {
  size_t instance = 0;
  std::vector<Eigen::Vector3f> points;
  TimeStamp support_end = 0;
};

// Collect the closed states' surfaces once per round. A state owns its background
// duplicate while it stands, so it enters here when its death is known and its
// last support lies before the round; a state still supported at `latest` is present,
// not closed.
std::vector<ClosedStateSurface> closedStateSurfaces(
    const PersistentObjectState& objects, TimeStamp latest);

// Remove the background vertices owned by the closed states. A vertex is owned when
// it lies within one map cell of the state's surface and was measured no later than
// the state's support: the duplicate layer the state left in the background. The
// removal is atomic, so a closed state never leaves half of its duplicate behind.
size_t markClosedObjectBackground(
    const spark_dsg::Mesh& background,
    const std::vector<ClosedStateSurface>& states,
    float map_resolution,
    BackgroundChanges& changes);

}  // namespace khronos
