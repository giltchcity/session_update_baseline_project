#pragma once

#include "session_core/state/persistent_object_state.h"
#include "khronos/backend/change_state.h"

namespace khronos {

// Remove the duplicate background surface of closed physical states. A closed
// state owns every background vertex measured while it stood and coincident
// with its own surface (meshing tolerance); the removal is atomic, so a state
// that closed no longer leaves half of its background layer behind. Does not
// alter the registry or history.
size_t markClosedObjectBackground(
    const spark_dsg::Mesh& background,
    const PersistentObjectState& objects,
    float map_resolution,
    TimeStamp latest,
    BackgroundChanges& changes);

}  // namespace khronos
