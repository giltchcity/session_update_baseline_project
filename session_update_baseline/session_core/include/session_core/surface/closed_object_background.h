#pragma once

#include "session_core/state/persistent_object_state.h"
#include "khronos/backend/change_detection/ray_verificator.h"
#include "khronos/backend/change_state.h"

namespace khronos {

// README principle 9: a background element is kept or removed by its own observations only. The
// surface of a placement committed ended retrieves the background elements near it (candidates,
// within sqrt(3) map voxels: retrieval, not a deletion); the frames after an element's own last
// observation are its evidence, and the element is marked absent only when they see through it
// more often than they hit it and, as one look, the evidence of an ended surface exceeds
// ln((1 - alpha) / alpha) (7). Does not alter the registry or history.
size_t markClosedObjectBackground(
    const spark_dsg::Mesh& background,
    const PersistentObjectState& objects,
    const RayVerificator& verificator,
    float map_resolution,
    TimeStamp latest,
    BackgroundChanges& changes);

}  // namespace khronos
