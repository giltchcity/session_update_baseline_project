#pragma once

#include "session_core/state/persistent_object_state.h"
#include "khronos/backend/change_detection/ray_verificator.h"
#include "khronos/backend/change_state.h"

namespace khronos {

// README (15a): bind old obligations and newly closed states to this exact mesh.
std::vector<PersistentObjectState::BackgroundObligation> closedObjectBackgroundObligations(
    const spark_dsg::Mesh& background, const PersistentObjectState& objects,
    float map_resolution, TimeStamp latest);

// README principle 9: a background element that coincides with the surface of a closed placement
// has that placement's closure posterior as its prior change probability; the frames after its
// last support are its rounds, and it is marked absent once the posterior odds reach (1-alpha)/alpha.
// Does not alter the registry or history.
size_t markClosedObjectBackground(
    const spark_dsg::Mesh& background,
    const PersistentObjectState& objects,
    const RayVerificator& verificator,
    float map_resolution,
    TimeStamp latest,
    BackgroundChanges& changes);

}  // namespace khronos
