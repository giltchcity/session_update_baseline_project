#pragma once

#include "khronos/backend/change_detection/ray_change_detector.h"
#include "session_core/state/persistent_object_state.h"
#include "khronos/backend/change_state.h"

namespace khronos {

// README (15a): bind old obligations and newly closed states to this exact mesh.
std::vector<PersistentObjectState::BackgroundObligation> closedObjectBackgroundObligations(
    const spark_dsg::Mesh& background, const PersistentObjectState& objects,
    float map_resolution, TimeStamp latest);

// Extend ordinary background change detection to the duplicate background
// surfaces of closed physical states. Does not alter the registry or history.
size_t markClosedObjectBackground(
    const spark_dsg::Mesh& background,
    const PersistentObjectState& objects,
    const RayVerificator& verificator,
    const RayChangeDetector& change_detector,
    float map_resolution,
    TimeStamp latest,
    BackgroundChanges& changes);

}  // namespace khronos
