#pragma once

#include "khronos/backend/change_detection/ray_change_detector.h"
#include "khronos/backend/reconciliation/persistent_object_state.h"
#include "khronos/backend/change_state.h"

namespace khronos {

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
