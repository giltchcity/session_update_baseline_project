# Vendor Source Manifest

Exported: 2026-07-04

This file records where the copied source files came from. These are local
working copies for the session-update baseline. The original repositories remain
the reference.

## Khronos

Original root:

```text
/home/jixian/ros2_ws/src/khronos/khronos
```

Copied files:

```text
include/khronos/backend/change_state.h
src/backend/change_state.cpp
include/khronos/backend/change_detection/ray_verificator.h
src/backend/change_detection/ray_verificator.cpp
include/khronos/backend/change_detection/background/background_change_detector.h
include/khronos/backend/change_detection/background/ray_background_change_detector.h
src/backend/change_detection/background/ray_background_change_detector.cpp
include/khronos/backend/change_detection/objects/object_change_detector.h
include/khronos/backend/change_detection/objects/ray_object_change_detector.h
src/backend/change_detection/objects/ray_object_change_detector.cpp
include/khronos/backend/reconciliation/mesh/change_merger.h
src/backend/reconciliation/mesh/change_merger.cpp
```

Purpose:

```text
ray / free-space evidence
object/background absence and persistence checks
mesh deletion predicates
```

## Panoptic Mapping

Original root:

```text
/home/jixian/Desktop/FT/baselines/panoptic_mapping/panoptic_mapping
```

Copied files:

```text
include/panoptic_mapping/common/common.h
include/panoptic_mapping/map/submap.h
src/map/submap.cpp
include/panoptic_mapping/map/submap_collection.h
src/map/submap_collection.cpp
include/panoptic_mapping/map/submap_id.h
src/map/submap_id.cpp
include/panoptic_mapping/map_management/tsdf_registrator.h
src/map_management/tsdf_registrator.cpp
include/panoptic_mapping/map_management/map_manager.h
src/map_management/map_manager.cpp
src/integration/projective_tsdf_integrator.cpp
```

Purpose:

```text
semantic submaps
active / inactive map update
submap conflict, absent, persistent decisions
projective TSDF integration and freespace behavior
```

