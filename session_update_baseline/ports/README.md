# Ports

Date: 2026-07-04

`vendor/` is the raw copied source. `ports/` is the modified code that should
actually be used by our baseline.

The rule is:

```text
vendor = original local source copy, kept for traceability
ports  = adapted code with a stable interface for this project
src    = experiment/session glue that calls ports
```

## Current Ports

### `panoptic_core`

Purpose:

```text
Panoptic Mapping style submap conflict / persistent / absent evidence,
without depending on ROS1 nodes or Panoptic's full runtime.
```

Source logic:

```text
vendor/panoptic_mapping/src/map_management/tsdf_registrator.cpp
  TsdfRegistrator::submapsConflict()
  TsdfRegistrator::checkSubmapForChange()

vendor/panoptic_mapping/include/panoptic_mapping/common/common.h
  PanopticLabel
  ChangeState
  IsoSurfacePoint
```

Adaptation:

```text
Panoptic voxblox TSDF lookup -> abstract DistanceQuery interface
Panoptic Submap class        -> lightweight SubmapSurface descriptor
ROS1/config_utilities        -> plain C++ config struct
```

This lets us reuse the Panoptic decision rule inside a Khronos/ROS2/Jazzy
environment once we provide a query backend from Khronos mesh/TSDF/exported
geometry.

