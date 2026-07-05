# Implementation Design

Date: 2026-07-04

## Core Idea

The baseline should be a thin session-update layer over existing systems:

```text
complete traversal/session input
  -> original Khronos or Panoptic mapping run
  -> exported map state
  -> ported Khronos/Panoptic update logic
  -> session-level compare/update/probability state
  -> diagnostics and visualization
```

The new code should not replace Khronos or Panoptic reconstruction. It should
only connect complete sessions and expose update evidence.

## Experiment Objects

### Session

A session is a complete traversal, not a slice:

```text
Session A:
  input trajectory + RGB-D/depth/segmentation
  produced map
  exported submaps / global mesh / object states

Session B:
  complete revisit traversal
  may load or compare against Session A
  produced updated map
```

### Update Evidence

Each spatial unit should get evidence from one or more sources:

```text
present evidence:
  observed surface / object / submap support

absent evidence:
  ray/free-space evidence from Khronos
  conflict/absence state from Panoptic submap update

unknown:
  no later observation through that region
```

### Spatial Unit

Start with two supported units:

```text
submap-level:
  Panoptic-style semantic/instance submaps

mesh-region-level:
  Khronos-style background mesh vertices / regions
```

Object-level probability can be added after the two units above are stable.

## Reused Code Paths

### Khronos path

Use copied Khronos code as the source for:

```text
RayVerificator:
  ray / point verification

RayBackgroundChangeDetector:
  background vertex absent / persistent / unobserved evidence

RayObjectChangeDetector:
  object surface absent / persistent / unobserved evidence

ChangeMerger:
  which change states delete or keep mesh vertices
```

### Panoptic path

Use copied Panoptic code as the source for:

```text
TsdfRegistrator:
  active-vs-inactive submap conflict check
  absent / persistent state assignment

MapManager:
  active/inactive submap lifecycle and merging

Submap / SubmapCollection:
  unit of update and matching

ProjectiveTsdfIntegrator:
  projective TSDF and freespace behavior
```

## Minimal New Modules

The modified reusable logic lives under `ports/`. The new project-specific code
should be limited to these adapters:

```text
src/session_manifest.*
  describes Session A/B/C paths, timestamps, and outputs

src/map_export_index.*
  reads exported Khronos/Panoptic artifacts into a common index

src/submap_matcher.*
  matches old/new submaps by label, bbox, centroid, and surface distance

src/update_belief.*
  small probability/log-odds wrapper around present/absent/unknown evidence

src/diagnostics.*
  writes CSV/JSON summaries and simple visualization exports
```

## Current Port Status

### Panoptic-style submap update

Implemented under:

```text
ports/panoptic_core/
```

This is adapted from:

```text
vendor/panoptic_mapping/src/map_management/tsdf_registrator.cpp
```

It keeps the original Panoptic conflict / match / absent / persistent
predicates but replaces Panoptic's concrete `voxblox::Interpolator<TsdfVoxel>`
with an abstract `DistanceQuery` interface. That is the part that lets the same
decision rule run from Khronos/Jazzy-side exported geometry.

## Docker / Runtime

Use shared `/ft`, not one forced mixed ROS install:

```text
Panoptic Mapping:
  ROS1 Noetic container
  /ft mounted
  runs official Panoptic launch/scripts

Khronos:
  ROS2 Jazzy host workspace or separate ROS2 container
  /ft and /ros2_ws mounted
  runs official Khronos launch/scripts
```

Reason:

```text
Panoptic is ROS1/Noetic.
Khronos is ROS2/Jazzy.
One shared data folder is stable; one universal container is fragile.
```

## First Implementation Target

Build the first adapter around exported map artifacts, not around live ROS
callbacks:

1. Run complete Session A with original system.
2. Save/export map.
3. Run complete Session B with original system.
4. Save/export map.
5. Compare A and B using vendor-copied update predicates.
6. Output:

```text
updated_regions.csv
submap_matches.csv
evidence_summary.json
present_absent_unknown.ply
```

This gives a baseline result without destabilizing the original systems.
