# Session Update Baseline

Exported: 2026-07-04

This folder is the single workspace for the session-based map-update baseline.
It is intentionally not a from-scratch reimplementation. Reusable logic is
copied from Khronos and Panoptic Mapping into `vendor/`, then adapted in
`ports/` so the pieces can cooperate in our baseline.

## Target Problem

We want to test map update under two change modes:

1. **Within-session change**
   The robot observes a place, the scene changes while the same session is
   still running, then the robot revisits or looks back and updates the map.

2. **Cross-session change**
   Session A completes and saves a map. The robot leaves, the scene changes,
   then Session B starts from the old map and updates it during a full revisit.

Do not reduce this to "cut one trajectory in half". The intended experiment
uses complete traversals and explicit revisit/update behavior.

## Folder Layout

```text
session_update_baseline/
  README.md
  docker/
    compose.yaml
  vendor/
    khronos/
      include/
      src/
    panoptic_mapping/
      include/
      src/
    SOURCES.md
  ports/
    panoptic_core/
      # modified Panoptic-style update logic with a stable baseline interface
    khronos_core/
      # Khronos ray/change logic adapters
  src/
    # our adapters/glue only
  scripts/
    # run orchestration and export helpers
  configs/
    # experiment configs
  runs/
    # local run outputs for this baseline
```

## Reuse Boundary

### Khronos-derived logic

Use Khronos for ray/free-space change evidence and mesh cleanup behavior:

- `ChangeState`
- `RayVerificator`
- `RayBackgroundChangeDetector`
- `RayObjectChangeDetector`
- `ChangeMerger`

These are copied under `vendor/khronos`. Runtime adapters should live under
`ports/khronos_core`, not by editing the raw vendor copy.

### Panoptic-derived logic

Use Panoptic Mapping for semantic submap state/update behavior:

- `ChangeState`
- `Submap`
- `SubmapCollection`
- `TsdfRegistrator`
- `MapManager`
- `ProjectiveTsdfIntegrator`

These are copied under `vendor/panoptic_mapping`. The first usable port is
`ports/panoptic_core`, which adapts Panoptic's submap conflict/update rule into
a ROS-independent C++ library that can be used from a Khronos/Jazzy workflow.

## Runtime Strategy

Khronos and Panoptic Mapping should not be forced into one binary/container at
the start:

- Panoptic Mapping is ROS1/Noetic.
- Khronos is ROS2/Jazzy in the current workspace.

The stable setup is:

```text
one shared FT folder mounted as /ft
  + Panoptic Noetic container for Panoptic runs
  + Khronos ROS2/Jazzy host/workspace or separate container for Khronos runs
  + shared outputs under /ft/runs and /ft/session_update_baseline/runs
```

This keeps the two original systems runnable while giving our adapters a single
place to live.

## Implementation Rule

When adding code:

1. First check whether Khronos or Panoptic already has the logic.
2. If yes, copy the relevant file or function into `vendor/` and modify the
   copy.
3. Modify the copied logic in `ports/`, not directly in `vendor/`.
4. Keep our own code small: session runner, map exporter, submap/object matcher,
   probability wrapper, diagnostics.
5. Do not modify the original source trees unless the change is explicitly an
   upstream experiment.
