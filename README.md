# Session Update Baseline Project

Khronos-based object-guided scene-memory reconciliation for within-session and
cross-session map update.

This repository contains the implementation, adapted Khronos/Panoptic logic,
configuration, design documents, and compact records for the current canonical
experiment. The CVPR paper is maintained separately at:

```text
https://github.com/giltchcity/myncv---rpsl---daicma
```

## Start Here

1. `session_update_baseline/README.md`
2. `session_update_baseline/base1.md`
3. `session_update_baseline/THREE_MODE_UNIFIED_MODEL.md`
4. `session_update_baseline/experiment_records/current/RUN_SUMMARY.md`
5. `session_update_baseline/EXPERIMENT_RETENTION_POLICY.md`

## Runtime Entry Points

```text
session_update_baseline/app/run_session_update_baseline.cpp
session_update_baseline/src/base1/object_guided_map_reconciler.cpp
session_update_baseline/scripts/run_base1_khronos_env.sh
session_update_baseline/scripts/run_session_update_baseline.sh
```

## Repository Boundary

The repository intentionally excludes datasets, ROS bags, `.4dmap`/`.panmap`
files, PLY sequences, complete DSG dumps, model weights, build products, and
generated run directories. Compact configs, metrics, summaries, and selected
presentation figures are retained under `experiment_records/current/`.
