# Base1 Loop Summary

Export time: 2026-07-04 23:48:09 CST

Title: Object-Guided Khronos Baseline, first implementation loop set

## Goal

Base1 is a Khronos-based baseline runner, not a one-off manual repair script. It takes a Khronos-compatible `.4dmap`, applies `ObjectGuidedMapReconciler`, and writes:

```text
original_final.4dmap
improved_final.4dmap
object_memory.json
mesh_update_summary.csv
object_update_summary.csv
evidence_summary.json
```

The first version focuses on final current map quality:

```text
object-guided cleanup
object audit
safe no-op IO
repair/injection accounting
cross-session prior diagnostics
```

## Two Dynamic Modes

Base1 keeps the two requested dynamic-map modes separate.

### Within-session Change

This is the Khronos-style situation:

```text
robot is still in the same run
scene changes while it is away or looking elsewhere
robot revisits / turns back
map should update using current evidence
```

Current Base1 support:

```text
--dynamic_mode within_session
```

Implemented behavior:

```text
use current Khronos final map
use current object private meshes and bboxes
remove global/background vertices that look like object-background leakage
do not use prior session memory
```

Not implemented yet:

```text
ray/free-space absence evidence
true temporal belief update inside one running backend
```

### Cross-session Revisit Change

This is the session-memory situation:

```text
Session A finishes and saves a map
scene changes while robot is gone
Session B starts with previous final map / object memory
Session B updates its final current map
```

Current Base1 support:

```text
--dynamic_mode cross_session
--prior_map A/improved_final.4dmap
--prior_object_memory A/object_memory.json
```

Implemented behavior:

```text
load and validate prior map
count prior object memory entries
record prior evidence in diagnostics
```

Not implemented yet:

```text
using prior object belief in remove/keep/repair scoring
old-object absent / persistent / unobserved state transition
```

## Implemented Loops

| Loop | Status | Result |
|---|---:|---|
| 1 | done | `base1.md`, runner skeleton, no-op, audit, cleanup-only, summary outputs |
| 2 | done | bbox containment guard for cleanup |
| 3 | done | ablation sweep script |
| 4 | done | threshold sweep smoke test |
| 5 | done | vertex-level deletion diagnostics |
| 6 | done | cross-session prior loading diagnostics |
| 7 | done | injection / repair dry-run accounting |
| 8 | done | no-op/full regression and reload check |
| 9 | done | reproducible loop summary and Khronos eval preparation entrypoint |

## Current Smoke Map

All loop smoke tests used the current valid Khronos map:

```text
/home/jixian/Desktop/FT/khronos_runs/CURRENT_VALID_RUN/final.4dmap
```

## Loop 4 Threshold Sweep

Summary file:

```text
session_update_baseline/runs/loop04_sweep/sweep_summary.csv
```

Observed cleanup behavior:

```text
0.02 m: removed 3788 vertices
0.03 m: removed 5126 vertices
0.05 m: removed 6874 vertices
0.08 m: removed 8323 vertices
same-label 0.05 m: removed 0 vertices
```

Important interpretation:

```text
same-label cleanup is not useful on this map because sampled global mesh vertex labels are mostly 0 while object labels are nonzero.
```

## Loop 8 Regression

No-op:

```text
run: session_update_baseline/runs/loop08_noop
original size: 235394534 bytes
improved size: 235394534 bytes
vertices: 261925 -> 261925
removed vertices: 0
```

Full:

```text
run: session_update_baseline/runs/loop08_full
mode: full
object_distance_m: 0.03
vertices: 261925 -> 256799
removed vertices: 5126
improved size: 85786378 bytes
```

Reload check:

```text
run: session_update_baseline/runs/loop08_load_full
input: session_update_baseline/runs/loop08_full/improved_final.4dmap
vertices: 256799 -> 256799
removed vertices: 0
```

Important interpretation:

```text
Cleanup/full currently saves a final-current single-timestep `.4dmap`.
No-op preserves the original full `.4dmap`.
This is acceptable for Base1's first milestone because the stated first target is final current map quality, not full 4D reconstruction.
```

## Current Commands

No-op:

```bash
session_update_baseline/scripts/run_session_update_baseline.sh \
  --map_file /home/jixian/Desktop/FT/khronos_runs/CURRENT_VALID_RUN/final.4dmap \
  --output_dir session_update_baseline/runs/new_noop \
  --mode no_op \
  --dynamic_mode within_session
```

Cleanup:

```bash
session_update_baseline/scripts/run_session_update_baseline.sh \
  --map_file /home/jixian/Desktop/FT/khronos_runs/CURRENT_VALID_RUN/final.4dmap \
  --output_dir session_update_baseline/runs/new_cleanup_003 \
  --mode cleanup \
  --dynamic_mode within_session \
  --object_distance_m 0.03
```

Full accounting:

```bash
session_update_baseline/scripts/run_session_update_baseline.sh \
  --map_file /home/jixian/Desktop/FT/khronos_runs/CURRENT_VALID_RUN/final.4dmap \
  --output_dir session_update_baseline/runs/new_full_003 \
  --mode full \
  --dynamic_mode within_session \
  --object_distance_m 0.03
```

Cross-session diagnostic:

```bash
session_update_baseline/scripts/run_session_update_baseline.sh \
  --map_file /home/jixian/Desktop/FT/khronos_runs/CURRENT_VALID_RUN/final.4dmap \
  --output_dir session_update_baseline/runs/new_cross_003 \
  --mode full \
  --dynamic_mode cross_session \
  --prior_map session_update_baseline/runs/loop08_full/improved_final.4dmap \
  --prior_object_memory session_update_baseline/runs/loop08_full/object_memory.json \
  --object_distance_m 0.03
```

## Khronos Eval Preparation

The evaluator expects:

```text
experiment_dir/final.4dmap
```

Base1 writes:

```text
run_dir/improved_final.4dmap
```

Prepare an eval directory without running the heavy evaluator:

```bash
session_update_baseline/scripts/prepare_base1_khronos_eval_dir.sh \
  session_update_baseline/runs/loop08_full \
  session_update_baseline/runs/loop08_full_eval_ready \
  improved_final.4dmap
```

Prepare the matching original map from the same Base1 run:

```bash
session_update_baseline/scripts/prepare_base1_khronos_eval_dir.sh \
  session_update_baseline/runs/loop08_full \
  session_update_baseline/runs/loop08_full_original_eval_ready \
  original_final.4dmap
```

Then, if you want to run final-map-only evaluation manually:

```bash
source /home/jixian/ros2_ws/install/setup.bash
/home/jixian/ros2_ws/install/khronos_eval/lib/khronos_eval/exp_pipeline \
  /home/jixian/ros2_ws/install/khronos_eval/share/khronos_eval/config/pipeline/office.yaml \
  session_update_baseline/runs/loop08_full_eval_ready true true true
```

## What Is Still Missing

These are not done yet and should not be claimed:

```text
ray/free-space absence evidence
Panoptic-style active/inactive conflict state transition
POCD-style stationarity probability update
actual object mesh injection into global mesh
true prior object memory scoring for Session B
official Background Precision/Recall/F1 comparison
```

## Current Truth

Base1 currently proves that we can:

```text
load Khronos `.4dmap`
audit object/private mesh vs global mesh
remove global mesh vertices near object private meshes with bbox guard
save a loadable improved final-current `.4dmap`
record exact vertex/object evidence
prepare output for Khronos evaluator
```

It does not yet prove metric improvement. The next mandatory step is official final-map-only eval on `original_final.4dmap` vs `improved_final.4dmap`, using the same Khronos evaluator and the same config.
