# Base1 Exploratory Round 03 Report

Date: 2026-07-05

Scope: final-current map snapshot only. This round fixes the evaluation
container mismatch found after Round 02: `original_final.4dmap` and
`improved_final.4dmap` are now both saved as one selected DSG at the same
timestamp.

Status correction, 2026-07-05: this is an exploratory/partial round, not a
strict completed round under the final user definition. It fixes final-snapshot
evaluation but does not run independent Session A and Session B Khronos
executions.

## Round 03 Change

Runner changes:

```text
--map_time latest|earliest|index:N|timestamp:NS
original_final.4dmap is now saved via saveMapWithSingleDsg(...)
improved_final.4dmap is saved at the same selected timestamp
```

This fixes the Round 02 issue where original eval had multiple map timestamps
but improved eval had one timestamp.

## Gate Status

| Gate | Status | Evidence |
|---|---|---|
| Gate 1: single-session Khronos run | GREEN after rerun | First attempt produced a tiny map and was rejected; second attempt with `PLAY_RATE=2` saved a full map |
| Gate 2: single-session official metrics | GREEN | `gate2_single/single_metrics.md`; both evals have one identical timestamp |
| Gate 3: multi-session wiring + prior matching | GREEN | `prior_map_loaded=true`, `prior_memory_objects=166`, `prior_matched_objects=166` |
| Gate 4: multi-session official metrics | GREEN | `gate4_multi_metrics/multi_metrics.md`; both evals have one identical timestamp |

## Gate 1

First Gate 1 attempt was rejected:

```text
final.4dmap: 75 MB
dsg.json: 97 KB
```

This indicated an incomplete map despite `SAVE_SERVICE_STATUS=0`.

Accepted Gate 1 command:

```bash
RUN_MODE=mesh_only USE_GT_FRAME=true PLAY_RATE=2 POST_WAIT=25 \
OUT_ROOT=/home/jixian/Desktop/FT/session_update_baseline/rounds/round03_single_snapshot/gate1_khronos \
scripts/khronos/run_pgmo_official_compare.sh
```

Accepted output:

```text
final.4dmap: 858 MB
dsg.json: 277 MB
SAVE_SERVICE_STATUS=0
Saved 7 time steps
```

## Gate 2: Single-Session Official Metrics

Selected timestamp:

```text
224043000000
```

Both original and improved evals contain exactly this one timestamp.

| label | background_p | background_r | background_f1 | object_p | object_r | object_f1 | change_p | change_r | change_f1 | dynamic_p | dynamic_r | dynamic_f1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| khronos_original | 0.987428 | 0.986685 | 0.987056 | 0.989474 | 0.666667 | 0.796610 | 0.461538 | 0.750000 | 0.571429 | 0.846154 | 0.444231 | 0.582598 |
| our_single | 0.996383 | 0.987344 | 0.991843 | 0.989474 | 0.666667 | 0.796610 | 0.461538 | 0.750000 | 0.571429 | 0.846154 | 0.444231 | 0.582598 |

Verdict: GREEN. Background F1 improves from `0.987056` to `0.991843`.
Object, Change, and Dynamic F1 are unchanged.

Mechanism:

```text
mode: injection
objects_total: 166
injected_objects: 162
injected_vertices: 1944855
injected_faces: 648285
removed_vertices: 0
```

## Gate 3: Multi-Session Wiring

Session A saves:

```text
gate3_multi/session_A/improved_final.4dmap
gate3_multi/session_A/object_memory.json
```

Session B loads:

```text
prior_map: gate3_multi/session_A/improved_final.4dmap
prior_object_memory: gate3_multi/session_A/object_memory.json
```

Prior matching result:

```text
prior_map_loaded: true
prior_memory_objects: 166
prior_memory_objects_used_for_matching: 166
prior_matched_objects: 166
prior_unmatched_current_objects: 0
session_state counts:
  persistent_prior_matched: 166
```

Verdict: GREEN.

## Gate 4: Multi-Session Official Metrics

Selected timestamp:

```text
224043000000
```

Both B_from_scratch and B_ours_with_memory evals contain exactly this one
timestamp.

| label | background_p | background_r | background_f1 | object_p | object_r | object_f1 | change_p | change_r | change_f1 | dynamic_p | dynamic_r | dynamic_f1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B_from_scratch | 0.987428 | 0.986685 | 0.987056 | 0.989474 | 0.666667 | 0.796610 | 0.461538 | 0.750000 | 0.571429 | 0.846154 | 0.444231 | 0.582598 |
| B_ours_with_memory | 0.996383 | 0.987344 | 0.991843 | 0.989474 | 0.666667 | 0.796610 | 0.461538 | 0.750000 | 0.571429 | 0.846154 | 0.444231 | 0.582598 |

Verdict: GREEN. Background F1 improves from `0.987056` to `0.991843`.

## Figures

```text
figures/round03_single_f1.png
figures/round03_multi_f1.png
figures/round03_prior_states.png
```

## Round 03 Conclusion

Round 03 completes all four gates with a stricter single-snapshot evaluation
contract.

What is now cleanly established:

```text
1. Khronos office run produces a valid final map.
2. Original and improved maps are evaluated at the same single final timestamp.
3. Object private mesh injection improves final-snapshot Background F1.
4. Multi-session plumbing loads and matches prior object memory.
```

Remaining limitation:

```text
Session A and Session B in this round are both the latest office snapshot, so
the prior state is persistent-only. This is not yet a changed-cross-session
case with new/moved/absent objects.
```
