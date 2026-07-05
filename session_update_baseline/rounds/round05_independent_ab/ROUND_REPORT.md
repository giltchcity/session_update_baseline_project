# Base1 Strict Round 05 Report

Date: 2026-07-05

Scope: final-current map snapshot only. This is the first strict round after
reclassifying Round 01-04 as exploratory/partial.

## Strict Round Definition

This round follows the user gate definition:

```text
Gate 1: Khronos on native office dataset produces final map.
Gate 2: our_single vs khronos_original, official evaluator, real P/R/F1 table
        at 10 cm and 20 cm.
Gate 3: independent Session A saves final map + object memory; independent
        Session B loads A prior and outputs final map.
Gate 4: B_ours(with memory) vs B_from_scratch, official evaluator, real P/R/F1 table
        at 10 cm and 20 cm.
```

## What Counts As Session A/B Here

Available official-GT data is only the office bag:

```text
/home/jixian/datasets/khronos/tesse_cd/ros2_converted/tesse_cd_office
```

Therefore this strict round uses independent executions on that bag:

```text
Session A: independent Khronos run, truncated to 80s bag playback.
Session B: independent Khronos run, full office bag playback.
```

This is stronger than Round 04 because A/B are not two snapshots extracted from
one already-built `.4dmap`.

## Gate Status

| Gate | Status | Evidence |
|---|---|---|
| Gate 1: B full Khronos run | GREEN | `gate1_B_full_khronos/mesh_only/final.4dmap`, 891 MB, `SAVE_SERVICE_STATUS=0` |
| Gate 2: single-session official metrics | GREEN | `gate2_single/single_metrics.md`; one timestamp |
| Gate 3: independent A/B multi-session wiring | GREEN | A independent memory saved; B independent full run loaded A memory |
| Gate 4: multi-session official metrics | GREEN | `gate4_multi_metrics/multi_metrics.md`; one timestamp |

## Gate 1: B Full Session

Command:

```bash
RUN_MODE=mesh_only USE_GT_FRAME=true PLAY_RATE=2 POST_WAIT=25 \
OUT_ROOT=/home/jixian/Desktop/FT/session_update_baseline/rounds/round05_independent_ab/gate1_B_full_khronos \
scripts/khronos/run_pgmo_official_compare.sh
```

Output:

```text
final.4dmap: 891 MB
dsg.json: 238 MB
SAVE_SERVICE_STATUS=0
Saved 7 time steps
```

## Gate 2: Single-Session Official Metrics

Selected timestamp:

```text
223242999999
```

Both original and improved evals contain exactly this one timestamp.

| label | Background P@10cm | Background R@10cm | Background F1@10cm | Background P@20cm | Background R@20cm | Background F1@20cm | Object F1 | Change F1 | Dynamic F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| khronos_original | 0.946311 | 0.932604 | 0.939408 | 0.987914 | 0.979944 | 0.983913 | 0.677966 | 0.500000 | 0.438438 |
| our_single | 0.972461 | 0.936352 | 0.954065 | 0.996476 | 0.981114 | 0.988735 | 0.677966 | 0.500000 | 0.438438 |

Verdict: GREEN for the measured injection sub-round. Background @10cm improves:
Precision `0.946311 -> 0.972461`, Recall `0.932604 -> 0.936352`,
F1 `0.939408 -> 0.954065`. Background @20cm also improves:
F1 `0.983913 -> 0.988735`.
Object, Change, and Dynamic F1 are unchanged.

Mechanism:

```text
mode: injection
objects_total: 167
injected_objects: 164
injected_vertices: 1916187
injected_faces: 638729
removed_vertices: 0
```

Note: because `removed_vertices=0`, this round does not complete the deletion
part of Base1.1. It verifies the 10 cm injection signal.

## Gate 3: Independent Session A -> Independent Session B

Session A command:

```bash
RUN_MODE=mesh_only USE_GT_FRAME=true PLAY_RATE=2 PLAYBACK_DURATION=80 POST_WAIT=20 \
OUT_ROOT=/home/jixian/Desktop/FT/session_update_baseline/rounds/round05_independent_ab/gate3_A_early_khronos \
scripts/khronos/run_pgmo_official_compare.sh
```

Session A output:

```text
final.4dmap: 168 MB
dsg.json: 97 MB
SAVE_SERVICE_STATUS=0
selected_stamp: 81843019999
objects_total: 75
injected_objects: 73
```

Session B loads Session A:

```text
prior_map: gate3_multi/session_A_independent/improved_final.4dmap
prior_object_memory: gate3_multi/session_A_independent/object_memory.json
```

Session B prior matching:

```text
prior_map_loaded: true
prior_memory_objects: 75
prior_memory_objects_used_for_matching: 75
objects_total: 167
prior_matched_objects: 68
prior_unmatched_current_objects: 99
```

Session-state counts:

```text
persistent_prior_matched: 68
new_or_moved_no_prior_match: 99
```

Verdict: GREEN. This gate uses two independent Khronos executions.

## Gate 4: Multi-Session Official Metrics

Selected timestamp:

```text
223242999999
```

Both B_from_scratch and B_ours_with_memory evals contain exactly this one
timestamp.

| label | Background P@10cm | Background R@10cm | Background F1@10cm | Background P@20cm | Background R@20cm | Background F1@20cm | Object F1 | Change F1 | Dynamic F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B_from_scratch | 0.946311 | 0.932604 | 0.939408 | 0.987914 | 0.979944 | 0.983913 | 0.677966 | 0.500000 | 0.438438 |
| B_ours_with_memory | 0.972461 | 0.936352 | 0.954065 | 0.996476 | 0.981114 | 0.988735 | 0.677966 | 0.500000 | 0.438438 |

Verdict: GREEN numerically at 10 cm and 20 cm. The geometry gain is still the
same current-session injection gain as Gate 2.

## Figures

```text
figures/round05_single_f1.png
figures/round05_multi_f1.png
figures/round05_prior_states.png
```

## Strict Round 05 Conclusion

This is the first strict round that satisfies all four gates with independent
Session A and Session B Khronos executions.

What is established:

```text
1. Khronos full office B session runs and saves a final map.
2. Base1 improves final-snapshot Background P/R/F1 at 10 cm and 20 cm by
   injecting Khronos object private meshes into the final global mesh.
3. A separate truncated Session A saves final map + object memory.
4. A separate full Session B loads Session A memory and assigns persistent vs
   new/moved object states.
5. B_ours_with_memory beats B_from_scratch on Background P/R/F1 at 10 cm and
   20 cm under the official evaluator.
```

What is not established yet:

```text
Object F1, Change F1, and Dynamic F1 do not improve in this strict round.
Prior memory affects object-state assignment, but the geometry improvement still
comes from current-session object mesh injection, not from absent-object deletion
or prior-driven geometric cleanup.
```
