# Base1 Exploratory Round 04 Report

Date: 2026-07-05

Scope: final-current map snapshot only. This round tests a more realistic
cross-session protocol by using an early office snapshot as Session A memory and
the latest office snapshot as Session B current observation.

Status correction, 2026-07-05: this is still exploratory/partial. It uses an
early snapshot and latest snapshot from one Khronos execution, not independent
Session A and Session B Khronos executions. It must not be reported as the final
strict round.

## Round 04 Change

Protocol change:

```text
Session A: map_time=index:0
Session B: map_time=latest
```

Matching fix:

```text
Prior object matching is one-to-one greedy by semantic label + bbox center
distance. This prevents one prior object from matching multiple current objects.
```

## Gate Status

| Gate | Status | Evidence |
|---|---|---|
| Gate 1: single-session Khronos run | GREEN | `SAVE_SERVICE_STATUS=0`, final map 811 MB |
| Gate 2: single-session official metrics | GREEN | `gate2_single/single_metrics.md`; one timestamp |
| Gate 3: multi-session early-to-latest | GREEN | A early memory: 10 objects; B latest: 126 objects; 10 persistent, 116 new/moved |
| Gate 4: multi-session official metrics | GREEN | `gate4_multi_metrics/multi_metrics.md`; one timestamp |

## Gate 1

Command:

```bash
RUN_MODE=mesh_only USE_GT_FRAME=true PLAY_RATE=2 POST_WAIT=25 \
OUT_ROOT=/home/jixian/Desktop/FT/session_update_baseline/rounds/round04_early_to_latest/gate1_khronos \
scripts/khronos/run_pgmo_official_compare.sh
```

Output:

```text
final.4dmap: 811 MB
dsg.json: 278 MB
SAVE_SERVICE_STATUS=0
Saved 7 time steps
```

## Gate 2: Single-Session Official Metrics

Selected timestamp:

```text
212143000000
```

Both original and improved evals contain exactly this one timestamp.

| label | background_p | background_r | background_f1 | object_p | object_r | object_f1 | change_p | change_r | change_f1 | dynamic_p | dynamic_r | dynamic_f1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| khronos_original | 0.987189 | 0.985353 | 0.986270 | 0.973684 | 0.500000 | 0.660714 | 0.384615 | 0.625000 | 0.476190 | 0.846743 | 0.425000 | 0.565941 |
| our_single | 0.997624 | 0.985686 | 0.991619 | 0.973684 | 0.500000 | 0.660714 | 0.384615 | 0.625000 | 0.476190 | 0.846743 | 0.425000 | 0.565941 |

Verdict: GREEN. Background F1 improves from `0.986270` to `0.991619`.
Object, Change, and Dynamic F1 are unchanged.

Mechanism:

```text
mode: injection
objects_total: 126
injected_objects: 121
injected_vertices: 1489902
injected_faces: 496634
removed_vertices: 0
```

## Gate 3: Multi-Session Early-To-Latest

Session A:

```text
map_time: index:0
selected_stamp: 31793019999
objects_total: 10
injected_objects: 8
```

Session B:

```text
map_time: latest
selected_stamp: 212143000000
objects_total: 126
prior_map_loaded: true
prior_memory_objects: 10
prior_memory_objects_used_for_matching: 10
prior_matched_objects: 10
prior_unmatched_current_objects: 116
```

Session-state counts:

```text
persistent_prior_matched: 10
new_or_moved_no_prior_match: 116
```

Verdict: GREEN. This is the first round where Session B is not simply the same
object state as Session A. It exercises the cross-session memory path:

```text
old memory from early visit
new/latest observation
persistent vs new/moved object state assignment
```

## Gate 4: Multi-Session Official Metrics

Selected timestamp:

```text
212143000000
```

Both B_from_scratch and B_ours_with_memory evals contain exactly this one
timestamp.

| label | background_p | background_r | background_f1 | object_p | object_r | object_f1 | change_p | change_r | change_f1 | dynamic_p | dynamic_r | dynamic_f1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B_from_scratch | 0.987189 | 0.985353 | 0.986270 | 0.973684 | 0.500000 | 0.660714 | 0.384615 | 0.625000 | 0.476190 | 0.846743 | 0.425000 | 0.565941 |
| B_ours_with_memory | 0.997624 | 0.985686 | 0.991619 | 0.973684 | 0.500000 | 0.660714 | 0.384615 | 0.625000 | 0.476190 | 0.846743 | 0.425000 | 0.565941 |

Verdict: GREEN. Background F1 improves from `0.986270` to `0.991619`.

## Figures

```text
figures/round04_single_f1.png
figures/round04_multi_f1.png
figures/round04_prior_states.png
```

## Round 04 Conclusion

Round 04 completes all four gates with a true early-to-latest cross-session
state split.

What is established:

```text
1. Base1 can evaluate single final snapshots with a clean timestamp contract.
2. Object private mesh injection improves final-snapshot Background F1.
3. Session A memory can be loaded by Session B.
4. Session B can classify current objects as persistent-prior-matched or
   new/moved-no-prior-match.
5. B_ours_with_memory beats B_from_scratch on Background F1 under the official
   evaluator.
```

Current limitation:

```text
The prior object state affects diagnostics and object-state assignment. The
map-quality gain still comes from object private mesh injection, not yet from
deleting absent prior objects or using prior memory to change geometry decisions.
```
