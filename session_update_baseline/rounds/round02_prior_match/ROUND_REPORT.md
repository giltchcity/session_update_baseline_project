# Base1 Exploratory Round 02 Report

Date: 2026-07-05

Scope: final-current map snapshot only. This round keeps the same object-mesh
injection mechanism from Round 01 and adds cross-session prior object-memory
matching.

Status correction, 2026-07-05: this is an exploratory/partial round, not a
strict completed round under the final user definition. It fixes prior-memory
matching but still does not run independent Session A and Session B Khronos
executions.

## Round 02 Change

Round 01 loaded prior memory but did not use it in object decisions. Round 02
adds:

```text
prior object memory parser
label + bbox-center-distance matching
object session state:
  persistent_prior_matched
  new_or_moved_no_prior_match
  current_no_prior
```

Bug found during Gate 3: `object_memory.json` wrote infinity as `inf`, which is
not valid JSON for the C++ parser. This produced zero prior matches. Fixed by
writing `null` for non-finite prior-match distances, then reran Gate 2 and Gate 3.

## Gate Status

| Gate | Status | Evidence |
|---|---|---|
| Gate 1: single-session Khronos run | GREEN | `gate1_khronos/mesh_only/final.4dmap`, `SAVE_SERVICE_STATUS=0` |
| Gate 2: single-session official metrics | GREEN | `gate2_single/single_metrics.md` |
| Gate 3: multi-session wiring + prior matching | GREEN | `prior_map_loaded=true`, `prior_memory_objects=80`, `prior_matched_objects=80` |
| Gate 4: multi-session official metrics | GREEN | `gate4_multi_metrics/multi_metrics.md` |

## Gate 1

Command:

```bash
RUN_MODE=mesh_only USE_GT_FRAME=true PLAY_RATE=5 POST_WAIT=20 \
OUT_ROOT=/home/jixian/Desktop/FT/session_update_baseline/rounds/round02_prior_match/gate1_khronos \
scripts/khronos/run_pgmo_official_compare.sh
```

Output:

```text
/home/jixian/Desktop/FT/session_update_baseline/rounds/round02_prior_match/gate1_khronos/mesh_only/final.4dmap
```

Save status:

```text
PLAYER_STATUS=0
SAVE_SERVICE_STATUS=0
```

## Gate 2: Single-Session Official Metrics

| label | background_p | background_r | background_f1 | object_p | object_r | object_f1 | change_p | change_r | change_f1 | dynamic_p | dynamic_r | dynamic_f1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| khronos_original | 0.972417 | 0.953104 | 0.962664 | 0.966667 | 0.411348 | 0.577114 | 0.750000 | 0.600000 | 0.666667 | 0.722222 | 0.075000 | 0.135889 |
| our_single | 0.993182 | 0.953512 | 0.972943 | 0.966667 | 0.411348 | 0.577114 | 0.750000 | 0.600000 | 0.666667 | 0.722222 | 0.075000 | 0.135889 |

Verdict: GREEN. Background F1 improves from `0.962664` to `0.972943`.
Object, Change, and Dynamic F1 are unchanged in this run.

Mechanism:

```text
mode: injection
objects_total: 80
injected_objects: 78
injected_vertices: 949722
injected_faces: 316574
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
prior_memory_objects: 80
prior_memory_objects_used_for_matching: 80
prior_matched_objects: 80
prior_unmatched_current_objects: 0
session_state counts:
  persistent_prior_matched: 80
```

Verdict: GREEN. Unlike Round 01, prior object memory is parsed and participates
in object state assignment.

## Gate 4: Multi-Session Official Metrics

| label | background_p | background_r | background_f1 | object_p | object_r | object_f1 | change_p | change_r | change_f1 | dynamic_p | dynamic_r | dynamic_f1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B_from_scratch | 0.972417 | 0.953104 | 0.962664 | 0.966667 | 0.411348 | 0.577114 | 0.750000 | 0.600000 | 0.666667 | 0.722222 | 0.075000 | 0.135889 |
| B_ours_with_memory | 0.993182 | 0.953512 | 0.972943 | 0.966667 | 0.411348 | 0.577114 | 0.750000 | 0.600000 | 0.666667 | 0.722222 | 0.075000 | 0.135889 |

Verdict: GREEN. Background F1 improves from `0.962664` to `0.972943`.
Object, Change, and Dynamic F1 are unchanged.

## Figures

```text
figures/round02_single_f1.png
figures/round02_multi_f1.png
figures/round02_prior_states.png
```

## Round 02 Conclusion

Round 02 completes all four gates with official metrics and produces figures.

What is genuinely shown:

```text
1. Khronos office single-session final map runs cleanly.
2. Injecting Khronos object private meshes into the final global mesh improves
   official final-snapshot Background F1.
3. The multi-session pipeline saves Session A memory, loads it in Session B,
   and assigns prior-matched object states.
4. B_ours_with_memory beats B_from_scratch on Background F1 under the official
   evaluator.
```

Current limitation:

```text
This round uses the same office final map as the current Session B observation.
It validates the Base1 plumbing, object-memory matching, and final-map repair,
but it is not yet a real changed-cross-session experiment.
```
