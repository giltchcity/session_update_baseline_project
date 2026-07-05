# Base1 Exploratory Round 01 Report

Date: 2026-07-05

Scope: final-current map snapshot only. The maps are stored in Khronos `.4dmap`
containers so that the official Khronos evaluator can load them, but Base1 does
not evaluate full 4D temporal queries in this round.

Status correction, 2026-07-05: this is an exploratory/partial round, not a
strict completed round under the final user definition. Gate 3 only verifies
multi-session plumbing and does not run independent Session A and Session B
Khronos executions.

## Gate Status

| Gate | Status | Evidence |
|---|---|---|
| Gate 1: single-session Khronos run | GREEN with save-timeout warning | `gate1_khronos/mesh_only/final.4dmap` exists, 674 MB |
| Gate 2: single-session official metrics | GREEN | `gate2_single/single_metrics.md` |
| Gate 3: multi-session wiring | GREEN, mechanical | `session_A/object_memory.json` saved; `session_B_ours` loaded prior map and prior memory; `session_B_ours/improved_final.4dmap` saved |
| Gate 4: multi-session official metrics | GREEN | `gate4_multi_metrics/multi_metrics.md` |

Important caveat: Gate 3 is a wiring pass, not yet a true prior-driven update.
The runner loads the prior map and counts the prior object memory, but Round 01
does not yet use prior memory in the reconciliation decision. The metric gain in
this round comes from object-private-mesh injection into the final global mesh.

## Gate 1

Command:

```bash
RUN_MODE=mesh_only USE_GT_FRAME=true PLAY_RATE=5 POST_WAIT=20 \
OUT_ROOT=/home/jixian/Desktop/FT/session_update_baseline/rounds/round01_fresh/gate1_khronos \
scripts/khronos/run_pgmo_official_compare.sh
```

Output:

```text
/home/jixian/Desktop/FT/session_update_baseline/rounds/round01_fresh/gate1_khronos/mesh_only/final.4dmap
```

Observed warning:

```text
SAVE_SERVICE_STATUS=124
```

The final map was nevertheless written and loaded by the official evaluator.

## Gate 2: Single-Session Metrics

| label | background_p | background_r | background_f1 | object_p | object_r | object_f1 | change_p | change_r | change_f1 | dynamic_p | dynamic_r | dynamic_f1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| khronos_original | 0.984319 | 0.852898 | 0.913908 | 0.971014 | 0.446667 | 0.611872 | 0.300000 | 0.600000 | 0.400000 | 0.732984 | 0.269231 | 0.393812 |
| our_single | 0.995875 | 0.854081 | 0.919544 | 0.972222 | 0.457516 | 0.622222 | 0.375000 | 0.600000 | 0.461538 | 0.732984 | 0.269231 | 0.393812 |

Verdict: GREEN. Background, Object, and Change F1 improved under the official
final-map evaluator. Dynamic metrics are unchanged.

Mechanism in this round:

```text
mode: injection
injected_objects: 116
injected_vertices: 1424001
injected_faces: 474667
removed_vertices: 0
```

## Gate 3: Multi-Session Wiring

Session A was copied from Gate 2 output:

```text
gate3_multi/session_A/improved_final.4dmap
gate3_multi/session_A/object_memory.json
```

Session B command loaded:

```text
prior_map: gate3_multi/session_A/improved_final.4dmap
prior_object_memory: gate3_multi/session_A/object_memory.json
```

Evidence:

```text
prior_map_loaded: true
prior_memory_objects: 121
session_B_ours/improved_final.4dmap saved
```

Verdict: GREEN for plumbing. Not yet green as a scientific claim that prior
memory changes decisions.

## Gate 4: Multi-Session Metrics

| label | background_p | background_r | background_f1 | object_p | object_r | object_f1 | change_p | change_r | change_f1 | dynamic_p | dynamic_r | dynamic_f1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B_from_scratch | 0.984319 | 0.852898 | 0.913908 | 0.971014 | 0.446667 | 0.611872 | 0.300000 | 0.600000 | 0.400000 | 0.732984 | 0.269231 | 0.393812 |
| B_ours_with_memory | 0.995875 | 0.854081 | 0.919544 | 0.972222 | 0.457516 | 0.622222 | 0.375000 | 0.600000 | 0.461538 | 0.732984 | 0.269231 | 0.393812 |

Verdict: GREEN numerically under the stated gate. However, this is still driven
by the same object-mesh injection mechanism as Gate 2, not by prior-memory
decision logic.

## Round 01 Conclusion

Round 01 satisfies the four mechanical gates and has official evaluator numbers.
The first useful effect is confirmed:

```text
Khronos object private meshes can be injected into the final global mesh and
improve Background/Object/Change F1 on the office final snapshot.
```

The next round must make cross-session memory participate in the update instead
of merely loading it.
