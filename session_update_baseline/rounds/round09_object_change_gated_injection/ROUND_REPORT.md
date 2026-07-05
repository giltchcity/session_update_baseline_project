# Base1.1 Round 09: Object-Change-Gated Injection

Date: 2026-07-05

Scope: Mode A single-session office final-current map.

## Mechanism

Round08 showed that injection epsilon tuning alone does not beat old full append.
Round09 uses Khronos object-level change evidence instead.

Input evidence:

```text
session_update_baseline/rounds/round05_independent_ab/gate1_B_full_khronos/mesh_only/object_changes.csv
```

Rule:

```text
For each object:
  if last_absent > last_persistent:
    skip injecting that object private mesh
  else:
    inject the object private mesh
```

This is an object-level gate from real Khronos change output, not a geometric
distance-only heuristic. It follows the paper-search lesson: decide object
state first, then update geometry.

Code:

```text
session_update_baseline/src/base1/object_guided_map_reconciler.cpp
loadAbsentObjectIds(...)
```

CLI:

```bash
--object_changes_csv <path>/object_changes.csv
--injection_min_separation_m 0
```

## Object Gate Stats

```text
object_changes rows:       164
absent-like skipped:       22
injected_objects:          142
injected_vertices:         1716396
old full injected_objects:  164
old full injected_vertices: 1916187
```

## Official 10 cm Results

Same round05 B full Khronos map, official evaluator, final-current snapshot.

| mechanism | injected objects | injected vertices | P@10cm | R@10cm | F1@10cm | P@20cm | R@20cm | F1@20cm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw Khronos | 0 | 0 | 0.946311 | 0.932604 | 0.939408 | 0.987914 | 0.979944 | 0.983913 |
| old full append | 164 | 1916187 | 0.972461 | 0.936352 | 0.954065 | 0.996476 | 0.981114 | 0.988735 |
| eps=0.005, no change gate | 164 | 1911326 | 0.972441 | 0.936352 | 0.954055 | 0.996482 | 0.981114 | 0.988738 |
| change gate, eps=0 | 142 | 1716396 | 0.976665 | 0.936233 | 0.956022 | 0.997421 | 0.981081 | 0.989184 |
| change gate, eps=0.005 | 142 | 1711856 | 0.976626 | 0.936233 | 0.956003 | 0.997415 | 0.981081 | 0.989181 |
| change gate, eps=0.020 | 142 | 1571718 | 0.975495 | 0.936221 | 0.955455 | 0.997264 | 0.981081 | 0.989106 |
| change gate + deletion d=0.01 | 142 | 1716396 | 0.976661 | 0.936228 | 0.956017 | 0.997420 | 0.981081 | 0.989183 |

## Verdict

This is a real improvement over the previous best.

```text
Old best F1@10cm: 0.954065
Round09 F1@10cm:  0.956022
Delta:            +0.001957

Old best P@10cm:  0.972461
Round09 P@10cm:   0.976665
Delta:            +0.004204
```

Recall drops slightly:

```text
Old best R@10cm:  0.936352
Round09 R@10cm:   0.936233
Delta:            -0.000119
```

The tradeoff is acceptable for this precision-side improvement because the
precision gain is much larger than the recall loss.

## Next Loop

Do not continue epsilon tuning. The best result is object-change-gated full
append among non-absent objects.

Do not enable the current distance-shell deletion by default. Even at `d=0.01`,
it slightly underperforms injection-only:

```text
change gate injection-only F1@10cm: 0.956022
change gate + deletion F1@10cm:    0.956017
```

Next target:

```text
Use absent-like object ids not just to skip injection, but to delete background
ghost geometry with a conservative bbox/free-space guard.
```

This should be tested separately because previous distance-only deletion failed.
