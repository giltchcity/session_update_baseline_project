# Base1.1 Round 10: Object Move Probability

Date: 2026-07-05

Scope: Mode A single-session office final-current map.

## Method

This round adds an object-level move probability instead of treating
`object_changes.csv` as a binary skip list.

For each object, the change evidence is converted to a log-odds score:

```text
has_absent     = first_absent > 0 or last_absent > 0
has_persistent = first_persistent > 0 or last_persistent > 0

logit(P_moved) starts with a static prior.
recent absent evidence raises the logit.
recent persistent evidence lowers the logit.
P_moved = sigmoid(logit)
```

The code supports three decisions:

```text
hard:
  skip injection iff last_absent > last_persistent

probability:
  skip injection iff P_moved >= object_move_skip_probability

expected_utility:
  missing_probability = clamp(1 - global_vertices_in_bbox / object_mesh_vertices, 0, 1)
  static_repair_gain = (1 - P_moved) * missing_probability
  moved_hallucination_risk = P_moved
  skip injection iff moved_hallucination_risk > static_repair_gain
```

The intended clean method is `expected_utility`: inject object private mesh only
when the object is probably still static and the current background appears to
be missing that object geometry.

## Code

```text
session_update_baseline/include/session_update_baseline/base1/object_guided_map_reconciler.h
session_update_baseline/src/base1/object_guided_map_reconciler.cpp
session_update_baseline/app/run_session_update_baseline.cpp
```

New CLI:

```bash
--object_move_decision hard|probability|expected_utility
--object_move_skip_probability FLOAT
--object_move_time_scale_s FLOAT
```

`hard` is the default to preserve the Round09 behavior.

## Official 10 cm Results

Official evaluator, same final-current office map.

| method | injected objects | injected vertices | P@10cm | R@10cm | F1@10cm | P@20cm | R@20cm | F1@20cm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Round09 hard gate | 142 | 1716396 | 0.976665 | 0.936233 | 0.956022 | 0.997421 | 0.981081 | 0.989184 |
| probability gate, best observed | 145 | 1739895 | 0.976880 | 0.936252 | 0.956135 | 0.997451 | 0.981081 | 0.989198 |
| expected utility | 140 | 1721355 | 0.976681 | 0.936182 | 0.956003 | 0.997428 | 0.981081 | 0.989187 |

## Verdict

The clean `expected_utility` method is mathematically better structured than the
Round09 hard gate, but this implementation does not improve the metric yet.

It over-skips some objects because `global_vertices_in_bbox / object_mesh_vertices`
is only a coarse proxy for whether object geometry is already represented in the
background. The proxy measures occupancy in a box, not whether the injected
object surface would be an evaluator inlier.

The useful signal from this round:

```text
Round09 hard gate skipped 22 objects.
Probability posterior showed that 3 of those were weak move cases.
Injecting those 3 back produced the best observed score:
F1@10cm 0.956135 vs 0.956022.
```

## Next Method

Do not keep tuning thresholds.

The next real method should replace the coarse bbox support proxy with an
object-surface likelihood:

```text
For object o:
  transform private mesh vertices into world frame
  sample object surface points
  compute distance-to-current-background distribution
  estimate P(surface_is_already_supported) and P(surface_would_be_hallucinated)
  combine those with P_moved in the expected utility decision
```

That would make the decision depend on object surface evidence, not just bbox
counts or a tuned probability threshold.
