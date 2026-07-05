# Base1.1 Round12: Footprint-Guarded Horizontal Plane Fill

Date: 2026-07-05

Scope: Base1.1 / Mode A, single-session Khronos office final-current map.

## Goal Restatement

Per `base1.1_mode_a_single_session.md`, this round must improve the final-current
background map with official evaluator numbers, reporting Background Precision
and Recall separately at 10cm and 20cm. Per user request, 5cm is also reported
as a diagnostic column.

The user-required practical target for this loop:

```text
F1@10cm must improve over raw Khronos by more than 0.02.
```

## Why The Method Changed

Object-only methods were not enough. Round09 injected 1.7M object-private-mesh
vertices but only reached:

```text
F1@10cm delta vs raw: +0.016614
```

Error audit showed the remaining 10cm false negatives are mostly outside object
bboxes:

```text
Round09 GT outliers @10cm:          192588
inside current object bbox:          10056
outside current object bbox:        182532
```

So the next mechanism must repair large background planes, not keep tuning
object selection.

## Method

Selected mechanism:

```text
Footprint-guarded horizontal plane fill.
```

Algorithm:

```text
1. Quantize current background mesh z values at 8cm.
2. Pick the strongest supported lower and upper horizontal planes.
3. Collect observed support cells for each plane in an 8cm x/y grid.
4. Candidate cell (ix, iy) is fillable only if:
   - ix lies inside the observed support interval for row iy
   - iy lies inside the observed support interval for column ix
5. Add the candidate vertex only if its nearest existing mesh vertex is >8cm away.
6. Then run the Round09 object-change-gated object mesh injection.
```

This fixes the failed rectangle-fill version, which overfilled whole bounding
rectangles and collapsed Precision@10.

## Official Metrics

Official evaluator paths:

```text
raw:
session_update_baseline/rounds/round05_independent_ab/gate2_single/khronos_original_eval/results/background_mesh.csv

our round:
session_update_baseline/rounds/round12_horizontal_plane_fill/gate2_single/our_single_eval_footprint/results/background_mesh.csv
```

| method | P@5 | R@5 | F1@5 | P@10 | R@10 | F1@10 | P@20 | R@20 | F1@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw Khronos | 0.864665 | 0.633729 | 0.731401 | 0.946311 | 0.932604 | 0.939408 | 0.987914 | 0.979944 | 0.983913 |
| Round09 object gate | 0.815716 | 0.646468 | 0.721297 | 0.976665 | 0.936233 | 0.956022 | 0.997421 | 0.981081 | 0.989184 |
| rectangle plane fill | 0.724162 | 0.685866 | 0.704494 | 0.868201 | 0.969674 | 0.916136 | 0.890773 | 0.991297 | 0.938350 |
| footprint plane fill | 0.809938 | 0.662148 | 0.728624 | 0.970028 | 0.953758 | 0.961824 | 0.991823 | 0.987738 | 0.989776 |

## Delta

Footprint plane fill vs raw:

```text
P@10:  +0.023717
R@10:  +0.021154
F1@10: +0.022416
```

This exceeds the requested 2 percentage point improvement at 10cm.

Footprint plane fill vs Round09:

```text
P@10:  -0.006637
R@10:  +0.017525
F1@10: +0.005802
```

The method trades some precision for a much larger recall gain. Precision@10
remains above raw Khronos.

5cm diagnostic:

```text
F1@5 vs raw: -0.002777
```

So this is a real 10cm/20cm Base1.1 improvement, but not a strict 5cm surface
quality improvement. 5cm remains a diagnostic problem for later local alignment
or finer plane placement.

## Mesh Update Counts

```text
horizontal_planes_filled: 2
horizontal_plane_vertices: 20385
horizontal_plane_faces: 21026
injected_objects: 142
injected_vertices: 1716396
removed_vertices: 0
```

Mechanism status:

```text
Injection/fill moved Recall@10.
Deletion is still not implemented in this successful round: removed_vertices=0.
```

## G Gates

G1 Khronos office single-session final map produced:

```text
session_update_baseline/rounds/round05_independent_ab/gate1_B_full_khronos/mesh_only/final.4dmap
```

G2 Our improved map produced:

```text
session_update_baseline/rounds/round12_horizontal_plane_fill/gate2_single/our_single_eval_footprint/final.4dmap
```

G3 No-op IO safe:

```text
session_update_baseline/rounds/round12_horizontal_plane_fill/gate2_single/no_op/noop_metrics.md
```

Raw and no-op official metrics are identical.

G4 Official evaluator run on both:

```text
session_update_baseline/rounds/round12_horizontal_plane_fill/gate2_single/round12_footprint_metrics.md
```

G5 Real mechanism effect:

```text
horizontal plane fill raised R@10 from 0.932604 raw to 0.953758.
```

## End-of-Round Self Check

Q1 Official evaluator actually run?

```text
Yes.
session_update_baseline/rounds/round12_horizontal_plane_fill/gate2_single/our_single_eval_footprint/results/background_mesh.csv
```

Q2 Precision and Recall separately at 20cm and 10cm?

```text
Yes. See table above. 5cm is also reported.
```

Q3 If injection/fill ran, did Background Recall move?

```text
Yes.
R@10 raw: 0.932604
R@10 round: 0.953758
delta: +0.021154
```

Q4 If deletion ran, is removed_vertices > 0 and ghost/leakage separated?

```text
No deletion in this round.
removed_vertices=0.
Mechanism (2) is not complete yet.
```

Q5 Does the scoring/deletion criterion cite paper equations?

```text
No deletion criterion used.
Plane fill is justified by planar-prior reconstruction papers and Khronos's 8cm
resolution; see method_search_notes.md.
```

Q6 One-line verdict:

```text
Footprint-guarded horizontal plane fill moved Background Recall@10 by +0.021154
and F1@10 by +0.022416 over raw Khronos, exceeding the requested 2 percentage
point improvement, while keeping Precision@10 above raw.
```
