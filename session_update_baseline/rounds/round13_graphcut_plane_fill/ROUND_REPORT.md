# Base1.1 Round13: Graph-Cut Structural Plane Fill

Date: 2026-07-05

Scope: Base1.1 / Mode A, single-session Khronos office final-current map.

## Goal

Try the max-flow / graph-cut direction discussed with the user, while keeping
the Base1.1 reporting contract:

```text
Report Background Precision, Recall, and F1 at 5cm, 10cm, and 20cm.
```

The practical target from the previous loop remains:

```text
F1@10cm must improve over raw Khronos by more than 0.02.
```

Round12 met that 10cm target but failed 5cm F1. Round13 specifically tries to
repair that 5cm failure.

## Method

Selected mechanism:

```text
Graph-cut horizontal structural plane fill.
```

Implementation:

```text
1. Detect strongest lower/upper horizontal structural z bins.
2. In graph-cut mode, also consider the immediately lower adjacent z-bin if it
   has real support. This targets the observed 0.96/1.04 and 3.92/4.00 layer
   split that hurts 5cm.
3. For each selected plane, build a 2D grid over the support footprint.
4. Define x_i in {FILL, EMPTY}.
5. Minimize:
      E(x) = sum_i D_i(x_i) + lambda * sum_(i,j) [x_i != x_j]
   with an s-t max-flow/min-cut.
6. Use robust median support height in graph-cut mode instead of raw 8cm bin
   center.
7. Add selected fill vertices with a 4cm novelty guard, then run the Round09
   hard object-change-gated object mesh injection.
```

Paper/source notes are in:

```text
session_update_baseline/rounds/round13_graphcut_plane_fill/method_search_notes.md
```

## Official Metrics

Official evaluator paths:

```text
raw:
session_update_baseline/rounds/round05_independent_ab/gate2_single/khronos_original_eval/results/background_mesh.csv

round13:
session_update_baseline/rounds/round13_graphcut_plane_fill/gate2_single/our_single_eval_graphcut/results/background_mesh.csv
```

| method | P@5 | R@5 | F1@5 | P@10 | R@10 | F1@10 | P@20 | R@20 | F1@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw Khronos | 0.864665 | 0.633729 | 0.731401 | 0.946311 | 0.932604 | 0.939408 | 0.987914 | 0.979944 | 0.983913 |
| Round12 footprint | 0.809938 | 0.662148 | 0.728624 | 0.970028 | 0.953758 | 0.961824 | 0.991823 | 0.987738 | 0.989776 |
| Round13 graph-cut | 0.808010 | 0.719144 | 0.760991 | 0.969173 | 0.956683 | 0.962887 | 0.991096 | 0.988034 | 0.989563 |

## Delta

Round13 vs raw:

```text
5cm:  P -0.056655   R +0.085415   F1 +0.029591
10cm: P +0.022862   R +0.024079   F1 +0.023480
20cm: P +0.003182   R +0.008090   F1 +0.005650
```

Round13 vs Round12:

```text
5cm:  P -0.001928   R +0.056996   F1 +0.032367
10cm: P -0.000855   R +0.002925   F1 +0.001063
20cm: P -0.000727   R +0.000296   F1 -0.000214
```

Verdict:

```text
Round13 improves F1 at 5cm, 10cm, and 20cm over raw Khronos.
It also keeps the required >2 percentage point F1@10 improvement over raw.
It does not improve Precision@5 over raw, and it does not add another +2 points
over Round12 at 10cm.
```

## Mesh Update Counts

```text
horizontal_planes_filled: 3
horizontal_plane_vertices: 65626
horizontal_plane_faces: 39722
horizontal_plane_graph_cut_cells: 137273
horizontal_plane_graph_cut_fill_cells: 137273
injected_objects: 142
injected_vertices: 1716396
removed_vertices: 0
```

Important self-audit:

```text
The current graph-cut unary selected the full supported footprint for the chosen
planes. The 5cm gain mostly comes from adding the adjacent structural layer and
using 4cm novelty/median height, not from a strong internal cut boundary.
```

So this is a useful graph-cut framework and a better metric result, but not yet
the final elegant free-space-aware max-flow segmentation.

## End-of-Round Self Check

Q1 Official evaluator actually run?

```text
Yes.
session_update_baseline/rounds/round13_graphcut_plane_fill/gate2_single/our_single_eval_graphcut/results/background_mesh.csv
```

Q2 Precision and Recall separately at 5cm, 10cm, and 20cm?

```text
Yes. See table above.
```

Q3 Did 5cm improve?

```text
F1@5 improved from 0.731401 raw to 0.760991.
Recall@5 improved from 0.633729 raw to 0.719144.
Precision@5 did not improve.
```

Q4 Did 10cm stay above the required +2 percentage points?

```text
Yes.
F1@10 raw:     0.939408
F1@10 round13: 0.962887
delta:        +0.023480
```

Q5 Did deletion run?

```text
No.
removed_vertices=0.
Base1.1 mechanism (2), deletion of ghost/leakage, is still unfinished.
```

Q6 One-line verdict:

```text
Graph-cut structural plane fill makes the first round where F1 improves at
5cm, 10cm, and 20cm over raw Khronos, while preserving the >2 point F1@10 gain.
```
