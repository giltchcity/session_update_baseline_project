# Base1.1 Round 07: Paper-Guided Deletion Attempt

Date: 2026-07-05

Scope: Mode A single-session office final-current map. This round followed the
paper notes in `method_search_notes.md` and tested a minimal gated deletion rule.

## Paper-Guided Rule Tested

The implementation kept Round06 duplicate-suppressed injection and added a
small deletion pass:

```text
1. Inject object private mesh vertices with min separation epsilon = 0.02 m.
2. For deletion, only use objects whose bbox already has enough global mesh support.
3. Delete only original background vertices, never vertices injected in this pass.
4. Candidate deletion vertex must be inside the object bbox and near the object private mesh.
```

The object support gate reuses:

```text
repair_global_vertex_ratio_threshold = 0.05
```

Objects below this ratio are treated as missing-background / repair targets, not
deletion targets.

Code:

```text
session_update_baseline/src/base1/object_guided_map_reconciler.cpp
collectCleanupSources(...)
```

## Official 10 cm Results

Same round05 B full Khronos map, official evaluator, final-current snapshot.

| mechanism | removed vertices | injected vertices | P@10cm | R@10cm | F1@10cm | P@20cm | R@20cm | F1@20cm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw Khronos | 0 | 0 | 0.946311 | 0.932604 | 0.939408 | 0.987914 | 0.979944 | 0.983913 |
| Round06 injection only | 0 | 1758633 | 0.971556 | 0.936339 | 0.953622 | 0.996435 | 0.981113 | 0.988715 |
| gated deletion d=0.01 | 1294 | 1758633 | 0.971549 | 0.936337 | 0.953618 | 0.996433 | 0.981113 | 0.988714 |
| gated deletion d=0.02 | 6152 | 1758633 | 0.971529 | 0.936326 | 0.953603 | 0.996428 | 0.981113 | 0.988711 |
| gated deletion d=0.03 | 11832 | 1758633 | 0.971553 | 0.936313 | 0.953608 | 0.996419 | 0.981111 | 0.988706 |

## Verdict

This deletion attempt is not a metric win. It satisfies `removed_vertices > 0`,
but it does not improve Background Precision @10cm over Round06 injection-only.

Failure hypothesis:

```text
The deleted near-object background vertices are mostly still GT inliers under
the official background metric. A distance-to-object shell is not enough evidence
of ghost/leakage.
```

This matches the paper search: Khronos/Panoptic/POCD all require absence,
conflict, or stationarity evidence before deletion. Our current rule only has
local proximity and weak object support.

## Next Loop

Do not continue tuning `object_distance_m` alone.

The next small mechanism should add one actual evidence term:

```text
ray/free-space proxy:
  use robot trajectory + final mesh/object bbox;
  mark a vertex deletable only if it lies in a re-observed ray corridor before
  the observed surface depth, or if a Khronos RayBackgroundChangeDetector output
  can be exported and joined to vertex ids.
```

Expected next proof:

```text
removed_vertices > 0
Background P@10cm improves over Round06 injection-only
Background R@10cm drop is smaller than the precision gain
```
