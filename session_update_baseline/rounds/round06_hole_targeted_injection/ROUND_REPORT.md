# Base1.1 Round 06: Injection Mechanism Cleanup

Date: 2026-07-05

Scope: Mode A single-session office final-current map. This round only changes
the injection mechanism; it does not implement deletion.

## Current Mechanism

The old round05 mechanism was:

```text
M_ours = M_background union all object_private_meshes
```

That is simple and scores well, but it is mathematically blunt. It improves
Background Precision because the injected object vertices are usually closer to
GT than the average background vertex:

```text
P_new = (TP_bg + TP_obj) / (N_bg + N_obj)
```

So `P_new > P_bg` whenever the object private mesh inlier rate is better than
the current background mesh inlier rate. It also improves Recall when object
vertices cover GT vertices that the background mesh missed.

## New Minimal Rule

Round06 replaces full append with duplicate-suppressed union:

```text
M_ours = M_background union { o in M_object : distance(o, M_background) > epsilon }
```

Default:

```text
epsilon = 0.02 m
```

`epsilon <= 0` restores the old full append.

This is still one local operation, not a new module. It constructs one nearest
neighbor index over the raw background mesh, then appends only object vertices
that are at least `epsilon` away from that background. Faces are appended only
when all three face vertices survive the filter.

Code:

```text
session_update_baseline/src/base1/object_guided_map_reconciler.cpp
appendObjectMeshToGlobal(...)
```

CLI:

```text
--injection_min_separation_m 0.02
```

## 10 cm Sweep

Official Khronos evaluator, same round05 B full map, final-current snapshot.

| mechanism | injected vertices | final vertices | P@10cm | R@10cm | F1@10cm | P@20cm | R@20cm | F1@20cm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw Khronos | 0 | 304326 | 0.946311 | 0.932604 | 0.939408 | 0.987914 | 0.979944 | 0.983913 |
| old full append, eps <= 0 | 1916187 | 2220513 | 0.972461 | 0.936352 | 0.954065 | 0.996476 | 0.981114 | 0.988735 |
| eps = 0.02 | 1758633 | 2062959 | 0.971556 | 0.936339 | 0.953622 | 0.996435 | 0.981113 | 0.988715 |
| eps = 0.04 | 1162554 | 1466880 | 0.966468 | 0.936287 | 0.951138 | 0.995896 | 0.981112 | 0.988449 |
| eps = 0.06 | 399262 | 703588 | 0.952593 | 0.936068 | 0.944258 | 0.994092 | 0.981107 | 0.987557 |
| eps = 0.08 | 134150 | 438476 | 0.939673 | 0.935577 | 0.937621 | 0.991222 | 0.981101 | 0.986136 |

## Verdict

`eps = 0.02` is the current default because it keeps almost all of the 10 cm
gain while removing the most obvious near-duplicate appends:

```text
F1@10cm: 0.939408 -> 0.953622
P@10cm:  0.946311 -> 0.971556
R@10cm:  0.932604 -> 0.936339
```

It is slightly below the old full append (`0.954065`), but it injects 157554
fewer vertices and is mathematically cleaner.

`eps = 0.08` is closer to pure hole filling, but it fails the 10 cm objective:
Recall improves, Precision drops, and F1 falls below raw Khronos. That means the
old 10 cm gain is not pure hole filling; it is mostly high-quality object-surface
union with a smaller recall contribution.

## Not Done

Deletion is still not implemented in this round:

```text
removed_vertices = 0
```

The next precision-side improvement should not be another distance-only cleanup.
It needs reliable absence/free-space or active/inactive conflict evidence before
removing background vertices.
