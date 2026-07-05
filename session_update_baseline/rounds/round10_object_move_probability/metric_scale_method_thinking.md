# 10cm / 5cm Method Thinking Before More Experiments

Date: 2026-07-05

## What The Target Really Is

The target is not to win a tiny threshold sweep.

The target is to improve the final-current background mesh in a way that holds
at both:

```text
10cm: task-level map correctness.
5cm: geometric surface correctness.
```

An update that improves 10cm but damages 5cm is probably adding approximate
surface coverage without real alignment. That is not a strong method.

## Current Metric Diagnosis

Official evaluator results:

| method | F1@5cm | F1@10cm | F1@20cm |
|---|---:|---:|---:|
| raw Khronos | 0.731401 | 0.939408 | 0.983913 |
| old full object append | 0.718690 | 0.954065 | 0.988735 |
| Round09 hard object-change gate | 0.721297 | 0.956022 | 0.989184 |
| Round10 probability best observed | 0.721887 | 0.956135 | 0.989198 |
| Round10 expected utility | 0.721580 | 0.956003 | 0.989187 |

Interpretation:

```text
Object private mesh injection helps 10cm and 20cm.
The same injection hurts 5cm compared with raw Khronos.
```

So the injected object geometry is often close enough to count at 10cm, but not
accurate enough to count at 5cm. That points to surface alignment / duplicate
surface / object-frame geometry quality, not merely object presence.

## What Methods Should Be Searched

Methods should be searched according to the error type they can fix.

### 1. Surface-Likelihood Injection

Question:

```text
Will this object private surface become a 5cm/10cm inlier if injected?
```

Method shape:

```text
sample object private mesh surface
transform samples to world
query nearest current background and/or GT-style support proxy
keep only surface patches whose local support predicts true missing geometry
```

Why this is better than object probability alone:

```text
P_moved decides whether the object may exist.
Surface likelihood decides whether the actual triangles are geometrically useful.
```

Expected metric behavior:

```text
10cm should stay near Round09 or improve.
5cm must not drop relative to raw Khronos.
```

### 2. Local Surface Registration Before Injection

Question:

```text
Is the object mesh good but slightly misaligned?
```

Method shape:

```text
for each object patch, estimate a small rigid correction against nearby background
accept correction only if it improves nearest-neighbor residual distribution
inject corrected surface, not raw private mesh
```

Why it targets 5cm:

```text
5cm is sensitive to small pose/surface offsets that 10cm hides.
```

### 3. Free-Space / Ray-Based Ghost Deletion

Question:

```text
Are 10cm/5cm precision errors caused by old surfaces that current rays saw through?
```

Method shape:

```text
use absence evidence along current observation rays
delete only vertices with repeated free-space evidence
protect vertices with current persistent/object support
```

Why previous deletion failed:

```text
distance-to-object-shell deletion deletes evaluator inliers.
Ray/free-space evidence is the missing guard.
```

### 4. TSDF / Signed-Distance Conflict Update

Question:

```text
Can we decide add/delete by local signed-distance agreement instead of object boxes?
```

Method shape:

```text
build local TSDF/confidence around current observations
add surfaces where current TSDF has strong zero-crossing and map lacks support
delete surfaces where map conflicts with free space or opposite signed distance
```

This is the most principled route if the data needed for rays/depth is available.

## What Not To Do

Do not continue:

```text
random object skip thresholds
epsilon sweeps around duplicate suppression
distance-shell deletion around object meshes
F1-only reporting without 5cm/10cm precision and recall
```

Those are too weak for the observed failure mode.

## Acceptance Criteria For The Next Real Method

A method is worth keeping only if it satisfies at least one:

```text
F1@10cm improves by >= 0.003 without lowering F1@5cm.
F1@5cm improves by >= 0.005 while keeping F1@10cm within 0.001 of Round09.
P@5cm improves materially without a large R@10cm collapse.
```

Tiny changes like +0.0001 are not success.

## Next Step

Before implementing another map updater, run a 5cm/10cm error audit:

```text
FP@5 / FP@10: ours surface points not supported by GT
FN@5 / FN@10: GT surface points not covered by ours
bucket errors by object bbox, semantic label, and spatial region
identify whether biggest losses are object surfaces, walls/floor, or global offset
```

Only after that audit should a new method be chosen.
