# Round07 Method Search Notes: Paper-Guided Deletion

Date: 2026-07-05

Scope: Base1.1 / Mode A, single-session office final-current map. Round06 made
injection cleaner with duplicate-suppressed object/background union. The next
step is deletion: remove ghost/leakage background vertices only when there is
real absence/conflict evidence.

## Failure Symptom

Round06 injection improved 10 cm metrics, but did not implement deletion:

```text
raw Khronos F1@10cm:        0.939408
round06 eps=0.02 F1@10cm:   0.953622
round06 removed_vertices:   0
```

The earlier cleanup-only attempt removed vertices by object distance and hurt
recall. Therefore the next deletion rule must not be a bare distance threshold.

## Papers Actually Read

### 1. Khronos: A Unified Approach for Spatio-Temporal Metric-Semantic SLAM in Dynamic Environments

Source:

```text
https://arxiv.org/html/2402.13817v2
```

Original sentence that matters:

> "To resolve this evidence-of-absence vs. absence-of-evidence problem, we perform an additional geometric verification step."

Implementation hint:

Khronos says deletion should be evidence gated. For Base1.1, do not delete a
background vertex just because it is near an object mesh. A deletion candidate
needs a local re-observation / free-space / ray-style absence proxy.

Minimal mechanism to try:

```text
delete(v_bg) only if:
  v_bg is near an object private mesh or inside its bbox,
  AND the object/background pair has a strong conflict score,
  AND the region is not unobserved.
```

### 2. Panoptic Multi-TSDFs: a Flexible Representation for Online Multi-resolution Volumetric Mapping and Long-term Dynamic Scene Consistency

Source:

```text
https://ar5iv.labs.arxiv.org/html/2109.10165
```

Original sentence that matters:

> "If they conflict with any of them, their state is set to absent."

Implementation hint:

Panoptic separates matching from conflicting. For Base1.1, a background vertex
near an object mesh is not automatically wrong. It should be removed only if the
object/background geometry is in conflict, not merely overlapping.

Minimal mechanism to try:

```text
For each object bbox:
  sample background vertices in/near bbox;
  compare distance-to-object-surface distribution;
  mark a small shell of duplicate/leakage vertices as conflict candidates;
  require conflict_count/object_support_count above threshold before deleting.
```

### 3. POCD: Probabilistic Object-Level Change Detection and Volumetric Mapping in Semi-Static Scenes

Source:

```text
https://arxiv.org/pdf/2205.01202
```

Original sentence that matters:

> "Objects with a high stationarity score are used to generate the new map"

Implementation hint:

POCD's useful lesson here is object-level gating. Do not make an isolated
vertex decision first. First decide whether the object region is reliable /
stationary / conflicting enough, then apply a narrow vertex rule.

Minimal mechanism to try:

```text
object_confidence(o) =
  private_mesh_vertices large enough
  bbox not tiny / degenerate
  object has enough local background support or enough conflict support

Only objects passing object_confidence can delete background vertices.
```

## Chosen Next Mechanism

Implement a small, paper-guided deletion rule:

```text
duplicate/leakage deletion =
  object-level gate
  + local object/background conflict shell
  + unobserved guard by requiring enough local background/object support
```

Expected metric movement:

```text
Background Precision @10cm should increase.
Background Recall @10cm must not drop more than the precision gain.
removed_vertices must be > 0.
```

## What Not To Do

```text
- Do not reintroduce the old cleanup-only distance threshold as the final rule.
- Do not delete whole object bboxes.
- Do not claim ghost deletion unless the object is absent/re-observed.
- Do not call leakage deletion complete unless removed_vertices > 0 and P@10cm moves up.
```

## First Code Loop

Keep code small:

```text
1. Add one leakage deletion pass inside ObjectGuidedMapReconciler.
2. Reuse the existing object private mesh nearest-neighbor search.
3. Candidate: background vertex inside object bbox and within a very thin object-surface shell.
4. Object gate: require enough object mesh vertices and enough local candidates.
5. Run official 10 cm/20 cm evaluator on round05 B full map.
```

If precision does not move, write a one-line failure hypothesis before changing
thresholds again.
