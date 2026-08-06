# Base1 Unified Three-Mode Scene Update Model

Date: 2026-07-30

## Scope

Base1 treats the following as three observation regimes of one scene-memory
problem:

1. **Observed motion**: an object moves while it remains visible.
2. **Hidden within-session change**: an object changes during an observation gap
   in one continuous run.
3. **Cross-session change**: a saved map is loaded in a later run; the later run
   can itself contain observed motion and hidden changes.

The session boundary changes persistence and time scale. It does not introduce a
different definition of `persistent`, `absent`, `unobserved`, or `new`.

## State And Output

For every scene element, maintain the same state. A scene element is either an
object instance or a structural surface patch:

```text
element_kind in {object, structural_patch}
state in {dynamic_observed, persistent, absent, unobserved, new}
stationarity belief Beta(alpha, beta)
last observed interval
geometry reference
semantic label when available
session provenance
```

Structural patches are geometric units such as connected planar or locally
smooth wall, floor, ceiling, door, or large background regions. A semantic label
is useful but not required to detect their geometric change.

The final current static map contains only geometry whose current state is
`persistent` or `new`. `unobserved` prior geometry is retained conservatively.
Observed dynamic objects are exported as trajectories, temporal bounding boxes,
and optional per-frame point clouds; they are not injected into the static mesh.
Vertices accidentally fused from a dynamic entity are tracked separately as
dynamic-residue candidates. They may be removed only when supported by both the
dynamic track and reliable later free-space or surface-conflict evidence.

## Evidence Paths

### Mode 1: observed motion

Use Khronos active-window motion detection and fragment tracking:

```text
continuous dynamic observations
-> trajectory timestamps + centroids
-> temporal bbox / point-cloud sequence
-> dynamic_observed
```

Do not infer a static private mesh for a currently moving person.

### Mode 2: hidden within-session change

At re-observation:

```text
old surface support + semantic/object association
-> persistent evidence

reliable free-space rays through the old location
-> absent evidence

no reliable re-observation
-> unobserved, keep

unmatched stable current observation
-> new
```

If identity cannot be established after a large displacement, report
`absent@A + new@B`; do not claim an observed trajectory from A to B.

The same logic applies to non-object structural changes:

```text
old wall patch + later free-space through its support
-> absent structural patch

new stable surface with repeated support
-> new structural patch

old patch not revisited
-> unobserved, keep
```

Wall semantics are not a prerequisite for these decisions. When available,
`wall`, `floor`, and `ceiling` labels provide structural protection and patch
grouping priors. They must not override contradictory geometric evidence.

### Mode 3: cross-session change

Load the prior global mesh, object memory, state belief, and provenance. Apply
the same Mode 1 and Mode 2 updates using the current session as evidence:

```text
prior current map
+ current-session dynamic tracks
+ current-session presence/absence observations
-> updated current map and updated memory
```

The current Base1 implementation restores the prior map at reconciliation time.
It does not yet hot-start Khronos PGMO, ray hash, or active-window threads.

## Evidence Fusion

The hard state machine is the first implementation. The intended probabilistic
upgrade follows POCD:

```text
v_i ~ Beta(alpha_i, beta_i)

alpha_i <- alpha_i + weighted persistent evidence
beta_i  <- beta_i  + weighted absent/change evidence

E[v_i] = alpha_i / (alpha_i + beta_i)
```

Free-space evidence is valid only when the old location was actually
re-observed. Semantic class provides a prior, not a deletion command. Structural
background receives a conservative prior; movable objects may adapt faster.

Dynamic residue uses a stricter conjunction:

```text
dynamic_residue =
  near recorded dynamic support
  AND fused during the dynamic observation interval
  AND later contradicted by free-space or a stable replacement surface
```

A trajectory-swept bounding box alone is never deletion evidence because it can
intersect valid floors, walls, and furniture.

## Geometry Reconciliation

The prior global mesh is the working map. Current-session background geometry is
merged by:

1. deleting prior vertices only with reliable absence evidence;
2. retaining unobserved prior vertices;
3. welding current vertices to nearby retained prior vertices;
4. adding genuinely new current vertices;
5. retaining mapped current triangle topology to repair holes across overlap
   boundaries;
6. applying object-guided cleanup and optional private-mesh repair.

Structural changes use the same merge but operate on connected surface patches
instead of object boxes. Large removals require spatially coherent absence
evidence; large additions require repeated stable surface support.

## Literature Basis

- [Khronos (RSS 2024)](https://www.roboticsproceedings.org/rss20/p081.pdf)
  factorizes short-term continuous motion into local active-window
  estimation and long-term abrupt change into global reconciliation. Dynamic
  surfaces are point-cloud sequences; static objects use TSDF meshes.
- [Panoptic Multi-TSDFs (ICRA 2022)](https://arxiv.org/abs/2109.10165)
  uses active/inactive entity submaps and the states
  `persistent`, `unobserved`, and `absent`; inactive state changes require
  overlap with active observations.
- [POCD (RSS 2022)](https://arxiv.org/abs/2205.01202) jointly models
  object geometric change and stationarity with a
  Gaussian-Beta representation and Bayesian updates, then removes objects whose
  stationarity falls below a threshold.
- [POV-SLAM (RSS 2023)](https://arxiv.org/abs/2307.00488) jointly
  estimates object consistency and robot pose. Base1 defers
  this localization extension until the map-only three-mode model is validated.
- [Detection and Tracking of General Movable Objects in Large 3D Maps
  (T-RO 2019)](https://arxiv.org/abs/1712.08409) explicitly models objects that
  may move while the robot observes only another part of a large environment,
  using linked local-motion and global-jump processes. This is relevant to
  identity-level Mode 2 tracking, but Base1 must keep `absent@A + new@B` until
  it has enough association evidence.
- [3D VSG (ICRA 2023)](https://arxiv.org/abs/2209.07896) predicts semantic
  scene variability for object position, semantic state, and scene composition.
  Such variability is suitable as a prior over stationarity, not as direct
  geometric deletion evidence.
- [PlaneSDF-based Change Detection (RA-L 2022)](https://arxiv.org/abs/2207.08323)
  represents long-term dense maps with planes and associated SDF components.
  It motivates geometric structural patches for walls and other large surfaces
  instead of forcing every change into an object instance.

## Required Tests

```text
Mode 1:
  dynamic track count, trajectory length, bbox sequence, no static-mesh injection

Mode 2:
  persistent / absent / unobserved / new decisions within one session

Mode 3:
  B starts from A global mesh, B evidence changes A, and B can also emit Mode 1
  dynamic tracks

Structural:
  wall/floor patch persistent / absent / unobserved / new decisions, with and
  without semantic labels

Dynamic residue:
  suspected candidates, later-free-space-confirmed removals, and protected
  structural vertices

Geometry:
  A->A idempotence, overlap face retention, floor-hole coverage, official map
  metrics where valid GT timing exists
```
