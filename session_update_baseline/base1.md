# Base1: Object-Guided Khronos Baseline

Date: 2026-07-04

> NOTE (scope): We only read/modify the **final-map snapshot** stored inside the Khronos
> `.4dmap` container (current-time background mesh + object outputs). We do **not** touch 4D
> timestamps, temporal queries, or 4D reconciliation. "4D" is out of scope for Base1.

## Goal

Implement a minimal, mathematically clean baseline on top of Khronos.

The goal is not to build a full 4D map and not to solve pose optimization first. The immediate
goal, in order, is:

```text
STEP 1 (single-session optimization):
  run Khronos on one session -> take its final map
  apply object-guided reconciliation -> improved final map
  evaluate improved vs original Khronos with the OFFICIAL evaluator (real numbers)

STEP 2 (multi-session):
  session A saves final map + object memory
  session B loads A's memory as prior -> updates final map
  evaluate B(with memory) vs B(from scratch) with the OFFICIAL evaluator (real numbers)
```

The baseline should use Khronos object information to improve the final global/background map,
especially by reducing ghost geometry, duplicated geometry, and stale object-looking residues in
the main mesh, and by repairing missing geometry.

The system should be presented as a baseline runner, not as a manual post-processing script:

```text
run_session_update_baseline
  -> runs / loads Khronos-compatible map state
  -> applies object-guided reconciliation before finalization
  -> saves improved final map
```

Offline operation is allowed for debugging and ablation, but the main command should behave like
our baseline pipeline rather than "run Khronos first, then externally repair the map".

## Non-goals for the first version

Do not implement these in the first version:

```text
full 4D map reconstruction
online backend rewrite
pose graph / loop closure optimization
moved-object identity tracking
semantic open-vocabulary reasoning
large structural scene graph
neural or learned module
```

The first version should focus on:

```text
final map quality
object-guided cleanup
object-guided repair
safe preservation of unobserved geometry
```

## Two dynamic-map modes

Base1 must explicitly support two different dynamic-map situations. Do not collapse them into one
generic "second run" story.

### Mode A: within-session change

```text
The robot is still in the same run.
It observes an object / region.
The scene changes while the session is active (e.g. moved while occluded / out of view).
The robot later looks back or revisits that region.
The map should update the current final map using the new evidence.
```

This is closest to Khronos's native strength:

```text
active window
object tracks/fragments
ray/free-space evidence
background change detection
reconciliation
```

> IMPORTANT: Khronos ALREADY detects the within-session change itself (this is its office-scene
> behavior: object moved while out of view, detected on loop-back). So in Mode A our job is NOT
> to re-detect the change. Mode A IS a real optimization target: our single-session final map
> MUST beat raw Khronos. The lever is that Khronos's final map uses ONLY the background/global
> mesh, while its object private meshes are a separate, often cleaner geometry that never helps
> the main map. We use the object meshes to repair the background mesh in BOTH directions:

```text
(1) INJECT good object mesh into the main map:
    Khronos's final map is background mesh only; object private meshes are not merged in.
    Where an object's private mesh is reliable and the background mesh is missing / incomplete
    there, fuse the object mesh into the main map -> fills holes -> raises recall.

(2) DELETE background residue with object-mesh help (two distinct cases, each must be accurate):
    (2a) ghost residue: the object is absent / moved away, but Khronos left object-looking
         geometry behind in the background mesh -> use the object's known bbox/mesh + absence
         evidence to remove that ghost -> raises precision.
    (2b) object->background leakage: the object is still present, but background vertices got
         smeared onto / around it (leakage, double surface) -> shave off the leaked background
         geometry near the object surface -> raises precision.
    Both deletions must be ACCURATE: never delete unless the region was re-observed and the
    absence/leakage evidence is reliable (see unobserved guard).

(3) DON'T punch holes in the floor:
    cleanup + change detection tend to erode floor / large planar background, leaving holes.
    Protect floor / wall / ceiling / large planar background, and where a hole remains, fill it
    back using object mesh or a plane/background prior -> preserves recall.
```

These three mechanisms map directly to the three update actions below (repair/injection = (1),
cleanup = (2), protection = (3)).

### Mode B: cross-session revisit change

```text
Session A finishes and saves final map / object memory.
The robot leaves.
The scene changes while the robot is absent.
Session B starts later in the same scene.
Session B should start with previous memory and update the final map using new observations.
```

> IMPORTANT: This is where Khronos CANNOT help natively — it only reads one session and has no
> way to load a prior map and continue. Mode B is the real contribution and where the method
> firepower goes. Panoptic and POCD-style ideas matter most here:

```text
Panoptic: active-vs-inactive submap conflict / persistent / absent states.
POCD: object stationarity / object-level belief over persistent vs changed.
Khronos: object private meshes, bbox, temporal attributes, ray/free-space cleanup.
```

Base1 should combine them as:

```text
Khronos object/private mesh evidence
+ Panoptic-style active/inactive conflict tests
+ POCD-style object memory belief
-> object-guided cleanup / protection / repair of the global map
```

The first implementation may run this offline from saved maps, but the data model must preserve the
distinction:

```text
within_session_evidence
cross_session_prior_evidence
current_session_observation_evidence
```

## Code organization

Use a clean source structure:

```text
vendor/
  Original Khronos / Panoptic source snapshots for reference only.

ports/
  Minimal copied or adapted Khronos/Panoptic logic.
  Keep this small and traceable.

src/
  Our baseline runner.
  Map IO.
  Khronos adapters.
  Evidence extraction.
  Object-guided reconciler.
  Evaluation and diagnostics.
```

Do not randomly copy large Khronos modules. Only copy or adapt functions that are directly needed
and document their source file / line.

## Core module

Implement:

```text
ObjectGuidedMapReconciler
```

Inputs:

```text
Khronos final map state
  global/background DSG mesh
  object nodes / object attributes
  object private meshes if available
  object bounding boxes
  semantic labels
  timestamps / observation metadata if available

Optional prior map
  previous improved map
  previous object memory
```

Output:

```text
improved final map (loadable by the same Khronos evaluator)
mesh_update_summary.csv
object_update_summary.csv
evidence_summary.json
```

The output must be evaluable by the same Khronos evaluator.

## Mathematical model

Avoid pure heuristic rules. Use a unified evidence model.

For each global mesh vertex or small mesh patch `i`, define a latent decision:

```text
y_i in {keep, remove, repair, unobserved}
```

Compute evidence:

```text
z_i =
  d_object: distance to nearby object private mesh
  b_object: whether inside / near object bounding box
  s_sem: semantic compatibility if available
  e_free: free-space / ray conflict evidence if available
  e_dup: duplicate surface evidence
  e_vis: visibility / re-observation confidence
  e_bg: background / structural protection evidence
  e_time: recency / last-seen evidence
```

Use either Bayesian scoring or log-odds:

```text
P(y_i | z_i) proportional to P(z_i | y_i) P(y_i)
```

or:

```text
L_i(remove) =
  w_obj * object_residue_score
+ w_free * free_space_conflict
+ w_dup * duplicate_score
- w_vis * unobserved_score
- w_bg  * background_protection_score
```

Decision rules:

```text
remove:
  high stale/object-residue/duplicate evidence
  and visibility is reliable

keep:
  high background evidence
  or low change evidence

repair:
  object is reliable and present
  object private mesh exists
  corresponding global mesh is missing or incomplete

unobserved:
  visibility is insufficient
  do not delete
```

The most important safety rule:

```text
Do not delete geometry merely because it is not matched.
Only delete if the region was actually re-observed or has reliable conflict evidence.
```

Accuracy is the top priority for every keep/remove/repair decision. Both delete cases (ghost
residue vs. leakage) and both repair cases must be ACCURATE — a wrong deletion (erasing real
background or a present object) is worse than doing nothing. Before implementing any decision
criterion, the agent must FIND THE MOST ACCURATE PUBLISHED METHOD for it (read Khronos ray
absence Sec. V-C, Panoptic SDF conflict Sec. III-D, POCD change measure Sec. IV-A, and search
for object-aware TSDF fusion / mesh residue removal / plane-based hole filling), and cite it in
the code. Do not settle for a bare distance threshold when a more accurate published criterion
exists.

### Every probabilistic term MUST cite a paper equation

To keep the math honest and non-fabricated, any code that computes a probability / belief /
stationarity / change score must carry a comment pointing to the exact paper equation it
implements. Below each mapping we include the verbatim source text so it can be checked against
the original PDF.

```text
Gaussian-Beta object state + Bayesian update  -> POCD, Eq. (11)-(12) (and state model Eq. 5)
persistent / absent / unobserved state machine -> Panoptic Multi-TSDF, Sec. III-D (Map Management)
free-space / ray absence evidence (d_r, d_d)    -> Khronos, Eq. (18)-(19) + Sec. V-C (Reconciliation)
class-conditioned moveability prior             -> POCD stationarity class prior / 3D-VSG variability
```

**POCD (Qian et al., RSS 2022) — Gaussian-Beta object state + Bayesian update.**
Source excerpts to verify against:
- State model, Eq. (5): "p(l, v | z1 . . . zT ) := q(l, v | μT , σT , βT , αT ) :=
  N(l | μT , σ2_T) Beta(v | αT , βT)".
- Bayesian update, Eq. (11): "p(l, v | ∆T , sT , μ, σ, α, β) = η p(∆T | l, v) p(sT | v) q(l, v | μ, σ, α, β)".
- Posterior mixture, Eq. (12): "C1 N(l | m, γ2) Beta(v | α + ksT + 1, β + k(1−sT)) +
  C2 N(l | μ, σ2) Beta(v | α + ksT , β + k(1−sT) + 1)", where "The weights, C1 and C2, are the
  probability of the measurement, zT , being an inlier and outlier, respectively."
- Meaning (Sec. IV-B): "μT and σ2_T represent the mean and variance of l ... αT and βT are the
  number of observed inlier and outlier measurements ... object pruning decisions can be made
  based on the expectation of the stationarity, E[v]."

**Panoptic Multi-TSDF (Schmid et al., ICRA 2022) — persistent / absent / unobserved.**
Source excerpts to verify against (Sec. III-D, Map Management):
- "Inactive submaps are frozen except for their change state C(S) ∈ {persistent, unobserved, absent}."
- "If |sdf(p)| < ξsdf ... the point counts as agreeing with the surface. Otherwise, sdf(p) < −ξsdf
  indicates intersections with object maps and sdf(p) > ξsdf indicates conflicts with free space maps."
- "If they conflict with any of them, their state is set to absent. Otherwise, if they match with
  any of them, their state is set to persistent ... Far back in time submaps are unknown, and can
  become absent or persistent again when observed."

**Khronos (Schmid et al., RSS 2024) — free-space / ray absence evidence + reconciliation.**
Source excerpts to verify against (Sec. V-C, Deformable Change Detection):
- Eq. (18): "dr = ‖ (pq − pr) × (pr − pv) ‖ / ‖ pq − pr ‖".
- Eq. (19): "dd = (pq − pr) · (pv − pr) / ‖ pq − pr ‖".
- Absence logic: "Distances dd longer than that of the vertex indicate the point was occluded,
  distances similar (within dray = 30 cm) to the vertex indicate geometric consistency ... and
  short distances indicate evidence of absence."
- Reliability threshold: "a timestamp is only considered reliable evidence of absence if at least
  cray = 60% percent of rays within a temporal window of τray = 5 s mark the object as absent."

**Class-conditioned moveability prior — POCD stationarity class / 3D-VSG variability.**
Source excerpts to verify against:
- POCD (Sec. III-D): "a stationarity class, st,j , can be mapped from ct,j ... st,j = 0 denotes a
  dynamic object and ... st,j = 1 denotes a static object. For example, an object with ct,j = robot
  will have st,j = 0 whereas an object with ct,j = shelf will have st,j = 1."
- 3D-VSG (Looper et al., ICRA 2023): "the Variable Scene Graph (VSG), which augments existing 3D
  Scene Graph (SG) representations with the variability attribute, representing the likelihood of
  discrete long-term change events." (arXiv 2209.07896)

Rule: if you cannot name the source equation/section AND quote the line for a scoring term, that
term is invented -> delete it and re-derive from the paper. Do not ship "probabilistic-looking"
if-else chains.

## Update actions

### 1. Cleanup (object-mesh-guided, two accurate deletion cases)

Remove global/background mesh vertices in TWO distinct cases, each must be accurate:

```text
(1a) GHOST residue: object is absent / moved away, but Khronos left object-looking geometry in
     the background mesh. Use the object's known bbox/mesh + reliable absence evidence to delete
     the ghost.

(1b) LEAKAGE: object is still present, but background vertices are smeared onto / around it
     (double surface, object->background leakage). Shave off the leaked background geometry that
     lies on the object surface.

also: duplicated object-like geometry; old geometry contradicted by reliable current evidence.
```

Accuracy requirement: distinguish (1a) from (1b) before deleting (the target differs — a
vanished object's ghost vs. a present object's smear). Never delete a region that was not
re-observed, or where absence/leakage evidence is unreliable (see unobserved guard). Find and
use the most accurate deletion criterion from the literature (Panoptic SDF conflict, Khronos ray
absence, POCD change measure) rather than a bare distance threshold.

> Note: cleanup mainly raises PRECISION. Khronos background precision is already very high
> (office P=96.3, apartment P=96.8), so cleanup alone will NOT move background F1 much and may
> cost recall. Treat cleanup as the precision-side action, paired with repair below.

### 2. Protection (and floor-hole filling)

Protect vertices likely belonging to:

```text
floor
wall
ceiling
large background surfaces
unobserved regions
```

This is necessary to avoid floor holes and over-cleaning. Additionally, where cleanup / change
detection has already punched a hole in the floor or a large planar surface, fill it back using
the object mesh or a plane / background prior. Goal: the floor stays continuous, no holes.

### 3. Repair / injection (PRIMARY lever for beating Khronos)

Khronos's final map is background/global mesh ONLY; its object private meshes never enter the
main map. Fuse good object mesh into the main map where the background is missing/incomplete:

```text
if object private mesh is reliable
and global mesh lacks corresponding geometry
then inject or attach object geometry into the final current map
```

> Note: repair/injection is the PRIMARY lever for improving the score, not an optional extra.
> Khronos's weak spots are RECALL (office background R=73.7) and OBJECT recall, not precision.
> Injecting reliable object private mesh into missing global geometry raises recall, which is
> where the real F1 gain lives. Ablate it separately, but expect it (not cleanup) to be the
> action that actually beats Khronos.

### 4. Prior-map evidence

For the second session:

```text
load previous improved object memory
compare current objects with prior objects
use prior object state as a soft prior
```

Do not force identity tracking. For now:

```text
old location absent + new location present
```

is acceptable.

This prior evidence is the initial Base1 version of a POCD-style object memory:

```text
object_memory.json stores object state, geometry, confidence, and last evidence.
Session B reads it as a soft prior, not as ground truth.
```

## Two-session protocol

### Session A

```text
Run our baseline on Session A.
Save:
  A/original_final  (raw Khronos single-session final map)
  A/improved_final  (our object-guided reconciliation)
  A/object_memory.json
  A/mesh_update_summary.csv
```

Acceptance target (single-session):

```text
PRIMARY:   A/improved  Object Recall / Change F1  >  A/original
SECONDARY: A/improved  Background F1  >=  A/original  (must NOT drop; it is already saturated)
```

Do NOT use "background F1 up" as the only success signal. Background precision is saturated in
Khronos; report recall and change metrics as the primary evidence of improvement.

### Session B

```text
Run our baseline on Session B.
Load:
  A/improved_final
  A/object_memory.json

Use current Session B evidence to update the map.
Save:
  B/from_scratch    (Khronos on B alone, no prior memory = Khronos's ceiling, since it cannot
                     load a prior map)
  B/improved_final  (our baseline, starting from A's memory)
  B/object_memory.json
```

Acceptance target (multi-session — this is the real claim):

```text
B/improved (with A's memory)  >  B/from_scratch (no memory)
on background recall / completeness and change F1.
```

It is not required that:

```text
B/improved > A/improved
```

because Session B may be harder or observe a different state.

### Comparison groups (report all three)

```text
H0  Khronos original                 : raw single-session final map
H1  Khronos + naive load/replay      : load previous final map, continue, NO object-guided update
Ours Khronos + object memory + object-guided cleanup/protection/repair + session update
```

## Tests

### Test 0: No-op IO test

Load and save the final map without changes.

Expected:

```text
evaluation metrics identical or nearly identical
mesh vertex count unchanged
object count unchanged
```

This test must pass before implementing update logic.

### Test 1: Object audit

Dump for every object:

```text
object id
semantic label
bbox
pose
private mesh vertices / faces
nearby global mesh vertices
visibility / timestamp if available
```

Output:

```text
object_audit.csv
```

Purpose:

```text
verify that Khronos object information is actually usable
```

### Test 2: Cleanup only

Run only removal decisions.

Compare:

```text
Khronos original
cleanup-only
```

Metrics:

```text
background precision
background recall
background F1
mesh vertex count
deleted vertex count
deleted region summary
```

Pass condition:

```text
precision improves and background F1 does not drop, AND recall does not collapse.
(Do not expect big F1 gains from cleanup alone; see Update action 1 note.)
```

### Test 3: Injection only

Run only object private mesh repair / injection.

Compare:

```text
Khronos original
injection-only
```

Purpose:

```text
check whether object private mesh can repair missing global geometry and RAISE RECALL.
This is the action expected to actually improve F1 over Khronos.
```

### Test 4: Cleanup + protection

Enable cleanup but protect floor/wall/background-like regions.

Purpose:

```text
avoid holes and over-deletion
```

### Test 5: Full baseline

Run:

```text
cleanup + protection + repair + prior evidence
```

Compare against Khronos original.

## Ablation table

At minimum, produce:

```text
Khronos original
No-op save
Cleanup only
Injection only
Cleanup + protection
Cleanup + protection + injection
Cleanup + protection + injection + prior
```

Metrics:

```text
Background Precision / Recall / F1
Object Precision / Recall / F1
Change Precision / Recall / F1
Mesh-to-GT distance if available
Vertex count
Deleted vertices
Injected vertices
```

## What counts as ONE round (hard gates)

A "round" is NOT "edited a file and it compiled". A round is complete only when ALL of these
gates are green, with REAL official-evaluator numbers:

```text
Gate 1  single-session runs:   Khronos on native dataset (office) produces a final map.
Gate 2  single-session numbers: our_improved vs khronos_original, OFFICIAL evaluator,
                                 Background + Object + Change P/R/F1 all in a table (numbers).
Gate 3  multi-session runs:     A saves final map + object_memory -> B loads it -> B produces
                                 a final map.
Gate 4  multi-session numbers:  B_ours(with memory) vs B_from_scratch, OFFICIAL evaluator,
                                 metrics in a table (numbers).
```

### Hard anti-laziness rules (read every round)

```text
- No round is "complete" without REAL numbers from the official Khronos evaluator.
  "eval dir prepared" != done. "byte size matches" != done. Only P/R/F1 in a CSV counts.
- Fixing evaluator errors (hardcoded GT path, missing experiment_log.txt, dir format) is PART
  of the task. Solve them until numbers come out. Do not defer them to "next step".
- Do not fabricate numbers, do not cherry-pick favorable samples, do not report F1 without
  running the evaluator.
- Do not randomly tune thresholds. If a metric does not improve, write a one-line failure
  hypothesis BEFORE the next code change.
- Every scoring term must cite a paper equation (see Mathematical model). No citation = delete.
- End of every round: post (a) a metrics comparison table (ours vs baseline, all metrics),
  (b) one-line verdict: "up / flat / down, because X".
```

## Build phases within a round

These are implementation phases, NOT rounds. A round only ends when the 5 gates above are green.

### Phase 1

```text
No-op IO
object audit
baseline runner command
```

If no-op fails, stop and fix map IO.

### Phase 2

```text
cleanup-only
FIRST official eval comparison (must produce real numbers)
diagnostic CSV explaining changes
```

If cleanup does not improve or creates holes, inspect deleted regions and improve evidence /
protection.

### Phase 3

```text
add protection and repair/injection
run ablations
run Session A and Session B with official eval
```

If after two meaningful tuning rounds the method still fails to improve metrics:

```text
stop coding
re-read Khronos source and paper sections on map finalization, object extraction, change detection,
and mesh reconciliation
re-read POCD object update and stationarity sections
re-read Panoptic submap update / absent-state mechanism
search for related methods on object-guided map cleanup, object-aware TSDF fusion, semantic map
maintenance, and mesh reconciliation
write a short failure analysis before coding again
```

Do not keep randomly changing thresholds without a written failure hypothesis.

## Reporting

Every run must save:

```text
config.yaml
command.txt
git diff or changed file list
original metrics (official evaluator)
improved metrics (official evaluator)
mesh_update_summary.csv
object_update_summary.csv
evidence_summary.json
```

Every round must additionally post:

```text
a metrics comparison table (ours vs baseline, all metrics)
one-line verdict: up / flat / down, because X
```

Every update decision should be explainable:

```text
which object caused it
which evidence supported it
which vertices were removed or injected
whether the region was visible or unobserved
```

## Success definition

One round is successful only if:

```text
1. Single-session: our improved final map runs end-to-end and is saved.
2. No-op IO is safe.
3. Single-session OFFICIAL metrics are produced: our_improved vs khronos_original
   (Background + Object + Change P/R/F1), with improvement shown on Object Recall / Change F1
   and Background F1 not dropping.
4. Multi-session works: A saves memory -> B loads it -> B produces a final map.
5. Multi-session OFFICIAL metrics are produced: B(with memory) vs B(from scratch), with the
   improvement explained by object-guided cleanup / protection / repair summaries.

If any of these lack REAL official-evaluator numbers, the round is NOT complete.
```