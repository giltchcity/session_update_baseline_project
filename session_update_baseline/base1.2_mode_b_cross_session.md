# Agent Context: Base1 Tracks Are Split On Purpose

Base1 is not one giant task. It is split into three parallel tracks so a new agent can work on one part without silently claiming the whole project is solved.

```text
Base1.1 / Mode A:
  Single-session geometric optimization on existing Khronos office data.
  This is available now and does not need a new dataset.

Base1.2 / Mode B:
  Real cross-session memory update on a true A/B change dataset.
  Do not use office slices as proof.

Base1.3 / NSS Drive:
  Dataset construction work for a future larger benchmark.
  Its rounds are data-product gates, not method-score gates.
```

If you are assigned this file, you are working only on **Base1.2 / Mode B**. This track must use real cross-session change data. Do not use Khronos office time slices, and do not relabel Base1.1 injection gains as memory gains.

# Base1.2 — Mode B: Cross-Session Memory (real change data)

Date: 2026-07-05
Scope: final-current map snapshot only. Data: REAL cross-session change datasets, not toy,
not time-slices of one office bag.

## Why this track exists

Khronos office is ONE bag with NO true cross-session change, so it cannot produce the required
two-visit A->B experiment. To test Mode B we need data where the scene actually changed between
two independent visits. Two such datasets already exist, with labeled changes:

```text
Panoptic flat (Schmid et al., Panoptic Multi-TSDF, Sec. IV):
  "two trajectories in a flat, where 8 objects are moved, 5 are added and 4 are removed
   between the runs." + real RIO dataset (scans 466/27, 2 and 4 runs).

POCD TorWIC (Qian et al., POCD):
  warehouse, 18 trajectories, changes = removal/addition/shift/rotation of boxes/fences,
  stitchable into routes with changed object locations. POCD itself ran exactly the
  "prior-map continued run vs from-scratch reference run" comparison (Sec. V-C).
```

Priority: Panoptic **flat** first (clean sim, full GT, labeled add/remove/move, exactly two
A/B trajectories) -> then POCD **TorWIC** (real, harder). Drop the manual toy-change idea.

## Goal

Prove that our A->B update produces a better / more up-to-date final map than the official
A->B update on the same data. The effect must be driven by memory-based decisions, not by the
same object-mesh injection as Base1.1.

```text
Session A: run, save final map + object_memory (persistent objects, geometry, confidence).
[scene changes between A and B — this is real in flat/TorWIC]
Session B: load A's memory as PRIOR, observe current scene, and UPDATE:
   - prior object still present  -> persistent, keep/refine
   - prior object now absent     -> mark absent, remove its stale geometry from the map
   - new object present          -> add
Main comparison:
   official A->B  = official mapper loads A map, runs B, saves final map
   ours A->B      = our updater loads A memory/map, runs B update, saves final map
```

## The core test (this is what earlier rounds FAILED to show)

```text
MEMORY MUST CHANGE THE OUTPUT.
Required evidence every round: ours A->B final map must differ from official A->B final map,
and the difference must be LOCATED ON the objects that actually changed between A and B.
If ours A->B is identical to official A->B, our update did nothing -> round is a FAILURE
regardless of other work.
```

## Metric

```text
Use the change dataset's own evaluation where possible (flat / TorWIC ship GT + protocols):
  - map reconstruction Precision / Recall / FPR vs the post-change reference (TorWIC style),
  - or coverage/error vs GT (flat style).
Report Precision and Recall SEPARATELY (not just F1 / not just one aggregate).
The headline number is: does ours A->B beat official A->B on the CHANGED regions?
Anchor: POCD Table II reports Precision / Recall(TPR) / FPR — match that shape.
```

Standard-only rule:

```text
Do not use local/proxy pointcloud P/R, namespace vertex deltas, current-only mesh exports, or
handmade object-count audits as experiment results. They are not substitutes for a dataset or
official evaluator. If the public dataset release does not provide the required changed-region
Precision/Recall protocol, mark that gate RED and do not invent a metric to fill it.
```

2026-07-05 clarification:

```text
The official Panoptic evaluator remains the headline full-map metric. Additional F-score /
voxel / changed-label tables are allowed only when they are generated from standard `.panmap`
outputs plus dataset GT/raw depth/segmentation, with scripts and CSVs saved in the round. These
tables must be labeled as standard extensions, not as official Panoptic paper numbers.
```

## Optimization advice

```text
- Data adapter FIRST, method SECOND. flat/TorWIC are NOT tesse_cd format. Round 1 of this track
  is a data-feasibility spike: get ONE session of flat to run through the pipeline and produce
  a final map + evaluable output. Do not write reconciler logic until data flows end to end.
- Memory must be USED, not just loaded+counted. Earlier office rounds loaded prior memory and
  matched objects but never let it change a geometry decision -> output was identical. The whole
  point of this track is to break that.
- The absent-object case is the cleanest win: a prior object that is gone in B should have its
  stale geometry removed more correctly than the official A->B update. That is a visible,
  localizable difference between ours A->B and official A->B. Start there.
- Object memory belief should be probabilistic and cite POCD (stationarity/persistence,
  Eq. 5, 11-12) and Panoptic states {persistent, unobserved, absent} (Sec. III-D). No
  paper-less "belief" scores.
- Keep the unobserved guard: do not blanket-convert `Unobserved` to `Persistent` or `Absent`.
  Resolve it only with current-session evidence such as same-object current support or
  free-space/depth visibility conflict.
```

## Current best standard round

```text
session_update_baseline/rounds/base12_pocd_panmap_round01

Main variant:
  ours A->B visibility/conflict policy

Key result vs official A->B:
  official Panoptic RMSE: 0.0131068 vs 0.0131076
  UnknownPoints:         621,655 vs 653,662
  F1@5/10/20cm:          0.896059 / 0.929896 / 0.966558
  moved-object F1@20cm:  0.594258 vs 0.481788
  removed stale points:  0

Remaining yellow gate:
  POCD-style voxel recall improves, but 20cm voxel precision/FPR are slightly worse than
  official A->B. Validate the visibility/conflict threshold on TorWIC/POCD before
  calling it general.
```

## Method-search checkpoint (mandatory before changing memory logic)

Mode B is easy to fake by merely loading prior memory. If memory fails to change B output, do
not keep renaming states or changing thresholds. A search/re-read step is required in either
case below:

```text
TRIGGER A: ours A->B == official A->B after our update was applied.
TRIGGER B: ours A->B differs from official A->B, but the diff is not located on the labeled
           changed objects.
TRIGGER C: absent prior objects cannot be removed without also deleting unobserved or
           persistent objects.
```

When triggered, stop coding and create:

```text
session_update_baseline/rounds/<round_name>/method_search_notes.md
```

The notes must contain:

```text
1. Failure symptom:
   changed object ids, prior states, diff-map location, Precision/Recall, and whether memory
   actually changed a geometry decision.

2. Source re-read:
   Panoptic Multi-TSDF Sec. III-D active/inactive submap conflict and states
     {persistent, unobserved, absent};
   POCD stationarity / persistence belief update, especially object change evidence;
   Khronos V-C evidence-of-absence vs absence-of-evidence logic.

3. Method search:
   search terms used, at least 5 candidate mechanisms, and why each is kept/rejected.
   Suggested query families:
     object-level change detection volumetric mapping
     probabilistic object map maintenance
     long-term dynamic scene consistency
     active inactive submap conflict
     stationarity belief object mapping
     absent object removal free-space evidence

4. Chosen next mechanism:
   one memory-driven mechanism only, with paper/code citation and expected changed-region
   metric movement. It must specify how it handles:
     persistent;
     absent;
     new;
     unobserved.
```

No new memory-update strategy is allowed after a trigger unless this file exists.

## What counts as ONE round (hard gates — all GREEN with real numbers)

```text
G1  A real change dataset session (flat run 1) flows through the pipeline -> final map produced.
G2  Session A saved (final map + object_memory); Session B loads it.
G3  Memory PARTICIPATES: prior object states assigned AND at least one geometry decision in B
      is driven by prior memory (e.g. an absent prior object's geometry removed).
G4  ours A->B final map != official A->B final map, and the diff is located on changed objects
      (dump the diff / the changed-object list as evidence).
G5  OFFICIAL / dataset evaluator run on ours A->B vs official A->B, Precision & Recall on the
      changed regions in a table.
```

## End-of-round self-check (answer ALL in writing; any "no" = round NOT done)

```text
Q1  Did a REAL change dataset (flat/TorWIC) actually run, or was this a time-slice of office?
Q2  Did prior memory change at least one geometry decision in Session B?          (which one?)
Q3  Is ours A->B != official A->B, and is the difference ON the changed objects?   (paste diff)
Q4  Did the evaluator emit real Precision/Recall on changed regions?               (paste table)
Q5  Does the memory belief cite POCD / Panoptic equations?
Q6  One-line verdict: what did our update change, and did ours A->B beat official A->B on changes?
```

## Forbidden (laziness paths)

```text
- Using an office time-slice and calling it cross-session A/B. (It is not — office has no
  cross-session change.)
- Loading prior memory, matching objects, and reporting GREEN while ours A->B == official A->B.
  Loading != using. Matching != deciding.
- Reusing Base1.1 object-mesh injection as the source of the gain and labeling it "memory".
- Claiming a round complete without the changed-object diff AND real evaluator numbers.
- Using local/proxy pointcloud P/R, namespace deltas, or handmade current-only exports as if
  they were official/dataset evaluator results.
- Inventing belief scores with no paper equation.
- Changing memory logic after a failed round without writing method_search_notes.md.
```

## Definition of done (whole track)

```text
On a real change dataset (flat, then TorWIC), our A->B update produces a final map that
(a) differs from official A->B specifically on the changed objects, and (b) beats official A->B
on changed-region Precision/Recall under the dataset's evaluator — with the gain attributable
to memory-driven decisions (absent-object removal etc.), not to injection alone.
```
