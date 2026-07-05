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

If you are assigned this file, you are working only on **Base1.1 / Mode A**. Do not claim cross-session memory is solved from this track. This track proves whether object private meshes can improve the final current map of a single Khronos run.

# Base1.1 — Mode A: Single-Session Geometric Optimization

Date: 2026-07-05
Scope: final-current map snapshot only. Data: existing Khronos office (tesse_cd_office),
one bag, no A/B needed. This track does NOT depend on any change dataset — run it now.

## Goal

Make our single-session final map beat raw Khronos by using object private meshes to repair
the background/global mesh in BOTH directions. Khronos's final map uses ONLY the background
mesh; its object private meshes never help the main map. We exploit that.

Three mechanisms (all required, do not stop after one):

```text
(1) INJECT good object mesh into background where background is MISSING/INCOMPLETE
    -> raises completeness (recall). Must land in HOLES, not on already-covered surface.
(2) DELETE background residue, two distinct accurate cases:
    (2a) ghost: object absent/moved, Khronos left object-looking geometry behind -> remove it.
    (2b) leakage: object still present, background smeared onto/around it -> shave the smear.
    -> raises precision.
(3) PROTECT + FILL floor/wall/ceiling/large planes so cleanup does not punch holes.
    -> preserves completeness.
```

## Metric (this is where the previous rounds went wrong — fix it)

Report Background PRECISION and RECALL SEPARATELY, not the combined F1. F1@20cm is saturated
(~0.98) so it hides everything.

```text
PRIMARY:   Background Recall  (= completeness, GT-vertex-covered-within-threshold)
             -> this is what INJECT (mechanism 1) must move.
           Background Precision
             -> this is what DELETE (mechanism 2) must move.
THRESHOLDS: report at 20cm (Khronos paper's official threshold, VI-A) AND at 10cm (stricter,
             near the ν=8cm resolution, un-saturates the number).
DO NOT report only F1. DO NOT invent a threshold other than 20 / 10 cm.
DO NOT use 5cm unless the actual mean vertex spacing of Khronos office mesh is verified finer
   than 5cm (below the 8cm voxel it would measure quantization noise, not method quality).
```

Paper anchor (verify against Khronos PDF, VI-A): "For the background, we consider each vertex
that has a corresponding ground truth (GT) vertex within 20cm a positive." Resolution ν = 8 cm.

## Optimization advice

```text
- The real headroom is RECALL. Khronos office background precision is already ~96-99%; you
  cannot win much there. Completeness/recall is where injection can actually help.
- INJECTION MUST BE HOLE-TARGETED. Before injecting, detect where the background mesh is
  missing (low local completeness vs GT, or gaps in the mesh). Only inject there. Injecting
  onto already-covered surface just games precision and does NOT raise recall — that is a
  failure, not a result.
- Sanity check every injection round: if injected_vertices is large but Background Recall
  barely moves, the injection is landing on covered surface -> it is gaming, fix the targeting.
- DELETION MUST BE ACCURATE. Separate ghost (2a) from leakage (2b) before deleting; the target
  differs. Use the most accurate published criterion, not a bare distance threshold:
    Khronos ray absence (Eq. 18-19, Sec. V-C), Panoptic SDF conflict (Sec. III-D),
    POCD change measure (Sec. IV-A). Cite the one you use in code.
- NEVER delete a region that was not re-observed / lacks reliable conflict evidence
  (unobserved guard).
- Track removed_vertices and injected_vertices every round. removed_vertices=0 means mechanism
  (2) is not implemented yet — say so, do not claim (2) is done.
```

## Method-search checkpoint (mandatory when stuck)

Do not keep tuning thresholds if a mechanism fails. A search/re-read step is required in either
case below:

```text
TRIGGER A: two consecutive injection rounds add many vertices but Background Recall @10cm
           does not improve.
TRIGGER B: two consecutive deletion rounds either remove zero vertices or hurt Recall more
           than they improve Precision.
```

When triggered, stop coding and create:

```text
session_update_baseline/rounds/<round_name>/method_search_notes.md
```

The notes must contain:

```text
1. Failure symptom:
   exact metrics, injected_vertices, removed_vertices, and affected object ids.

2. Source re-read:
   Khronos V-C ray/free-space change detection and ChangeMerger code;
   Panoptic Multi-TSDF map-management / active-inactive conflict;
   POCD object-level change / stationarity update.

3. Method search:
   search terms used, at least 5 candidate methods or code ideas, and why each is kept/rejected.
   Suggested query families:
     object-aware TSDF fusion
     semantic mesh cleanup
     object-guided background mesh removal
     volumetric change detection free-space carving
     submap conflict TSDF consistency
     mesh hole filling from instance mesh

4. Chosen next mechanism:
   one mechanism only, with paper/code citation and expected metric movement:
     injection should move Recall;
     deletion should move Precision;
     protection should prevent Recall loss.
```

No new implementation strategy is allowed after a trigger unless this file exists.

## What counts as ONE round (hard gates — all must be GREEN with real numbers)

```text
G1  Khronos office single-session final map produced (valid, full-size map, not a stub).
G2  Our improved single-session map produced and saved.
G3  No-op IO safe: load+save with no change -> metrics identical (float-level).
G4  OFFICIAL evaluator run on BOTH (khronos_original, our_single), numbers in a table:
      Background Precision @20cm, @10cm
      Background Recall    @20cm, @10cm
      (also dump Object/Change/Dynamic for completeness, but they are NOT the target here)
G5  At least one mechanism produced a real, explained effect this round:
      injection -> Background Recall moved up (with hole-targeting evidence), OR
      deletion  -> Background Precision moved up AND removed_vertices > 0.
```

## End-of-round self-check (answer ALL in writing; any "no" = round NOT done)

```text
Q1  Did the official Khronos evaluator actually run and emit numbers this round?   (paste path)
Q2  Are Background Precision AND Recall reported SEPARATELY at 20cm AND 10cm?       (paste table)
Q3  If injection ran: did Background RECALL move? If injected_vertices is large but recall is
    flat, is that acknowledged as gaming (not a win)?                              (yes/no + numbers)
Q4  If deletion ran: is removed_vertices > 0, and is ghost (2a) separated from leakage (2b)?
    If removed_vertices = 0, is it stated that mechanism (2) is NOT yet implemented?
Q5  Does every scoring/deletion criterion cite a paper equation (Khronos/Panoptic/POCD)?
Q6  One-line verdict: which mechanism moved which metric by how much, on real numbers.
```

## Forbidden (these are the laziness paths — do none of them)

```text
- Claiming a round complete without real official-evaluator numbers.
- Reporting only F1 (it is saturated; it hides the result).
- Calling injection a success when injected_vertices is large but recall is flat.
- Claiming deletion is done while removed_vertices = 0.
- Using a distance threshold when a more accurate published criterion exists.
- Inventing a "probabilistic" score with no paper equation behind it.
- Tuning thresholds randomly. If a metric does not move, write a one-line failure hypothesis
  BEFORE the next change. Two flat rounds in a row -> stop coding, re-read the relevant paper
  sections, write a short failure analysis and the method_search_notes.md described above.
```

## Definition of done (whole track)

```text
Single-session improved map beats raw Khronos on Background RECALL @10cm (from hole-targeted
injection) AND on Background PRECISION @10cm (from accurate ghost/leakage deletion), with
official numbers, no-op safe, and every decision explained by object/mesh evidence + paper cite.
```
