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

If you are assigned this file, you are working only on **Base1.3 / NSS Dataset Construction**. This track does not prove the method yet. It builds the benchmark substrate that later lets Mode B graduate to larger real changes.

# Base1.3 — NSS Dataset Construction Drive

Date: 2026-07-05
Owner: you lead, agent assists. This track's rounds are defined by DATA PRODUCTS, not by map
metrics (there is no method to score yet — the point is to build the benchmark).

## Why this track exists

Neither office (no cross-session change) nor flat/TorWIC (small object-level change only) covers
the full spectrum you want: short-blind-spot + long-blind-spot changes AND large structural
change (walls / big furniture / room-level rearrangement). NSS (Nothing Stands Still, Sun et al.,
Khronos ref [50]) provides real multi-time scans with large geometric + temporal change. You
derive a change-detection / map-maintenance benchmark from it.

## Goal

Turn NSS (GT meshes, multiple time states) into inputs your Khronos-based pipeline can consume,
with a labeled A/B change protocol and an evaluation contract, spanning:

```text
blind-spot duration  x  change scale
  short blind spot (occlusion / look-back)   x   small (single object moved)
  long blind spot  (cross-session revisit)   x   large (wall / room-level rearrangement)
```

## Build stages (each stage = one round; a round ends only when its product exists AND loads)

```text
S1  Source audit:    which NSS scenes/time-pairs exist; what GT is available (mesh, poses,
                     semantics?); pick ~6-10 stage pairs with clear change.  Product: a written
                     inventory table (scene, times, change type, change scale).
S2  Renderable input: from NSS GT mesh, produce a depth (and color if available) stream + poses
                     for ONE scene state.  Product: a depth/pose sequence that opens correctly.
S3  Khronos ingest:  wrap S2 into the topics/TF Khronos needs; run Khronos on ONE state ->
                     final map.  Product: a valid final map from NSS-derived input.
S4  A/B pair:        produce two states (A, B) of the SAME scene with a KNOWN change between
                     them.  Product: two inputs + a written change label (what moved/appeared/
                     disappeared, where, scale).
S5  Semantics/instances: attach semantic/instance labels (open-vocab lift is acceptable) so
                     object-level reasoning works.  Product: labeled A/B inputs.
S6  Eval contract:   define GT + metric for the derived benchmark (Precision/Recall on changed
                     regions; how unobserved regions are EXCLUDED from scoring).  Product: an
                     eval config that runs on an A/B pair and emits numbers.
```

## Optimization advice

```text
- Do NOT block Base1.1 / Base1.2 on this. This is the long pole; 1.1 and 1.2 run in parallel on
  data that already exists.
- One scene state end-to-end (S2->S3) before scaling to many. Prove the render->ingest path on a
  single state; do not batch-convert 10 scenes then discover the format is wrong.
- The hardest correctness issue is the UNOBSERVED region: NSS scans are static snapshots, not a
  robot trajectory. Decide early how "not observed in B" is represented and how it is EXCLUDED
  from scoring, or every unseen region will look like a false change.
- Keep A/B change labels explicit and machine-readable from S4 on — the benchmark's value is the
  labels, not the meshes.
```

## What counts as ONE round (per stage — product must EXIST and be VERIFIED)

```text
A stage-round is done only when its Product above:
  - exists on disk (paste path),
  - opens / loads / runs (paste the command and its success output),
  - is described in one written paragraph (what it is, what change it encodes, what's missing).
Producing a script that "should" work is NOT done. The product must be demonstrated loading.
```

## End-of-round self-check (answer ALL; any "no" = round NOT done)

```text
Q1  Which stage (S1-S6) did this round complete, and where is the product on disk?
Q2  Did the product actually load / run (not just get written)?  (paste the proof)
Q3  For S4+: is the A/B change explicitly labeled (type, location, scale) in a readable file?
Q4  For S3+: does Khronos actually produce a valid final map from the NSS-derived input?
Q5  Is the unobserved-region handling decided/documented (so it won't fake changes)?
Q6  What is the single next blocking unknown for the following stage?
```

## Forbidden (laziness paths)

```text
- Marking a stage done because a conversion script was written, without showing the product load.
- Batch-converting many scenes before one end-to-end path is proven.
- Skipping the unobserved-region decision (S6) — it silently corrupts every change metric.
- Vague change labels ("scene changed") instead of typed, located, scaled labels.
```

## Definition of done (whole track)

```text
An NSS-derived benchmark: >=6 labeled A/B scene pairs spanning small->large structural change,
ingestible by the Khronos-based pipeline, with an eval contract that emits Precision/Recall on
changed regions and correctly excludes unobserved regions. This is the real Mode B / benchmark
substrate that Base1.2 graduates onto once it exists.
```
