# Base1.1 Round11 Method Search: Temporal Background Union

Date: 2026-07-05

This note is required by `base1.1_mode_a_single_session.md` because the previous
rounds triggered both failure conditions:

```text
Trigger A: object injection added many vertices but Recall@10 barely moved.
Trigger B: deletion rounds either removed zero useful vertices or hurt Recall more
           than they improved Precision.
```

## 1. Failure Symptom

Official evaluator, final-current office map.

| method | injected_vertices | removed_vertices | P@5 | R@5 | P@10 | R@10 | P@20 | R@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw Khronos | 0 | 0 | 0.864665 | 0.633729 | 0.946311 | 0.932604 | 0.987914 | 0.979944 |
| old full object append | 1916187 | 0 | 0.807808 | 0.647282 | 0.972461 | 0.936352 | 0.996476 | 0.981114 |
| Round09 hard object-change gate | 1716396 | 0 | 0.815716 | 0.646468 | 0.976665 | 0.936233 | 0.997421 | 0.981081 |
| Round10 probability best | 1739895 | 0 | 0.817075 | 0.646563 | 0.976880 | 0.936252 | 0.997451 | 0.981081 |
| Round10 expected utility | 1721355 | 0 | 0.816674 | 0.646322 | 0.976681 | 0.936182 | 0.997428 | 0.981081 |

Diagnosis:

```text
Object private mesh append improves Precision@10 by a lot.
Recall@10 moves only about +0.0036.
5cm is reported as requested, but used as diagnostic: object append hurts
strict surface accuracy relative to raw Khronos.
```

This means the previous mechanism mostly adds approximate object surfaces. It
does not solve the main Base1.1 requirement: hole-targeted recall repair.

## 2. Source Re-Read

### Khronos

Paper: `Khronos: A Unified Approach for Spatio-Temporal Metric-Semantic SLAM in Dynamic Environments`

URL: https://arxiv.org/html/2402.13817v2

Short source phrases:

```text
"object fragments"
"evidence-of-absence vs. absence-of-evidence"
"within 20cm a positive"
```

Relevant code:

```text
/home/jixian/ros2_ws/src/khronos/khronos/src/backend/change_detection/ray_verificator.cpp
/home/jixian/ros2_ws/src/khronos/khronos/src/backend/change_detection/ray_change_detector.cpp
/home/jixian/ros2_ws/src/khronos/khronos/src/backend/change_detection/background/ray_background_change_detector.cpp
/home/jixian/ros2_ws/src/khronos/khronos/src/backend/reconciliation/mesh/change_merger.cpp
```

Implementation facts from code:

```text
RayVerificator labels a point absent when a ray passes through it beyond depth_tolerance.
RayChangeDetector requires a majority over a temporal window.
RayBackgroundChangeDetector only marks vertices that are re-observed.
ChangeMerger deletes absent or duplicated/persistent background vertices.
```

### Panoptic Multi-TSDFs

Paper: `Panoptic Multi-TSDFs: a Flexible Representation for Online Multi-resolution Volumetric Mapping and Long-term Dynamic Scene Consistency`

URL: https://ar5iv.labs.arxiv.org/html/2109.10165

Short source phrases:

```text
"object as the minimal unit of change"
"active and inactive submaps"
"re-use of prior measurements"
```

Useful math / rule:

```text
Inactive submaps are compared to active submaps by SDF agreement/conflict.
Matching inactive submaps are fused, letting prior measurements fill coverage.
Conflicting inactive submaps are marked absent.
```

### POCD

Paper: `POCD: Probabilistic Object-Level Change Detection and Volumetric Mapping in Semi-Static Scenes`

URL: https://arxiv.org/abs/2205.01202

Short source phrase:

```text
"stationarity score and a TSDF change measure"
```

Useful rule:

```text
Object state must combine geometric TSDF disagreement with object-level stationarity.
Pure object id probability is not enough.
```

### Dynablox / Free-Space Family

Paper/code family:

```text
Dynablox: Real-time Detection of Diverse Dynamic Objects in Complex Environments
FreeDOM: Online Dynamic Object Removal Framework for Static Map Construction
```

URLs:

```text
https://github.com/ethz-asl/dynablox
https://arxiv.org/html/2504.11073v1
```

Useful rule:

```text
Free-space evidence must be conservative before removing geometry.
```

## 3. Method Search

Search terms used:

```text
object-aware TSDF fusion
semantic mesh cleanup
object-guided background mesh removal
volumetric change detection free-space carving
submap conflict TSDF consistency
mesh hole filling from instance mesh
conservative free space dynamic object removal
```

### Candidate A: object skip probability

Reject.

Reason:

```text
It only changes which object private meshes are appended.
It does not identify background holes.
It produced only +0.000113 F1@10 over Round09.
It violates the Base1.1 warning against invented probabilistic scores without a
paper equation.
```

### Candidate B: distance-shell deletion around object meshes

Reject.

Reason:

```text
Already tested. It mostly deletes evaluator inliers and does not separate ghost
from leakage.
```

### Candidate C: ray/free-space deletion of absent object bboxes

Keep for later deletion mechanism.

Expected metric movement:

```text
Precision@10 should rise.
removed_vertices must be > 0.
```

Reason not selected now:

```text
Current biggest unsolved Base1.1 issue is Recall@10. Deletion alone cannot give
the required >2% total gain unless recall repair also happens.
```

### Candidate D: object surface likelihood before injection

Keep for later injection refinement.

Expected metric movement:

```text
Preserve Precision@10 and improve 5cm diagnostic quality.
```

Reason not selected now:

```text
It still depends only on object private mesh. Previous evidence says object mesh
append barely moves Recall@10.
```

### Candidate E: temporal background union from same 4dmap

Select.

Reason:

```text
Base1.1 is single-session. The same Khronos run contains multiple DSG snapshots
before the final timestamp. Earlier background meshes may contain surfaces missing
from the latest map. This directly targets Recall@10 without inventing new data.
```

Paper link:

```text
Khronos represents the scene through time using fragments.
Panoptic Multi-TSDFs fuses matching inactive submaps to re-use prior measurements.
```

Expected metric movement:

```text
Primary: Recall@10 up by >= 0.02 absolute over raw Khronos target direction.
Secondary: Precision@10 must not collapse; moved-object bboxes are protected
using Khronos object_changes.csv.
5cm is reported as diagnostic.
```

## 4. Chosen Next Mechanism

One mechanism only:

```text
Temporal background union with paper-grounded guards.
```

Algorithm:

```text
1. Load the same single-session 4dmap.
2. Extract latest DSG as the final-current target.
3. For every earlier DSG timestamp:
   - consider its background/global mesh as a prior background fragment.
   - append only vertices farther than 0.08m from the latest background mesh.
     This is tied to Khronos office resolution nu = 8cm, not a tuned sweep.
   - keep faces only if all three vertices survive.
4. Guard moved/absent regions:
   - load Khronos object_changes.csv.
   - if last_absent > last_persistent for an object, do not inject temporal
     background vertices inside that object's bbox.
5. Also allow Round09 object-change-gated object mesh injection after temporal
   background repair, because Base1.1 requires object private meshes to be used.
```

Expected official table:

```text
Report P/R separately at 5cm, 10cm, 20cm.
10cm Recall must move materially.
If F1@10 does not beat raw by >2 percentage points, this mechanism is not enough
and must be reported as such.
```
