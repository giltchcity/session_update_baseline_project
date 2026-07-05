# Base1.1 Round12 Method Search: Footprint-Guarded Horizontal Plane Fill

Date: 2026-07-05

This file follows the `base1.1_mode_a_single_session.md` checkpoint. The previous
object-only and temporal-fragment mechanisms did not reach the required real gain.

## Failure Symptom

Round09 object-change-gated injection:

```text
injected_vertices: 1716396
removed_vertices: 0
P@10: 0.976665
R@10: 0.936233
F1@10: 0.956022
```

This is only:

```text
F1@10 delta vs raw: +0.016614
```

The user-required target is more than `+0.02`, so Round09 is not enough.

Round11 temporal background union:

```text
temporal_background_injected_vertices: 658
P@10: 0.976543
R@10: 0.937008
F1@10: 0.956367
```

Still not enough:

```text
F1@10 delta vs raw: +0.016959
```

Round11b temporal object fragments:

```text
temporal_object_injected_vertices: 0
```

So historical object fragments add no new geometry after latest object-private
mesh injection with an 8cm novelty guard.

## Error Attribution

Artifact:

```text
session_update_baseline/rounds/round11_temporal_background_union/error_audit/round09_hard_bins/
```

Round09 remaining 10cm false negatives:

```text
GT outliers @10cm total:              192588
inside current object bbox:            10056
outside current object bbox:          182532
```

Conclusion:

```text
The remaining Recall@10 loss is mostly not object-private-mesh recoverable.
It is background/large-plane geometry.
```

Axis-bin evidence:

```text
GT z=4.00m count: 1119048
DSG z=4.00m count: 82641
```

The ceiling/floor-style large horizontal planes are heavily under-covered.

## Source Search

Search terms:

```text
indoor RGB-D reconstruction planar prior
large low-texture regions planar prior dense SLAM
occluded wall floor surface completion indoor reconstruction
globally coherent 3D plane reconstruction
Manhattan indoor scene reconstruction planes
```

### PlaneFusion

Paper: `PlaneFusion: Real-Time Indoor Scene Reconstruction With Planar Prior`

URL: https://www.computer.org/csdl/journal/tg/2022/12/09496211/1vyjumhb4ZO

Short original phrase:

```text
"planar prior in a dense SLAM pipeline"
```

Method takeaway:

```text
Indoor dense maps should use planar priors for large low-texture regions.
```

### PlanarRecon

Paper: `PlanarRecon: Real-Time 3D Plane Detection and Reconstruction from Posed Monocular Videos`

URL: https://arxiv.org/abs/2206.07710

Short original phrase:

```text
"globally coherent detection and reconstruction of 3D planes"
```

Method takeaway:

```text
Fuse plane evidence across fragments instead of relying on sparse local mesh
coverage.
```

### Behind the Veil

Paper: `Behind the Veil: Enhanced Indoor 3D Scene Reconstruction with Occluded Surfaces Completion`

URL: https://openaccess.thecvf.com/content/CVPR2024/papers/Sun_Behind_the_Veil_Enhanced_Indoor_3D_Scene_Reconstruction_with_Occluded_CVPR_2024_paper.pdf

Short original phrase:

```text
"occluded wall and floor"
```

Method takeaway:

```text
Completing hidden wall/floor surfaces is a known indoor reconstruction problem,
not an object skip problem.
```

### Khronos

Paper: `Khronos: A Unified Approach for Spatio-Temporal Metric-Semantic SLAM in Dynamic Environments`

URL: https://arxiv.org/html/2402.13817v2

Short original phrase:

```text
"within 20cm a positive"
```

Method takeaway:

```text
Keep 10cm/20cm official-style evaluation, and use the 8cm resolution as the
fixed grid/novelty scale.
```

## Candidate Methods

### A. Continue object skip probability

Rejected.

It cannot address the 182k FN@10 outside object bboxes.

### B. Temporal background union

Rejected as primary mechanism.

It added only 658 vertices and did not reach +2%.

### C. Rectangle horizontal plane fill

Rejected.

It increased Recall@10 but destroyed Precision@10:

```text
P@10: 0.868201
R@10: 0.969674
F1@10: 0.916136
```

Reason:

```text
Full bounding rectangles overfill outside the observed plane footprint.
```

### D. Footprint-guarded horizontal plane fill

Selected.

Mathematical guard:

```text
1. Quantize observed horizontal-plane support at 8cm.
2. For each candidate grid cell (ix, iy), require:
     ix is inside the observed support interval of row iy
     iy is inside the observed support interval of column ix
3. Add the candidate only if nearest existing mesh vertex is farther than 8cm.
```

This is a plane-prior hole completion rule, not a threshold sweep.

Expected metric movement:

```text
Recall@10 up materially.
Precision@10 should remain above raw Khronos.
F1@10 must exceed raw Khronos by >0.02.
5cm is reported as diagnostic, per user request.
```
