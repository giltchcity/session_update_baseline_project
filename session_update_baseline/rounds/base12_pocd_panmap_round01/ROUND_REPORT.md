# Base1.2 POCD-Style Panmap Round 01

Date: 2026-07-05

Scope: Base1.2 / Mode B only. Dataset is Panoptic flat run1 -> run2. Khronos
office results are intentionally out of scope for this round.

## Goal

Redo Base1.2 without proxy metrics or private map formats:

- Session A is the official flat run1 `.panmap`.
- Session B baselines are official flat run2 from-scratch and run2 loading run1.
- Ours outputs standard `.panmap`.
- Evaluation includes the official Panoptic evaluator, threshold F-score at
  5/10/20 cm, POCD-style voxel Precision/Recall/FPR, and changed-label metrics.

The previous `keep_unobserved` result is kept only as a rejected negative
baseline because it turns all `Unobserved` submaps into `Persistent`.

## Implemented Method

Added standard-map tools:

```text
scripts/panoptic_mapping/run_base12_pocd_panmap_update.sh
scripts/panoptic_mapping/flat_visibility_audit.py
scripts/panoptic_mapping/standard_mesh_metrics.py
scripts/panoptic_mapping/extract_flat_label_clouds.py
scripts/panoptic_mapping/changed_object_metrics.py
```

Main result:

```text
ours_with_prior_visibility_c0326
```

Inputs:

```text
prior/base map:  base12_panoptic_flat_round01/B_with_prior/run2.panmap
current B map:   base12_panoptic_flat_round01/B_from_scratch/run2.panmap
visibility:      run2 pose/depth projection over prior namespace meshes
output:          eval_input/ours_with_prior_visibility_c0326/ours_with_prior_visibility_c0326.panmap
```

Policy:

- Current-supported prior submaps become present.
- Prior instance submaps without same-name current support are checked against
  run2 depth visibility.
- Depth-consistent memory with low free-space conflict is restored.
- High-conflict memory is marked `Absent`.
- Missing current B instance submaps are copied in as `New`.
- `changes.txt` is not read by the updater; it is used only for changed-label
  evaluation.

Parameters:

```text
theta_dist = 0.9 m
theta_stat = 0.4
tau = 0.20 m
lambda_diff = 1.6
delta_max = 4.0 m
VIS_CONSISTENT_MIN = 0.8
VIS_CONFLICT_MAX = 0.326
```

Audit for the main result:

```text
input_prior_submaps = 57
input_current_submaps = 43
output_submaps = 60
associated_prior_submaps = 38
state_to_persistent = 13
state_to_absent = 3
copied_current_submaps = 3
```

Important decisions:

```text
drop_visibility_conflict:
  SM_Bed_a, SM_Floor_Lamp_a, SM_Journal_b

restore_visibility_supported_memory:
  SM_Cup_a, SM_Bed_lamp_a, SM_Bed_table_b, SM_Bed_lamp_b,
  SM_Digital_Clock_a, SM_Decor_a, SM_Journal_a, SM_Plant_a

restore_current_supported_prior:
  SM_Bed_table_a, SM_Office_Chair_base_a, SM_Table_b

copied current B submaps:
  SM_Picture_b, SM_Picture_f, SM_Stack_of_Books_b
```

## Official Panoptic Results

Source:

```text
standard_eval_summary_final.csv
```

| variant | MeanError m | RMSE m | Unknown | Truncated | Observed-ish |
|---|---:|---:|---:|---:|---:|
| official_from_scratch | 0.00911097 | 0.0133199 | 1,022,198 | 89,375 | 2,004,989 |
| official_with_prior | 0.00908544 | 0.0131076 | 653,662 | 197,605 | 2,265,295 |
| ours_from_scratch | 0.00929112 | 0.0134687 | 1,013,355 | 89,116 | 2,014,091 |
| ours_with_prior_visibility_c0326 | 0.00909517 | 0.0131068 | 621,655 | 208,494 | 2,286,413 |
| negative_keep_unobserved | 0.00907146 | 0.0130769 | 621,804 | 238,463 | 2,256,295 |

Main vs official prior:

```text
RMSE:     0.0131068 vs 0.0131076  -> slightly better
Unknown: 621,655 vs 653,662       -> -32,007 (-4.9%)
Trunc:   208,494 vs 197,605       -> +10,889 (+5.5%)
```

Negative baseline still has the best official RMSE, but it is rejected because
it keeps unobserved changed-object geometry and produces stale removed-object
map points.

## F-Score And POCD-Style Voxel Metrics

Source:

```text
standard_mesh_metrics_final.csv
```

| variant | F1@5cm | F1@10cm | F1@20cm | Voxel P@20 | Voxel R@20 | Voxel FPR@20 |
|---|---:|---:|---:|---:|---:|---:|
| official_from_scratch | 0.831386 | 0.873404 | 0.925258 | 0.927744 | 0.723034 | 0.014927 |
| official_with_prior | 0.892460 | 0.928317 | 0.966545 | 0.934087 | 0.800361 | 0.014970 |
| ours_from_scratch | 0.833209 | 0.874398 | 0.925336 | 0.927716 | 0.724840 | 0.014970 |
| ours_with_prior_visibility_c0326 | 0.896059 | 0.929896 | 0.966558 | 0.932914 | 0.803645 | 0.015318 |
| negative_keep_unobserved | 0.884653 | 0.920035 | 0.958542 | 0.911348 | 0.801675 | 0.020671 |

Interpretation:

- Main improves F1 at 5/10/20 cm over official prior.
- Main improves voxel recall at 20 cm.
- Main does not improve 20 cm voxel precision/FPR; this is the remaining
  POCD-style tradeoff.
- `ours_with_prior_pruned_no_copy` improves 5/10 cm voxel precision/FPR but
  loses official RMSE/coverage, so it is rejected as the main result.

## Changed-Object Metrics

GT changed-label clouds are extracted from flat run2 segmentation/depth/pose.
`changes.txt` is used only to group eval rows.

Source:

```text
changed_object_summary_final.csv
```

| variant | change type | labels | P@20 | R@20 | F1@20 | removed stale map points |
|---|---|---:|---:|---:|---:|---:|
| official_with_prior | moved | 8 | 0.800485 | 0.623896 | 0.481788 | |
| official_with_prior | new | 7 | 0.882117 | 0.980701 | 0.926857 | |
| official_with_prior | removed | 4 | 1.000000 | 1.000000 | 1.000000 | 0 |
| ours_with_prior_visibility_c0326 | moved | 8 | 0.777688 | 0.748936 | 0.594258 | |
| ours_with_prior_visibility_c0326 | new | 7 | 0.880298 | 0.996816 | 0.932541 | |
| ours_with_prior_visibility_c0326 | removed | 4 | 1.000000 | 1.000000 | 1.000000 | 0 |
| negative_keep_unobserved | moved | 8 | 0.756236 | 0.623936 | 0.450185 | |
| negative_keep_unobserved | new | 7 | 0.882117 | 0.980701 | 0.926857 | |
| negative_keep_unobserved | removed | 4 | 0.500000 | 1.000000 | 0.500000 | 3,052 |

Main vs official prior:

```text
moved F1@20: 0.594258 vs 0.481788  -> +0.112470
new   F1@20: 0.932541 vs 0.926857  -> +0.005684
removed stale map points: 0 vs 0
```

The rejected `negative_keep_unobserved` row shows why the old policy is invalid:
it leaves 3,052 stale map points on removed labels.

## Gate Status

| Gate | status | note |
|---|---|---|
| G1 real A/B change dataset | GREEN | Panoptic flat run1/run2 |
| G2 A saved and B loads it | GREEN | official prior map loads run1 |
| G3 memory participates | GREEN | 13 state restorations, 3 absent decisions, 3 current copies |
| G4 B_ours differs from scratch on changed objects | GREEN | changed moved/new metrics improve; copied current labels are changed labels |
| G5 official evaluator | GREEN | official `evaluate_panmap` run on all standard `.panmap` rows |
| 5/10/20cm F-score | GREEN | main improves all F1 thresholds vs official prior |
| POCD voxel P/R/FPR | YELLOW | recall improves; 20cm precision/FPR slightly worse |
| unobserved guard | GREEN | no blanket `Unobserved -> Persistent`; visibility/conflict gate resolves memory |

## Verdict

This round produces the first standard `.panmap` Base1.2 result that beats the
official prior on official RMSE, Unknown count, 5/10/20cm F-score, and
changed-object moved/new F1 without keeping removed-object stale geometry.

It is not a perfect POCD-style win because 20cm voxel precision/FPR are slightly
worse than official prior. The next round should validate the visibility/conflict
threshold on TorWIC/POCD or another held-out Panoptic/RIO sequence before treating
`VIS_CONFLICT_MAX=0.326` as general.

Sources used for the method shape:

```text
POCD RSS 2022: https://www.roboticsproceedings.org/rss18/p013.pdf
Panoptic Mapping: https://ar5iv.labs.arxiv.org/html/2109.10165
Panoptic code/data: https://github.com/ethz-asl/panoptic_mapping
                 https://projects.asl.ethz.ch/datasets/panoptic-mapping/
```
