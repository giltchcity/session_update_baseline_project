# Base1.2 Ours Panmap Policy Round 01

Date: 2026-07-05

Scope: Base1.2 / Mode B only. This round fixes the missing "ours" rows by
forcing our output back into the standard Panoptic `.panmap` format and running
the official Panoptic evaluator. No pointcloud P/R, namespace delta, or local
mesh-export metric is used.

## Goal

Produce standard-evaluable rows for:

```text
ours_from_scratch
ours_with_prior
```

Both must be `.panmap` files and both must be evaluated by:

```text
roslaunch panoptic_mapping_utils evaluate_panmap.launch
```

## Method

Added:

```text
scripts/panoptic_mapping/rewrite_panmap_policy.sh
```

It loads a standard `.panmap`, rewrites only Panoptic `ChangeState` metadata,
saves a standard `.panmap`, and then the official evaluator consumes that map.

The main policy tested here:

```text
keep_unobserved:
  Unobserved -> Persistent
```

This is a deliberately narrow memory-policy test. It is standard-evaluable, but
it is not automatically a valid final Base1.2 method because it may preserve
unobserved prior geometry.

## Commands

Ours from scratch:

```bash
scripts/panoptic_mapping/rewrite_panmap_policy.sh \
  session_update_baseline/rounds/base12_panoptic_flat_round01/B_from_scratch/run2.panmap \
  session_update_baseline/rounds/base12_ours_panmap_policy_round01/eval_input/ours_from_scratch_keep_unobserved/ours_from_scratch_keep_unobserved.panmap \
  keep_unobserved
```

Ours with prior:

```bash
scripts/panoptic_mapping/rewrite_panmap_policy.sh \
  session_update_baseline/rounds/base12_panoptic_flat_round01/B_with_prior/run2.panmap \
  session_update_baseline/rounds/base12_ours_panmap_policy_round01/eval_input/ours_with_prior_keep_unobserved/ours_with_prior_keep_unobserved.panmap \
  keep_unobserved
```

Official evaluator:

```bash
GT=/ft/datasets/panoptic_mapping/flat_dataset/flat_2_gt_10000.ply \
scripts/panoptic_mapping/evaluate_panmap_batch.sh \
  /ft/session_update_baseline/rounds/base12_ours_panmap_policy_round01/eval_input \
  official_scratch official_prior \
  ours_from_scratch_keep_unobserved ours_with_prior_keep_unobserved
```

## Standard Results

Summary CSV:

```text
session_update_baseline/rounds/base12_ours_panmap_policy_round01/standard_eval_summary.csv
```

| variant | MeanError m | RMSE m | Unknown | Truncated | Observed-ish |
|---|---:|---:|---:|---:|---:|
| official_scratch | 0.00911097 | 0.0133199 | 1,022,198 | 89,375 | 2,004,989 |
| official_prior | 0.00908544 | 0.0131076 | 653,662 | 197,605 | 2,265,295 |
| ours_from_scratch_keep_unobserved | 0.00911097 | 0.0133199 | 1,022,198 | 89,375 | 2,004,989 |
| ours_with_prior_keep_unobserved | 0.00907146 | 0.0130769 | 621,804 | 238,463 | 2,256,295 |

Ours with prior vs ours from scratch:

| metric | delta | relative |
|---|---:|---:|
| MeanError | -0.00003951 m | -0.43% |
| RMSE | -0.00024300 m | -1.82% |
| UnknownPoints | -400,394 | -39.17% |
| TruncatedPoints | +149,088 | +166.81% |
| Observed-ish | +251,306 | +12.53% |

Ours with prior vs official prior:

| metric | delta | relative |
|---|---:|---:|
| MeanError | -0.00001398 m | -0.15% |
| RMSE | -0.00003070 m | -0.23% |
| UnknownPoints | -31,858 | -4.87% |
| TruncatedPoints | +40,858 | +20.68% |
| Observed-ish | -9,000 | -0.40% |

## Semantic Audit

`keep_unobserved` changes 9 `Unobserved` submaps to `Persistent` in the
prior-loaded map. The affected labels include:

```text
SM_Floor_Lamp_a  moved
SM_Journal_b     removed
SM_Bed_a         removed
SM_Cup_a         unchanged_or_unlisted
SM_Bed_table_b   unchanged_or_unlisted
SM_Bed_lamp_b    unchanged_or_unlisted
SM_Plant_a       unchanged_or_unlisted
FreeSpace        unchanged_or_unlisted
```

This is why the round is not a clean Base1.2 success: the standard evaluator
score improves, but the policy violates the intended unobserved guard by
including prior geometry for changed labels.

## Gate Status

| Gate | status | note |
|---|---|---|
| G1 real change dataset | GREEN | Panoptic flat run1/run2 |
| G2 A saved and B loads it | GREEN | official prior map from `run1.panmap` loaded into run2 |
| G3 ours standard map output | GREEN | ours variants are `.panmap` files |
| G4 official evaluator on ours | GREEN | official `evaluate_panmap` CSV exists for ours rows |
| G5 ours_with_prior beats ours_from_scratch full-map official eval | GREEN/PARTIAL | RMSE and Unknown improve, Truncated worsens |
| G6 changed-region/object official evaluator | RED | public Panoptic release does not provide changed-region P/R table |
| G7 unobserved guard | RED | `keep_unobserved` preserves unobserved changed-object prior geometry |

## Verdict

We now have standard-evaluable "ours" rows. The previous failure was missing:
only official Panoptic rows existed.

This round still does not finish Base1.2. The valid conclusion is:

```text
ours_keep_unobserved improves full-map official RMSE/Unknown vs ours_from_scratch,
but it is not an acceptable final memory policy because it keeps unobserved
prior geometry, including moved/removed labels.
```

Next valid method target: keep prior memory internally, but include it in the
current final map only when present/persistent evidence exists; unobserved prior
must stay out of scoring unless visibility/free-space evidence resolves it.
