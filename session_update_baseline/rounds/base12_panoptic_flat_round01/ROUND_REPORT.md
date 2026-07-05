# Base1.2 Panoptic Flat Round 01

Date: 2026-07-05

Scope: true A/B cross-session data only. This round uses Panoptic flat:
Session A is run1, Session B is run2.

## Commands

Session B from scratch:

```bash
PLAY_RATE=1 POST_WAIT=10 VISUALIZE=false \
scripts/panoptic_mapping/run_flat_save_monotonic.sh \
  run2 \
  /home/jixian/Desktop/FT/session_update_baseline/rounds/base12_panoptic_flat_round01/B_from_scratch
```

Session A:

```bash
PLAY_RATE=1 POST_WAIT=10 VISUALIZE=false \
scripts/panoptic_mapping/run_flat_save_monotonic.sh \
  run1 \
  /home/jixian/Desktop/FT/session_update_baseline/rounds/base12_panoptic_flat_round01/session_A
```

Session B with prior:

```bash
LOAD_MAP=true \
LOAD_FILE=/home/jixian/Desktop/FT/session_update_baseline/rounds/base12_panoptic_flat_round01/session_A/run1.panmap \
PLAY_RATE=1 POST_WAIT=10 VISUALIZE=false \
scripts/panoptic_mapping/run_flat_save_monotonic.sh \
  run2 \
  /home/jixian/Desktop/FT/session_update_baseline/rounds/base12_panoptic_flat_round01/B_with_prior
```

Official Panoptic eval:

```bash
GT=/ft/datasets/panoptic_mapping/flat_dataset/flat_2_gt_10000.ply \
scripts/panoptic_mapping/evaluate_panmap_batch.sh \
  /ft/session_update_baseline/rounds/base12_panoptic_flat_round01/eval_input \
  B_from_scratch B_with_prior
```

## Map Artifacts

| artifact | path | size | save evidence |
|---|---|---:|---|
| Session A run1 | `session_A/run1.panmap` | 44 MB | saved 47 submaps |
| B_from_scratch run2 | `B_from_scratch/run2.panmap` | 43 MB | saved 43 submaps |
| B_with_prior run2 | `B_with_prior/run2.panmap` | 60 MB | loaded A map; saved 57 submaps |

`B_with_prior` used the official Panoptic `map_loader`; the log shows it waited
for `/panoptic_mapper/load_map`, called the service, and finished cleanly before
run2 playback.

## Official Full-Map Evaluation vs Post-Change GT

Reference:

```text
datasets/panoptic_mapping/flat_dataset/flat_2_gt_10000.ply
```

| label | MeanError m | StdError m | RMSE m | TotalPoints | UnknownPoints | TruncatedPoints | Observed-ish |
|---|---:|---:|---:|---:|---:|---:|---:|
| B_from_scratch | 0.00911097 | 0.00972860 | 0.0133199 | 3116562 | 1022198 | 89375 | 2085009 |
| B_with_prior | 0.00908544 | 0.00945884 | 0.0131076 | 3116562 | 653662 | 197605 | 2439790 |

Official comparison:

| metric | B_with_prior - B_from_scratch | relative change |
|---|---:|---:|
| MeanError | -0.00002553 m | -0.28% |
| RMSE | -0.00021230 m | -1.59% |
| UnknownPoints | -368,536 | -36.05% |
| Observed-ish | +354,781 | +17.02% |
| TruncatedPoints | +108,230 | +121.10% |

Interpretation: prior update changes output and improves full-map RMSE and
unknown-point count under the official Panoptic evaluator. No local P/R or
local audit result is part of this report anymore.

## Deleted Local Artifacts

The previous local audit artifacts were non-official for this comparison and
were deleted. Only the official maps and official evaluator CSV are valid in
this round.

## Gate Status

| Gate | status | note |
|---|---|---|
| G1 real change dataset flows through pipeline | GREEN | flat run1/run2 ran and saved maps |
| G2 Session A saved and Session B loads it | GREEN | `session_A/run1.panmap`; B log confirms map_loader |
| G3 official evaluator available | GREEN | `evaluate_panmap` produced full-map MeanError/RMSE/Unknown CSV |
| G4 standard from-scratch vs with-prior comparison | GREEN | official full-map comparison table above |
| G5 changed-region/object evaluator | RED | no official changed-region P/R table is available in the public Panoptic release |
| G6 our Base1.2 method evaluated under a standard evaluator | MOVED | see `base12_ours_panmap_policy_round01` for standard `.panmap` ours rows |

## Verdict

This round proves the official Panoptic flat data path is runnable for A/B:
Session A is saved, Session B loads it, and the official evaluator runs.

It does not complete Base1.2 method evaluation, because the public Panoptic
release does not provide an official changed-region Precision/Recall table. Our
standard-evaluable `.panmap` policy rows are reported separately in
`base12_ours_panmap_policy_round01`.
