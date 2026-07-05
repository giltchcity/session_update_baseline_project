# Base1.2 Dataset Audit

Date: 2026-07-05

Scope: Base1.2 only. Khronos office/apartment are intentionally excluded because
they are Base1.1 material, not true A/B cross-session change datasets.

## Usable Now

### Panoptic flat

Local path:

```text
datasets/panoptic_mapping/flat_dataset
```

Status: installed and runnable.

Contents checked:

```text
run1 color frames: 400
run2 color frames: 362
run1 files: 2401
run2 files: 2173
GT: flat_1_gt_10000.ply, flat_2_gt_10000.ply
change log: changes.txt
```

Local `changes.txt` labels:

```text
moved:   8
new:     7 labels
removed: 4
```

Note: the paper/source summary uses "5 added" at object-change level, while the
local label file lists 7 new labels. For machine checks in this repo, use the
local `changes.txt` labels.

### Panoptic RIO add-ons

Local path:

```text
datasets/panoptic_mapping/rio_data
```

Status: add-on data present for 6 sequences. This is secondary for Base1.2
because flat has the clean two-run A/B change setup and full synthetic GT.

## Usable Later After Download

### TorWIC / POCD

Local path:

```text
baselines/TorWICDataset
```

Status: repository, papers, figures, and utility scripts are present. The actual
trajectory data are not present locally.

Expected mapping dataset from the official repo:

```text
18 trajectories in 4 scenarios
WarehouseSequences/
  rgb/
  depth/
  segmentation/
  laser scans/
  poses.txt
  imu.txt
  odom.txt
```

Current local check found no `WarehouseSequences`, no `.bag`, no `.pcd`,
no `poses.txt`, no `odom.txt`, and no real image/depth sequence under
`baselines/TorWICDataset`.

Verdict: TorWIC is a real candidate, but it is not locally runnable yet.
Do not block Panoptic flat work on it.
