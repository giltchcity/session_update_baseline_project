# Base1.1 Existing Results Check

Date: 2026-07-05
Scope: Mode A / single-session office final-current map.

## Where the previous results are

Primary previous round:

```text
session_update_baseline/rounds/round05_independent_ab/
```

Important files:

```text
gate2_single/khronos_original_eval/results/background_mesh.csv
gate2_single/our_single_eval/results/background_mesh.csv
gate2_single/single_metrics.csv
gate2_single/ours/mesh_update_summary.csv
ROUND_REPORT.md
```

Earlier loop with explicit 5/10/20/50 cm final-current table:

```text
session_update_baseline/base1_eval_summary.md
session_update_baseline/runs/loop08_full_original_eval_ready/results/background_mesh.csv
session_update_baseline/runs/loop08_full_eval_ready/results/background_mesh.csv
```

## Key verification point

The per-round `single_metrics.csv` files only captured the 20 cm official background columns.
The official evaluator output `results/background_mesh.csv` contains the stricter 10 cm columns:

```text
Accuracy@0.1      = Background Precision @10cm
Completeness@0.1  = Background Recall @10cm
Accuracy@0.2      = Background Precision @20cm
Completeness@0.2  = Background Recall @20cm
```

## 10 cm and 20 cm check across existing rounds

| round | P@10 original | P@10 ours | delta | R@10 original | R@10 ours | delta | P@20 original | P@20 ours | delta | R@20 original | R@20 ours | delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| round01_fresh | 0.943164 | 0.973096 | +0.029932 | 0.799831 | 0.802850 | +0.003019 | 0.984319 | 0.995875 | +0.011556 | 0.852898 | 0.854081 | +0.001183 |
| round02_prior_match | 0.925612 | 0.968057 | +0.042445 | 0.886505 | 0.888373 | +0.001868 | 0.972417 | 0.993182 | +0.020765 | 0.953104 | 0.953512 | +0.000408 |
| round03_single_snapshot | 0.948903 | 0.972634 | +0.023731 | 0.941440 | 0.944687 | +0.003247 | 0.987428 | 0.996383 | +0.008955 | 0.986685 | 0.987344 | +0.000659 |
| round04_early_to_latest | 0.945009 | 0.980585 | +0.035576 | 0.939967 | 0.942318 | +0.002351 | 0.987189 | 0.997624 | +0.010435 | 0.985353 | 0.985686 | +0.000333 |
| round05_independent_ab | 0.946311 | 0.972461 | +0.026150 | 0.932604 | 0.936352 | +0.003748 | 0.987914 | 0.996476 | +0.008562 | 0.979944 | 0.981114 | +0.001170 |

Latest strict round F1, computed from the same 10 cm/20 cm P/R:

| round | F1@10 original | F1@10 ours | delta | F1@20 original | F1@20 ours | delta |
|---|---:|---:|---:|---:|---:|---:|
| round05_independent_ab | 0.939408 | 0.954065 | +0.014657 | 0.983913 | 0.988735 | +0.004822 |

## What this proves

The existing injection implementation has a real positive official-evaluator signal at 10 cm.
For round05, Background Precision @10cm improves by +0.026150 and Background Recall @10cm
improves by +0.003748.

This is stronger than the 20 cm-only table because 10 cm is less saturated.

## What this does not prove yet

The current implementation is not a completed Base1.1 round under the new split document:

```text
1. Injection is broad object-private-mesh append, not verified hole-targeted injection.
2. removed_vertices = 0 in round05, so ghost/leakage deletion is not implemented in this round.
3. There is no floor/wall/ceiling protection or repair proof attached to round05.
4. The report text still leans on Background F1; future reports must lead with P/R at 20 cm and 10 cm.
```

Code pointer:

```text
session_update_baseline/src/base1/object_guided_map_reconciler.cpp
appendObjectMeshToGlobal(...)
mode == "injection" appends every object mesh with enough vertices.
```

Round05 command:

```text
session_update_baseline/rounds/round05_independent_ab/gate2_single/ours/command.txt
```

## Recommended next Base1.1 split

Run the next 1.1 work as two separate sub-rounds:

```text
A. Injection / recall sub-round
   Keep the existing positive 10 cm signal as baseline.
   Replace broad append with hole-targeted injection.
   Required proof: Background Recall @10cm moves up, with object ids and local hole evidence.

B. Deletion / precision sub-round
   Implement ghost/leakage deletion separately.
   Use Khronos ray absence/free-space evidence or Panoptic/POCD conflict logic, not a bare
   object-distance threshold.
   Required proof: removed_vertices > 0 and Background Precision @10cm moves up without a
   meaningful Recall drop.
```

Do not let a future agent use `single_metrics.csv` alone for 10 cm claims. Always read
`results/background_mesh.csv`.
