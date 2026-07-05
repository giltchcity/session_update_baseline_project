# Base1 Evaluation Summary

Export time: 2026-07-04 23:55 CST

## Evaluation Setup

Config:

```text
/home/jixian/Desktop/FT/configs/khronos/eval_office_local.yaml
```

Original eval directory:

```text
session_update_baseline/runs/loop08_full_original_eval_ready
```

Improved eval directory:

```text
session_update_baseline/runs/loop08_full_eval_ready
```

Evaluator command shape:

```bash
source /home/jixian/ros2_ws/install/setup.bash
/home/jixian/ros2_ws/install/khronos_eval/lib/khronos_eval/exp_pipeline \
  /home/jixian/Desktop/FT/configs/khronos/eval_office_local.yaml \
  <eval_dir> true true true
```

Important caveat:

```text
original_final.4dmap contains 5 DSG time steps.
improved_final.4dmap currently contains 1 final-current DSG time step.
The comparison below uses original final row Name=4 vs improved row Name=0.
This is a final-current map comparison, not full 4D table reproduction.
```

## Background Mesh Final-Current Metrics

Here Accuracy is treated as precision and Completeness as recall.

| Map | P@0.05 | R@0.05 | F1@0.05 | P@0.1 | R@0.1 | F1@0.1 | P@0.2 | R@0.2 | F1@0.2 | P@0.5 | R@0.5 | F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original final | 0.820739 | 0.553427 | 0.661083 | 0.915575 | 0.826288 | 0.868643 | 0.966349 | 0.894568 | 0.929074 | 0.995331 | 0.953995 | 0.974225 |
| improved final | 0.822149 | 0.551202 | 0.659948 | 0.914665 | 0.824287 | 0.867127 | 0.965759 | 0.894214 | 0.928610 | 0.995238 | 0.953988 | 0.974177 |
| delta improved-original | +0.001410 | -0.002225 | -0.001135 | -0.000910 | -0.002001 | -0.001516 | -0.000590 | -0.000354 | -0.000464 | -0.000093 | -0.000007 | -0.000048 |

## Chamfer

| Map | Chamfer@0.05 | Chamfer@0.1 | Chamfer@0.2 | Chamfer@0.5 |
|---|---:|---:|---:|---:|
| original final | 0.0508264 | 0.0653921 | 0.0790499 | 0.103408 |
| improved final | 0.0506087 | 0.0651512 | 0.0790502 | 0.103610 |
| delta improved-original | -0.0002177 | -0.0002409 | +0.0000003 | +0.0002020 |

## Object Metrics

Simple final-row detected/missed/hallucinated accounting:

| Map | Static detected | Static missed | Static hallucinated | Static P | Static R | Static F1 |
|---|---:|---:|---:|---:|---:|---:|
| original final | 22 | 140 | 0 | 1.0 | 0.135802 | 0.239130 |
| improved final | 22 | 140 | 0 | 1.0 | 0.135802 | 0.239130 |

Dynamic object final-row accounting:

| Map | Dynamic detected | Dynamic missed | Dynamic hallucinated | Dynamic P | Dynamic R | Dynamic F1 |
|---|---:|---:|---:|---:|---:|---:|
| original final | 0 | 520 | 19 | 0.0 | 0.0 | 0.0 |
| improved final | 0 | 520 | 19 | 0.0 | 0.0 | 0.0 |

## Verdict

Current Base1 cleanup does not improve official final-current background F1.

What it did:

```text
removed 5126 global mesh vertices near object private meshes with bbox guard
slightly improved Chamfer at 0.05m and 0.1m
slightly reduced background recall
slightly reduced F1 at all reported thresholds
left object/dynamic final-row counts unchanged
```

Interpretation:

```text
This first cleanup is too naive as an improvement method.
It proves map IO + object-guided vertex diagnostics work, but it does not yet meet the acceptance target:
improved_final.4dmap background F1 > original_final.4dmap background F1.
```

Next implementation should add true visibility/free-space evidence and background protection before deleting vertices.
