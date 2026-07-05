# Session Update Baseline Project Snapshot

Date: 2026-07-05

User-provided tag:

```text
SHA256:sljiPcpSD7NX+i7cBbJqgPIxRgHjVfrmj+YYaJqlyos
```

Agent/task name:

```text
1.1agent
```

This project snapshot preserves the Base1 / Base1.1 baseline work history:

- source code for the current baseline tools
- Base1.1 task definition and checkpoints
- round reports and method-search notes
- official metric CSV/MD outputs
- command/config/evidence/summary artifacts needed to understand each run
- the original pasted task-split text under `references/`

Large reproducible map artifacts are intentionally excluded from this upload
package. See `EXCLUDED_ARTIFACTS.md`.

## Current Main Result

The strongest final-current single-map result in this snapshot is Round13:

```text
session_update_baseline/rounds/round13_graphcut_plane_fill/ROUND_REPORT.md
```

Official background F1 versus raw Khronos:

```text
F1@5:  +0.029591
F1@10: +0.023480
F1@20: +0.005650
```

Important caveat:

```text
removed_vertices = 0
```

So final-map filling is useful, but ghost/leakage deletion is still not finished.

## Method State

The method that should be carried forward is:

```text
temporal object gate
+ structural plane graph-cut fill
+ future visibility/free-space deletion
```

Older probability/expected-utility and temporal-union attempts are kept as
history, not as the recommended clean final method.
