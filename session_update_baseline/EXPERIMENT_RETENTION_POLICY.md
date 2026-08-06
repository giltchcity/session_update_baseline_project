# Experiment Retention Policy

Date: 2026-08-06

## Rule

`rounds/` contains exactly one current canonical experiment. Older experiments
must not remain there as numbered `roundXX`, `loopXX`, smoke, or visualization
variants.

When a newer canonical experiment replaces the current one:

1. Preserve compact records: reports, commands, configs, metrics, summaries,
   logs, object memory, and representative comparison figures.
2. Move those compact records to `archive/records_YYYYMMDD/`, preserving their
   relative paths.
3. Delete reproducible bulk artifacts: intermediate `.4dmap` files, map
   checkpoints, DSG dumps, per-frame PLY/JSON, superseded visualizations,
   temporary runs, and build products.
4. Keep only the latest original/improved final maps and the latest complete
   visualization needed for evaluation or presentation.
5. Update `rounds/CURRENT.md` and the archive cleanup report.

## Naming

Use one descriptive name:

```text
<dataset>_<session_protocol>_<main_feature>_<YYYYMMDD>
```

Do not use ambiguous sequences such as `round01`, `loop08`, `test_v7`, or
`final_final`. Temporary outputs belong outside `rounds/` and should be removed
after the result has been promoted or rejected.

## Current Canonical Experiment

See `rounds/CURRENT.md`.
