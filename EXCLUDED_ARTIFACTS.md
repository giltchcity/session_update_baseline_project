# Excluded Artifacts

Updated: 2026-08-06

This repository preserves source code and compact evidence, not generated map
storage. The following remain outside Git:

```text
build/
archive/
rounds/
runs/
visual_compare/
__pycache__/
*.pyc
*.4dmap
*.panmap
*.bag
*.db3
*.mcap
*.sparkdsg
*.dgrf
*.ply
large DSG and per-frame visualization dumps
datasets and model weights
```

The local generated-output retention policy is documented in
`session_update_baseline/EXPERIMENT_RETENTION_POLICY.md`.
