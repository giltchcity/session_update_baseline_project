# Excluded Artifacts

This project snapshot is meant to preserve code, method notes, and metrics, not
the full binary map cache.

Excluded patterns:

```text
build/
__pycache__/
*.pyc
*.4dmap
*.panmap
*.sparkdsg
*.dgrf
*.ply
dsg.json
dsg_with_mesh.json
shared_dsg.json
*/maps/*.json
*/frontend/*.json
*/backend/*.sparkdsg
*/_logs/
*/timing/
*_timing_raw.csv
timing_stats.csv
*.log
eval_visualization/
```

Reason:

```text
The source `session_update_baseline/` directory is about 72G because it contains
many generated final maps and intermediate DSG/mesh files. The upload snapshot
is about 30M and keeps the reproducible history: code, markdown, commands,
configs, summaries, and official metric outputs.
```

The excluded files still exist in the local working tree at:

```text
/home/jixian/Desktop/FT/session_update_baseline
```
