"""Real-data D1 metric of the TSDF version (Dynamic person ABC + C cabinet I2 D1) on the layer's D1 trajectories.

  python -m update_layer.eval.d1_person RUN_DIR OUT_DIR

The scorer is tools/abc_eval_final/score_dynamic.py, run unchanged on a scratch root (it writes its tables into its
root, which is the published results directory): the root gets the published input_manifest.json and observations/
(symlinked, read only) and predictions/<s>/dynamic_samples.json built from RUN_DIR/session_<s>/d1_trajectories.npz
-- every trajectory sample (stamp, centroid of its dynamic cluster's points = MeshObjectExtractor trajectory
position, track id, semantic), the native dynamic samples' fields. Writes OUT_DIR/{DYNAMIC*.csv, DYNAMIC_PROTOCOL.json}.
"""
from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

PUBLISHED = Path("/mnt/d/3Study/ETH/FT/results/abc_eval_final")
TOOLS = Path("/home/jixian/Desktop/FT/tools/abc_eval_final")


def main():
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(PUBLISHED / "input_manifest.json", out / "input_manifest.json")
    (out / "gt_objects").mkdir(exist_ok=True)
    if not (out / "observations").exists():
        (out / "observations").symlink_to(PUBLISHED / "observations")
    for s in "abc":
        f = run / f"session_{s}" / "d1_trajectories.npz"
        samples = []
        if f.exists():
            z = np.load(f, allow_pickle=True)
            for tr, sem, st, pos in zip(z["track"], z["semantic"], z["stamps"], z["positions"]):
                for t, p in zip(np.asarray(st).tolist(), np.asarray(pos).reshape(-1, 3).tolist()):
                    samples.append(dict(id=str(int(tr)), timestamp_ns=int(t), semantic=int(sem), position=p))
        d = out / "predictions" / s
        d.mkdir(parents=True, exist_ok=True)
        (d / "dynamic_samples.json").write_text(json.dumps(samples))
        print(f"session {s}: {len(samples)} dynamic samples", flush=True)
    sys.path.insert(0, str(TOOLS))
    common = importlib.import_module("eval_common")
    common.P = out
    common.C = json.loads((out / "input_manifest.json").read_text())
    score = importlib.import_module("score_dynamic")
    score.P, score.C = common.P, common.C
    score.run()


if __name__ == "__main__":
    main()
