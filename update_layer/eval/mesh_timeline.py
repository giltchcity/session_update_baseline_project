"""A two-snapshot timeline from two exported meshes, for scorers that read only the map at the session start and at the
session end (the ghost cohort: cleanup_evaluate_h.py 'initial supported' at the start, the final row at the session
end), e.g. to score the ghost of median-depth TSDF exports with the unchanged harness.

  python -m update_layer.eval.mesh_timeline PREV_MESH.ply T_START CUR_MESH.ply T_END OUT.pkl

The map of [T_START, T_END) is PREV_MESH (the previous session's final export), the map at T_END is CUR_MESH; every
vertex is background (no object nodes: change and object scores of this timeline mean nothing).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from update_layer.eval.scene import EvaluationScene  # noqa: E402
from update_layer.eval.scenelist import SceneListTimeline  # noqa: E402


def scene(path: str, t: int) -> EvaluationScene:
    m = o3d.io.read_triangle_mesh(path)
    m.compute_vertex_normals()
    P = np.asarray(m.vertices, np.float32)
    N = np.asarray(m.vertex_normals, np.float32)
    return EvaluationScene(t, P, np.zeros(len(P), np.uint32), [], N)


def main():
    prev, t0, cur, t1, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
    tl = SceneListTimeline([t0, t1], [scene(prev, t0), scene(cur, t1)])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(tl, f)
    print(f"{out}: {t0} {len(tl.scenes[0].background)} vertices, {t1} {len(tl.scenes[1].background)} vertices")


if __name__ == "__main__":
    main()
