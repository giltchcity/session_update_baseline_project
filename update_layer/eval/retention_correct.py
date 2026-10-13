"""Retention of the previous session's correct surface, real data (the synthetic STABLE_ACCUMULATION analogue).

  python -m update_layer.eval.retention_correct PREV CUR PREV_MAP.ply OUT.json LABEL=CUR_MAP.ply [LABEL=CUR_MAP.ply ...]

Synthetic (eval/synthetic_base_v38/harness/evaluate_cross_session_geometry.py): the stable reference is the A-observed
GT surface of the structure and of the objects that did not move between A and B; retained_fraction_of_A = of the
stable reference points the A map covers (within th), the share the B map still covers. Real data have no object
poses, so here the stable reference = the previous session's G1 reference vertices (results/abc_eval_v2/geometry/
<S>_reference_1cm.ply, 1 cm) inside both sessions' crops that lie within 5 cm of the current session's reference (the
surface that is there in both sessions). Coverage = distance to the map's surface (Open3D raycast distance to a mesh, as
geometry_g1; point maps: nearest point) <= th, th = 5 / 10 / 20 cm.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, "/home/jixian/Desktop/FT/tools/abc_eval_v2")
sys.path.insert(0, "/home/jixian/Desktop/FT/eval")
import geometry_score as G  # noqa: E402
from geometry_g1 import rotate  # noqa: E402

FT = Path("/home/jixian/Desktop/FT")


def inside(p: np.ndarray, stage: str) -> np.ndarray:
    c = json.loads((FT / f"eval/crops/real_{stage}.json").read_text())
    r = rotate(p, c["yaw_deg"])
    return ~((r < np.array(c["lo"])) | (r > np.array(c["hi"]))).any(1)


def distance(points: np.ndarray, path: str) -> np.ndarray:
    m = o3d.io.read_triangle_mesh(path)
    if len(m.triangles):
        return G.distances(points, m)
    P = np.asarray(o3d.io.read_point_cloud(path).points)
    return cKDTree(P).query(points)[0]


def main():
    a = sys.argv[1:]
    prev, cur, prev_map, out = a[0].upper(), a[1].upper(), a[2], Path(a[3])
    maps = [x.split("=", 1) for x in a[4:]]
    rp = np.asarray(o3d.io.read_triangle_mesh(str(FT / f"results/abc_eval_v2/geometry/{prev}_reference_1cm.ply")).vertices)
    rc = np.asarray(o3d.io.read_triangle_mesh(str(FT / f"results/abc_eval_v2/geometry/{cur}_reference_1cm.ply")).vertices)
    rp = rp[inside(rp, prev) & inside(rp, cur)].astype(np.float64)
    stable = rp[cKDTree(rc).query(rp)[0] <= 0.05]
    dp = distance(stable, prev_map)
    res = dict(prev=prev, cur=cur, prev_map=prev_map, stable_reference_points=int(len(stable)),
               stable_share_of_prev_reference=round(len(stable) / max(len(rp), 1), 4), maps={})
    for label, path in maps:
        dc = distance(stable, path)
        r = {}
        for th in (0.05, 0.10, 0.20):
            ca, cb = dp <= th, dc <= th
            r[f"{int(th * 100)}cm"] = dict(coverage_prev=round(float(ca.mean()), 4), coverage_cur=round(float(cb.mean()), 4),
                                         retained_fraction_of_prev=round(float(cb[ca].mean()), 4) if ca.any() else None,
                                         lost=int((ca & ~cb).sum()), new=int((cb & ~ca).sum()))
        res["maps"][label] = dict(path=path, **r)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
