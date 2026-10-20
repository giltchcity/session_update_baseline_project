"""G1 on Flat. Flat's ground truth is a point cloud per run (panoptic_mapping's flat_dataset: flat_1_gt_10000.ply for run1,
flat_2_gt_10000.ply for run2, the flat's surface sampled at 10000 points per m2), not a mesh; this is the G1 definition
(FT/eval/geometry_g1.py) with that reference:
  * the prediction is surface-sampled by area (geometry_score.surface_sample: one sample per 2 cm surface voxel) and the
    reference points are reduced the same way (per 2 cm voxel the point nearest the cell centre);
  * P = share of prediction samples within t of the reference (its nearest point: the points, about 1 cm apart, stand
    for the surface), R = share of reference samples within t of the prediction (closest triangle), t = 5 / 10 / 20 cm;
  * no crop: the reference holds the flat's surface only (G1's crop removes what lies outside the scanned room).

  python -m update_layer.eval.flat_geometry PRED.ply REF.ply OUT.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, "/home/jixian/Desktop/FT/tools/abc_eval_v2")
import geometry_score as G  # noqa: E402

SPACING = 0.02                    # geometry_score.surface_sample's voxel


def voxel_union(points: np.ndarray, spacing: float = SPACING) -> np.ndarray:
    """geometry_score.surface_sample's union: per voxel the point nearest the cell centre."""
    keys = np.floor(points / spacing).astype(np.int32)
    error = np.sum((points - (keys + .5) * spacing) ** 2, axis=1)
    order = np.lexsort((error, keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    first = np.r_[True, np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)]
    return points[order[first]]


def main() -> None:
    pred_path, ref_path, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    ref = np.asarray(o3d.io.read_point_cloud(ref_path).points, dtype=np.float64)
    pred = o3d.io.read_triangle_mesh(pred_path)
    pp, meta = G.surface_sample(pred)
    rp = voxel_union(ref)
    dp = cKDTree(ref).query(pp, k=1, workers=4)[0] if len(pp) else np.empty(0)
    dr = G.distances(rp.astype(np.float32), pred)
    row = dict(protocol="G1_flat_point_reference", prediction=pred_path, reference=ref_path,
               pred_area_m2=round(float(meta["area_m2"]), 1), pred_samples=int(len(pp)), ref_points=int(len(ref)),
               ref_samples=int(len(rp)))
    for t, tag in ((.05, "05"), (.10, "10"), (.20, "20")):
        p, r = float((dp <= t).mean()) if len(dp) else 0., float((dr <= t).mean())
        row.update({f"P_{tag}cm": round(p, 4), f"R_{tag}cm": round(r, 4),
                    f"F1_{tag}cm": round(2 * p * r / (p + r) if p + r else 0., 4)})
    out.write_text(json.dumps(row, indent=2))
    print(json.dumps(row))


if __name__ == "__main__":
    main()
