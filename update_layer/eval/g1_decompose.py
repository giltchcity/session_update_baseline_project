"""Where G1@5 is lost: the precision errors and recall misses of one map, by kind (real data).

  python -m update_layer.eval.g1_decompose STAGE PRED.ply OUT.json [--ref REF.ply] [--crop CROP.json]

Same samples, crop and distances as eval/geometry_g1.py (area-sampled at 2 cm, oriented room crop, Open3D raycast
distance), so the shares add up to 1 - P and 1 - R of G1@5.
Precision errors (prediction samples farther than 5 cm from the reference), by the closest reference point q and its
face normal n, oriented towards the nearest camera of the session (the free side; 1 fps frames):
  change      within 10 cm of an old or new GT state of an event of the session (results/abc_eval_final_baseline:
              events.json, gt_objects/<state>.npy)
  behind      s = (p - q).n < 0: behind the observed surface (expected-depth bias, thickness, interior)
  in_front    s > 0 and the offset is mostly along the normal (|s| > half the distance): floaters in free space
  lateral     the rest: off the reference surface sideways (beyond its extent: surface the reference lacks, or
              over-extended borders)
Recall misses (reference samples farther than 5 cm from the prediction):
  change      as above
  unobserved  no 1 fps frame of the session measured it on its surface (|depth - z| < 5 cm): not reconstructible
  observed    measured by a frame, missing from the map
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, "/home/jixian/Desktop/FT/tools/abc_eval_v2")
sys.path.insert(0, "/home/jixian/Desktop/FT/eval")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import geometry_score as G  # noqa: E402
from geometry_g1 import rotate  # noqa: E402

FT = Path("/home/jixian/Desktop/FT")
BASE = FT / "results/abc_eval_final_baseline"


def change_points(stage: str) -> np.ndarray:
    ev = json.loads((BASE / "events.json").read_text())
    pts = []
    for e in ev:
        if e["session"] != stage.upper():
            continue
        for k in ("old", "new"):
            st = e[k]
            if isinstance(st, str):
                import ast
                st = ast.literal_eval(st)
            if not st:                               # appeared / disappeared: one side has no state
                continue
            f = BASE / "gt_objects" / f"{st['id']}.npy"
            if f.exists():
                pts.append(np.load(f)[:, :3])
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def session_frames(stage: str, step: int = 30):
    from update_layer.frames import FlatSession, real_session
    fs = FlatSession(real_session(stage.lower()), pixel_step=2)
    return [fs.load(i) for i in range(0, len(fs.ids), step)]


def observed(points: np.ndarray, frames, tol: float = 0.05) -> np.ndarray:
    seen = np.zeros(len(points), bool)
    for f in frames:
        T = np.linalg.inv(f.T_world_cam)
        cam = points @ T[:3, :3].T + T[:3, 3]
        z = cam[:, 2]
        ok = z > 0.1
        u = np.full(len(points), -1)
        v = np.full(len(points), -1)
        u[ok] = np.floor(f.K.fx * cam[ok, 0] / z[ok] + f.K.cx - f.K.offset + 0.5).astype(int)
        v[ok] = np.floor(f.K.fy * cam[ok, 1] / z[ok] + f.K.cy - f.K.offset + 0.5).astype(int)
        ok &= (u >= 0) & (u < f.K.width) & (v >= 0) & (v < f.K.height)
        d = np.full(len(points), np.nan)
        d[ok] = f.depth[v[ok], u[ok]]
        seen |= ok & np.isfinite(d) & (np.abs(d - z) < tol)
    return seen


def main():
    args = sys.argv[1:]
    stage, pred_path, out = args[0].lower(), args[1], Path(args[2])
    S = stage.upper()
    ref_path = args[args.index("--ref") + 1] if "--ref" in args else str(FT / f"results/abc_eval_v2/geometry/{S}_reference_1cm.ply")
    crop_path = args[args.index("--crop") + 1] if "--crop" in args else str(FT / f"eval/crops/real_{S}.json")
    crop = json.loads(Path(crop_path).read_text())
    lo, hi, yaw = np.array(crop["lo"]), np.array(crop["hi"]), crop["yaw_deg"]
    ref = o3d.io.read_triangle_mesh(ref_path)
    pred = o3d.io.read_triangle_mesh(pred_path)
    rp, _ = G.surface_sample(ref)
    pp, meta = G.surface_sample(pred)
    keep_r = ~((rotate(rp, yaw) < lo) | (rotate(rp, yaw) > hi)).any(1)
    keep_p = ~((rotate(pp, yaw) < lo) | (rotate(pp, yaw) > hi)).any(1)
    pp, rp = pp[keep_p].astype(np.float64), rp[keep_r].astype(np.float64)
    dp = G.distances(pp, ref)
    dr = G.distances(rp, pred)
    frames = session_frames(S)
    centres = np.stack([f.T_world_cam[:3, 3] for f in frames])
    from scipy.spatial import cKDTree
    ctree = cKDTree(centres)
    chg = change_points(S)
    chtree = cKDTree(chg) if len(chg) else None

    # precision errors
    err = dp > 0.05
    pe = pp[err]
    ref.compute_triangle_normals()
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(ref))
    cl = scene.compute_closest_points(o3d.core.Tensor(pe.astype(np.float32)))
    q = cl["points"].numpy().astype(np.float64)
    n = np.asarray(ref.triangle_normals)[cl["primitive_ids"].numpy()]
    _, ci = ctree.query(q)
    flip = ((centres[ci] - q) * n).sum(1) < 0
    n[flip] *= -1
    off = pe - q
    s = (off * n).sum(1)
    dist = np.linalg.norm(off, axis=1)
    in_change = np.zeros(len(pe), bool)
    if chtree is not None:
        dc, _ = chtree.query(pe, distance_upper_bound=0.10)
        in_change = np.isfinite(dc)
    behind = ~in_change & (s < 0) & (np.abs(s) > 0.5 * dist)
    front = ~in_change & (s > 0) & (np.abs(s) > 0.5 * dist)
    lateral = ~in_change & ~behind & ~front
    nP = len(pp)
    prec = dict(samples=nP, P_05cm=round(1 - err.mean(), 4), error_share=round(err.mean(), 4),
                change=round(in_change.sum() / nP, 4), behind=round(behind.sum() / nP, 4),
                in_front=round(front.sum() / nP, 4), lateral=round(lateral.sum() / nP, 4))
    for name, m in (("behind", behind), ("in_front", front), ("lateral", lateral)):
        if m.any():
            prec[f"{name}_offset_m_median"] = round(float(np.median(dist[m])), 3)

    # recall misses
    miss = dr > 0.05
    rm = rp[miss]
    r_change = np.zeros(len(rm), bool)
    if chtree is not None:
        dc, _ = chtree.query(rm, distance_upper_bound=0.10)
        r_change = np.isfinite(dc)
    seen = observed(rm, frames)
    nR = len(rp)
    rec = dict(samples=nR, R_05cm=round(1 - miss.mean(), 4), miss_share=round(miss.mean(), 4),
               change=round(r_change.sum() / nR, 4), unobserved=round((~r_change & ~seen).sum() / nR, 4),
               observed=round((~r_change & seen).sum() / nR, 4))
    res = dict(stage=S, prediction=pred_path, reference=ref_path, crop=crop_path, pred_area_m2=round(meta["area_m2"], 1),
               precision=prec, recall=rec)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
