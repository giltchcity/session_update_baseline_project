"""G1: plain two-mesh F-score inside a room-oriented crop box (the primary geometry metric).

Definition (same for every method and stage of a dataset):
  * the crop box is defined ONCE per stage: yaw = the rotation (0..90 deg, 0.5 deg steps)
    that minimises the top-down bounding area of the reference mesh vertices (1..99 pct);
    the box = [p2, p98] percentile bounds of the rotated reference vertices in x/y/z;
  * prediction and reference are both surface-sampled by area (geometry_score.surface_sample),
    rotated into the box frame and cropped with the same box;
  * distances are exact closest-triangle distances in both directions;
  * P = share of cropped prediction samples within t of the reference, R = share of cropped
    reference samples within t of the prediction, t = 5/10/20 cm.
This is the oriented crop recovered from the 2026-09-21 analysis (recovered_obb_crop.py, p2 row),
extended with the 10/20 cm thresholds, the box parameters persisted in a json.

usage: geometry_g1.py --calibrate DATASET STAGE REF.ply OUT.json
       geometry_g1.py score DATASET STAGE REF.ply PRED.ply CROP.json OUT.json
"""
import json, sys
from pathlib import Path
import numpy as np
import open3d as o3d
sys.path.insert(0, str(Path(__file__).resolve().parent))
import geometry_score as G

PCT = 2.0


def rot2(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def rotate(points, deg):
    out = np.array(points, dtype=np.float64, copy=True)
    out[:, [0, 2]] = out[:, [0, 2]] @ rot2(deg).T
    return out


def best_yaw(xz):
    best, angle = None, 0.
    for deg in np.arange(0, 90, .5):
        p = xz @ rot2(deg).T
        area = float(np.prod(np.percentile(p, 99, 0) - np.percentile(p, 1, 0)))
        if best is None or area < best:
            best, angle = area, float(deg)
    return angle


def calibrate(dataset, stage, ref_path, out):
    ref = o3d.io.read_triangle_mesh(str(ref_path))
    v = np.asarray(ref.vertices)
    yaw = best_yaw(v[:, [0, 2]])
    r = rotate(v, yaw)
    crop = dict(dataset=dataset, stage=stage, yaw_deg=yaw, percentile=PCT,
                lo=np.percentile(r, PCT, 0).tolist(), hi=np.percentile(r, 100 - PCT, 0).tolist(),
                reference=str(ref_path), axes='rotation about world y applied to (x, z); bounds in the rotated frame')
    Path(out).write_text(json.dumps(crop, indent=2))
    print(json.dumps(crop))


def score(dataset, stage, ref_path, pred_path, crop_path, out):
    crop = json.loads(Path(crop_path).read_text())
    lo, hi, yaw = np.array(crop['lo']), np.array(crop['hi']), crop['yaw_deg']
    ref = o3d.io.read_triangle_mesh(str(ref_path))
    pred = o3d.io.read_triangle_mesh(str(pred_path))
    rp, _ = G.surface_sample(ref)
    pp, meta = G.surface_sample(pred)
    keep_r = ~((rotate(rp, yaw) < lo) | (rotate(rp, yaw) > hi)).any(1)
    keep_p = ~((rotate(pp, yaw) < lo) | (rotate(pp, yaw) > hi)).any(1)
    dp = G.distances(pp[keep_p], ref)
    dr = G.distances(rp[keep_r], pred)
    row = dict(protocol='G1_oriented_room_crop_two_mesh', dataset=dataset, stage=stage, prediction=str(pred_path),
               reference=str(ref_path), crop=crop_path, pred_area_m2=round(float(meta['area_m2']), 1),
               pred_samples_inside=float(round(keep_p.mean(), 4)), ref_samples_inside=float(round(keep_r.mean(), 4)))
    for t, tag in ((.05, '05'), (.10, '10'), (.20, '20')):
        p, r = float((dp <= t).mean()), float((dr <= t).mean())
        row.update({'P_%scm' % tag: round(p, 4), 'R_%scm' % tag: round(r, 4),
                    'F1_%scm' % tag: round(2 * p * r / (p + r) if p + r else 0., 4)})
    Path(out).write_text(json.dumps(row, indent=2))
    print(json.dumps(row))


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == '--calibrate':
        calibrate(a[1], a[2], a[3], a[4])
    elif a[0] == 'score':
        score(a[1], a[2], a[3], a[4], a[5], a[6])
    else:
        sys.exit(__doc__)
