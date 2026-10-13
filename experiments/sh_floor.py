"""Can a view-dependent colour model explain how the photos disagree on one surface point? (CPU, data only, no map.)

For a target frame: each pixel with measured depth is lifted to the world, projected into every STRIDE-th frame of the
same session, and kept where the source's measured depth agrees within TOL. Per pixel and channel, the source colours
are fitted as a function of the viewing direction with the real spherical harmonics of degree 0..3 (the basis and sign
convention of 3DGS sh_utils.eval_sh, direction = point - camera centre). The fit never sees the target frame; it is
evaluated at the target's own viewing direction. Degree 0 = the mean of the sources = colour_floor.py's floor.

usage: sh_floor.py OUT_DIR SESSION:INDEX [SESSION:INDEX ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from update_layer.frames import FlatSession, synthetic_session  # noqa: E402

TOL = 0.02
STRIDE = 5
RIDGE = 1e-3        # relative ridge on the normal equations (numerical only: keeps unseen directions at the mean)
GRID = 3            # every 3rd pixel in each image direction of the target (36k pixels: the MAE is a mean)
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
C0, C1 = 0.28209479177387814, 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
      1.445305721320277, -0.5900435899266435)


def basis(d):
    """Real SH basis up to degree 3 at unit directions d (N, 3), in eval_sh's order and signs."""
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
    return np.stack([
        np.full_like(x, C0),
        -C1 * y, C1 * z, -C1 * x,
        C2[0] * xy, C2[1] * yz, C2[2] * (2 * zz - xx - yy), C2[3] * xz, C2[4] * (xx - yy),
        C3[0] * y * (3 * xx - yy), C3[1] * xy * z, C3[2] * y * (4 * zz - xx - yy), C3[3] * z * (2 * zz - 3 * xx - 3 * yy),
        C3[4] * x * (4 * zz - xx - yy), C3[5] * z * (xx - yy), C3[6] * x * (xx - 3 * yy)], axis=1).astype(np.float64)


def lift(frame):
    K, z = frame.K, frame.depth
    h, w = z.shape
    v, u = np.indices((h, w), dtype=np.float32)
    ok = np.isfinite(z) & (z > 0) & (u % GRID == 0) & (v % GRID == 0)
    x = (u + K.offset - K.cx) / K.fx * z
    y = (v + K.offset - K.cy) / K.fy * z
    cam = np.stack((x[ok], y[ok], z[ok], np.ones(ok.sum(), np.float32)), axis=1)
    return (cam @ frame.T_world_cam.T)[:, :3].astype(np.float64), ok


def direction(points, frame):
    d = points - frame.T_world_cam[:3, 3]
    return d / np.linalg.norm(d, axis=1, keepdims=True)


results = {}
for target in sys.argv[2:]:
    session, index = target[0], int(target[1:])
    fs = FlatSession(synthetic_session(session), pixel_step=1)
    tgt = fs.load(index, color=True)
    world, ok = lift(tgt)
    photo = tgt.color[ok].astype(np.float64)
    n = len(world)
    ata = np.zeros((n, 16, 16))
    atc = np.zeros((n, 16, 3))
    count = np.zeros(n)
    for s in range(0, len(fs.ids), STRIDE):
        if s == index:
            continue
        src = fs.load(s, color=True)
        cam = np.c_[world, np.ones(n)] @ np.linalg.inv(src.T_world_cam).T
        K, h, w = src.K, *src.depth.shape
        z = cam[:, 2]
        front = z > 1e-3
        zs = np.where(front, z, 1.0)
        u = np.floor(K.fx * cam[:, 0] / zs + K.cx - K.offset + 0.5).astype(np.int64)
        v = np.floor(K.fy * cam[:, 1] / zs + K.cy - K.offset + 0.5).astype(np.int64)
        hit = np.flatnonzero(front & (u >= 0) & (u < w) & (v >= 0) & (v < h))
        measured = src.depth[v[hit], u[hit]]
        hit = hit[np.isfinite(measured) & (np.abs(measured - z[hit]) <= TOL)]
        y = basis(direction(world[hit], src))
        ata[hit] += y[:, :, None] * y[:, None, :]
        atc[hit] += y[:, :, None] * src.color[v[hit], u[hit]].astype(np.float64)[:, None, :]
        count[hit] += 1
    has = count >= 2
    yt = basis(direction(world, tgt))
    rec = dict(covered=float(has.mean()), sources_median=float(np.median(count[has])))
    for degree in range(4):
        m = (degree + 1) ** 2
        a = ata[has][:, :m, :m]
        a = a + RIDGE * np.trace(a, axis1=1, axis2=2)[:, None, None] / m * np.eye(m)
        coef = np.linalg.solve(a, atc[has][:, :m, :])
        pred = np.einsum('pk,pkc->pc', yt[has][:, :m], coef)
        rec[f'degree_{degree}_MAE_255'] = float(np.abs(np.clip(pred, 0, 255) - photo[has]).mean())
    results[target] = rec
    print(target, json.dumps(rec), flush=True)
(OUT / 'sh_floor.json').write_text(json.dumps(results, indent=2) + '\n')
