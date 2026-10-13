"""How much the photos themselves disagree on the colour of one surface point (CPU, data only, no model).

For a target frame: every pixel with measured depth is lifted to the world with the target's depth and pose, projected into
source frames of the same session, and kept where the source's measured depth agrees within TOL (same surface, visible).
The fused colour of a pixel = mean of its sources' colours. MAE(target photo, fused colour) is what a perfect
view-independent fused model of those views would score on the target photo. Windows: sources within +-W seconds.

usage: colour_floor.py OUT_DIR SESSION:INDEX [SESSION:INDEX ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from update_layer.frames import FlatSession, synthetic_session  # noqa: E402

TOL = 0.02          # m: same surface (synthetic depth is rendered, not sensed)
STRIDE = 5          # every 5th frame of the session as a source (30 Hz -> 6 Hz)
WINDOWS = (0.5, 2.0, 20.0, None)   # seconds around the target; None = the whole session
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)


def lift(frame):
    K, z = frame.K, frame.depth
    h, w = z.shape
    v, u = np.indices((h, w), dtype=np.float32)
    ok = np.isfinite(z) & (z > 0)
    x = (u + K.offset - K.cx) / K.fx * z
    y = (v + K.offset - K.cy) / K.fy * z
    cam = np.stack((x[ok], y[ok], z[ok], np.ones(ok.sum(), np.float32)), axis=1)
    return (cam @ frame.T_world_cam.T)[:, :3], ok


results = {}
for target in sys.argv[2:]:
    session, index = target[0], int(target[1:])
    fs = FlatSession(synthetic_session(session), pixel_step=1)
    tgt = fs.load(index, color=True)
    world, ok = lift(tgt)
    photo = tgt.color[ok].astype(np.float64)
    t0 = fs.stamp_ns(index)
    sums = {w: np.zeros_like(photo) for w in WINDOWS}
    counts = {w: np.zeros(len(photo)) for w in WINDOWS}
    for s in range(0, len(fs.ids), STRIDE):
        if s == index:
            continue
        dt = abs(fs.stamp_ns(s) - t0) / 1e9
        if not any(w is None or dt <= w for w in WINDOWS):
            continue
        src = fs.load(s, color=True)
        cam = np.c_[world, np.ones(len(world))] @ np.linalg.inv(src.T_world_cam).T
        K, h, w = src.K, *src.depth.shape
        z = cam[:, 2]
        front = z > 1e-3
        zs = np.where(front, z, 1.0)
        u = np.floor(K.fx * cam[:, 0] / zs + K.cx - K.offset + 0.5).astype(np.int64)
        v = np.floor(K.fy * cam[:, 1] / zs + K.cy - K.offset + 0.5).astype(np.int64)
        inside = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        hit = np.flatnonzero(inside)
        measured = src.depth[v[hit], u[hit]]
        same = np.isfinite(measured) & (np.abs(measured - z[hit]) <= TOL)
        hit = hit[same]
        colour = src.color[v[hit], u[hit]].astype(np.float64)
        for win in WINDOWS:
            if win is None or dt <= win:
                sums[win][hit] += colour
                counts[win][hit] += 1
    rec = dict(pixels_with_depth=int(ok.sum()), pixels_total=int(ok.size))
    for win in WINDOWS:
        n = counts[win]
        has = n >= 2
        fused = sums[win][has] / n[has, None]
        rec[f'window_{win}s'] = dict(covered=float(has.mean()), sources_median=float(np.median(n[has])) if has.any() else 0,
                                     floor_MAE_255=float(np.abs(photo[has] - fused).mean()) if has.any() else None)
        mask = np.zeros(ok.shape, bool)
        mask.flat[np.flatnonzero(ok.ravel())[has]] = True
        np.save(OUT / f'covered_{target}_{win}.npy', mask)
    results[target] = rec
    print(target, json.dumps(rec), flush=True)
(OUT / 'colour_floor.json').write_text(json.dumps(results, indent=2) + '\n')
