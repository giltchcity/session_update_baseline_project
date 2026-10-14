"""What a session's keyframes measure of the session-start map, as E1 (evsnap 86bf77e) measures it (GPU, read-only).
The accepted A loaded as the chain loads it (prior_from_state + kf_strip.restore, then start_session of B: the snapshot is built); for each
given frame of B: the keyframe sample exactly as GameBackend._sample makes it, the snapshot's first echo (probe_render_fe), and, as a check
of the snapshot class, the first echo of the live model restricted to the rows alive at B's start (identical before any training).
Per frame: shares of the L2 classification (on |delta| <= tol, seen through delta > tol, occluded delta < -tol) by range bin, and PNGs of
the raw RGB, B's depth, the first-echo depth (same colour scale) and the classification (green on, red seen through, blue occluded).
usage: snapshot_first_echo.py CODE_DIR A_CKPT DATASET SESSION OUT_DIR FRAME [FRAME ...]
"""
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import torch

CODE, A_CKPT, DS, SESS, OUT = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], Path(sys.argv[5])
FRAMES = [int(x) for x in sys.argv[6:]]
sys.path.insert(0, str(CODE))
from update_layer.run import dataset_config  # noqa: E402
from update_layer.backends.game import game as G  # noqa: E402
from update_layer.eval.kf_strip import restore  # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
cfg, info, specs = dataset_config(DS)
spec = next(sp for sp in specs if sp.name.endswith('_' + SESS))
be = G.GameBackend(info, own_update=False)
ck = torch.load(A_CKPT, map_location='cpu', weights_only=False)
g = be.prior_from_state(restore(dict(backend=ck['backend']), DS)['backend'])
be.start_session(spec, g)
snap = be._ev_snap
print(f'snapshot rows {len(snap.opacity)} of {len(g.uid)}; tolerance {be.tolerance} m; scale {be.scale}', flush=True)


def colour_depth(d, lo, hi):
    x = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    img = cv.applyColorMap((255 * x).astype(np.uint8), cv.COLORMAP_TURBO)
    img[d <= 0] = 0
    return img


tol = be.tolerance
for i in FRAMES:
    sample, instance, dyn = be._sample(i)
    kf = {"color": G.gu.np2torch(sample["color"], device="cuda").permute(2, 0, 1) / 255.0,
          "depth": G.gu.np2torch(sample["depth"], device="cuda"),
          "pose": G.gu.np2torch(sample["pose"], device="cuda"), "intrinsics": sample["intrinsics"]}
    view = G.gu.flashsplat_cam(kf["color"], kf["depth"], None, kf["intrinsics"], kf["pose"].cpu(), None)
    with torch.no_grad():
        fe = G.probe_render_fe(view, snap)
        gm = g.gaussian_model
        gm.alive = g.alive_at(be.session_start)
        fe_live = G.probe_render_fe(view, gm)
        gm.alive = None
    h, w = sample["depth"].shape
    med = fe["median"].reshape(h, w).cpu().numpy() / be.scale
    mi = fe["median_index"].reshape(h, w).cpu().numpy()
    med_live = fe_live["median"].reshape(h, w).cpu().numpy() / be.scale
    d = sample["depth"] / be.scale
    ok = (mi >= 0) & (d > 0)
    delta = np.where(ok, d - med, np.nan)
    same = np.abs(med - med_live)[ok]
    on = ok & (np.abs(delta) <= tol)
    past = ok & (delta > tol)
    occ = ok & (delta < -tol)
    n = max(int(ok.sum()), 1)
    print(f'\n== {DS} {SESS} frame {i}: valid depth {int((d > 0).sum())} px, with a first echo {int(ok.sum())} px; '
          f'snapshot vs live first echo max |diff| {float(same.max()) if len(same) else 0:.2e} m', flush=True)
    print(f'   on {100 * on.sum() / n:.1f} %  seen through {100 * past.sum() / n:.1f} %  occluded {100 * occ.sum() / n:.1f} %;  '
          f'B depth {np.percentile(d[ok], 5):.2f}-{np.percentile(d[ok], 95):.2f} m, first echo {np.percentile(med[ok], 5):.2f}-{np.percentile(med[ok], 95):.2f} m')
    for lo_, hi_ in ((0, 0.25), (0.25, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 5), (5, 15)):
        m = ok & (d >= lo_) & (d < hi_)
        k = int(m.sum())
        if k:
            print(f'     B depth {lo_:4.2f}-{hi_:4.2f} m: {k:6d} px  on {100 * (on & m).sum() / k:5.1f} %  seen through {100 * (past & m).sum() / k:5.1f} %  '
                  f'occluded {100 * (occ & m).sum() / k:5.1f} %  median delta {100 * np.nanmedian(delta[m]):7.2f} cm  median first echo {np.median(med[m]):.2f} m')
    lo, hi = np.percentile(d[d > 0], 1), np.percentile(d[d > 0], 99)
    cls = np.zeros((h, w, 3), np.uint8)
    cls[on] = (0, 200, 0); cls[past] = (0, 0, 230); cls[occ] = (230, 80, 0)
    rgb = cv.cvtColor(sample["color"], cv.COLOR_RGB2BGR)
    row = np.concatenate([rgb, colour_depth(d, lo, hi), colour_depth(np.where(mi >= 0, med, 0), lo, hi), cls], 1)
    cv.imwrite(str(OUT / f'{DS}_{SESS}_{i:05d}.png'), row)
print('\nSNAPSHOT_FIRST_ECHO_DONE')
