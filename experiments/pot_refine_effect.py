"""What the refinement did to the plant (id 109) vs a static object (id 72), from checkpoints and raw frames only (CPU).

Part 1 (refined checkpoints): rows alive at the end, split into ORIGINAL (uid present before the refinement) and BORN in the
refinement (densification), and by distance to the plant surface of the last keyframe 3447: counts, opacity, axis ratio, size.
Part 2 (pre-refinement checkpoint + the raw frames of its keyframes): for every row and every keyframe whose stamp lies in the
row's lifetime, the keyframe's measured depth at the row's pixel: ON the row's band (|d - z| <= 5 cm + 3 sigma_max), SEEN THROUGH
(d - z > band), or neither. Same classification as the refinement's [RT] rule.
usage: pot_refine_effect.py PRE_CKPT LABEL=REFINED_CKPT ...
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from update_layer.frames import FlatSession, synthetic_session  # noqa: E402

INT64_MAX = 9223372036854775807
SCALE = 10.0
sess = {0: FlatSession(synthetic_session('a'), pixel_step=1), 1: FlatSession(synthetic_session('b'), pixel_step=1)}


def lift(f, mask):
    K, z = f.K, f.depth
    v, u = np.indices(z.shape, dtype=np.float32)
    ok = mask & np.isfinite(z) & (z > 0)
    cam = np.stack(((u[ok] + K.offset - K.cx) / K.fx * z[ok], (v[ok] + K.offset - K.cy) / K.fy * z[ok], z[ok], np.ones(ok.sum())), 1)
    return (cam @ f.T_world_cam.T)[:, :3]


f3447 = sess[1].load(3447)
tree = cKDTree(lift(f3447, f3447.instance == 109))


def rows(path):
    ck = torch.load(path, mmap=True, map_location='cpu', weights_only=False)
    b, t, m = ck['backend'], int(ck['prev_final']), ck['backend']['model']
    end = np.minimum(np.minimum(b['death_state'].numpy(), b['death_evidence'].numpy()), b['death_prune'].numpy())
    sb = b['state_birth'].numpy()
    born = np.maximum(b['created'].numpy(), np.where(sb < INT64_MAX, sb, -INT64_MAX))
    return dict(b=b, t=t, xyz=m[1].detach().numpy() / SCALE, s=np.exp(m[4].detach().numpy()) / SCALE,
                o=1 / (1 + np.exp(-m[6].detach().numpy()[:, 0])), uid=b['uid'].numpy(), ident=b['identity'].numpy(),
                born=born, end=end, alive=(born <= t) & (end > t))


pre = rows(sys.argv[1])
pre_uids = pre['uid']
print('PART 1  rows alive at the end; dist = distance to the plant surface seen at keyframe 3447')
for arg in sys.argv[2:]:
    label, path = arg.split('=', 1)
    r = rows(path)
    for ident in (109, 72):
        sel = r['alive'] & (r['ident'] == ident)
        d = tree.query(r['xyz'][sel], k=1)[0]
        orig = np.isin(r['uid'][sel], pre_uids)
        for name, part in (('original', orig), ('born in refinement', ~orig)):
            for gname, g in ((('<=3cm', d <= 0.03), ('3-10cm', (d > 0.03) & (d <= 0.10)), ('>10cm', d > 0.10)) if ident == 109 else (('all', d >= 0),)):
                q = part & g
                if not q.any():
                    continue
                s, o = r['s'][sel][q], r['o'][sel][q]
                print(f'  {label} id{ident} {name:18s} {gname:6s} {q.sum():6d} rows | opacity median {np.median(o):.2f} | '
                      f'axis ratio median {np.median(s.max(1) / s.min(1)):.2f} | largest axis median {np.median(s.max(1)) * 1000:.1f} mm')
print('PART 2  pre-refinement rows vs the keyframes that render them (stamp inside the row lifetime)')
b = pre['b']
kfs = sorted(b['keyframes'].items(), key=lambda kv: b['kf_stamp'][kv[0]])
for ident in (109, 72):
    sel = np.flatnonzero(pre['alive'] & (pre['ident'] == ident))
    xyz, band = pre['xyz'][sel], 0.05 + 3 * pre['s'][sel].max(1)
    on = np.zeros(len(sel), int); through = np.zeros(len(sel), int); nkf = 0
    for kid, kf in kfs:
        stamp = b['kf_stamp'][kid]
        live = (pre['born'][sel] <= stamp) & (pre['end'][sel] > stamp)
        if not live.any() or b['kf_session'][kid] != 1:
            continue
        nkf += 1
        f = sess[1].load(int(kf['index']))
        cam = np.c_[xyz[live], np.ones(live.sum())] @ np.linalg.inv(f.T_world_cam).T
        K, (h, w) = f.K, f.depth.shape
        z = cam[:, 2]
        zs = np.where(z > 1e-3, z, 1.0)
        u = np.floor(K.fx * cam[:, 0] / zs + K.cx - K.offset + 0.5).astype(int)
        v = np.floor(K.fy * cam[:, 1] / zs + K.cy - K.offset + 0.5).astype(int)
        inside = (z > 1e-3) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        d = np.full(len(z), np.nan)
        d[inside] = f.depth[v[inside], u[inside]]
        res = d - z
        idx = np.flatnonzero(live)
        on[idx] += np.nan_to_num(np.abs(res) <= band[live], nan=0).astype(int)
        through[idx] += np.nan_to_num(res > band[live], nan=0).astype(int)
    dist = tree.query(xyz, k=1)[0]
    groups = (('<=3cm', dist <= 0.03), ('3-10cm', (dist > 0.03) & (dist <= 0.10)), ('>10cm', dist > 0.10)) if ident == 109 else (('all', dist >= 0),)
    print(f'  id{ident}: {len(sel)} rows, {nkf} B keyframes render at least one of them')
    for gname, g in groups:
        print(f'    {gname:6s} {g.sum():6d} rows | keyframes ON median {np.median(on[g]):.0f} | SEEN THROUGH median {np.median(through[g]):.0f} | '
              f'rows seen through more often than on {np.mean(through[g] > on[g]) * 100:.1f} % | rows with both {np.mean((through[g] > 0) & (on[g] > 0)) * 100:.1f} %')
