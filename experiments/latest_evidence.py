"""Offline check of one existence rule on the pre-refinement B (CPU; checkpoint + raw frames of its own keyframes only).

Rule: going through the keyframes in time order, a row whose measured depth puts it ON its band (|d - z| <= 5 cm + 3 sigma_max)
is supported at that stamp; a later keyframe that SEES THROUGH it (d - z > band) after its last support ends it at that stamp.
Rows never supported are not ended by this rule. Only keyframes whose stamp lies in the row's current lifetime count (T1).
Reported: share of rows the rule would end, for the rows on the surface of every acceptance view (within 3 cm of the view's
raw depth), for the plant (by distance to its surface at keyframe 3447), the table-top items, the chairs, and all rows.
usage: latest_evidence.py PRE_CKPT [synthetic|real] [REFERENCE_PLY|-] [range]
With "label": a see-through is absence evidence for an object row (identity > 0) only where the pixel's instance label is no longer the
row's identity: a sensor that reads through a dark or thin object still labels the pixel with that object (RGB segmentation sees it), a
place the object left is labelled with what is there now. The observation is (depth, instance), the same pair the layer's object test
reads. Rows of identity <= 0 stay under the existing background rule and are not ended here.
With "trunc": a reading counts as SEEN THROUGH only beyond the TSDF's truncation distance (run.dataset_config truncation + 3 sigma):
inside +-truncation a TSDF writes a signed distance (surface noise), only beyond it free space (the TSDF's own semantics).
With "range": a see-through counts only at a range <= the farthest range at which the row was ON its band before
(the sensor has shown that it sees this surface at that range; farther, a read-through is the sensor, not absence).
With a reference cloud: the share of the ended rows that lie within 5 cm of it (on the reference = the rule would end a correct row).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from update_layer.frames import FlatSession, synthetic_session, real_session  # noqa: E402
from update_layer.run import INPUT_STEP  # noqa: E402

INT64_MAX = 9223372036854775807
SCALE = 10.0
DATASET = sys.argv[2] if len(sys.argv) > 2 else 'synthetic'
make = synthetic_session if DATASET == 'synthetic' else real_session      # the data source only
sess = {i: FlatSession(make(s), pixel_step=INPUT_STEP[DATASET]) for i, s in enumerate(('a', 'b', 'c'))
        if DATASET == 'real' or s != 'c'}
RANGE = sess[0].spec.depth_range                                          # the sensor's range (data)
ck = torch.load(sys.argv[1], mmap=True, map_location='cpu', weights_only=False)
b, t, m = ck['backend'], int(ck['prev_final']), ck['backend']['model']
end = np.minimum(np.minimum(b['death_state'].numpy(), b['death_evidence'].numpy()), b['death_prune'].numpy())
sb = b['state_birth'].numpy()
born = np.maximum(b['created'].numpy(), np.where(sb < INT64_MAX, sb, -INT64_MAX))
alive = (born <= t) & (end > t)
idx = np.flatnonzero(alive)
xyz = (m[1].detach().numpy() / SCALE)[idx].astype(np.float64)
band = 0.05 + 3 * (np.exp(m[4].detach().numpy()) / SCALE)[idx].max(1)
ident, born, end = b['identity'].numpy()[idx], born[idx], end[idx]
n = len(idx)
last_on = np.full(n, -1, np.int64)
first_thr = np.full(n, INT64_MAX, np.int64)
n_on = np.zeros(n, np.int32)
RANGE_RULE = len(sys.argv) > 4 and sys.argv[4] == 'range'
TRUNC_RULE = len(sys.argv) > 4 and sys.argv[4] == 'trunc'
LABEL_RULE = len(sys.argv) > 4 and sys.argv[4] == 'label'
from update_layer.run import dataset_config
TR = float(dataset_config(DATASET)[0].truncation)
thr_band = (TR if TRUNC_RULE else 0.05) + 3 * (np.exp(m[4].detach().numpy()) / SCALE)[idx].max(1)
end_res = np.full(n, np.nan)
far_on = np.full(n, -1.0)
n_thr = np.zeros(n, np.int32)
kfs = sorted(b['keyframes'].items(), key=lambda kv: b['kf_stamp'][kv[0]])
hom = np.c_[xyz, np.ones(n)]
for i, (kid, kf) in enumerate(kfs):
    stamp = int(b['kf_stamp'][kid])
    live = np.flatnonzero((born <= stamp) & (end > stamp))
    f = sess[int(b['kf_session'][kid])].load(int(kf['index']))
    cam = hom[live] @ np.linalg.inv(f.T_world_cam).T
    K, (h, w) = f.K, f.depth.shape
    z = cam[:, 2]
    front = z > 1e-3
    zs = np.where(front, z, 1.0)
    u = np.floor(K.fx * cam[:, 0] / zs + K.cx - K.offset + 0.5).astype(np.int64)
    v = np.floor(K.fy * cam[:, 1] / zs + K.cy - K.offset + 0.5).astype(np.int64)
    ins = np.flatnonzero(front & (u >= 0) & (u < w) & (v >= 0) & (v < h))
    d = f.depth[v[ins], u[ins]]
    lab = f.instance[v[ins], u[ins]]
    ok = np.isfinite(d) & (d > RANGE[0]) & (d < RANGE[1])
    ins, d, lab = ins[ok], d[ok], lab[ok]
    res = d - z[ins]
    rows = live[ins]
    on = rows[np.abs(res) <= band[rows]]
    through = res > thr_band[rows]
    if LABEL_RULE:
        through &= (ident[rows] > 0) & (lab != ident[rows])
    thr = rows[through]
    thr_res = res[through]
    if RANGE_RULE:
        zr = z[ins]
        on_z = zr[np.abs(res) <= band[rows]]
        np.maximum.at(far_on, on, on_z)
        keep = zr[res > thr_band[rows]] <= far_on[thr]
        thr, thr_res = thr[keep], thr_res[keep]
    last_on[on] = stamp
    first_thr[on] = INT64_MAX
    n_on[on] += 1
    n_thr[thr] += 1
    sel_new = (last_on[thr] >= 0) & (first_thr[thr] == INT64_MAX)
    newly = thr[sel_new]
    first_thr[newly] = stamp
    end_res[newly] = thr_res[sel_new]
    if i % 200 == 0:
        print(f'keyframe {i}/{len(kfs)}', flush=True)
ended = first_thr < INT64_MAX
B0 = 141e9
frame_b = lambda ns: (ns - B0) / 1e9 * 30


def report(name, sel):
    if not sel.any():
        print(f'  {name:34s} 0 rows'); return
    e = ended[sel]
    extra = f' | ended at B frame median {np.median(frame_b(first_thr[sel][e])):.0f}' if e.any() else ''
    print(f'  {name:34s} {sel.sum():8d} rows | would end {e.sum():7d} = {e.mean() * 100:5.1f} % | '
          f'never supported {np.mean(n_on[sel] == 0) * 100:5.1f} %{extra}', flush=True)


def surface(session, index):
    f = sess[session].load(index)
    K, z = f.K, f.depth
    vv, uu = np.indices(z.shape, dtype=np.float32)
    ok = np.isfinite(z) & (z > RANGE[0]) & (z < RANGE[1])
    cam = np.stack(((uu[ok] + K.offset - K.cx) / K.fx * z[ok], (vv[ok] + K.offset - K.cy) / K.fy * z[ok], z[ok], np.ones(ok.sum())), 1)
    return cam @ f.T_world_cam.T, f


print(f'{DATASET}: rows alive at the end {n}; keyframes {len(kfs)}')
report('ALL rows', np.ones(n, bool))
if DATASET == 'real':
    manifest = json.loads(Path('/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/real_row4f3e5_r9h/visual/render_manifest.json').read_text())
    views = {f'c{k}': (2, int(k)) for k in sorted(manifest['sessions']['c']['frames'], key=int)}
else:
    views = {'bathroom a2700': (0, 2700), 'bathroom a2850': (0, 2850), 'bathroom a2985': (0, 2985),
             'kitchen wall b2625': (1, 2625), 'chairs b1395': (1, 1395), 'chairs b2955': (1, 2955),
             'curtain b1830': (1, 1830), 'curtain b1845': (1, 1845), 'curtain b1860': (1, 1860),
             'b345': (1, 345), 'b2190': (1, 2190), 'b3135': (1, 3135), 'last keyframe b3447': (1, 3447)}
for name, (s_, i_) in views.items():
    pts, _ = surface(s_, i_)
    d = cKDTree(pts[:, :3]).query(xyz, k=1, distance_upper_bound=0.03)[0]
    report(f'surface of {name}', np.isfinite(d))
if DATASET == 'synthetic':
    pts, f = surface(1, 3447)
    valid = np.isfinite(f.depth) & (f.depth > RANGE[0]) & (f.depth < RANGE[1])
    plant = cKDTree(pts[f.instance[valid] == 109][:, :3]).query(xyz, k=1)[0]
    report('plant 109 <=3cm of its 3447 surface', (ident == 109) & (plant <= 0.03))
    report('plant 109 3-10cm', (ident == 109) & (plant > 0.03) & (plant <= 0.10))
    report('plant 109 >10cm (trail)', (ident == 109) & (plant > 0.10))
    report('table items 79 80 81 117 170 181', np.isin(ident, [79, 80, 81, 117, 170, 181]))
    for i, nm in ((69, 'chair_0001'), (70, 'chair_0002'), (72, 'chair_0004'), (164, 'table_0003'), (147, 'pillow_0006')):
        report(f'{nm} ({i})', ident == i)
report('background (id 0)', ident == 0)
ids, cnt = np.unique(ident[ended], return_counts=True)
top = np.argsort(-cnt)[:10]
print('  identities with the most ended rows:', ', '.join(f'id{ids[k]} {cnt[k]} of {int((ident == ids[k]).sum())}' for k in top))
if len(sys.argv) > 3 and sys.argv[3] != '-':
    from plyfile import PlyData
    v = PlyData.read(sys.argv[3])['vertex']
    ref = cKDTree(np.stack((v['x'], v['y'], v['z']), 1).astype(np.float64))
    d = ref.query(xyz, k=1)[0]
    for nm, sel_ in (('ended ON the reference', ended & (d <= 0.05)), ('ended OFF the reference (>10 cm)', ended & (d > 0.10))):
        if sel_.any():
            print(f'  {nm}: {sel_.sum()} rows | measured depth beyond the row at the ending reading: p10 {np.nanpercentile(end_res[sel_], 10):.2f} '
                  f'median {np.nanmedian(end_res[sel_]):.2f} p90 {np.nanpercentile(end_res[sel_], 90):.2f} m | share beyond the truncation {TR} m: '
                  f'{np.mean(end_res[sel_] > TR) * 100:.1f} %')
    print(f'  reference {sys.argv[3]}: all rows within 5 cm {np.mean(d <= 0.05) * 100:.1f} % | ENDED rows within 5 cm '
          f'{np.mean(d[ended] <= 0.05) * 100:.1f} %, beyond 10 cm {np.mean(d[ended] > 0.10) * 100:.1f} % ({ended.sum()} rows)')
    for k in top[:6]:
        sel = ended & (ident == ids[k])
        print(f'    id{ids[k]}: ended {sel.sum()} | within 5 cm of the reference {np.mean(d[sel] <= 0.05) * 100:.1f} %')
