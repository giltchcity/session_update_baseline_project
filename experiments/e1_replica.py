"""E1 replica (GPU renders, then CPU distances; read-only).

The E1 decision of evsnap 86bf77e is run offline on one B session's keyframes, without the map's training, for two counting rules on
the same measurements. The ended rows are then checked against the scorer's reference.

The measurement
  Every B keyframe of an existing B run (its stamps, matched to the session's frames) is sampled exactly as GameBackend._sample does.
  The session-start snapshot (GameBackend._ev_snapshot, built by start_session) is rendered without the rows ended so far.
  delta = measured depth - first-echo depth, in metres.

Rule E1 (86bf77e)
  against = delta > tol (seen through); for = |delta| <= tol; delta < -tol is no evidence.
Rule E1f (the L2 footprint, classify_frames): per row and keyframe one verdict over the row's first-echo pixels:
  hit (on its surface) when ANY of its pixels has |delta| <= tol; seen through when EVERY one of its pixels is valid and has
  delta > tol; otherwise no verdict.
Rule E2 (the TSDF band)
  against = delta > tol, or -trunc <= delta < -tol: the present measures a surface nearer, inside the TSDF truncation band, where
  TSDF fusion would replace the row's surface by the present one.
  for = |delta| <= tol; delta < -trunc is no evidence.
  tol = EvidenceConfig.surface_match_tolerance; trunc = the layer's TSDF truncation of the dataset (run.dataset_config).

Rounds
  Every layer round (cfg.round_s of session time) runs the same row EM and LOG99 commit as GameBackend._ev_round.
  The rows the layer ended in the B run (its checkpoint's death_state, matched by uid) are hidden from the round they ended in, as
  the online masks do.

Check
  Each ended row's session-start position is compared with the scorer's reference:
    real       B_reference_1cm.ply
    synthetic  the exact GT of B's last frame (cvpr_eval_common.worldmesh, every render-visible entity)
  wrong = farther than 5 cm from the reference (G1@5 / F1@5).

usage: e1_replica.py CODE_DIR DATASET A_CKPT B_CKPT
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

CODE, DS, A_CKPT, B_CKPT = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
sys.path.insert(0, str(CODE))
from update_layer.run import dataset_config  # noqa: E402
from update_layer.backends.game import game as G  # noqa: E402
from update_layer.eval.kf_strip import restore  # noqa: E402
from update_layer.core.l2.evidence import LOG99  # noqa: E402

INT64_MAX = np.iinfo(np.int64).max
cfg, info, specs = dataset_config(DS)
spec_b = next(sp for sp in specs if sp.name.endswith('_b'))
be = G.GameBackend(info, own_update=False)
cka = torch.load(A_CKPT, map_location='cpu', weights_only=False)
g = be.prior_from_state(restore(dict(backend=cka['backend']), DS)['backend'])
be.start_session(spec_b, g)
snap = be._ev_snap
rows0 = torch.nonzero(g.root >= 0).squeeze(1)                 # snapshot index j <-> live row rows0[j] (root numbered ascending)
snap_uid = g.uid[rows0].cpu().numpy()
n0 = len(snap_uid)
xyz0 = (snap.get_xyz / be.scale).cpu().numpy().astype(np.float64)
tol, trunc = be.tolerance, float(cfg.truncation)
print(f'{DS}: snapshot {n0} rows; tol {tol} m; TSDF truncation {trunc} m; layer round {cfg.round_s} s', flush=True)

ckb = torch.load(B_CKPT, map_location='cpu', mmap=True, weights_only=False)
bb = ckb['backend']
b_session = len(bb['session_starts']) - 1
kf_stamps = sorted(int(bb['kf_stamp'][k]) for k in bb['keyframes'] if int(bb['kf_session'].get(k, -1)) == b_session)
fs = be.session
frame_stamps = np.array([fs.stamp_ns(i) for i in range(len(fs.ids))], dtype=np.int64)
kf_index = np.searchsorted(frame_stamps, kf_stamps)
assert np.all(frame_stamps[np.minimum(kf_index, len(frame_stamps) - 1)] == np.array(kf_stamps)), 'keyframe stamps not found'
uid_b = bb['uid'].numpy()
ds_b = bb['death_state'].numpy()
pos = {int(u): i for i, u in enumerate(uid_b)}
layer_end = np.array([ds_b[pos[int(u)]] if int(u) in pos else INT64_MAX for u in snap_uid], dtype=np.int64)
layer_end_t = torch.as_tensor(layer_end, device='cuda')
print(f'B keyframes {len(kf_stamps)}; snapshot rows ended by the layer\'s state ends in the B run: {int((layer_end < INT64_MAX).sum())}',
      flush=True)


class Rule:
    def __init__(self, name, band, footprint=False):
        self.name, self.band, self.footprint = name, band, footprint
        self.on = torch.zeros(n0, device='cuda')
        self.against = torch.zeros(n0, device='cuda')
        self.ended = torch.zeros(n0, dtype=torch.bool, device='cuda')
        self.px = torch.zeros(n0, device='cuda')          # first-echo pixels each row owned, over the keyframes that measured it
        self.rounds = []


def decide(rule, stamp):
    """GameBackend._ev_round on this rule's counts (the same EM and commit)."""
    open_ = ~rule.ended & (layer_end_t > stamp)
    n = rule.on + rule.against
    ev = open_ & (n >= 1)
    if int(ev.sum()) < 2:
        return
    k, nn = rule.against[ev], n[ev]
    frac = k / nn
    hi = frac >= 0.5
    if not hi.any() or hi.all():
        return
    qp = min(max(float(frac[~hi].mean()), 1e-4), 1 - 1e-4)
    qa = min(max(float(frac[hi].mean()), 1e-4), 1 - 1e-4)
    w = min(max(float(hi.float().mean()), 1e-4), 1 - 1e-4)
    for _ in range(200):
        l1 = math.log(w) + k * math.log(qa) + (nn - k) * math.log1p(-qa)
        l0 = math.log(1 - w) + k * math.log(qp) + (nn - k) * math.log1p(-qp)
        pa = torch.sigmoid(l1 - l0)
        qa_n = min(max(float((pa * k).sum() / (pa * nn).sum().clamp(min=1e-9)), 1e-4), 1 - 1e-4)
        qp_n = min(max(float(((1 - pa) * k).sum() / ((1 - pa) * nn).sum().clamp(min=1e-9)), 1e-4), 1 - 1e-4)
        w_n = min(max(float(pa.mean()), 1e-4), 1 - 1e-4)
        done = abs(qa_n - qa) < 1e-7 and abs(qp_n - qp) < 1e-7 and abs(w_n - w) < 1e-7
        qp, qa, w = qp_n, qa_n, w_n
        if done:
            break
    if qa <= qp:
        qp, qa = min(qp, qa), max(qp, qa)
    llr = math.log(w / (1 - w)) + k * math.log(qa / qp) + (nn - k) * math.log((1 - qa) / (1 - qp))
    gone = torch.nonzero(ev).squeeze(1)[llr >= LOG99]
    rule.ended[gone] = True
    rule.rounds.append((stamp, int(ev.sum()), qp, qa, w, len(gone)))


rules = [Rule('E1 (seen through)', False), Rule('E2 (seen through or nearer within the TSDF band)', True),
         Rule('E1f (L2 footprint: hit = any pixel on, through = every pixel valid and beyond)', False, True)]
round_ns = int(cfg.round_s * 1e9)
next_round = be.session_start + round_ns
for kf_i, (stamp, index) in enumerate(zip(kf_stamps, kf_index.tolist())):
    while next_round <= stamp:
        for r in rules:
            decide(r, next_round)
        next_round += round_ns
    sample, instance, dyn = be._sample(int(index))
    kf = {"color": G.gu.np2torch(sample["color"], device="cuda").permute(2, 0, 1) / 255.0,
          "depth": G.gu.np2torch(sample["depth"], device="cuda"),
          "pose": G.gu.np2torch(sample["pose"], device="cuda"), "intrinsics": sample["intrinsics"]}
    view = G.gu.flashsplat_cam(kf["color"], kf["depth"], None, kf["intrinsics"], kf["pose"].cpu(), None)
    dk = kf["depth"].reshape(-1)
    with torch.no_grad():
        for r in rules:
            snap.shown = ~r.ended & (layer_end_t > stamp)
            fe = G.probe_render_fe(view, snap)
            mi = fe["median_index"].reshape(-1).long()
            med = fe["median"].reshape(-1)
            if r.footprint:
                fe_ok = mi >= 0
                rr = mi[fe_ok]
                valid = dk[fe_ok] > 0
                delta = (dk[fe_ok] - med[fe_ok]) / be.scale
                n_all = torch.zeros(n0, device='cuda').index_add_(0, rr, torch.ones_like(delta))
                n_on = torch.zeros(n0, device='cuda').index_add_(0, rr, (valid & (delta.abs() <= tol)).float())
                n_beyond = torch.zeros(n0, device='cuda').index_add_(0, rr, (valid & (delta > tol)).float())
                hit = n_on > 0
                through = (n_all > 0) & (n_beyond == n_all)
                r.on += hit.float()
                r.against += through.float()
                r.px += n_all
                continue
            ok = (mi >= 0) & (dk > 0)
            rr = mi[ok]
            delta = (dk[ok] - med[ok]) / be.scale
            on = (delta.abs() <= tol).float()
            against = (delta > tol) | ((delta < -tol) & (delta >= -trunc)) if r.band else (delta > tol)
            against = against.float()
            n_px = torch.zeros(n0, device='cuda').index_add_(0, rr, torch.ones_like(on))
            s_on = torch.zeros(n0, device='cuda').index_add_(0, rr, on)
            s_ag = torch.zeros(n0, device='cuda').index_add_(0, rr, against)
            seen = n_px > 0
            r.on[seen] += s_on[seen] / n_px[seen]
            r.against[seen] += s_ag[seen] / n_px[seen]
            r.px += n_px
    if kf_i % 50 == 0:
        print(f'  keyframe {kf_i}/{len(kf_stamps)}: ' + '; '.join(f'{r.name.split()[0]} ended {int(r.ended.sum())}' for r in rules), flush=True)
for r in rules:
    decide(r, next_round)

# the scorer's reference
if DS == 'real':
    import open3d as o3d
    ref = o3d.io.read_triangle_mesh('/home/jixian/Desktop/FT/results/abc_eval_v2/geometry/B_reference_1cm.ply')
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(ref))
    ref_name = 'B_reference_1cm.ply'
else:
    sys.path.insert(0, '/mnt/d/3Study/ETH/FT/Synthetic_ABC/scripts')
    import open3d as o3d
    import cvpr_eval_common as C
    frame = len(C.GT['b']['time_s']) - 1
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    n_ent = 0
    for i in C.GT['b']['instance_ids'].tolist():
        i = int(i)
        if i not in C.IX['b'] or not C.GT['b']['render_visible'][frame, C.IX['b'][i]]:
            continue
        v, f = C.worldmesh('b', i, frame)
        if len(f):
            scene.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(v), o3d.core.Tensor(f)))
            n_ent += 1
    ref_name = f'the exact GT of B\'s last frame ({n_ent} render-visible entities)'
dist = scene.compute_distance(o3d.core.Tensor(xyz0.astype(np.float32))).numpy()
wrong = dist > 0.05
print(f'\nreference: {ref_name}; snapshot rows farther than 5 cm from it: {100 * wrong.mean():.1f} %')
ended_sets = {}
for r in rules:
    e = r.ended.cpu().numpy()
    px = r.px.cpu().numpy()
    ended_sets[r.name] = e
    ev = ((r.on + r.against) >= 1).cpu().numpy()
    print(f'\n== {r.name}: rounds {len(r.rounds)}, rows with evidence {int(ev.sum())}, ended {int(e.sum())}')
    if r.rounds:
        qs = np.array([x[2:5] for x in r.rounds])
        print(f'   learned rates over the rounds: q_p median {np.median(qs[:, 0]):.4f}, q_a median {np.median(qs[:, 1]):.4f} '
              f'(min {qs[:, 1].min():.3f}), w median {np.median(qs[:, 2]):.4f}')
    for lab, m in (('ended', e), ('kept, with evidence', ev & ~e)):
        if m.any():
            print(f'   {lab:22s} rows {int(m.sum()):8d}: wrong {100 * (wrong & m).sum() / m.sum():5.1f} %;  weighted by first-echo pixels: '
                  f'wrong {100 * (px * (wrong & m)).sum() / max((px * m).sum(), 1):5.1f} %  ({(px * m).sum():.3g} px)')
e1, e2, e1f = ended_sets[rules[0].name], ended_sets[rules[1].name], ended_sets[rules[2].name]
m = e1 & ~e1f
if m.any():
    print(f'   ended by E1 not by E1f rows {int(m.sum()):8d}: wrong {100 * (wrong & m).sum() / m.sum():5.1f} %;  median distance {100 * np.median(dist[m]):.1f} cm')
for lab, m in (('ended by E2 only', e2 & ~e1), ('ended by E1 only', e1 & ~e2)):
    if m.any():
        print(f'   {lab:22s} rows {int(m.sum()):8d}: wrong {100 * (wrong & m).sum() / m.sum():5.1f} %;  median distance to the reference '
              f'{100 * np.median(dist[m]):.1f} cm')
print('E1_REPLICA_DONE')
