"""Are the A rows behind real B's precision errors part of A's own readout surface? (GPU for A's renders, then CPU, read-only)
1. A's readout rows: every stored keyframe of A rendered with A's final map (the rows alive at A's end, T1), the first echo
   (probe_render_fe median_index, the row the median-depth readout shows at that pixel); the union over A's keyframes = the rows
   that are A's readout surface somewhere. Rows never the first echo in any of A's own views are behind or inside A's surfaces.
2. For each B run: the prediction samples of real B's export (G1@5 samples, crop, distances as g1_by_row_group.py), attributed
   to the nearest row alive at B's end; for the samples whose row is an A row (uid of A, created before B), the share whose row
   is / is not an A readout row, split into correct (<= 5 cm from B's reference) and errors.
usage: a_readout_rows.py CODE_DIR A_CKPT OUT_NPZ NAME=B_CKPT:PRED_PLY [...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

CODE, A_CKPT, OUT = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
sys.path.insert(0, str(CODE))
from update_layer.run import dataset_config  # noqa: E402
from update_layer.backends.game import game as G  # noqa: E402

cfg, info, specs = dataset_config('real')
if OUT.exists():
    readout_uids = np.load(OUT)['uids']
    print(f'A readout rows from {OUT}: {len(readout_uids)}', flush=True)
else:
    be = G.GameBackend(info, own_update=False)
    ck = torch.load(A_CKPT, map_location='cpu', weights_only=False)
    g = be.prior_from_state(ck['backend'])
    t_end = int(ck['prev_final'])
    gm = g.gaussian_model
    seen = torch.zeros(len(g.uid), dtype=torch.bool, device='cuda')
    with torch.no_grad():
        gm.alive = g.alive_at(t_end)
        alive_n = int(gm.alive.sum()) if gm.alive is not None else len(g.uid)
        for kid, kf in g.keyframes.items():
            h, w = kf['depth_shape'][-2:] if 'depth_shape' in kf else kf['depth'].shape[-2:]
            color = torch.zeros(3, h, w, device='cuda'); depth = torch.zeros(h, w, device='cuda')
            pose = kf['pose'] if torch.is_tensor(kf['pose']) else torch.as_tensor(kf['pose'])
            view = G.gu.flashsplat_cam(color, depth, None, kf['intrinsics'], pose.cpu(), None)
            mi = G.probe_render_fe(view, gm)['median_index'].reshape(-1).long()
            seen[mi[mi >= 0]] = True
        gm.alive = None
    readout_uids = g.uid[seen].cpu().numpy()
    np.savez(OUT, uids=readout_uids)
    print(f'A: {len(g.uid)} rows, {alive_n} alive at A\'s end, {len(g.keyframes)} keyframes; readout rows {len(readout_uids)} '
          f'({100 * len(readout_uids) / max(alive_n, 1):.1f} % of the alive rows)', flush=True)
    del g, be, gm
    torch.cuda.empty_cache()

sys.path.insert(0, "/home/jixian/Desktop/FT/tools/abc_eval_v2")
sys.path.insert(0, "/home/jixian/Desktop/FT/eval")
import open3d as o3d  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
import geometry_score as GS  # noqa: E402
from geometry_g1 import rotate  # noqa: E402
FT = Path("/home/jixian/Desktop/FT")
uidA = torch.load(A_CKPT, map_location='cpu', mmap=True, weights_only=False)['backend']['uid'].numpy()
crop = json.loads((FT / "eval/crops/real_B.json").read_text())
lo, hi, yaw = np.array(crop["lo"]), np.array(crop["hi"]), crop["yaw_deg"]
ref = o3d.io.read_triangle_mesh(str(FT / "results/abc_eval_v2/geometry/B_reference_1cm.ply"))
for arg in sys.argv[4:]:
    name, rest = arg.split('=', 1)
    ckp, pred_path = rest.split(':', 1)
    ck = torch.load(ckp, map_location='cpu', mmap=True, weights_only=False)
    b = ck['backend']
    s0, s1 = int(b['session_start']), int(ck['prev_final'])
    xyz = b['model'][1].detach().numpy() / float(b['model'][11])
    uid = b['uid'].numpy()
    created, de, dp_, ds_ = (b[k].numpy() for k in ('created', 'death_evidence', 'death_prune', 'death_state'))
    alive = (de > s1) & (dp_ > s1) & (ds_ > s1)
    a_row = (created < s0) & np.isin(uid, uidA)
    readout = np.isin(uid, readout_uids)
    tree = cKDTree(xyz[alive])
    pred = o3d.io.read_triangle_mesh(pred_path)
    pp, _ = GS.surface_sample(pred)
    pp = pp[~((rotate(pp, yaw) < lo) | (rotate(pp, yaw) > hi)).any(1)].astype(np.float64)
    err = GS.distances(pp, ref) > 0.05
    _, ip = tree.query(pp, workers=4)
    ia = np.nonzero(alive)[0][ip]
    isA, isR = a_row[ia], readout[ia]
    n = len(pp)
    print(f'\n== {name}: {n} prediction samples, errors {100 * err.mean():.2f} %; alive A rows at B\'s end {int((alive & a_row).sum())}, '
          f'of them A readout rows {int((alive & a_row & readout).sum())}')
    for lab, m in (('A row, A readout surface', isA & isR), ('A row, never A\'s first echo', isA & ~isR)):
        print(f'   {lab:30s} samples {100 * m.sum() / n:6.2f} %   errors {100 * (m & err).sum() / n:5.2f} pp   '
              f'error rate {100 * (m & err).sum() / max(m.sum(), 1):5.1f} %')
print('A_READOUT_ROWS_DONE')
