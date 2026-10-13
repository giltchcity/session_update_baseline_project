# F3 real-side bound: the earlier-session background rows that B/C pruned, classified by the single-pixel verdicts of the
# session's frames (through >= 3 & no on -> carved by the element rule; on-surface readings -> refined in place; front-only
# -> hidden behind the surface; unmeasured) crossed with their distance to the reference. The risk group = 'on' & > 5 cm.
import sys, math, numpy as np, torch, open3d as o3d
from scipy.spatial import cKDTree
sys.path.insert(0, "/home/jixian/Desktop/FT/session_update_baseline_project")
from update_layer.run import dataset_config
from update_layer.frames import FlatSession
R = "/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006"; FT = "/home/jixian/Desktop/FT"; MAX = torch.iinfo(torch.int64).max
s, step, sess_len = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
cfg, info, specs = dataset_config("real"); spec = [sp for sp in specs if sp.name.endswith("_" + s)][0]; sess = FlatSession(spec, 1, 1)
ck = torch.load(f"{R}/real_row4k/checkpoint_{s}.pt", map_location="cpu", weights_only=False, mmap=True); b = ck["backend"]; t = int(ck["prev_final"]); t0 = t - int(sess_len * 1e9)
ten = [x for x in b["model"] if torch.is_tensor(x) and x.dim() == 2 and x.shape[0] == b["uid"].numel()]
xyz = (ten[0].detach().float() / 10.0).numpy(); scal = [x for x in ten if x.shape[1] == 3][1].detach().float(); rot = [x for x in ten if x.shape[1] == 4][0].detach().float()
ext = (torch.exp(scal).max(dim=1).values * 3 / 10.0).numpy()
q = rot / rot.norm(dim=1, keepdim=True); w, x, y, z = q.unbind(1)
Rm = torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y), 2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x), 2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], 1).reshape(-1, 3, 3)
kmin = scal.argmin(dim=1); nrm = Rm[torch.arange(len(kmin)), :, kmin].numpy()
cr = b["created"].numpy(); dp = b["death_prune"].numpy(); ident = b["identity"].numpy()
ref = np.asarray(o3d.io.read_point_cloud(f"{FT}/results/abc_eval_v2/geometry/{s.upper()}_reference_1cm.ply").points); tree = cKDTree(ref); lo, hi = ref.min(0) - 0.3, ref.max(0) + 0.3
inside = np.all((xyz > lo) & (xyz < hi), axis=1)
pr = (cr < t0) & (dp >= t0) & (dp < MAX) & (ident <= 0) & inside
rows = np.nonzero(pr)[0]; rng_ = np.random.default_rng(0); rows = rows[rng_.choice(len(rows), min(120000, len(rows)), replace=False)]
tol = 0.05; min_cos = math.cos(math.radians(60.0)); frames = list(range(0, len(sess.ids), step)); r = np.zeros((len(rows), 3), np.int64)
for i in frames:
    f = sess.load(i); T = f.T_world_cam; K = f.K; off = float(getattr(K, "offset", 0.0)); d = f.depth; H, W = d.shape
    Rcw = T[:3, :3].T; tcw = -Rcw @ T[:3, 3]; campos = T[:3, 3]; p = xyz[rows]; cam = p @ Rcw.T + tcw; zc = cam[:, 2]
    u = np.floor(K.fx * cam[:, 0] / np.maximum(zc, 1e-6) + K.cx - off + 0.5).astype(int); v = np.floor(K.fy * cam[:, 1] / np.maximum(zc, 1e-6) + K.cy - off + 0.5).astype(int)
    ins = (zc > 0.25) & (u >= 0) & (u < W) & (v >= 0) & (v < H); meas = np.full(len(rows), np.nan); meas[ins] = d[v[ins], u[ins]]; measured = ins & np.isfinite(meas)
    delta = meas - zc; view = p - campos; view /= np.linalg.norm(view, axis=1, keepdims=True); facing = np.abs((nrm[rows] * view).sum(1)) >= min_cos; band = tol + ext[rows]
    r[:, 0] += measured & (delta > band) & facing; r[:, 1] += measured & (np.abs(delta) <= band); r[:, 2] += measured & (delta < -band)
dref, _ = tree.query(xyz[rows], k=1, distance_upper_bound=1.0); off5 = dref > 0.05
carved = (r[:, 0] >= 3) & (r[:, 1] == 0); on = r[:, 1] >= 1; front = (r[:, 0] < 3) & (r[:, 1] == 0) & (r[:, 2] >= 1); unm = r.sum(1) == 0
n = len(rows); tot = pr.sum()
print(f"real {s.upper()}: pruned carried background rows {tot} (sampled {n}), frames every {step}")
for sel, name in ((carved, "carved (>=3 through, 0 on)"), (on, "on-surface readings (refined)"), (front, "front-only (hidden)"), (unm, "unmeasured")):
    print(f"  {name:32s} {100*sel.mean():5.1f} % of pruned rows; of these > 5 cm off the reference {100*(off5[sel].mean() if sel.any() else 0):4.1f} %  -> as a share of ALL alive rows: {100*sel.sum()/n*tot/int(((b['death_evidence']==MAX)&(b['death_state']==MAX)&(dp==MAX)).sum()):.2f} % (off > 5 cm: {100*(sel&off5).sum()/n*tot/int(((b['death_evidence']==MAX)&(b['death_state']==MAX)&(dp==MAX)).sum()):.2f} %)")
onoff = on & off5
print(f"  RISK group (on-surface & > 5 cm): {onoff.sum()} sampled -> ~{int(onoff.mean()*tot)} rows; on-reading count median {np.median(r[onoff,1]) if onoff.any() else 0:.0f}; distance 5-10 cm {100*((dref[onoff]>0.05)&(dref[onoff]<=0.10)).mean() if onoff.any() else 0:.0f} %, >10 cm {100*(dref[onoff]>0.10).mean() if onoff.any() else 0:.0f} %; extent median {np.median(ext[rows[onoff]]) if onoff.any() else 0:.3f} m")
