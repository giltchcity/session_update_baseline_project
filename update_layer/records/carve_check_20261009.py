# Frame-level check of the element rule's single-pixel conditions on the rows that survive at old sites in synthetic B
# (no run, no GT selection): for each row, over B frames after the moves, count frames where it is measured and read
# through (delta > tol + ext, facing), on surface, occluded (something closer), or not measured.
import sys, math, numpy as np, torch, pickle
sys.path.insert(0, "/home/jixian/Desktop/FT/session_update_baseline_project")
from update_layer.run import dataset_config
from update_layer.frames import FlatSession
from update_layer.backends.game import t1
R = "/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006"
cfg, info, specs = dataset_config("synthetic")
print("min_object_volume", getattr(cfg, "min_object_volume", None), "max", getattr(cfg, "max_object_volume", None), "tol", getattr(cfg, "surface_match_tolerance", None), "incidence", getattr(cfg, "max_absence_incidence_deg", None))
spec = [sp for sp in specs if sp.name.endswith("_b")][0]; sess = FlatSession(spec, 1, 1)
ck = torch.load(f"{R}/syn_row4i/checkpoint_b.pt", map_location="cpu", weights_only=False, mmap=True); b = ck["backend"]; t = int(ck["prev_final"])
alive = t1.alive_at(t, b["identity"], b["created"], b["state_birth"], b["death_state"], b["death_evidence"], b.get("bg_birth"), b.get("death_prune"))
m = b["model"]; ten = [x for x in m if torch.is_tensor(x) and x.dim() == 2 and x.shape[0] == alive.numel()]
xyz = (ten[0].detach().float() / 10.0).numpy(); scal = [x for x in ten if x.shape[1] == 3][1].detach().float(); rot = [x for x in ten if x.shape[1] == 4][0].detach().float()
ext = (torch.exp(scal).max(dim=1).values * 3 / 10.0).numpy()
q = rot / rot.norm(dim=1, keepdim=True); w, x, y, z = q.unbind(1)
Rm = torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y), 2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x), 2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], 1).reshape(-1, 3, 3)
kmin = scal.argmin(dim=1); nrm = Rm[torch.arange(len(kmin)), :, kmin].numpy()
tl = pickle.load(open(f"{R}/syn_row4i/session_b/timeline.pkl", "rb")); sc0 = tl.scenes[0]
c156 = [np.asarray(o.points).mean(0) for o in sc0.objects if o.instance_id == 156][0]
gt = np.load("/home/jixian/Desktop/FT/datasets/synthetic_ab/session_b/gt/temporal_gt.npz", allow_pickle=True)
ids = list(gt["instance_ids"]); cw = gt["center_world"]
groups = {"sofa 156 old site, background rows": np.nonzero(alive.numpy() & (np.linalg.norm(xyz - c156, axis=1) < 0.5) & (b["identity"].numpy() == 0))[0]}
for inst in (79, 80, 81, 181, 117):
    c = cw[0, ids.index(inst)]
    groups[f"hidden inst {inst} site (GT centre at B start)"] = np.nonzero(alive.numpy() & (np.linalg.norm(xyz - c, axis=1) < 0.15))[0]
tol = float(getattr(cfg, "surface_match_tolerance", 0.05)); min_cos = math.cos(math.radians(float(getattr(cfg, "max_absence_incidence_deg", 60.0))))
frames = list(range(2280, 3550, 30))      # B-local 76 .. 118 s, one frame per second
res = {g: np.zeros((len(r), 4), np.int64) for g, r in groups.items()}   # through, on, front, unmeasured
for i in frames:
    f = sess.load(i); T = f.T_world_cam; K = f.K; off = float(getattr(K, "offset", 0.0)); d = f.depth; H, W = d.shape
    Rcw = T[:3, :3].T; tcw = -Rcw @ T[:3, 3]; campos = T[:3, 3]
    for g, rows in groups.items():
        if len(rows) == 0: continue
        p = xyz[rows]; cam = p @ Rcw.T + tcw; zc = cam[:, 2]
        u = np.floor(K.fx * cam[:, 0] / np.maximum(zc, 1e-6) + K.cx - off + 0.5).astype(int); v = np.floor(K.fy * cam[:, 1] / np.maximum(zc, 1e-6) + K.cy - off + 0.5).astype(int)
        inside = (zc > 0.25) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        meas = np.full(len(rows), np.nan); meas[inside] = d[v[inside], u[inside]]
        measured = inside & np.isfinite(meas)
        delta = meas - zc
        view = p - campos; view /= np.linalg.norm(view, axis=1, keepdims=True); facing = np.abs((nrm[rows] * view).sum(1)) >= min_cos
        band = tol + ext[rows]
        through = measured & (delta > band) & facing; on = measured & (np.abs(delta) <= band); front = measured & (delta < -band)
        r = res[g]; r[:, 0] += through; r[:, 1] += on; r[:, 2] += front; r[:, 3] += ~measured
for g, r in res.items():
    n = len(r)
    if n == 0: print(f"{g}: no rows"); continue
    print(f"{g}: {n} rows over {len(frames)} frames | rows with >=2 through frames (carvable): {(r[:,0]>=2).sum()} | >=1 through: {(r[:,0]>=1).sum()} | 0 through & >=1 on-surface: {((r[:,0]==0)&(r[:,1]>=1)).sum()} | 0 through & occluded only: {((r[:,0]==0)&(r[:,1]==0)&(r[:,2]>=1)).sum()} | never measured: {((r[:,:3].sum(1))==0).sum()} | mean per row: through {r[:,0].mean():.1f} on {r[:,1].mean():.1f} front {r[:,2].mean():.1f}")
