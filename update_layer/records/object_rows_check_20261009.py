# F4 offline check: this-session OBJECT rows (identity > 0 with a state, alive at the session end) under the element rule's
# single-pixel conditions over the session's frames: rows read through >= k times with no on-surface reading (carvable) vs
# rows with on-surface readings (protected). Their distance to the GT/reference says whether carvable rows are ghosts.
import sys, math, numpy as np, torch, json, open3d as o3d
from scipy.spatial import cKDTree
sys.path.insert(0, "/home/jixian/Desktop/FT/session_update_baseline_project")
from update_layer.run import dataset_config
from update_layer.frames import FlatSession
from update_layer.backends.game import t1
R = "/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006"; FT = "/home/jixian/Desktop/FT"; MAX = torch.iinfo(torch.int64).max
ds, run, s, step, sess_len = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5])
cfg, info, specs = dataset_config(ds); spec = [sp for sp in specs if sp.name.endswith("_" + s)][0]; sess = FlatSession(spec, 1, 1)
ck = torch.load(f"{R}/{run}/checkpoint_{s}.pt", map_location="cpu", weights_only=False, mmap=True); b = ck["backend"]; t = int(ck["prev_final"]); t0 = t - int(sess_len * 1e9)
alive = t1.alive_at(t, b["identity"], b["created"], b["state_birth"], b["death_state"], b["death_evidence"], b.get("bg_birth"), b.get("death_prune"))
ident = b["identity"]; sb = b["state_birth"]
obj = alive & (ident > 0) & (sb < MAX) & (sb >= t0)          # this session's object states' rows
ten = [x for x in b["model"] if torch.is_tensor(x) and x.dim() == 2 and x.shape[0] == alive.numel()]
xyz = (ten[0].detach().float() / 10.0).numpy(); scal = [x for x in ten if x.shape[1] == 3][1].detach().float(); rot = [x for x in ten if x.shape[1] == 4][0].detach().float()
ext = (torch.exp(scal).max(dim=1).values * 3 / 10.0).numpy()
q = rot / rot.norm(dim=1, keepdim=True); w, x, y, z = q.unbind(1)
Rm = torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y), 2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x), 2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], 1).reshape(-1, 3, 3)
kmin = scal.argmin(dim=1); nrm = Rm[torch.arange(len(kmin)), :, kmin].numpy()
rows = np.nonzero(obj.numpy())[0]; rng_ = np.random.default_rng(0)
if len(rows) > 80000: rows = rows[rng_.choice(len(rows), 80000, replace=False)]
print(f"{run} {s}: this-session object rows alive {int(obj.sum())} (sampled {len(rows)}), identities {len(torch.unique(ident[obj]))}")
tol = 0.05; min_cos = math.cos(math.radians(60.0)); frames = list(range(0, len(sess.ids), step))
r = np.zeros((len(rows), 3), np.int64)
for i in frames:
    f = sess.load(i); T = f.T_world_cam; K = f.K; off = float(getattr(K, "offset", 0.0)); d = f.depth; H, W = d.shape
    Rcw = T[:3, :3].T; tcw = -Rcw @ T[:3, 3]; campos = T[:3, 3]; p = xyz[rows]; cam = p @ Rcw.T + tcw; zc = cam[:, 2]
    u = np.floor(K.fx * cam[:, 0] / np.maximum(zc, 1e-6) + K.cx - off + 0.5).astype(int); v = np.floor(K.fy * cam[:, 1] / np.maximum(zc, 1e-6) + K.cy - off + 0.5).astype(int)
    inside = (zc > 0.25) & (u >= 0) & (u < W) & (v >= 0) & (v < H); meas = np.full(len(rows), np.nan); meas[inside] = d[v[inside], u[inside]]; measured = inside & np.isfinite(meas)
    delta = meas - zc; view = p - campos; view /= np.linalg.norm(view, axis=1, keepdims=True); facing = np.abs((nrm[rows] * view).sum(1)) >= min_cos; band = tol + ext[rows]
    r[:, 0] += measured & (delta > band) & facing; r[:, 1] += measured & (np.abs(delta) <= band); r[:, 2] += measured & (delta < -band)
need = 3 if ds == "real" else 2
carvable = (r[:, 0] >= need) & (r[:, 1] == 0); protected = r[:, 1] >= 1
print(f"  over {len(frames)} frames: carvable (>= {need} through, 0 on) {carvable.sum()} ({100*carvable.mean():.1f} %); with on-surface readings {protected.sum()} ({100*protected.mean():.1f} %); never measured {(r.sum(1)==0).sum()}")
if ds == "synthetic":
    D = "/home/jixian/Desktop/FT/datasets/synthetic_ab"; man = json.load(open(f"{D}/gt_geometry/manifest.json"))["b"]["objects"]; z = np.load(f"{D}/gt_geometry/b_local_meshes.npz"); g = np.load(f"{D}/session_b/gt/temporal_gt.npz"); ids = list(g["instance_ids"]); Tg = g["T_world_object"][-1]; vis = g["render_visible"][-1]
    pts = []
    for o in man:
        i = o["instance_id"]; j = ids.index(i)
        if not vis[j]: continue
        V = z[f"{i:04d}_vertices"].astype(np.float64); F = z[f"{i:04d}_faces"].astype(np.int32); V = (np.c_[V, np.ones(len(V))] @ Tg[j].T)[:, :3]
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F)); pts.append(np.asarray(m.sample_points_uniformly(number_of_points=max(50, int(m.get_surface_area() * 3000))).points))
    ref = np.vstack(pts)
else:
    ref = np.asarray(o3d.io.read_point_cloud(f"{FT}/results/abc_eval_v2/geometry/{s.upper()}_reference_1cm.ply").points)
tree = cKDTree(ref)
for sel, name in ((carvable, "carvable object rows"), (protected, "object rows with on readings")):
    if sel.sum() == 0: continue
    dd, _ = tree.query(xyz[rows[sel]], k=1, distance_upper_bound=1.0); h = np.histogram(np.minimum(dd, 1.0), [0, 0.05, 0.10, 0.20, 1.01])[0] / sel.sum() * 100
    ids_, cnt = np.unique(ident.numpy()[rows[sel]], return_counts=True); top = sorted(zip(cnt.tolist(), ids_.tolist()), reverse=True)[:6]
    print(f"  {name:30s}: distance to GT/reference <5cm {h[0]:.0f} %  5-10 {h[1]:.0f}  10-20 {h[2]:.0f}  >20 {h[3]:.0f}; top identities {top}")
