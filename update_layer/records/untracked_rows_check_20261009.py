# Real-side consequence check (offline): alive rows at the end of a session whose identity is a label the layer never
# tracked (no state); how many, and over the session's frames how many would be carved by the element rule's
# single-pixel conditions (>= 3 through frames for real at step -log(0.15) = 1.9 nats; >= 2 for synthetic).
import sys, math, numpy as np, torch
sys.path.insert(0, "/home/jixian/Desktop/FT/session_update_baseline_project")
from update_layer.run import dataset_config
from update_layer.frames import FlatSession
from update_layer.backends.game import t1
R = "/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006"
ds, run, s, step_frames = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
cfg, info, specs = dataset_config(ds); spec = [sp for sp in specs if sp.name.endswith("_" + s)][0]; sess = FlatSession(spec, 1, 1)
ck = torch.load(f"{R}/{run}/checkpoint_{s}.pt", map_location="cpu", weights_only=False, mmap=True); b = ck["backend"]; t = int(ck["prev_final"])
alive = t1.alive_at(t, b["identity"], b["created"], b["state_birth"], b["death_state"], b["death_evidence"], b.get("bg_birth"), b.get("death_prune"))
tl = b.get("tracked_labels"); tracked = set(tl.tolist()) if tl is not None else set()
ident = b["identity"]
untracked = alive & (ident > 0) & ~torch.isin(ident, torch.tensor(sorted(tracked), dtype=ident.dtype))
labs, cnt = torch.unique(ident[untracked], return_counts=True)
print(f"{run} {s}: alive {int(alive.sum())}, alive rows with an untracked label: {int(untracked.sum())} ({100*untracked.sum()/alive.sum():.2f} %), labels {dict(zip(labs.tolist(), cnt.tolist()))}")
m = b["model"]; ten = [x for x in m if torch.is_tensor(x) and x.dim() == 2 and x.shape[0] == alive.numel()]
xyz = (ten[0].detach().float() / 10.0).numpy(); scal = [x for x in ten if x.shape[1] == 3][1].detach().float(); rot = [x for x in ten if x.shape[1] == 4][0].detach().float()
ext = (torch.exp(scal).max(dim=1).values * 3 / 10.0).numpy()
q = rot / rot.norm(dim=1, keepdim=True); w, x, y, z = q.unbind(1)
Rm = torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y), 2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x), 2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], 1).reshape(-1, 3, 3)
kmin = scal.argmin(dim=1); nrm = Rm[torch.arange(len(kmin)), :, kmin].numpy()
rows = np.nonzero(untracked.numpy())[0]
if len(rows) > 60000: rows = rows[np.random.default_rng(0).choice(len(rows), 60000, replace=False)]
tol = 0.05; min_cos = math.cos(math.radians(60.0))
n_frames = len(sess.ids) if hasattr(sess, "ids") else 4000
frames = list(range(0, n_frames, step_frames))
r = np.zeros((len(rows), 4), np.int64)
for i in frames:
    f = sess.load(i); T = f.T_world_cam; K = f.K; off = float(getattr(K, "offset", 0.0)); d = f.depth; H, W = d.shape
    Rcw = T[:3, :3].T; tcw = -Rcw @ T[:3, 3]; campos = T[:3, 3]
    p = xyz[rows]; cam = p @ Rcw.T + tcw; zc = cam[:, 2]
    u = np.floor(K.fx * cam[:, 0] / np.maximum(zc, 1e-6) + K.cx - off + 0.5).astype(int); v = np.floor(K.fy * cam[:, 1] / np.maximum(zc, 1e-6) + K.cy - off + 0.5).astype(int)
    inside = (zc > 0.25) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    meas = np.full(len(rows), np.nan); meas[inside] = d[v[inside], u[inside]]; measured = inside & np.isfinite(meas)
    delta = meas - zc; view = p - campos; view /= np.linalg.norm(view, axis=1, keepdims=True); facing = np.abs((nrm[rows] * view).sum(1)) >= min_cos
    band = tol + ext[rows]
    r[:, 0] += measured & (delta > band) & facing; r[:, 1] += measured & (np.abs(delta) <= band); r[:, 2] += measured & (delta < -band); r[:, 3] += ~measured
need = 3 if ds == "real" else 2
print(f"  over {len(frames)} frames (every {step_frames}): rows with >= {need} through frames: {(r[:,0]>=need).sum()} of {len(rows)} ({100*(r[:,0]>=need).mean():.1f} %); of those, also on-surface in >= 1 frame: {((r[:,0]>=need)&(r[:,1]>=1)).sum()}; rows never through: {(r[:,0]==0).sum()}")
