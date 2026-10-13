# Element-rule accumulation, full session, two variants, on the dataset's own frames at 1 Hz (the rule's cadence):
#  A) as coded: one step per 10.8-s round from the round's last verdict (through adds, on resets), retire when c > LN99 and last verdict through
#  B) per stamp (the stated rule): every through stamp adds a step, every on stamp resets, retire when c > LN99
# Rows = alive at the session end + rows the run ended by evidence (at their final positions, checkpoint last_update).
# Correctness vs the exact GT (synthetic B end) / the official B reference (real). GT only for checking.
import sys, os, csv, math, numpy as np, torch, tifffile, open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, "/home/jixian/Desktop/FT/session_update_baseline_project")
from update_layer.backends.game import t1
R = "/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006"; G = "/home/jixian/Desktop/FT/results/abc_eval_v2/geometry"; D = "/home/jixian/Desktop/FT/datasets/synthetic_ab"; MAX = torch.iinfo(torch.int64).max; LN99 = math.log(99.0); tol = 0.05; ROUND = 10.8e9
def rows_of(run, s):
    ck = torch.load(f"{R}/{run}/checkpoint_{s}.pt", map_location="cpu", weights_only=False, mmap=True); b = ck["backend"]; t = int(ck["prev_final"]); st = int(b["session_starts"][-1])
    ten = [x for x in b["model"] if torch.is_tensor(x) and x.dim() == 2 and x.shape[0] == len(b["identity"])]; xyz = (ten[0].detach().float() / 10.0).numpy(); ext = 3.0 * (torch.exp([x for x in ten if x.shape[1] == 3][1].detach().float()).max(dim=1).values / 10.0).numpy()
    alive = t1.alive_at(t, b["identity"], b["created"], b["state_birth"], b["death_state"], b["death_evidence"], b.get("bg_birth"), b.get("death_prune")).numpy(); de = b["death_evidence"].numpy() < MAX
    sel = np.nonzero(alive | de)[0]
    n_, s_ = [float(x) for x in open(f"{R}/{run}/session_{s}/sensor_statistics.txt").read().strip().splitlines()[-1].split()[:2]]; p_miss = s_ / n_ if n_ >= 3 else 0.05
    return xyz[sel], ext[sel], b["last_update"].numpy()[sel], b["created"].numpy()[sel] < st, de[sel], alive[sel], b["identity"].numpy()[sel], st, p_miss
def simulate(name, P, E, LU, carried, de_run, alive_run, ident, frames, dtrue, p_miss):
    step = -math.log(min(0.995, max(0.005, p_miss))); N = len(P)
    cA = np.zeros(N); vlast = np.zeros(N, np.int8); retA = np.zeros(N, bool); cB = np.zeros(N); retB = np.zeros(N, bool); round_end = None
    for (stamp, Rw, tw, dep, fx, fy, cx, cy, W, H) in frames:
        if round_end is None: round_end = stamp + ROUND
        pc = (P - tw) @ Rw; zz = pc[:, 2]; ok = zz > 0.1; u = np.round(fx * pc[:, 0] / np.where(ok, zz, 1) + cx).astype(int); v = np.round(fy * pc[:, 1] / np.where(ok, zz, 1) + cy).astype(int)
        inv = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H); dm = np.zeros(N); dm[inv] = dep[v[inv], u[inv]]; meas = inv & (dm > 0); delta = dm - zz; later = stamp > LU
        on = meas & (np.abs(delta) <= tol + E) & later; thr = meas & (delta > tol + E) & later
        vlast[on & ~retA] = 1; vlast[thr & ~retA] = 2
        cB[on & ~retB] = 0; cB[thr & ~retB] += step; retB |= cB > LN99
        if stamp >= round_end:
            cA[(vlast == 1) & ~retA] = 0; cA[(vlast == 2) & ~retA] += step; retA |= (vlast == 2) & (cA > LN99); vlast[:] = 0; round_end = stamp + ROUND
    cA[(vlast == 1) & ~retA] = 0; cA[(vlast == 2) & ~retA] += step; retA |= (vlast == 2) & (cA > LN99)
    print(f"\n== {name}: rows {N} (alive at end {alive_run.sum()}, ended by evidence in the run {de_run.sum()}); p_miss {p_miss:.4f} step {step:.2f}")
    for vname, ret in (("run's own evidence ends", de_run), ("A) as coded (round)", retA), ("B) per stamp", retB)):
        on_t = dtrue[ret] <= 0.05
        print(f"   {vname:26s}: retired {ret.sum():7d} ({100*ret.mean():5.2f} %); carried {int((ret&carried).sum()):6d}, this session {int((ret&~carried).sum()):6d}; ON the true surface {100*on_t.mean() if ret.any() else 0:5.1f} % ({on_t.sum()} rows), off > 5 cm {int((~on_t).sum()):6d}, off > 10 cm {int((dtrue[ret]>0.10).sum()):6d}; plant-109 {int(((ident==109)&ret).sum())}")
    return retA, retB
# ---- synthetic B (syn_row4k)
P, E, LU, carried, de_run, alive_run, ident, st, p_miss = rows_of("syn_row4k", "b")
g = np.load(f"{D}/session_b/gt/temporal_gt.npz"); ids = list(g["instance_ids"]); z = np.load(f"{D}/gt_geometry/b_local_meshes.npz"); nb = len(g["render_visible"]); pts = []
for i in ids:
    j = ids.index(i)
    if not g["render_visible"][nb - 1, j]: continue
    T = g["T_world_object"][nb - 1, j]; V = z[f"{i:04d}_vertices"].astype(np.float64); F = z[f"{i:04d}_faces"].astype(np.int32); V = (np.c_[V, np.ones(len(V))] @ T.T)[:, :3]
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F)); pts.append(np.asarray(m.sample_points_uniformly(number_of_points=max(100, int(m.get_surface_area() * 4000))).points))
dtrue = cKDTree(np.vstack(pts)).query(P, k=1, distance_upper_bound=2.0)[0]
tum = np.loadtxt(f"{D}/session_b/trajectory_gt.tum"); Rq = Rot.from_quat(tum[:, 4:8]).as_matrix(); tw = tum[:, 1:4]; fx = fy = 588.897289890142
frames = []
for f in range(0, len(tum), 30):
    frames.append((st + int(f / 30 * 1e9), Rq[f], tw[f], tifffile.imread(f"{D}/session_b/rgbd/{f:06d}_depth.tiff").astype(np.float64), fx, fy, 340.0, 240.0, 680, 480))
simulate("synthetic B (syn_row4k)", P, E, LU, carried, de_run, alive_run, ident, frames, dtrue, p_miss)
# ---- real B (real_row4k)
P, E, LU, carried, de_run, alive_run, ident, st, p_miss = rows_of("real_row4k", "b")
root = "/home/jixian/Desktop/FT/datasets/local_ab"; rb = f"{root}/rgbd/session_b_20260810_030502_620_flat_rgbd_30hz_1080p"; Tba = np.loadtxt(f"{root}/alignment/session_b_to_session_a.txt")
ref = np.asarray(o3d.io.read_triangle_mesh(f"{G}/B_reference_1cm.ply").vertices, dtype=np.float64); dtrue = cKDTree(ref).query(P, k=1, distance_upper_bound=2.0)[0]
ts = [(int(r[0]), int(r[1])) for r in list(csv.reader(open(f"{rb}/timestamps.csv")))[1:] if r and r[0].isdigit()]; frames = []; last = None
for fid, stamp in ts:
    if last is not None and stamp - last < int(1e9): continue
    last = stamp; Tcw = Tba @ np.loadtxt(f"{rb}/{fid:06d}_pose.txt")
    frames.append((st + stamp, Tcw[:3, :3], Tcw[:3, 3], np.nan_to_num(tifffile.imread(f"{rb}/{fid:06d}_depth.tiff").astype(np.float64), nan=0.0), 958.539672852, 958.00012207, 956.520463638, 548.005616755, 1920, 1080))
simulate("real B (real_row4k)", P, E, LU, carried, de_run, alive_run, ident, frames, dtrue, p_miss)
