# smoke_synB (922bd6b, synthetic B 50 s): the rows created in each 10.8-s window of B: how many, where they stand relative to
# carried rows (within tol+extent of a carried row = on a memory surface) and to earlier rows of this session, their nearest
# current GT entity, and the carried rows' own state there. GT for attribution only.
import sys, json, numpy as np, torch, open3d as o3d
from scipy.spatial import cKDTree
sys.path.insert(0, "/home/jixian/Desktop/FT/session_update_baseline_project")
from update_layer.backends.game import t1
R = "/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006"; D = "/home/jixian/Desktop/FT/datasets/synthetic_ab"; MAX = torch.iinfo(torch.int64).max
MAN = {o["instance_id"]: o["name"] for o in json.load(open(f"{D}/gt_geometry/manifest.json"))["b"]["objects"]}
ck = torch.load(f"{R}/smoke_synB/checkpoint_b.pt", map_location="cpu", weights_only=False, mmap=True); b = ck["backend"]; t = int(ck["prev_final"]); st = int(b["session_starts"][-1])
ten = [x for x in b["model"] if torch.is_tensor(x) and x.dim() == 2 and x.shape[0] == len(b["identity"])]; xyz = (ten[0].detach().float() / 10.0).numpy(); ext = 3.0 * (torch.exp([x for x in ten if x.shape[1] == 3][1].detach().float()).max(dim=1).values / 10.0).numpy()
opa = torch.sigmoid([x for x in ten if x.shape[1] == 1][0].detach().float()).squeeze(1).numpy()
alive = t1.alive_at(t, b["identity"], b["created"], b["state_birth"], b["death_state"], b["death_evidence"], b.get("bg_birth"), b.get("death_prune")).numpy()
cr = (b["created"].numpy() - st) / 1e9; ident = b["identity"].numpy(); dp = (b["death_prune"].numpy() < MAX); de = b["death_evidence"].numpy() < MAX
car = cr < 0; print(f"rows {len(cr)}: carried {car.sum()} (alive {int((car&alive).sum())}), this session {int((~car).sum())} (alive {int((~car&alive).sum())}, prune-ended {int((~car&dp).sum())}, evidence-ended {int((~car&de).sum())}); session end {(t-st)/1e9:.1f} s")
g = np.load(f"{D}/session_b/gt/temporal_gt.npz"); ids = list(g["instance_ids"]); z = np.load(f"{D}/gt_geometry/b_local_meshes.npz"); f_end = min(int(((t - st) / 1e9) * 30), len(g["render_visible"]) - 1); pts, lab = [], []
for i in ids:
    j = ids.index(i)
    if not g["render_visible"][f_end, j]: continue
    T = g["T_world_object"][f_end, j]; V = z[f"{i:04d}_vertices"].astype(np.float64); F = z[f"{i:04d}_faces"].astype(np.int32); V = (np.c_[V, np.ones(len(V))] @ T.T)[:, :3]
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F)); p = np.asarray(m.sample_points_uniformly(number_of_points=max(100, int(m.get_surface_area() * 3000))).points); pts.append(p); lab.append(np.full(len(p), i))
gt = np.vstack(pts); gl = np.concatenate(lab); tgt = cKDTree(gt)
tc = cKDTree(xyz[car & alive]); ext_c = ext[car & alive]
for w in range(5):
    m = (~car) & (cr >= w * 10.8) & (cr < (w + 1) * 10.8); n = m.sum()
    if not n: continue
    d, j = tc.query(xyz[m], k=1, distance_upper_bound=1.0); band = 0.05 + ext_c[np.minimum(j, len(ext_c) - 1)]; onmem = d <= band
    earlier = (~car) & (cr < w * 10.8) & alive
    if earlier.sum(): de_, _ = cKDTree(xyz[earlier]).query(xyz[m], k=1, distance_upper_bound=1.0); onsess = de_ <= 0.02
    else: onsess = np.zeros(n, bool)
    dg, jg = tgt.query(xyz[m], k=1, distance_upper_bound=1.0); u, c = np.unique(gl[np.minimum(jg, len(gl) - 1)][dg < 1.0], return_counts=True); top = sorted(zip(c.tolist(), u.tolist()), reverse=True)[:5]
    print(f"window {w} ({w*10.8:.0f}-{(w+1)*10.8:.0f} s): created {n} (alive now {int((m&alive).sum())}, prune-ended {int((m&dp).sum())}); within tol+extent of a carried row {100*onmem.mean():.0f} %, within 2 cm of an earlier this-session row {100*onsess.mean():.0f} %; off the GT > 5 cm {100*(dg>0.05).mean():.0f} %; opacity p50 {np.median(opa[m]):.2f}; near: " + ", ".join(f"{MAN.get(int(i),'bg')} {k}" for k, i in top))
