# Residue sites of the synthetic B map (ours syn_row4j vs decided syn_row4s3cb): which rows stand behind the residue points.
import sys, json, pickle, numpy as np, open3d as o3d, torch
from scipy.spatial import cKDTree
sys.path.insert(0, "/home/jixian/Desktop/FT/session_update_baseline_project")
from update_layer.backends.game import t1
D = "/home/jixian/Desktop/FT/datasets/synthetic_ab"; R = "/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006"; MAX = torch.iinfo(torch.int64).max
G = {s: np.load(f"{D}/session_{s}/gt/temporal_gt.npz") for s in "ab"}; Z = {s: np.load(f"{D}/gt_geometry/{s}_local_meshes.npz") for s in "ab"}; IDS = {s: list(G[s]["instance_ids"]) for s in "ab"}
MAN = {o["instance_id"]: o["name"] for o in json.load(open(f"{D}/gt_geometry/manifest.json"))["b"]["objects"]}
def surf(s, i, f, per_m2=4000):
    T = G[s]["T_world_object"][f, IDS[s].index(i)]; V = Z[s][f"{i:04d}_vertices"].astype(np.float64); F = Z[s][f"{i:04d}_faces"].astype(np.int32); V = (np.c_[V, np.ones(len(V))] @ T.T)[:, :3]
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F)); return np.asarray(m.sample_points_uniformly(number_of_points=max(100, int(m.get_surface_area() * per_m2))).points)
nb = len(G["b"]["render_visible"]); na = len(G["a"]["render_visible"])
cur = np.vstack([surf("b", i, nb - 1) for i in IDS["b"] if G["b"]["render_visible"][nb - 1, IDS["b"].index(i)]]); tcur = cKDTree(cur)
sites = [("flower_0003 mid-route (fr 3424)", surf("b", 109, 3424)), ("flower_0003 B-start", surf("b", 109, 0)), ("sofa_0000 B-start", surf("b", 155, 0)), ("sofa_0001 A-end", surf("a", 156, na - 1)), ("sofa_0001 B-start", surf("b", 156, 0)),
         ("cabinet_0000 fr 2524", surf("b", 48, 2524)), ("chair_0004 B-start", surf("b", 72, 0)), ("throw_pillow_0000 B-start", surf("b", 173, 0)), ("television_0000 B-start", surf("b", 171, 0)), ("cabinet_0007 A-end", surf("a", 55, na - 1))]
def scene_points(path):
    tl = pickle.load(open(path, "rb")); sc = tl.scenes[-1]; bg = np.asarray(sc.background, dtype=np.float64); ob = [(np.asarray(x.points, dtype=np.float64), x.instance_id) for x in sc.objects if x.present and len(x.points)]
    return np.vstack([bg] + [p for p, _ in ob]), np.concatenate([np.zeros(len(bg), np.int64)] + [np.full(len(p), i, np.int64) for p, i in ob])
for name, run, L in (("ours", "syn_row4j", 118.5), ("decided", "syn_row4s3cb", 118.5)):
    v, lab = scene_points(f"{R}/{run}/session_b/timeline.pkl"); tv = cKDTree(v); dcur, _ = tcur.query(v, k=1, distance_upper_bound=1.0)
    ck = torch.load(f"{R}/{run}/checkpoint_b.pt", map_location="cpu", weights_only=False, mmap=True); b = ck["backend"]; t = int(ck["prev_final"]); t0 = t - int(L * 1e9)
    dp = b.get("death_prune"); alive = t1.alive_at(t, b["identity"], b["created"], b["state_birth"], b["death_state"], b["death_evidence"], b.get("bg_birth"), dp).numpy()
    ten = [x for x in b["model"] if torch.is_tensor(x) and x.dim() == 2 and x.shape[0] == len(alive)]; xyz = (ten[0].detach().float() / 10.0).numpy(); opa = torch.sigmoid([x for x in ten if x.shape[1] == 1][0].detach().float()).squeeze(1).numpy()
    ident = b["identity"].numpy(); cr = (b["created"].numpy() - t0) / 1e9; ds = b["death_state"].numpy(); de = b["death_evidence"].numpy(); dpn = dp.numpy() if dp is not None else np.full(len(alive), MAX)
    ra = np.nonzero(alive)[0]; ta = cKDTree(xyz[ra]); tall = cKDTree(xyz)
    print(f"\n==== {name}: prev_final {t} ({(t-t0)/1e9:.1f} s local end); rows {len(alive)}, alive {alive.sum()}")
    for sname, sp in sites:
        near = np.unique(np.concatenate(tv.query_ball_point(sp, 0.05)).astype(int)); res = near[dcur[near] > 0.05]
        if len(res) == 0: print(f"  {sname:32s}: residue 0"); continue
        ls, lc = np.unique(lab[res], return_counts=True)
        dr, jr = ta.query(v[res], k=1, distance_upper_bound=0.3); hit = dr < 0.03; rows = np.unique(ra[jr[hit]])
        # ALL rows (alive or not) at the site, by fate
        allr = np.unique(np.concatenate(tall.query_ball_point(sp, 0.05)).astype(int))
        fate = {"alive": int(alive[allr].sum()), "ended_by_state": int((ds[allr] < MAX).sum()), "ended_by_evidence": int((de[allr] < MAX).sum()), "ended_by_prune": int((dpn[allr] < MAX).sum())}
        ends = ds[allr][ds[allr] < MAX]; endt = f"state-end stamps local s: {np.unique(np.round((ends - t0)/1e9, 1))[:6].tolist()}" if len(ends) else "no state ends"
        print(f"  {sname:32s}: residue {len(res)} (scene labels {dict(zip(ls.tolist(), lc.tolist()))}); nearest alive row within 3 cm for {100*hit.mean():.0f} %")
        if len(rows): print(f"      alive rows behind them {len(rows)}: identity {dict(zip(*[x.tolist() for x in np.unique(ident[rows], return_counts=True)]))}; created in A {100*(cr[rows]<0).mean():.0f} %, in B local s {np.percentile(cr[rows][cr[rows]>=0], [10,50,90]).round(0).tolist() if (cr[rows]>=0).any() else '-'}; opacity median {np.median(opa[rows]):.2f} (>0.5: {100*(opa[rows]>0.5).mean():.0f} %)")
        print(f"      all rows within 5 cm of the site {len(allr)}: {fate}; {endt}")
