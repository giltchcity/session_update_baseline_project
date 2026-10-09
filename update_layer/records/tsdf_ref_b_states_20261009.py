import sys, json, numpy as np, open3d as o3d
from scipy.spatial import cKDTree
D = "/home/jixian/Desktop/FT/datasets/synthetic_ab"; N = f"{D}/eval_geo21_20260922/ours/native/b"
rec = [json.loads(l) for l in open(f"{N}/meshes.jsonl")][-1]
def verts(prefix): return np.fromfile(f"{N}/{prefix}.f32", dtype=np.float32).reshape(-1, 3).astype(np.float64)
import os
bg = verts(rec["background_prefix"]); obs = [(verts(o["prefix"]), o) for o in rec["objects"] if o["present"] and os.path.exists(f"{N}/{o['prefix']}.f32")]; print("present objects without a mesh file:", sum(1 for o in rec["objects"] if o["present"] and not os.path.exists(f"{N}/{o['prefix']}.f32")))
v = np.vstack([bg] + [p for p, _ in obs]); lab = np.concatenate([np.zeros(len(bg), np.int64)] + [np.full(len(p), i + 1, np.int64) for i, (p, _) in enumerate(obs)])
print(f"TSDF geo21 B scene (stamp {rec['stamp_ns']}): background {len(bg)} + {len(obs)} present objects {sum(len(p) for p,_ in obs)} = {len(v)} vertices")
G = {s: np.load(f"{D}/session_{s}/gt/temporal_gt.npz") for s in "ab"}; Z = {s: np.load(f"{D}/gt_geometry/{s}_local_meshes.npz") for s in "ab"}; IDS = {s: list(G[s]["instance_ids"]) for s in "ab"}
MAN = {o["instance_id"]: o["name"] for o in json.load(open(f"{D}/gt_geometry/manifest.json"))["b"]["objects"]}
def surf(s, i, f, per_m2=4000):
    T = G[s]["T_world_object"][f, IDS[s].index(i)]; V = Z[s][f"{i:04d}_vertices"].astype(np.float64); F = Z[s][f"{i:04d}_faces"].astype(np.int32); V = (np.c_[V, np.ones(len(V))] @ T.T)[:, :3]
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F)); return np.asarray(m.sample_points_uniformly(number_of_points=max(100, int(m.get_surface_area() * per_m2))).points)
nb = len(G["b"]["render_visible"]); na = len(G["a"]["render_visible"])
cur, curl = [], []
for i in IDS["b"]:
    if G["b"]["render_visible"][nb - 1, IDS["b"].index(i)]: p = surf("b", i, nb - 1); cur.append(p); curl.append(np.full(len(p), i))
cur = np.vstack(cur); curl = np.concatenate(curl); tcur = cKDTree(cur); tv = cKDTree(v)
d, j = tcur.query(v, k=1, distance_upper_bound=1.0); e = d > 0.05
print(f"errors > 5 cm: {e.sum()} ({100*e.mean():.2f} %), > 10 cm {int((d>0.10).sum())}")
sites = [("flower_0003 mid-route (fr 3424)", surf("b", 109, 3424)), ("flower_0003 fr 3393", surf("b", 109, 3393)), ("flower_0003 B-start", surf("b", 109, 0)), ("sofa_0000 B-start", surf("b", 155, 0)), ("sofa_0001 A-end", surf("a", 156, na - 1)), ("cabinet_0000 fr 2524", surf("b", 48, 2524)), ("chair_0004 B-start", surf("b", 72, 0)), ("throw_pillow_0000 B-start", surf("b", 173, 0)), ("television_0000 B-start", surf("b", 171, 0)), ("bed_0000 A-end", surf("a", 7, na - 1)), ("range_hood A-end", surf("a", 153, na - 1))]
for sname, sp in sites:
    near = np.unique(np.concatenate(tv.query_ball_point(sp, 0.05)).astype(int)); print(f"  {sname:32s}: residue {int((d[near] > 0.05).sum()):6d} | strict {int((d[near] > 0.10).sum()):6d}")
for i, nm in ((109, "flower_0003 end pose"), (155, "sofa_0000 end"), (5, "glass panel")):
    sp = surf("b", i, nb - 1); dd, _ = tv.query(sp, k=1, distance_upper_bound=0.05); print(f"  recall {nm:22s}: {100*(dd<0.05).mean():.1f} %")
ent = curl[np.minimum(j, len(curl) - 1)]; u, c = np.unique(ent[e], return_counts=True); top = sorted(zip(c.tolist(), u.tolist()), reverse=True)[:10]
print("errors by nearest entity (top 10):", ", ".join(f"{MAN.get(i,'?')} {n} ({100*n/len(v):.2f}%)" for n, i in top))
gp = surf("b", 5, nb - 1, 20000); tp = cKDTree(gp); dp, _ = tp.query(v, k=1, distance_upper_bound=0.6); n = dp < 0.6
print(f"panel: within 0.6 m {n.sum()} ({100*n.mean():.2f} %), 5-20 cm off {int(((dp>0.05)&(dp<0.2)).sum())} ({100*((dp>0.05)&(dp<0.2)).sum()/max(1,n.sum()):.0f} % of those)")
