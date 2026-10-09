# Where synthetic B's precision and recall are lost, geometrically: our map vs the exact GT at B's end (GT meshes posed
# at the last frame), error points (> 5 cm from GT) clustered (DBSCAN 10 cm), each cluster labelled by the nearest GT
# entity, and compared at the same place with the decided map and row 3's map. Recall misses likewise (GT > 5 cm from map).
import sys, json, numpy as np, open3d as o3d
from scipy.spatial import cKDTree
D = "/home/jixian/Desktop/FT/datasets/synthetic_ab"; R = "/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006"
man = json.load(open(f"{D}/gt_geometry/manifest.json"))["b"]["objects"]; z = np.load(f"{D}/gt_geometry/b_local_meshes.npz")
g = np.load(f"{D}/session_b/gt/temporal_gt.npz"); ids = list(g["instance_ids"]); T = g["T_world_object"][-1]; vis = g["render_visible"][-1]
pts, lab = [], []
rng = np.random.default_rng(0)
for o in man:
    i = o["instance_id"]; j = ids.index(i)
    if not vis[j]: continue
    V = z[f"{i:04d}_vertices"].astype(np.float64); F = z[f"{i:04d}_faces"].astype(np.int32)
    V = (np.c_[V, np.ones(len(V))] @ T[j].T)[:, :3]
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
    area = m.get_surface_area(); n = max(50, int(area * 4000))
    p = np.asarray(m.sample_points_uniformly(number_of_points=n).points)
    pts.append(p); lab.append(np.full(len(p), i))
gt = np.vstack(pts); gl = np.concatenate(lab); name = {o["instance_id"]: f'{o["name"]}' for o in man}
print(f"GT cloud at B end: {len(gt)} points from {len(pts)} visible entities", flush=True)
tree = cKDTree(gt)
def load(p, n=600000):
    m = o3d.io.read_triangle_mesh(p); v = np.asarray(m.vertices)
    if len(v) > n: v = v[rng.choice(len(v), n, replace=False)]
    return v
maps = {"ours 0c9c487": f"{R}/syn_row4i/session_b/tsdf_final.ply", "decided": f"{R}/syn_row4s3cb/session_b/tsdf_final.ply", "row 3": f"{R}/syn_row3v3/session_b/tsdf_final.ply"}
err = {}; mp = {}
for k, p in maps.items():
    v = load(p); d, j = tree.query(v, k=1, distance_upper_bound=1.0); e = d > 0.05
    mp[k] = v; err[k] = (v[e], gl[np.minimum(j[e], len(gl) - 1)], d[e])
    print(f"{k}: {len(v)} pts, precision proxy {100*(1-e.mean()):.2f} % (> 5 cm: {e.sum()})", flush=True)
# clusters of OUR error points
v, l, d = err["ours 0c9c487"]
pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(v)); cl = np.asarray(pc.cluster_dbscan(eps=0.10, min_points=30))
trees = {k: cKDTree(err[k][0]) for k in maps}
print("\nOUR precision-error clusters (size >= 300), nearest GT entity, and the error count of each map within 0.3 m of the cluster centroid:")
rows = []
for c in range(cl.max() + 1):
    s = cl == c
    if s.sum() < 300: continue
    cen = v[s].mean(0); labs, cnt = np.unique(l[s], return_counts=True); top = labs[np.argmax(cnt)]
    counts = {k: len(trees[k].query_ball_point(cen, 0.3)) for k in maps}
    rows.append((int(s.sum()), cen, top, float(d[s].mean()), counts))
rows.sort(key=lambda r: -r[0])
for n, cen, top, md, counts in rows[:30]:
    print(f"  n={n:6d} at ({cen[0]:.2f},{cen[1]:.2f},{cen[2]:.2f}) near {name.get(int(top),'?')} (mean err {md*100:.0f} cm) | ours {counts['ours 0c9c487']} decided {counts['decided']} row3 {counts['row 3']}")
# recall misses: GT points > 5 cm from each map, by entity
print("\nRecall misses by GT entity (GT points > 5 cm from the map), top 15 for ours, with decided / row 3 at the same entity:")
miss = {}
for k in maps:
    t2 = cKDTree(mp[k]); d2, _ = t2.query(gt, k=1, distance_upper_bound=1.0); miss[k] = d2 > 0.05
    print(f"  {k}: recall proxy {100*(1-miss[k].mean()):.2f} %")
ents = np.unique(gl)
tab = sorted(((miss["ours 0c9c487"][gl == e].sum(), e) for e in ents), reverse=True)[:15]
for n, e in tab:
    print(f"  {name.get(int(e),'?'):28s} ours {n:6d} / decided {miss['decided'][gl==e].sum():6d} / row3 {miss['row 3'][gl==e].sum():6d}  (of {int((gl==e).sum())} GT pts)")
