# Object STATES of the synthetic B map at B's end vs the exact GT: every GT change A->B (removed, new, moved between
# sessions, moved within B) and every object present at B's end. Per object: present in the map (recall of its GT surface
# at its B-end pose, 5 cm), residue at its old site(s) (map points within 5 cm of the old-pose surface that are > 5 cm /
# > 10 cm from every current GT surface). ours/decided = scored readout (timeline final scene); row 3 = tsdf_final.ply.
import sys, json, pickle, numpy as np, open3d as o3d
from scipy.spatial import cKDTree
sys.path.insert(0, "/home/jixian/Desktop/FT/session_update_baseline_project")
D = "/home/jixian/Desktop/FT/datasets/synthetic_ab"; R = "/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006"
G = {s: np.load(f"{D}/session_{s}/gt/temporal_gt.npz") for s in "ab"}; Z = {s: np.load(f"{D}/gt_geometry/{s}_local_meshes.npz") for s in "ab"}
MAN = {s: {o["instance_id"]: o for o in json.load(open(f"{D}/gt_geometry/manifest.json"))[s]["objects"]} for s in "ab"}
IDS = {s: list(G[s]["instance_ids"]) for s in "ab"}
def exists(s, i, f): return i in IDS[s] and bool(G[s]["render_visible"][f, IDS[s].index(i)])
def pose(s, i, f): return G[s]["T_world_object"][f, IDS[s].index(i)]
def surf(s, i, T, per_m2=4000):
    V = Z[s][f"{i:04d}_vertices"].astype(np.float64); F = Z[s][f"{i:04d}_faces"].astype(np.int32); V = (np.c_[V, np.ones(len(V))] @ T.T)[:, :3]
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
    return np.asarray(m.sample_points_uniformly(number_of_points=max(100, int(m.get_surface_area() * per_m2))).points)
nb = len(G["b"]["render_visible"]); na = len(G["a"]["render_visible"])
# current GT (B end) tree with labels
cur = []; curl = []
for i in IDS["b"]:
    if exists("b", i, nb - 1): p = surf("b", i, pose("b", i, nb - 1)); cur.append(p); curl.append(np.full(len(p), i))
cur = np.vstack(cur); curl = np.concatenate(curl); tcur = cKDTree(cur)
name = lambda i: MAN["b"].get(i, MAN["a"].get(i, {"name": "?"}))["name"]
# GT change list
changes = []
for i in sorted(set(IDS["a"]) | set(IDS["b"])):
    inA = exists("a", i, na - 1); inB0 = exists("b", i, 0); inB1 = exists("b", i, nb - 1)
    if not (inA or inB0 or inB1): continue
    tA = pose("a", i, na - 1)[:3, 3] if inA else None; tB0 = pose("b", i, 0)[:3, 3] if inB0 else None; tB1 = pose("b", i, nb - 1)[:3, 3] if inB1 else None
    kinds = []
    if inA and not inB1: kinds.append("removed")
    if not inA and inB1: kinds.append("new")
    if inA and inB0 and np.linalg.norm(tA - tB0) > 0.05: kinds.append(f"moved A->B {np.linalg.norm(tA-tB0)*100:.0f} cm")
    if inB0 and inB1:
        j = IDS["b"].index(i); tr = G["b"]["T_world_object"][:, j, :3, 3]; disp = np.linalg.norm(tr - tr[0], axis=1)
        if disp.max() > 0.05: kinds.append(f"moved within B {np.linalg.norm(tB1-tB0)*100:.0f} cm (max {disp.max()*100:.0f} cm, frames {int(np.argmax(disp>0.05))}-{int(len(disp)-1-np.argmax(disp[::-1]>0.05))})")
    if kinds: changes.append((i, kinds))
print(f"GT changes A->B: {len(changes)} objects"); 
for i, k in changes: print(f"  {name(i):26s} ({i:3d}): {'; '.join(k)}")
def scene_points(path):
    tl = pickle.load(open(path, "rb")); sc = tl.scenes[-1]
    bg = np.asarray(sc.background, dtype=np.float64); ob = [(np.asarray(x.points, dtype=np.float64), x.instance_id) for x in sc.objects if x.present and len(x.points)]
    return np.vstack([bg] + [p for p, _ in ob]), np.concatenate([np.zeros(len(bg), np.int64)] + [np.full(len(p), i, np.int64) for p, i in ob])
def ply_points(path):
    v = np.asarray(o3d.io.read_triangle_mesh(path).vertices, dtype=np.float64); return v, np.full(len(v), -1, np.int64)
maps = {"ours(scene)": scene_points(f"{R}/syn_row4j/session_b/timeline.pkl"), "decided(scene)": scene_points(f"{R}/syn_row4s3cb/session_b/timeline.pkl"), "row3(ply)": ply_points(f"{R}/syn_row3v3/session_b/tsdf_final.ply")}
res = {}
for k, (v, lab) in maps.items():
    d, j = tcur.query(v, k=1, distance_upper_bound=1.0); res[k] = (v, lab, d, cKDTree(v))
    print(f"{k}: {len(v)} points, > 5 cm from current GT: {int((d > 0.05).sum())} ({100*(d>0.05).mean():.1f} %), > 10 cm: {int((d > 0.10).sum())}")
# 1) old sites: residue
print("\nOLD SITES (map points within 5 cm of the old-pose surface that are > 5 cm | > 10 cm from every current GT surface):")
print(f"  {'object':26s} {'site':24s} " + " ".join(f"{k:>18s}" for k in maps))
for i, kinds in changes:
    sites = []
    if exists("a", i, na - 1) and ("removed" in kinds[0] or any("A->B" in x for x in kinds)): sites.append(("A-end pose", surf("a", i, pose("a", i, na - 1))))
    if any("within B" in x for x in kinds):
        j = IDS["b"].index(i); tr = G["b"]["T_world_object"][:, j, :3, 3]; f0 = int(np.argmax(np.linalg.norm(tr - tr[0], axis=1) > 0.05))
        sites.append(("B-start pose", surf("b", i, pose("b", i, 0))))
        f1 = int(len(tr) - 1 - np.argmax(np.linalg.norm(tr - tr[-1], axis=1)[::-1] > 0.05))
        mid = [f for f in range(f0, f1, max(1, (f1 - f0) // 4))][1:4]
        for f in mid: sites.append((f"frame {f} pose", surf("b", i, pose("b", i, f))))
    for sname, sp in sites:
        ts = cKDTree(sp); line = f"  {name(i):26s} {sname:24s} "
        for k, (v, lab, d, tv) in res.items():
            near = np.unique(np.concatenate(tv.query_ball_point(sp, 0.05)).astype(int)) if len(sp) else np.zeros(0, int)
            line += f"{int((d[near] > 0.05).sum()):9d}|{int((d[near] > 0.10).sum()):8d}"
        print(line)
# 2) present objects: recall at B-end pose (and label-consistent recall for the scenes)
print("\nPRESENT at B end: recall of the GT surface (5 cm) per map; '*' = changed object. Listed: every changed object + any object with recall < 90 % in ours")
print(f"  {'object':26s} {'cat':8s} " + " ".join(f"{k:>14s}" for k in maps) + "   ours label-consistent")
rows = []
for i in IDS["b"]:
    if not exists("b", i, nb - 1): continue
    sp = surf("b", i, pose("b", i, nb - 1)); rec = {}
    for k, (v, lab, d, tv) in res.items():
        dd, jj = tv.query(sp, k=1, distance_upper_bound=0.05); rec[k] = 100 * (dd < 0.05).mean()
    v, lab, d, tv = res["ours(scene)"]; dd, jj = tv.query(sp, k=1, distance_upper_bound=0.05); hit = dd < 0.05
    lc = 100 * (lab[jj[hit]] == i).mean() if hit.any() else float("nan")
    chg = any(c[0] == i for c in changes)
    if chg or rec["ours(scene)"] < 90: rows.append((i, chg, rec, lc, len(sp)))
for i, chg, rec, lc, n in sorted(rows, key=lambda r: r[2]["ours(scene)"]):
    print(f"  {('*' if chg else ' ') + name(i):26s} {i:3d}     " + " ".join(f"{rec[k]:13.1f}%" for k in maps) + f"   {lc:5.1f}% of hits labelled {i} (GT samples {n})")
# 3) where the error points are, by nearest current GT entity (top 12 of ours), with the other maps at the same entity
print("\nERROR POINTS (> 5 cm from current GT) by nearest current GT entity, top 12 of ours:")
v, lab, d, tv = res["ours(scene)"]; dj = tcur.query(v, k=1, distance_upper_bound=1.0)[1]; e = d > 0.05; ent = curl[np.minimum(dj, len(curl) - 1)]
others = {}
for k, (vv, ll, ddk, _) in res.items():
    jj = tcur.query(vv, k=1, distance_upper_bound=1.0)[1]; ee = ddk > 0.05; en = curl[np.minimum(jj, len(curl) - 1)]; u, c = np.unique(en[ee], return_counts=True); others[k] = (dict(zip(u.tolist(), c.tolist())), int(ee.sum()), len(vv))
u, c = np.unique(ent[e], return_counts=True); top = sorted(zip(c.tolist(), u.tolist()), reverse=True)[:12]
print(f"  {'entity':26s} " + " ".join(f"{k:>22s}" for k in maps))
for c, i in top:
    print(f"  {name(i):26s} " + " ".join(f"{others[k][0].get(i, 0):9d} ({100*others[k][0].get(i,0)/others[k][2]:4.2f}%)" for k in maps))
print("  total errors: " + ", ".join(f"{k} {others[k][1]} of {others[k][2]} ({100*others[k][1]/others[k][2]:.2f} %)" for k in maps))
