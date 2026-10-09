# The ours-specific B error cluster at bathroom_product_0002 (instance 5): what the scored readout points are (scene label,
# signed offset from the GT surface along its normal, which stored view produced them is not stored) and which alive rows of
# the checkpoint stand at those points (distance, identity, creation session, opacity, extent), ours vs decided.
import sys, json, pickle, numpy as np, open3d as o3d, torch
from scipy.spatial import cKDTree
sys.path.insert(0, "/home/jixian/Desktop/FT/session_update_baseline_project")
from update_layer.backends.game import t1
D = "/home/jixian/Desktop/FT/datasets/synthetic_ab"; R = "/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006"; MAX = torch.iinfo(torch.int64).max
sess = "b"; L = 118.5
man = json.load(open(f"{D}/gt_geometry/manifest.json"))[sess]["objects"]; z = np.load(f"{D}/gt_geometry/{sess}_local_meshes.npz"); g = np.load(f"{D}/session_{sess}/gt/temporal_gt.npz"); ids = list(g["instance_ids"]); T = g["T_world_object"][-1]
o = [x for x in man if x["instance_id"] == 5][0]; j = ids.index(5)
V = z["0005_vertices"].astype(np.float64); F = z["0005_faces"].astype(np.int32); V = (np.c_[V, np.ones(len(V))] @ T[j].T)[:, :3]
m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F)); m.compute_vertex_normals()
pc = m.sample_points_uniformly(number_of_points=max(2000, int(m.get_surface_area() * 20000))); gp = np.asarray(pc.points); gn = np.asarray(pc.normals); tree = cKDTree(gp)
bb = gp.min(0), gp.max(0); print(f"product GT: {len(gp)} samples, bbox {bb[0].round(2)} .. {bb[1].round(2)}, area {m.get_surface_area():.3f} m2, watertight {m.is_watertight()}")
# all-GT tree for the 'nearest entity' attribution as the cluster script did
pts, lab = [], []
for oo in man:
    i = oo["instance_id"]; jj = ids.index(i)
    if not g["render_visible"][-1, jj]: continue
    VV = z[f"{i:04d}_vertices"].astype(np.float64); FF = z[f"{i:04d}_faces"].astype(np.int32); VV = (np.c_[VV, np.ones(len(VV))] @ T[jj].T)[:, :3]
    mm = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(VV), o3d.utility.Vector3iVector(FF)); p = np.asarray(mm.sample_points_uniformly(number_of_points=max(50, int(mm.get_surface_area() * 4000))).points); pts.append(p); lab.append(np.full(len(p), i))
gt = np.vstack(pts); gl = np.concatenate(lab); tall = cKDTree(gt)
def scene_points(path):
    tl = pickle.load(open(path, "rb")); sc = tl.scenes[-1]
    bg = np.asarray(sc.background, dtype=np.float64); ob = [(np.asarray(x.points, dtype=np.float64), x.instance_id) for x in sc.objects if x.present and len(x.points)]
    v = np.vstack([bg] + [p for p, _ in ob]); src = np.concatenate([np.zeros(len(bg), np.int64)] + [np.full(len(p), i, np.int64) for p, i in ob]); return v, src
for name, run in (("ours", "syn_row4j"), ("decided", "syn_row4s3cb")):
    v, src = scene_points(f"{R}/{run}/session_{sess}/timeline.pkl")
    d, jn = tree.query(v, k=1, distance_upper_bound=0.6); near = d < 0.6
    dall, ja = tall.query(v[near], k=1, distance_upper_bound=1.0); errn = dall > 0.05; lab_near = gl[np.minimum(ja, len(gl) - 1)]
    e = np.zeros(len(v), bool); e[np.nonzero(near)[0][errn & (lab_near == 5)]] = True   # error points attributed to the product
    off = ((v[e] - gp[jn[e]]) * gn[jn[e]]).sum(1)
    labs, cnt = np.unique(src[e], return_counts=True)
    print(f"\n== {name}: readout points within 0.6 m of the product {near.sum()} (error > 5 cm, nearest entity = product: {e.sum()}); scene labels of the error points {dict(zip(labs.tolist(), cnt.tolist()))}")
    print(f"   signed offset along the GT normal (outside > 0): <-10 {100*(off<-0.10).mean():.0f} %, -10..-5 {100*((off>=-0.10)&(off<-0.05)).mean():.0f} %, 5..10 {100*((off>0.05)&(off<=0.10)).mean():.0f} %, 10..20 {100*((off>0.10)&(off<=0.20)).mean():.0f} %, >20 {100*(off>0.20).mean():.0f} %; |d| median {100*np.median(d[e]):.1f} cm")
    ck = torch.load(f"{R}/{run}/checkpoint_{sess}.pt", map_location="cpu", weights_only=False, mmap=True); b = ck["backend"]; t = int(ck["prev_final"]); t0 = t - int(L * 1e9)
    dp = b.get("death_prune"); alive = t1.alive_at(t, b["identity"], b["created"], b["state_birth"], b["death_state"], b["death_evidence"], b.get("bg_birth"), dp).numpy()
    ten = [x for x in b["model"] if torch.is_tensor(x) and x.dim() == 2 and x.shape[0] == len(alive)]; xyz = (ten[0].detach().float() / 10.0).numpy(); scal = [x for x in ten if x.shape[1] == 3][1].detach().float(); ext = (torch.exp(scal).max(dim=1).values * 3 / 10.0).numpy()
    opa = torch.sigmoid([x for x in ten if x.shape[1] == 1][0].detach().float()).squeeze(1).numpy()
    ident = b["identity"].numpy(); cr = (b["created"].numpy() - t0) / 1e9
    ra = np.nonzero(alive)[0]; rt = cKDTree(xyz[ra]); dr, jr = rt.query(v[e], k=1, distance_upper_bound=0.3); hit = dr < 0.03; rows = ra[jr[hit]]
    print(f"   nearest ALIVE row to each error point: within 3 cm {100*hit.mean():.0f} %, 3-10 cm {100*((dr>=0.03)&(dr<0.10)).mean():.0f} %, >10 cm {100*(dr>=0.10).mean():.0f} %")
    u, c = np.unique(rows, return_counts=True)
    print(f"   rows hit (unique {len(u)}): identity {dict(zip(*[x.tolist() for x in np.unique(ident[u], return_counts=True)]))}; created in A {100*(cr[u]<0).mean():.0f} % / in B {100*(cr[u]>=0).mean():.0f} %; opacity median {np.median(opa[u]):.2f} (<0.1: {100*(opa[u]<0.1).mean():.0f} %, >0.5: {100*(opa[u]>0.5).mean():.0f} %); extent median {100*np.median(ext[u]):.1f} cm, >10 cm {100*(ext[u]>0.10).mean():.0f} %")
    dg, _ = tree.query(xyz[u], k=1); print(f"   those rows' own distance to the product GT: <5 {100*(dg<0.05).mean():.0f} % 5-10 {100*((dg>=0.05)&(dg<0.1)).mean():.0f} % >10 {100*(dg>=0.1).mean():.0f} %")
    # all alive rows near the product by opacity, for the comparison of the populations
    da, _ = tree.query(xyz[ra], k=1, distance_upper_bound=0.6); n6 = da < 0.6; rr = ra[n6]
    print(f"   all alive rows within 0.6 m: {len(rr)}; opacity: <0.1 {100*(opa[rr]<0.1).mean():.0f} %, 0.1-0.5 {100*((opa[rr]>=0.1)&(opa[rr]<0.5)).mean():.0f} %, >0.5 {100*(opa[rr]>=0.5).mean():.0f} %; extent >10 cm {100*(ext[rr]>0.10).mean():.1f} %; rows >10 cm off with opacity >0.5: {int(((da[n6]>0.1)&(opa[rr]>0.5)).sum())}")
