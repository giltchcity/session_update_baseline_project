"""Full-session check of a GaME real map: final background samples against the session's measured depth
(every 10th frame), surface area, agreement with the points map; GaME keyframe state from the checkpoint."""
import sys, numpy as np, torch
sys.path.insert(0, "/home/jixian/Desktop/FT/session_update_baseline_project")
from scipy.spatial import cKDTree
from update_layer.eval.scenelist import load_timeline
from update_layer.frames import FlatSession, real_session
R = "/home/jixian/Desktop/FT/runs/"; s = sys.argv[1]; runs = dict(a.split("=", 1) for a in sys.argv[2:])
fs = FlatSession(real_session(s), pixel_step=4); frames = [fs.load(i) for i in range(0, len(fs.ids), 10)]
def classify(P):
    free = np.zeros(len(P), np.int32); on = np.zeros(len(P), np.int32); beh = np.zeros(len(P), np.int32)
    for f in frames:
        T = np.linalg.inv(f.T_world_cam); c = P @ T[:3, :3].T + T[:3, 3]; z = c[:, 2]
        u = np.floor(f.K.fx * c[:, 0] / np.maximum(z, 1e-6) + f.K.cx - f.K.offset + .5).astype(int)
        v = np.floor(f.K.fy * c[:, 1] / np.maximum(z, 1e-6) + f.K.cy - f.K.offset + .5).astype(int)
        ok = (z > 0.1) & (u >= 0) & (u < f.K.width) & (v >= 0) & (v < f.K.height)
        d = np.full(len(P), np.nan); d[ok] = f.depth[v[ok], u[ok]]; m = np.isfinite(d)
        free += m & (z < d - .05); on += m & (np.abs(z - d) <= .05); beh += m & (z > d + .05)
    seen = free + on + beh > 0
    return seen, seen & (on >= free) & (on >= beh), seen & (free > on)
rs = load_timeline(R + "update_layer_points_real_final_20260930/row4/session_%s/timeline.pkl" % s); rs = rs.scene(rs.stamps()[-1])
ref = np.concatenate([rs.background] + [o.points for o in rs.objects])
for name, rel in list(runs.items()) + [("points (control)", "update_layer_points_real_final_20260930/row4")]:
    tl = load_timeline(R + rel + f"/session_{s}/timeline.pkl"); sc = tl.scene(tl.stamps()[-1])
    P = np.concatenate([sc.background] + [o.points for o in sc.objects])
    area = len(np.unique(np.floor(P / 0.02).astype(np.int64), axis=0)) * 0.02 ** 2
    seen, on, free = classify(sc.background)
    d = cKDTree(ref).query(P, distance_upper_bound=1.0)[0]; d[~np.isfinite(d)] = 1.0
    r = cKDTree(P).query(ref, distance_upper_bound=0.05)[0]
    print(f"{name:22s} objects {len(sc.objects):2d} | surface ~{area:5.0f} m2 | bg vs measured: on {100*on[seen].mean():5.1f}% free {100*free[seen].mean():5.1f}%"
          f" | within 5cm of points {100*np.mean(d < .05):5.1f}% | covers points {100*np.mean(np.isfinite(r)):5.1f}%")
    try:
        ck = torch.load(R + rel + "/checkpoint.pt", map_location="cpu", mmap=True, weights_only=False)["backend"]
        print(f"{'':22s} keyframes {len(ck['keyframes'])} ignored {len(ck['ignored_frames'])} Gaussians {ck['model'][1].shape[0]}")
    except Exception:
        pass
