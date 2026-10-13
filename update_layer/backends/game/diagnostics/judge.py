"""Judge the 150-frame tests: background samples of the final snapshot against measured depth; GaME state from the checkpoint."""
import sys, numpy as np, torch
sys.path.insert(0, "/home/jixian/Desktop/FT/session_update_baseline_project")
from update_layer.eval.scenelist import load_timeline
from update_layer.frames import FlatSession, real_session
R = "/home/jixian/Desktop/FT/runs/"
fs = FlatSession(real_session("a"), pixel_step=4)
frames = [fs.load(i) for i in range(0, 150, 5)]
def classify(P):
    free = np.zeros(len(P), int); on = np.zeros(len(P), int); beh = np.zeros(len(P), int)
    for f in frames:
        T = np.linalg.inv(f.T_world_cam); c = P @ T[:3, :3].T + T[:3, 3]; z = c[:, 2]
        u = np.floor(f.K.fx * c[:, 0] / np.maximum(z, 1e-6) + f.K.cx - f.K.offset + .5).astype(int)
        v = np.floor(f.K.fy * c[:, 1] / np.maximum(z, 1e-6) + f.K.cy - f.K.offset + .5).astype(int)
        ok = (z > 0.1) & (u >= 0) & (u < f.K.width) & (v >= 0) & (v < f.K.height)
        d = np.full(len(P), np.nan); d[ok] = f.depth[v[ok], u[ok]]; m = np.isfinite(d)
        free += m & (z < d - .05); on += m & (np.abs(z - d) <= .05); beh += m & (z > d + .05)
    seen = free + on + beh > 0
    return seen, seen & (on >= free) & (on >= beh), seen & (free > on)
runs = {"aria, no fix (row 4)": "update_layer_game_real_20260930/row4_noC4C5_20261001",
        "aria + C4C5 (row 4)": "update_layer_game_c4c5_test_20261001/row4",
        "kinect + C4C5 + E1 (row 4)": "update_layer_game_fixall_test_20261001/row4",
        "kinect + C4C5 + E1 (row 3)": "update_layer_game_fixall_test_20261001/row3",
        "points backend (control)": "update_layer_points_real_final_20260930/row4"}
t = load_timeline(R + runs["kinect + C4C5 + E1 (row 4)"] + "/session_a/timeline.pkl").stamps()[-1]
for name, rel in runs.items():
    try:
        tl = load_timeline(R + rel + "/session_a/timeline.pkl")
    except FileNotFoundError:
        print(f"{name:28s} (missing)"); continue
    sc = tl.scene(t); P = sc.background; O = np.concatenate([o.points for o in sc.objects]) if sc.objects else np.zeros((0, 3))
    seen, on, free = classify(P)
    print(f"{name:28s} bg {len(P):7d} obj {len(O):7d} objects {len(sc.objects):2d} | bg on surface {100*on[seen].mean():5.1f}%  free space {100*free[seen].mean():5.1f}%  behind {100*(seen&~on&~free)[seen].mean():5.1f}%")
for r in (4, 3):
    try:
        ck = torch.load(R + f"update_layer_game_fixall_test_20261001/row{r}/checkpoint.pt", map_location="cpu", mmap=True, weights_only=False)["backend"]
        print(f"row {r} GaME state: frames {len(ck['estimated_poses'])} keyframes {len(ck['keyframes'])} ignored {len(ck['ignored_frames'])} occlusion masks {len(ck['occlusion_masks'])} Gaussians {ck['model'][1].shape[0]}")
    except FileNotFoundError:
        pass
