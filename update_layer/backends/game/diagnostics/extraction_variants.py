"""Which surface extraction from GaME's Gaussians is faithful? Variants on the C4+C5 test model (real A, 150 frames):
mean depth at alpha>0.5 (current) / alpha>0.9 / + depth-edge filter / + agreement with the keyframe's measured depth.
Judged per sample against measured depth of the session's frames (on surface / free space / behind)."""
import sys, numpy as np, torch
sys.path.insert(0, "/home/jixian/Desktop/FT/session_update_baseline_project")
from update_layer.run import dataset_config, make_backend
from update_layer.frames import FlatSession, real_session
from update_layer.backends.game.game import gu, flashsplat_render
cfg, info, specs = dataset_config("real")
be = make_backend("game", info, False)
ck = torch.load("/home/jixian/Desktop/FT/runs/update_layer_game_c4c5_test_20261001/row4/checkpoint.pt", weights_only=False)   # model tensors come back on the GPU, as in run.py --resume
g = be.prior_from_state(ck["backend"]); del ck
gm = g.gaussian_model; pipe, bgc = gu.flashsplat_pipe(), torch.zeros(3).cuda()
V = {k: [] for k in ("mean a>0.5 (current)", "a>0.5 + edge filter", "a>0.9 + edge filter", "a>0.5 + edge + only where view had depth", "a>0.9 + edge + only where view had depth")}
with torch.no_grad():
    for kid, kf in g.keyframes.items():
        K = kf["intrinsics"]; _, h, w = kf["color"].shape; pose = kf["pose"].cpu()
        view = gu.flashsplat_cam(torch.zeros((3, h, w), device="cuda"), torch.zeros((h, w), device="cuda"), None, K, pose, None)
        pkg = flashsplat_render(view, gm, pipe, bgc, obj_num=1)
        a = pkg["alpha"].squeeze(); D = pkg["depth"].squeeze() / a.clamp(min=1e-6)
        meas = kf["depth"].cuda()
        gy, gx = torch.zeros_like(D), torch.zeros_like(D)
        gx[:, 1:-1] = (D[:, 2:] - D[:, :-2]).abs(); gy[1:-1, :] = (D[2:, :] - D[:-2, :]).abs()
        edge = torch.maximum(gx, gy) > 0.05 * D           # >5% relative jump between neighbours
        masks = {"mean a>0.5 (current)": a > 0.5, "a>0.5 + edge filter": (a > 0.5) & ~edge,
                 "a>0.9 + edge filter": (a > 0.9) & ~edge,
                 "a>0.5 + edge + only where view had depth": (a > 0.5) & ~edge & (meas > 0),
                 "a>0.9 + edge + only where view had depth": (a > 0.9) & ~edge & (meas > 0)}
        Tw = torch.linalg.inv(pose.to(torch.float64)).to("cuda", torch.float32)
        for name, m in masks.items():
            m = m & (D > 0)
            v, u = torch.nonzero(m, as_tuple=True); z = D[v, u]
            x = (u.float() - K[0, 2]) / K[0, 0] * z; y = (v.float() - K[1, 2]) / K[1, 1] * z
            p = torch.stack([x, y, z], 1) @ Tw[:3, :3].T + Tw[:3, 3]
            k = torch.unique(torch.floor(p / 0.02).long(), dim=0, return_inverse=True)[1]
            first = torch.full((int(k.max()) + 1,), len(p), dtype=torch.int64, device="cuda").scatter_reduce_(0, k, torch.arange(len(p), device="cuda"), reduce="amin")
            V[name].append(p[first].cpu().numpy())
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
print(f"Gaussians {gm.get_xyz.shape[0]}, keyframes {len(g.keyframes)}")
for name, parts in V.items():
    P = np.concatenate(parts); k = np.unique(np.floor(P / 0.02).astype(np.int64), axis=0, return_index=True)[1]; P = P[k]
    seen, on, free = classify(P)
    print(f"{name:42s} samples {len(P):8d} | judged {100*seen.mean():4.0f}% | on surface {100*on[seen].mean():5.1f}%  free space {100*free[seen].mean():5.1f}%  behind {100*(seen&~on&~free)[seen].mean():5.1f}%")
