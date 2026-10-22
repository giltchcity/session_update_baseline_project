"""D1 as 4D Gaussians (user 10-23: "我们的D1能不能做成4DGS呢" / "是的 试试吧"): a piece of furniture the camera watches
being moved (the visible changes, D1_visible) is tracked frame by frame as a rigid body, so that the 4D map can show it
moving instead of jumping between two rounds.
  model   the object's Gaussians (its identity) in our latest round before the window (opacity over 0.2, within 1 m
          of their median: stray rows of the identity dropped);
  frames  every dataset frame of the window: the pixels of the object's instance label with depth, back-projected
          (the inputs the mapping reads);
  pose    world = R(yaw) (m - c) + c + tau for a model point m (c: the model's centre): furniture slides on the floor, so
          a yaw and a 3D translation; per frame trimmed ICP (the closest 80 % of the observed points, then those within
          10 cm) from the previous frame's pose; the first frame from a yaw search (every 15 deg) with the centres
          matched.
Writes OUT_DIR/<session>_<unit>.json: the model (uid of its rows, round) and per frame t, yaw, tau, rmse, inliers.
Checked against the ground truth's two states of the move (gt_syn): the model at the first and the last pose against
the state before and after (mean distance of the state's points to the posed model, and back).
  track_d1.py RUN SESSION UNIT T_FROM T_TO BEFORE_STATE AFTER_STATE OUT_DIR"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))                    # the repository's root
from update_layer.backends.game.snapshots import _replay  # noqa: E402
from update_layer.frames import FlatSession, backproject, synthetic_session  # noqa: E402
from update_layer.backends.game.game import CONFIGS  # noqa: E402
from update_layer.run import dataset_config  # noqa: E402

run, s, unit = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
t_from, t_to, before, after, out = float(sys.argv[4]), float(sys.argv[5]), sys.argv[6], sys.argv[7], Path(sys.argv[8])
out.mkdir(parents=True, exist_ok=True)
GT = Path("/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/analysis/scratch_keep_20261022/fourd/gt_syn")
import yaml  # noqa: E402
SCALE = float(yaml.safe_load(CONFIGS[dataset_config("synthetic")[1].name].read_text())["game"].get("scale", 1.0))   # map units per metre
spec = synthetic_session(s); fs = FlatSession(spec); rel = np.array(fs.rel_ns) / 1e9

# the model: the object's rows in our latest round before the window
rounds = run / "snapshots" / f"synthetic_{s}" / "rounds"
index = json.loads((rounds / "index.json").read_text())
k = max(j for j, r in enumerate(index) if (r["stamp"] - spec.start_ns) / 1e9 <= t_from)
_, stamp, uid, bits, ints = _replay(rounds, k)
p = bits.view(np.float32)
mine = (ints["identity"] == unit) & (p[:, 58] > np.log(0.2 / 0.8))
M = p[mine, :3].astype(np.float64) / SCALE
scale = SCALE
keep = np.linalg.norm(M - np.median(M, 0), axis=1) < 1.0
M, model_uid = M[keep], uid[mine][keep]
c = M.mean(0)
tree = cKDTree(M[:: max(1, len(M) // 6000)])
print(f"model: round {k + 1} ({(stamp - spec.start_ns) / 1e9:.1f} s), {len(M)} Gaussians of unit {unit}, centre {c.round(2)} m", flush=True)


def Rz(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1.0]])


def to_model(O, yaw, tau):
    return (O - c - tau) @ Rz(yaw) + c                                       # R^T (o - c - tau) + c, row vectors


def icp(O, yaw, tau, iters=40):
    for it in range(iters):
        Q = to_model(O, yaw, tau)
        d, j = tree.query(Q)
        cut = np.quantile(d, 0.8) if it < iters // 2 else min(np.quantile(d, 0.8), 0.10)
        w = d <= cut
        A, B = tree.data[j[w]] - c, O[w] - c                                    # model (centred) -> world (centred)
        a0, b0 = A.mean(0), B.mean(0)
        H = (A[:, :2] - a0[:2]).T @ (B[:, :2] - b0[:2])
        yaw = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
        tau = b0 - a0 @ Rz(yaw).T
    Q = to_model(O, yaw, tau)
    d, _ = tree.query(Q)
    inl = d < 0.03
    return yaw, tau, float(np.sqrt(np.mean(np.minimum(d, 0.10) ** 2))), float(inl.mean())


frames = [i for i in range(len(rel)) if t_from <= rel[i] <= t_to]
track, yaw, tau = [], None, None
for i in frames:
    f = fs.load(i)
    mask = (f.instance == unit) & np.isfinite(f.depth)
    if mask.sum() < 200:
        track.append(dict(t=float(rel[i]), seen=int(mask.sum())))
        continue
    O, _, _ = backproject(f, mask)
    O = O[:: max(1, len(O) // 3000)].astype(np.float64)
    if yaw is None:                                                            # the first view: a yaw search
        best = None
        for a in np.deg2rad(np.arange(0, 360, 15)):
            t0 = np.median(O, 0) - (np.median(M, 0) - c) @ Rz(a).T - c
            r = icp(O, a, t0, iters=20)
            if best is None or r[2] < best[2]:
                best = r
        yaw, tau, rmse, inl = best
    else:
        yaw, tau, rmse, inl = icp(O, yaw, tau)
    track.append(dict(t=float(rel[i]), seen=int(mask.sum()), yaw=float(yaw), tau=tau.tolist(), rmse=rmse, inliers=inl))
posed = [r for r in track if "yaw" in r]
print(f"{len(frames)} frames, {len(posed)} posed; rmse median {np.median([r['rmse'] for r in posed]) * 100:.1f} cm, "
      f"max {max(r['rmse'] for r in posed) * 100:.1f} cm; inliers (3 cm) median {np.median([r['inliers'] for r in posed]):.2f}", flush=True)


def check(state, r):
    G = np.load(GT / f"{state}.npy")
    W = (M - c) @ Rz(r["yaw"]).T + c + np.array(r["tau"])
    a = cKDTree(W).query(G)[0].mean(); b = cKDTree(G).query(W)[0].mean()
    return a * 100, b * 100


for name, state, r in (("first pose vs the state before", before, posed[0]), ("last pose vs the state after", after, posed[-1])):
    a, b = check(state, r)
    print(f"{name} ({state}, t = {r['t']:.2f} s): state to model {a:.1f} cm, model to state {b:.1f} cm", flush=True)
(out / f"{s}_{unit}.json").write_text(json.dumps(dict(
    session=s, unit=unit, model_round=k + 1, model_stamp=int(stamp), model_uid=model_uid.tolist(), centre=c.tolist(), scale=scale,
    window=[t_from, t_to], before=before, after=after, track=track), indent=0))
