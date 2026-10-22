"""The 4D map with D1's moving furniture (track_d1.py): at time t, our latest round at or before t, and for every tracked
object whose window has begun, its rows (its identity) replaced by its model posed at its latest tracked pose at or
before t (held after the window): the object moves with the frames that watched it. Nothing else is taken out of the
round (10-23: clearing the unlabelled rows of the place it left took a wall corner by the plant; what showed there was a
hole, the floor under the pot never seen, which the next round maps). Rendered from a camera JSON as
render_4d_rounds.py renders (crop box, white background). With --rounds-only the plain 4D map (no tracked objects).
Writes OUT_DIR/<session>_<t>.png and OUT_DIR/renders.json.
  render_d1_4d.py CAMERA.json RUN SESSION TIMES(a,b,... s | t0:t1:step) OUT_DIR [--tracks T.json ...] [--rounds-only]
                  [--lamps DONE.json]"""
import json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

W = str(Path(__file__).resolve().parents[3])                                    # the repository's root
sys.path.insert(0, W)
from update_layer.backends.game.game import GameBackend, _View, flashsplat_render, gu  # noqa: E402
from update_layer.backends.game.snapshots import _replay  # noqa: E402
from update_layer.frames import synthetic_session  # noqa: E402
from update_layer.run import dataset_config  # noqa: E402

cam = json.load(open(sys.argv[1])); run, s, out = Path(sys.argv[2]), sys.argv[3], Path(sys.argv[5])
spec = synthetic_session(s)
if ":" in sys.argv[4]:
    a, b, st = (float(v) for v in sys.argv[4].split(":"))
    times = [round(a + k * st, 4) for k in range(int(round((b - a) / st)) + 1)]
else:
    times = [float(v) for v in sys.argv[4].split(",")]
tracks = []
if "--tracks" in sys.argv:
    for f in sys.argv[sys.argv.index("--tracks") + 1:]:
        if f.startswith("--"):
            break
        tracks.append(json.load(open(f)))
rounds_only = "--rounds-only" in sys.argv
lamps = [(np.array(lo) - 0.1, np.array(hi) + 0.1) for _, lo, hi in json.load(open(sys.argv[sys.argv.index("--lamps") + 1]))["lamps_m"]] \
    if "--lamps" in sys.argv else []
out.mkdir(parents=True, exist_ok=True)
_, info, _ = dataset_config("synthetic")
ck = torch.load(sorted(run.glob("checkpoint_*.pt"))[0], map_location="cpu", weights_only=False)
to_gpu = lambda x: (torch.nn.Parameter(x.detach().cuda(), requires_grad=x.requires_grad)
                    if isinstance(x, torch.nn.Parameter) else x.cuda() if torch.is_tensor(x)
                    else type(x)(to_gpu(v) for v in x) if isinstance(x, (tuple, list))
                    else {k: to_gpu(v) for k, v in x.items()} if isinstance(x, dict) else x)
ck["backend"]["model"] = to_gpu(ck["backend"]["model"])
be = GameBackend(info, own_update=False)
be.game = be.prior_from_state(ck["backend"])
gm = be.game.gaussian_model
del ck
H, Wd = int(cam["height"]), int(cam["width"])
K = np.array(cam["K"], dtype=np.float32)
w2c = np.array(cam["T_world_to_camera"], dtype=np.float64); w2c[:3, 3] *= be.scale
view = gu.flashsplat_cam(torch.zeros((3, H, Wd), device="cuda"), torch.zeros((H, Wd), device="cuda"), None, K,
                         torch.from_numpy(w2c.astype(np.float32)), 0)
pipe, bg = gu.flashsplat_pipe(), torch.ones(3, device="cuda")
box = cam["crop_box_world"]
rounds_dir = run / "snapshots" / f"synthetic_{s}" / "rounds"
index = json.loads((rounds_dir / "index.json").read_text())
stamps = [(r["stamp"] - spec.start_ns) / 1e9 for r in index]

# each tracked object's model: its rows in its model round (canonical), and its poses
for tr in tracks:
    _, _, uid, bits, _ = _replay(rounds_dir, tr["model_round"] - 1)
    pos = np.searchsorted(uid, np.array(tr["model_uid"]))
    tr["P"] = bits.view(np.float32)[pos].copy()
    tr["posed"] = [r for r in tr["track"] if "yaw" in r]



def quat_mul(a, b):                                                        # (w, x, y, z)
    w1, x1, y1, z1 = a.T; w2, x2, y2, z2 = b.T
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], 1)


def posed(tr, t):
    """The object's rows at time t: the latest pose at or before t (None before its window)."""
    ps = [r for r in tr["posed"] if r["t"] <= t]
    if t < tr["window"][0] or not ps:
        return None
    r = ps[-1]
    P = tr["P"].copy()
    c, sc, yaw, tau = np.array(tr["centre"]), tr["scale"], r["yaw"], np.array(r["tau"])
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.0]])
    P[:, :3] = (((P[:, :3] / sc - c) @ R.T + c + tau) * sc).astype(np.float32)
    q = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])[None].repeat(len(P), 0)
    rot = P[:, 54:58].astype(np.float64)
    rot /= np.linalg.norm(rot, axis=1, keepdims=True)
    P[:, 54:58] = quat_mul(q, rot).astype(np.float32)
    return P, r["t"]


def inside(xyz):
    m = torch.ones(len(xyz), dtype=torch.bool, device=xyz.device)
    for i, k in enumerate("xyz"):
        lo, hi = box[k]
        m &= (xyz[:, i] >= lo * be.scale) & (xyz[:, i] <= hi * be.scale)
    for lo, hi in lamps:
        m &= ~((xyz >= torch.tensor(lo * be.scale, device=xyz.device, dtype=xyz.dtype)) &
               (xyz <= torch.tensor(hi * be.scale, device=xyz.device, dtype=xyz.dtype))).all(1)
    return m


state, done = None, []
for t in sorted(times):
    k = max((j for j, st in enumerate(stamps) if st <= t), default=None)
    if k is None:
        img = np.full((H, Wd, 3), 255, np.uint8); used = []; n = 0
    else:
        state = _replay(rounds_dir, k, state if state is not None and state[0] <= k else None)
        _, stamp, uid, bits, ints = state
        p = bits.view(np.float32)
        keep = np.ones(len(uid), bool); extra, used = [], []
        if not rounds_only:
            for tr in tracks:
                pt = posed(tr, t)
                if pt is not None:
                    keep &= ints["identity"] != tr["unit"]
                    extra.append(pt[0]); used.append(dict(unit=tr["unit"], pose_t=pt[1]))
        P = np.concatenate([p[keep]] + extra) if extra else p[keep]
        with torch.no_grad():
            q = torch.from_numpy(np.ascontiguousarray(P)).cuda(); n = len(P)
            xyz, dc, rest, scaling, rotation, opacity = torch.split(q, [3, 3, 45, 3, 4, 1], dim=1)
            v = _View(gm, xyz.contiguous(), dc.reshape(n, 1, 3).contiguous(), rest.reshape(n, 15, 3).contiguous(),
                      scaling.contiguous(), rotation.contiguous(), opacity.contiguous())
            mv = inside(v.get_xyz)
            v.get_opacity = v.get_opacity * mv[:, None].float()
            pkg = flashsplat_render(view, v, pipe, bg, obj_num=1)
            img = (torch.clamp(pkg["render"], 0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            del v, pkg, q, xyz, dc, rest, scaling, rotation, opacity, mv
        torch.cuda.empty_cache()
    name = f"{s}_{t:08.3f}.png"
    Image.fromarray(img).save(out / name)
    done.append(dict(t=t, file=name, round=None if k is None else k + 1, tracked=used))
    print("RENDERED", name, "round", None if k is None else k + 1, "tracked", used, flush=True)
(out / "renders.json").write_text(json.dumps(dict(run=str(run), session=s, camera=sys.argv[1], rounds_only=rounds_only,
                                                  tracks=[tr["unit"] for tr in tracks], renders=done), indent=1))
