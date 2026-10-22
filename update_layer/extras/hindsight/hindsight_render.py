"""The 4D map in hindsight at given times (hindsight_intervals.py): the session's background (its last round's unlabelled
rows), every object state whose interval holds t (its rows in its best-observed round), and every watched move from its
start on (track_d1.py's model: its first pose, then the track, then its last pose; its identity's states not drawn
then). Rendered from a camera JSON as render_d1_4d.py renders (crop box, lamps out, white background).
Writes OUT_DIR/<session>_<t>.png and OUT_DIR/renders.json.
  hindsight_render.py CAMERA.json RUN SESSION TIMES(a,b,... s | t0:t1:step) INTERVALS.json TRACK_DIR OUT_DIR [--lamps DONE.json]"""
import json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

W = str(Path(__file__).resolve().parents[3])                                    # the repository's root
sys.path.insert(0, W)
from update_layer.backends.game.game import GameBackend, _View, flashsplat_render, gu  # noqa: E402
from update_layer.backends.game.snapshots import _replay  # noqa: E402
from update_layer.run import dataset_config  # noqa: E402

cam = json.load(open(sys.argv[1])); run, s = Path(sys.argv[2]), sys.argv[3]
if ":" in sys.argv[4]:
    a, b, st = (float(v) for v in sys.argv[4].split(":"))
    times = [round(a + k * st, 4) for k in range(int(round((b - a) / st)) + 1)]
else:
    times = [float(v) for v in sys.argv[4].split(",")]
iv, tracks_dir, out = json.load(open(sys.argv[5])), Path(sys.argv[6]), Path(sys.argv[7])
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
rounds = run / "snapshots" / f"synthetic_{s}" / "rounds"
n_rounds = len(json.loads((rounds / "index.json").read_text()))

# the rows: the background (last round, identity 0), each drawn state's rows in its best round, the tracks' models
states = [r for r in iv["states"] if r["start"] is not None and r["start"] < r["end"]]
need = {}
for r in states:
    need.setdefault(r["best_round"] - 1, []).append(r)
tracks = {}
for t in iv["tracks"]:
    tr = json.loads((tracks_dir / t["file"]).read_text())
    tr["start"] = t["start"]
    tr["posed"] = [x for x in tr["track"] if "yaw" in x]
    need.setdefault(tr["model_round"] - 1, [])
    tracks[tr["unit"]] = tr
background, state_rows, cache = None, {}, None
for k in sorted(set(need) | {n_rounds - 1}):
    cache = _replay(rounds, k, cache)
    _, _, uid, bits, ints = cache
    P = bits.view(np.float32)
    for r in need.get(k, []):
        m = (ints["identity"] == r["identity"]) & (ints["state_birth"] == r["birth"])
        state_rows[(r["identity"], r["birth"])] = P[m].copy()
    for tr in tracks.values():
        if tr["model_round"] - 1 == k:
            tr["P"] = P[np.searchsorted(uid, np.array(tr["model_uid"]))].copy()
    if k == n_rounds - 1:
        background = P[ints["identity"] == 0].copy()
print(f"background {len(background)} rows; {len(states)} states drawn at some time; tracks {sorted(tracks)}", flush=True)


def quat_mul(a, b):                                                        # (w, x, y, z)
    w1, x1, y1, z1 = a.T; w2, x2, y2, z2 = b.T
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], 1)


def posed(tr, t):
    ps = [x for x in tr["posed"] if x["t"] <= t]
    r = ps[-1] if ps else tr["posed"][0]                                      # before the track: its first pose
    P = tr["P"].copy()
    c, sc, yaw, tau = np.array(tr["centre"]), tr["scale"], r["yaw"], np.array(r["tau"])
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.0]])
    P[:, :3] = (((P[:, :3] / sc - c) @ R.T + c + tau) * sc).astype(np.float32)
    rot = P[:, 54:58].astype(np.float64); rot /= np.linalg.norm(rot, axis=1, keepdims=True)
    P[:, 54:58] = quat_mul(np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])[None].repeat(len(P), 0), rot).astype(np.float32)
    return P


def inside(xyz):
    m = torch.ones(len(xyz), dtype=torch.bool, device=xyz.device)
    for i, k in enumerate("xyz"):
        lo, hi = box[k]
        m &= (xyz[:, i] >= lo * be.scale) & (xyz[:, i] <= hi * be.scale)
    for lo, hi in lamps:
        m &= ~((xyz >= torch.tensor(lo * be.scale, device=xyz.device, dtype=xyz.dtype)) &
               (xyz <= torch.tensor(hi * be.scale, device=xyz.device, dtype=xyz.dtype))).all(1)
    return m


done = []
for t in sorted(times):
    tracked = {u for u, tr in tracks.items() if t >= tr["start"]}
    parts = [background]
    drawn = []
    for r in states:
        if r["start"] <= t < r["end"] and r["identity"] not in tracked:
            parts.append(state_rows[(r["identity"], r["birth"])]); drawn.append([r["identity"], r["birth"]])
    for u in sorted(tracked):
        parts.append(posed(tracks[u], t))
    P = np.concatenate(parts)
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
    done.append(dict(t=t, file=name, states=len(drawn), tracked=sorted(tracked)))
    print("RENDERED", name, "states", len(drawn), "tracked", sorted(tracked), flush=True)
(out / "renders.json").write_text(json.dumps(dict(run=str(run), session=s, camera=sys.argv[1], intervals=sys.argv[5], renders=done), indent=1))
