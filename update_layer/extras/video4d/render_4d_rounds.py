"""The 4D video's right panel (user 10-22): every round of a run's 4D map (snapshots: the live map of each round),
rendered from one camera (camera JSON as render_view.py takes it: intrinsics, world-to-camera, size, crop box), white
background, the crop box applied (ceiling cut). The map of time t is the map of the latest round at or before t, so these
renders are the 4D map at every time. Writes OUT_DIR/<session>_<k>_<stamp>.png (k: the round in its session, from 1)
and OUT_DIR/rounds.json (every round of the run). A round whose image exists is not rendered again (its replay still
runs, on the CPU); the card's cache is emptied after every round (10-22: it grew to 15 GB over 17 rounds).
--lamps DONE.json: the Gaussians inside the lamps Isaac hides (render_video_isaac.py's lamps_m, padded 0.1 m) are not
drawn, as the ground truth does not draw them.
  render_4d_rounds.py CAMERA.json RUN OUT_DIR [--lamps DONE.json] [--only SESSION:K ...]"""
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

cam = json.load(open(sys.argv[1])); run, out = Path(sys.argv[2]), Path(sys.argv[3])
only = set(sys.argv[sys.argv.index("--only") + 1:]) if "--only" in sys.argv else None
opt = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
lamps = [(np.array(lo) - 0.1, np.array(hi) + 0.1) for _, lo, hi in json.load(open(opt("--lamps")))["lamps_m"]] if opt("--lamps") else []
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
w2c = np.array(cam["T_world_to_camera"], dtype=np.float64); w2c[:3, 3] *= be.scale            # map units
view = gu.flashsplat_cam(torch.zeros((3, H, Wd), device="cuda"), torch.zeros((H, Wd), device="cuda"), None, K,
                         torch.from_numpy(w2c.astype(np.float32)), 0)
pipe, bg = gu.flashsplat_pipe(), torch.ones(3, device="cuda")
box = cam["crop_box_world"]


def inside(xyz):
    m = torch.ones(len(xyz), dtype=torch.bool, device=xyz.device)
    for i, k in enumerate("xyz"):
        lo, hi = box[k]
        m &= (xyz[:, i] >= lo * be.scale) & (xyz[:, i] <= hi * be.scale)
    for lo, hi in lamps:
        m &= ~((xyz >= torch.tensor(lo * be.scale, device=xyz.device, dtype=xyz.dtype)) &
               (xyz <= torch.tensor(hi * be.scale, device=xyz.device, dtype=xyz.dtype))).all(1)
    return m



rounds = []
for sdir in sorted(p for p in (run / "snapshots").iterdir() if (p / "rounds" / "index.json").exists()):
    for k, r in enumerate(json.loads((sdir / "rounds" / "index.json").read_text()), 1):
        rounds.append(dict(session=sdir.name, k=k, stamp=int(r["stamp"]), rows=int(r["rows"])))
done, state = [], {}
for r in rounds:
    if only is not None and f"{r['session']}:{r['k']}" not in only:
        continue
    # each session's own rounds (map_at's tie rule gives A's last stamp to A, and B's first round is that stamp)
    state = {r["session"]: _replay(run / "snapshots" / r["session"] / "rounds", r["k"] - 1, state.get(r["session"]))}
    _, stamp, uid, bits, _ = state[r["session"]]
    name = f"{r['session']}_{r['k']:02d}_{r['stamp']}.png"
    if (out / name).exists():
        done.append(dict(r, file=name, rendered=None))
        continue
    with torch.no_grad():
        p = torch.from_numpy(np.ascontiguousarray(bits.view(np.float32))).cuda(); n = len(uid)
        xyz, dc, rest, scaling, rotation, opacity = torch.split(p, [3, 3, 45, 3, 4, 1], dim=1)
        v = _View(gm, xyz.contiguous(), dc.reshape(n, 1, 3).contiguous(), rest.reshape(n, 15, 3).contiguous(),
                  scaling.contiguous(), rotation.contiguous(), opacity.contiguous())
        mv = inside(v.get_xyz)
        v.get_opacity = v.get_opacity * mv[:, None].float()
        pkg = flashsplat_render(view, v, pipe, bg, obj_num=1)
        img = (torch.clamp(pkg["render"], 0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        shown = int(mv.sum())
        del v, pkg, p, xyz, dc, rest, scaling, rotation, opacity, mv
    torch.cuda.empty_cache()
    Image.fromarray(img).save(out / name)
    done.append(dict(r, file=name, rendered=shown))
    print("RENDERED", name, n, shown, f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
(out / "rounds.json").write_text(json.dumps(dict(run=str(run), camera=sys.argv[1], rounds=done), indent=1))
