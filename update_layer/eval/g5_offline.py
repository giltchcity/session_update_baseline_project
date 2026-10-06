"""Offline G5 test: the session-end memory test (core/session_end.py) on a finished session's GaME checkpoint.

  python -m update_layer.eval.g5_offline CHECKPOINT_PREV CHECKPOINT REAL_SESSION OUT_DIR [--max-frames N]

CHECKPOINT_PREV: the checkpoint the session started from (its layer prior seeds the registry);
CHECKPOINT: the one after the session (the GaME map that is tested). The layer is replayed over the session's
frames exactly as run.py feeds it (evidence frames every `step` frames, one decide() per round on the
checkpoint's elements: the object decisions do not read the backend's map), so the registry and the evidence
store are the session's. The memory test then runs on the checkpoint's elements; the elements it removes end
at the session's first stamp. Writes OUT_DIR/{g5.json, before.ply, after.ply} (median-depth TSDF exports of
the map at the session's end, as update_layer.eval.tsdf_export).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def export(g, be, info, t: int, out: Path, voxel: float = 0.02) -> int:
    import open3d as o3d
    from update_layer.backends.game.game import gu, probe_render
    gm = g.gaussian_model
    gm.alive = g.alive_at(t)
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=5 * voxel, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    with torch.no_grad():
        for kid, kf in g.keyframes.items():
            K = np.asarray(kf["intrinsics"], dtype=np.float64)
            _, h, w = kf["color"].shape
            view = gu.flashsplat_cam(kf["color"].cuda(), kf["depth"].cuda(), None, K, kf["pose"].cpu(), None)
            pkg = probe_render(view, gm)
            alpha = pkg["alpha"].squeeze()
            depth = pkg["median"].squeeze() / be.scale
            measured = kf["depth"].cuda().reshape(depth.shape) > 0
            depth = torch.where(measured & (alpha > 0), depth, torch.zeros_like(depth))
            color = (pkg["render"].clamp(0, 1).permute(1, 2, 0) * 255).byte().cpu().numpy()
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(color)),
                o3d.geometry.Image(np.ascontiguousarray(depth.cpu().numpy().astype(np.float32))),
                depth_scale=1.0, depth_trunc=float(info.depth_range[1]), convert_rgb_to_intensity=False)
            intr = o3d.camera.PinholeCameraIntrinsic(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
            w2c = kf["pose"].cpu().numpy().astype(np.float64).copy()
            w2c[:3, 3] /= be.scale
            vol.integrate(rgbd, intr, w2c)
    gm.alive = None
    mesh = vol.extract_triangle_mesh()
    o3d.io.write_triangle_mesh(str(out), mesh)
    return len(mesh.vertices)


def diagnose(layer, el, out: Path) -> None:
    """The evidence of every tested element, and the band's width: the share of blocked readings that fall
    within [extent, tau) in front of the element (the band of a TSDF layer) under tau = max(extent, sigma)."""
    from update_layer.core import session_end
    from update_layer.core.l2.evidence import EvidenceConfig
    start = layer.store.first_stamp()
    reg = layer.registry
    own = torch.zeros(int(el.identity.max()) + 1, dtype=torch.bool, device="cuda")
    own[0] = True
    for i in reg.tracked_ids():
        cur = reg.current_fragment(i)
        if cur is not None and cur.num_vertices and cur.birth_time < start and i < len(own):
            own[i] = True
    idx = torch.nonzero((el.created < start) & own[el.identity.clamp(min=0)]).squeeze(1)
    sigma = EvidenceConfig().surface_match_tolerance
    ext = el.extent[idx]
    res = {"tested": len(idx), "extent_quantiles_cm": {q: round(float(torch.quantile(ext, q)) * 100, 2)
                                                         for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
           "tau_equals_extent_share": float((ext >= sigma).float().mean())}
    for name, trunc in (("trunc=extent", ext), ("trunc=0.15 (TSDF background)", torch.full_like(ext, 0.15))):
        ev = session_end.memory_test(layer.store, el.xyz[idx], ext, trunc, sigma, layer.rejected)
        seen, hidden = session_end.decide(ev)
        b, bb = ev["blocked"].float(), ev["blocked_band"].float()
        res[name] = {"blocked>0": int((b > 0).sum()), "blocked_band>0": int((bb > 0).sum()),
                     "blocked_quantiles": [float(torch.quantile(b, q)) for q in (0.5, 0.9, 0.99)],
                     "band_share_of_blocked(mean over blocked>0)": float((bb[b > 0] / b[b > 0]).mean()) if (b > 0).any() else 0,
                     "hit>0": int((ev["hit"] > 0).sum()), "through>0": int((ev["through"] > 0).sum()),
                     "seen_through": int(seen.sum()), "hidden": int(hidden.sum())}
        if name == "trunc=extent":
            torch.save({k: v.cpu() for k, v in ev.items()} | {"extent": ext.cpu()}, out / "evidence.pt")
    (out / "diag.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint_prev")
    ap.add_argument("checkpoint")
    ap.add_argument("session", help="a | b | c")
    ap.add_argument("out")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--diagnose", action="store_true",
                    help="no exports; write the per-element evidence and its distributions (evidence.npz, diag.json)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from update_layer.run import dataset_config
    from update_layer.frames import FlatSession
    from update_layer.core.layer import UpdateLayer
    from update_layer.backends.game.game import GameBackend

    cfg, info, specs = dataset_config("real")
    cfg.core, cfg.g5 = "l2", True
    spec = next(s for s in specs if s.name.endswith("_" + a.session))
    prev = torch.load(a.checkpoint_prev, map_location="cuda", weights_only=False)
    ck = torch.load(a.checkpoint, map_location="cuda", weights_only=False)
    be = GameBackend(info, own_update=False)
    be.game = be.prior_from_state(ck["backend"])
    g = be.game
    el = be.elements()
    t_end = int(ck["prev_final"])

    t0 = time.time()
    layer = UpdateLayer(cfg)
    layer.start_session(spec, prev["layer"])
    layer_session = FlatSession(spec, pixel_step=cfg.pixel_step)
    step = max(1, int(round(30.0 / cfg.evidence_hz)))
    n = len(layer_session.ids) if not a.max_frames else min(a.max_frames, len(layer_session.ids))
    round_start = None
    for i in range(n):
        if i % step == 0:
            f = layer_session.load(i)
            layer.observe(f)
            stamp = f.stamp_ns
            if round_start is None:
                round_start = stamp
        last = i == n - 1
        if (i % step == 0 and stamp - round_start >= cfg.round_s * 1e9) or last:
            layer.decide(stamp, last, el, object_support=False)
            round_start = stamp
    t_layer = time.time() - t0
    if a.diagnose:
        diagnose(layer, el, out)
        return
    mem, start = layer.session_end_memory(el)
    t_test = time.time() - t0 - t_layer
    line = next(l for l in reversed(layer.log) if l.startswith("MEMORY_TEST"))
    extra = [l for l in layer.log if l.startswith(("SIGMA_CM", "DEPTH_SCALE"))]
    v_before = export(g, be, info, t_end, out / "before.ply")
    removed_identity = {}
    if len(mem):
        sel = torch.isin(el.ids, mem)
        ids, cnt = torch.unique(el.identity[sel], return_counts=True)
        removed_identity = {int(i): int(c) for i, c in zip(ids.tolist(), cnt.tolist())}
        be.retire(mem, start)
    v_after = export(g, be, info, t_end, out / "after.ply")
    res = dict(memory_test=line, diag=extra, elements=len(el), removed=len(mem), removed_by_identity=removed_identity,
               session_start=start, map_time=t_end, vertices_before=v_before, vertices_after=v_after,
               layer_replay_s=round(t_layer, 1), test_s=round(t_test, 1))
    (out / "g5.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
