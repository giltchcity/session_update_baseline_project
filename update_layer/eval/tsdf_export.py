"""Surface of a GaME map by TSDF fusion of its rendered depth (the mesh extraction of 2DGS).

  python -m update_layer.eval.tsdf_export CHECKPOINT.pt OUT.ply {real|synthetic|flat} [--voxel 0.02]

2DGS (Huang et al., SIGGRAPH 2024; utils/mesh_utils.py extract_mesh_bounded) renders the depth of every
training view from the final Gaussians and fuses the depth maps with Open3D's TSDF (voxel 0.004, sdf_trunc
0.02 = 5 voxels, depth_trunc 3 at DTU object scale). Here: the views are the map's stored keyframes, the
Gaussians are those alive at the map's time (T1: the map of the session's last stamp), the depth is the
median depth (default; --depth mean: the expected depth D / alpha, 2DGS's depth_ratio 0 variant), only pixels where the
keyframe has measured depth are
fused (GaME is trained only there, C4), voxel = the common map resolution of the other backends (2 cm),
sdf_trunc = 5 voxels as in 2DGS, depth_trunc = the sensor range of the dataset. Checkpoints with stripped keyframes
(eval/kf_strip.py) are read as they are: the measured depth is rebuilt from the dataset on the CPU. GaME's own checkpoint
(GaME.save: its model and keyframe ids, e.g. its official Flat run) is read too: every Gaussian of its model is its map
(GaME deletes what it removes), its keyframes are rebuilt from the dataset (game_keyframes).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def wait_for_gpu(free_gb: float = 6.0, poll: float = 30.0) -> None:
    """An evaluation next to a running GaME job (supervisor 12:42): start only when the GPU has >= free_gb free (GaME's
    refinement with densification is its peak), checked twice `poll` seconds apart."""
    import time
    ok = 0
    while ok < 2:
        free = torch.cuda.mem_get_info()[0] / 2 ** 30
        ok = ok + 1 if free >= free_gb else 0
        if ok < 2:
            if not ok:
                print(f"tsdf_export: {free:.1f} GB free GPU memory < {free_gb} GB, waiting", flush=True)
            time.sleep(poll)


def game_keyframes(ids, specs, info, scale: float) -> dict:
    """The keyframes of GaME's own checkpoint as stripped keyframe records (eval/kf_strip.py: the depth is rebuilt from the
    dataset). A keyframe id is the frame's place in GaME's training sequence (GaME.load: every frame of each run in order,
    the last run without its test split, datasets.py get_flat_datasets = run.py's G8 split); the pose is the dataset's
    (world-to-camera, translation x scale, as GameBackend._sample); the intrinsics are the backend crop's (the
    rasteriser's centred principal point, the one GaME's renders have)."""
    from update_layer.backends.game.game import STEP, Crop
    from update_layer.frames import FlatSession
    seq = []
    for sp in specs:
        fs = FlatSession(sp, pixel_step=1)
        seq += [(sp.name, fs, i) for i in range(len(fs.ids)) if sp is not specs[-1] or i % 10 != 0 or i == 0]
    out = {}
    for k in sorted(ids):
        name, fs, i = seq[k]
        c = Crop(fs.K, STEP[info.name])
        pose = np.linalg.inv(fs.pose(i))
        pose[:3, 3] *= scale
        out[k] = {"stripped": True, "session": name, "index": i, "scale": scale, "depth_shape": (c.rows, c.cols),
                  "zeroed": np.packbits(np.zeros(c.rows * c.cols, dtype=bool)),
                  "pose": torch.from_numpy(pose.astype(np.float32)), "intrinsics": c.K}
    return out


def main():
    wait_for_gpu()
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("dataset", choices=["real", "synthetic", "flat"])
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--time", type=int, default=None, help="map time (ns); default the checkpoint's last stamp")
    ap.add_argument("--depth", choices=["median", "mean"], default="median",
                    help="median: 2DGS depth_ratio=1 (where T first drops to 0.5); mean: expected depth D/alpha")
    ap.add_argument("--revive-at", type=int, default=None,
                    help="counterfactual: Gaussians retired exactly at this stamp (the session-end memory test retires "
                         "at the session's first stamp) count as alive; they missed the refinement after it (approximate)")
    ap.add_argument("--revive-all", action="store_true",
                    help="diagnostic: every Gaussian the layer retired (death_evidence set) counts as alive")
    ap.add_argument("--revive-states", action="store_true",
                    help="diagnostic: every Gaussian whose object state the layer ended (death_state set) counts as alive")
    ap.add_argument("--drop-uids", default=None,
                    help="diagnostic: Gaussians whose uid is in this .npy count as ended (a proxy of an online rule that "
                         "would have ended them)")
    ap.add_argument("--revive-uids", default=None,
                    help="diagnostic: Gaussians whose uid is in this .npy and that the layer retired count as alive "
                         "(an upper bound of not retiring them; they missed the optimisation after their retirement)")
    ap.add_argument("--measured", action="store_true",
                    help="control: fuse the keyframes' measured depth instead of the rendered map (same views, same TSDF)")
    ap.add_argument("--views", default=None,
                    help="diagnostic: fuse only the keyframes of this session (a, b, c; by the keyframe's stamp)")
    a = ap.parse_args()

    from update_layer.run import dataset_config
    from update_layer.backends.game.game import GameBackend, TrackedGaME, gu, probe_render

    _, info, specs = dataset_config(a.dataset)
    # the checkpoint on the CPU, only the Gaussian model on the GPU (the keyframes go to the GPU one at a time below):
    # a 13 GB checkpoint loaded whole to the GPU took ~10 GB next to a running GaME job
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    to_gpu = lambda x: (torch.nn.Parameter(x.detach().cuda(), requires_grad=x.requires_grad)
                        if isinstance(x, torch.nn.Parameter) else x.cuda() if torch.is_tensor(x)
                        else type(x)(to_gpu(v) for v in x) if isinstance(x, (tuple, list))
                        else {k: to_gpu(v) for k, v in x.items()} if isinstance(x, dict) else x)
    be = GameBackend(info, own_update=False)
    game_format = "keyframe_key_ids" in ck             # GaME's own checkpoint
    if game_format:
        if a.revive_at is not None or a.revive_all or a.revive_states or a.revive_uids or a.drop_uids or a.views or \
                a.time is not None:
            sys.exit("GaME's own checkpoint has no lifetimes, identities or keyframe stamps: only --voxel / --depth / --measured")
        g = TrackedGaME(dict(be.config))
        g.gaussian_model.restore(to_gpu(ck["model"]), g.opt_params)
        g.keyframes = game_keyframes(ck["keyframe_key_ids"], specs, info, be.scale)
        t = None
    else:
        ck["backend"]["model"] = to_gpu(ck["backend"]["model"])
        g = be.prior_from_state(ck["backend"])
        t = a.time if a.time is not None else int(ck["prev_final"])
    gm = g.gaussian_model
    if a.revive_at is not None:
        revived = g.death_evidence == a.revive_at
        print(f"revive: {int(revived.sum())} Gaussians retired at {a.revive_at} counted as alive", flush=True)
        g.death_evidence[revived] = torch.iinfo(torch.int64).max
    if a.revive_all:
        revived = g.death_evidence < torch.iinfo(torch.int64).max
        print(f"revive all: {int(revived.sum())} retired Gaussians counted as alive", flush=True)
        g.death_evidence[revived] = torch.iinfo(torch.int64).max
    if a.revive_states:
        revived = g.death_state < torch.iinfo(torch.int64).max
        print(f"revive states: {int(revived.sum())} Gaussians of ended states counted as alive", flush=True)
        g.death_state[revived] = torch.iinfo(torch.int64).max
    if a.revive_uids:
        rev = torch.isin(g.uid, torch.as_tensor(np.load(a.revive_uids), device=g.uid.device)) & \
            (g.death_evidence < torch.iinfo(torch.int64).max)
        print(f"revive uids: {int(rev.sum())} retired Gaussians counted as alive", flush=True)
        g.death_evidence[rev] = torch.iinfo(torch.int64).max
    if a.drop_uids:
        drop = torch.as_tensor(np.load(a.drop_uids), device=g.uid.device)
        gone = torch.isin(g.uid, drop)
        print(f"drop: {int(gone.sum())} Gaussians of {len(drop)} uids counted as ended", flush=True)
        g.death_evidence[gone] = torch.minimum(g.death_evidence[gone], torch.full_like(g.death_evidence[gone], t))
    gm.alive = None if game_format else g.alive_at(t)
    scale = be.scale
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=a.voxel, sdf_trunc=5 * a.voxel, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    pipe, bg = gu.flashsplat_pipe(), torch.zeros(3).cuda()
    n = 0
    sess = None
    if a.views:
        from update_layer.eval import kf_strip
        _, by_stamp = kf_strip._sources(a.dataset)
        own = {kid for kid, ts in g.kf_stamp.items() if ts in by_stamp and by_stamp[ts][0].split("_")[-1] == a.views}
        print(f"views: {len(own)} of {len(g.keyframes)} keyframes (session {a.views})", flush=True)
    with torch.no_grad():
        for kid, kf in g.keyframes.items():
            if a.views and kid not in own:
                continue
            K = np.asarray(kf["intrinsics"], dtype=np.float64)
            if isinstance(kf, dict) and kf.get("stripped"):            # keyframe images stripped (eval/kf_strip.py)
                if sess is None:
                    from update_layer.eval import kf_strip
                    sess, _ = kf_strip._sources(a.dataset)
                kf = dict(kf, depth=kf_strip.restore_depth(sess, kf))  # the measured depth, rebuilt on the CPU
                h, w = kf["depth"].shape[-2:]
                kf["color"] = torch.zeros((3, h, w))                    # only its size is used (the colour is rendered)
            _, h, w = kf["color"].shape
            view = gu.flashsplat_cam(kf["color"].cuda(), kf["depth"].cuda(), None, K, kf["pose"].cpu(), None)
            pkg = probe_render(view, gm)
            alpha = pkg["alpha"].squeeze()
            if a.depth == "median":
                depth = pkg["median"].squeeze() / scale
            else:
                depth = (pkg["depth"].squeeze() / alpha.clamp(min=1e-6)) / scale
            measured = kf["depth"].cuda().reshape(depth.shape) > 0
            depth = torch.where(measured & (alpha > 0), depth, torch.zeros_like(depth))
            if a.measured:
                depth = kf["depth"].cuda().reshape(depth.shape) / scale
            color = (pkg["render"].clamp(0, 1).permute(1, 2, 0) * 255).byte().cpu().numpy()
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(color)),
                o3d.geometry.Image(np.ascontiguousarray(depth.cpu().numpy().astype(np.float32))),
                depth_scale=1.0, depth_trunc=float(info.depth_range[1]), convert_rgb_to_intensity=False)
            intr = o3d.camera.PinholeCameraIntrinsic(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
            w2c = kf["pose"].cpu().numpy().astype(np.float64).copy()
            w2c[:3, 3] /= scale
            vol.integrate(rgbd, intr, w2c)
            n += 1
    gm.alive = None
    mesh = vol.extract_triangle_mesh()
    o3d.io.write_triangle_mesh(a.out, mesh)
    print(f"{n} keyframes fused, map time {t}, {len(mesh.vertices)} vertices -> {a.out}")


if __name__ == "__main__":
    main()
