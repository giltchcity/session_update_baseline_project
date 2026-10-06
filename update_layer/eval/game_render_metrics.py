"""[G8] GaME's own rendering metrics (GaME src/utils/mapping_eval.py evaluate_all_rendering) of a GaME checkpoint on
the last session of a run with --g8-holdout: 'train' = every frame of the session that was integrated, 'test' = the
held-out frames (every 10th but the first, GaME datasets.py:432; the stamps are in session_<s>/run.json).

  python -m update_layer.eval.game_render_metrics RUN SESSION_LETTER real|synthetic OUT.json

The checkpoint is RUN/checkpoint_<s>.pt (in a post_ref view of eval/post_ref_view.sh: the refined map). Per frame, as
GaME: render (black background), colour clamped to [0, 1]; PSNR = 20 log10(1 / sqrt(MSE)) (calc_psnr), LPIPS (alex,
normalize=True), MS-SSIM (data_range 1), depth L1 = mean |rendered - measured depth| in GaME scene units; each averaged
over the frames. Two things are stated rather than copied: (1) the frame is rendered with the map of its own time
(T1: the Gaussians alive at its stamp; rows without the layer: every Gaussian, which is GaME's final model as in
mapping_eval.py); (2) depth L1 only over pixels with measured depth (C4: real frames have invalid pixels, and people
pixels are zero in the backend's depth; on GaME's Flat every pixel has depth, so it is the same formula there).
Inputs = what the backend integrated (GameBackend._sample: crop, pixel step, scale); obj_num=1 (the label
accumulation buffer is not read; GaME passes 256).
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def calc_psnr(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """GaME mapping_eval.calc_psnr."""
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def main():
    run, s, ds, out = Path(sys.argv[1]), sys.argv[2].lower(), sys.argv[3], Path(sys.argv[4])
    info_run = json.loads((run / f"session_{s}" / "run.json").read_text())
    hold = info_run.get("g8_holdout")
    if not hold:
        sys.exit(f"{run}/session_{s}: no g8_holdout in run.json (run without --g8-holdout)")
    held = set(int(x) for x in hold["stamps"])
    from pytorch_msssim import ms_ssim
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from update_layer.backends.game.game import Crop, GameBackend, flashsplat_render, gu
    from update_layer.eval.tsdf_export import wait_for_gpu
    from update_layer.frames import FlatSession
    from update_layer.run import dataset_config
    wait_for_gpu()
    _, info, specs = dataset_config(ds)
    spec = next(sp for sp in specs if sp.name.split("_")[-1] == s)
    ck_path = run / f"checkpoint_{s}.pt"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    to_gpu = lambda x: (torch.nn.Parameter(x.detach().cuda(), requires_grad=x.requires_grad)
                        if isinstance(x, torch.nn.Parameter) else x.cuda() if torch.is_tensor(x)
                        else type(x)(to_gpu(v) for v in x) if isinstance(x, (tuple, list))
                        else {k: to_gpu(v) for k, v in x.items()} if isinstance(x, dict) else x)
    ck["backend"]["model"] = to_gpu(ck["backend"]["model"])
    be = GameBackend(info, own_update=False)
    be.game = be.prior_from_state(ck["backend"])
    del ck
    g, gm = be.game, be.game.gaussian_model
    be.session = FlatSession(spec, pixel_step=1)                     # as GameBackend.start_session
    be.crop = Crop(be.session.K, be.step)
    n_frames = int(info_run["frames"])
    lpips_model = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).cuda()
    pipe, bg = gu.flashsplat_pipe(), torch.zeros(3).cuda()
    acc = {"train": {"psnr": [], "lpips": [], "ssim": [], "l1": []}, "test": {"psnr": [], "lpips": [], "ssim": [], "l1": []}}
    with ThreadPoolExecutor(max_workers=4) as pool, torch.no_grad():
        futs = [pool.submit(be._sample, i) for i in range(min(16, n_frames))]
        for i in range(n_frames):
            sample, _, _ = futs[i].result()
            futs[i] = None
            if i + 16 < n_frames:
                futs.append(pool.submit(be._sample, i + 16))
            t = be.session.stamp_ns(i)
            split = "test" if t in held else "train"
            color = gu.np2torch(sample["color"], device="cuda").permute(2, 0, 1) / 255.0
            depth = gu.np2torch(sample["depth"], device="cuda")
            pose = gu.np2torch(sample["pose"], device="cuda")
            view = gu.flashsplat_cam(color, depth, None, sample["intrinsics"], pose.cpu(), i)
            gm.alive = g.alive_at(t)
            pkg = flashsplat_render(view, gm, pipe, bg, obj_num=1)
            gm.alive = None
            rc = torch.clamp(pkg["render"], 0.0, 1.0)
            rd = pkg["depth"].reshape(depth.shape)
            m = depth > 0
            a = acc[split]
            a["psnr"].append(calc_psnr(rc, color).mean().item())
            a["lpips"].append(lpips_model(rc[None], color[None]).mean().item())
            a["ssim"].append(ms_ssim(rc[None], color[None], data_range=1.0).item())
            a["l1"].append(torch.abs(rd - depth)[m].mean().item() if m.any() else float("nan"))
    res = {"run": str(run), "session": s, "checkpoint": str(ck_path.resolve()), "scale": be.scale,
           "protocol": "GaME mapping_eval.evaluate_all_rendering; frame rendered with the map of its stamp; depth L1 over measured pixels"}
    for split, a in acc.items():
        l1 = float(np.nanmean(a["l1"])) if a["l1"] else float("nan")
        res[split] = {"frames": len(a["psnr"]), "psnr": float(np.mean(a["psnr"])) if a["psnr"] else None,
                      "lpips": float(np.mean(a["lpips"])) if a["lpips"] else None,
                      "ms_ssim": float(np.mean(a["ssim"])) if a["ssim"] else None,
                      "depth_l1_scene_units": l1, "depth_l1_m": l1 / be.scale}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("train", "test")}))


if __name__ == "__main__":
    main()
