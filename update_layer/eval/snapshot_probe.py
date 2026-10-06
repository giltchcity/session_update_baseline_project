"""Diagnostic: the map's snapshot at a checkpoint's final stamp (backends/game/game.py _render_scene), per object
sample counts, and per object how many of its rendered pixels the snapshot drops as mixed identity (var >= 0.25).

  python -m update_layer.eval.snapshot_probe CHECKPOINT.pt real|synthetic OUT.json [--revive-at STAMP] [--ids 5,6,14,17]

--revive-at: Gaussians retired exactly at STAMP (the session-end memory test retires at the session's first stamp)
count as alive (as tsdf_export --revive-at). Pixels are counted over all keyframes of the checkpoint, as the snapshot
renders them: a pixel 'belongs' to an object when the rounded mean of the rendered identity code is that object.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("dataset", choices=["real", "synthetic"])
    ap.add_argument("out")
    ap.add_argument("--revive-at", type=int, default=None)
    ap.add_argument("--ids", default="5,6,14,17")
    a = ap.parse_args()
    from update_layer.eval.tsdf_export import wait_for_gpu
    wait_for_gpu()
    from update_layer.run import dataset_config
    from update_layer.backends.game.game import GameBackend, gu, flashsplat_render
    _, info, _ = dataset_config(a.dataset)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    to_gpu = lambda x: (torch.nn.Parameter(x.detach().cuda(), requires_grad=x.requires_grad)
                        if isinstance(x, torch.nn.Parameter) else x.cuda() if torch.is_tensor(x)
                        else type(x)(to_gpu(v) for v in x) if isinstance(x, (tuple, list))
                        else {k: to_gpu(v) for k, v in x.items()} if isinstance(x, dict) else x)
    ck["backend"]["model"] = to_gpu(ck["backend"]["model"])
    be = GameBackend(info, own_update=False)
    g = be.prior_from_state(ck["backend"])
    be.game = g
    t = int(ck["prev_final"])
    if a.revive_at is not None:
        rev = g.death_evidence == a.revive_at
        g.death_evidence[rev] = torch.iinfo(torch.int64).max
    ids = [int(x) for x in a.ids.split(",")]
    sc = be._render_scene(t)
    res = dict(checkpoint=a.checkpoint, stamp=t, revive_at=a.revive_at, background=int(len(sc.background)),
               objects={str(o.instance_id): int(len(o.points)) for o in sc.objects})
    # mixed-identity pixels per object over the keyframes (the rule of _render_scene_alive)
    gm = g.gaussian_model
    gm.alive = g.alive_at(t)
    idf = g.identity.clamp(min=0).to(torch.float32)
    codes = torch.stack([idf, idf * idf, torch.ones_like(idf)], dim=1)
    pipe, bg = gu.flashsplat_pipe(), torch.zeros(3).cuda()
    pure = {i: 0 for i in ids}
    mixed = {i: 0 for i in ids}
    with torch.no_grad():
        for kid, kf in g.keyframes.items():
            K = kf["intrinsics"]
            _, h, w = kf["color"].shape
            view = gu.flashsplat_cam(torch.zeros((3, h, w), device="cuda"), torch.zeros((h, w), device="cuda"),
                                     None, K, kf["pose"].cpu(), None)
            pkg = flashsplat_render(view, gm, pipe, bg, override_color=codes, obj_num=1)
            alpha = pkg["alpha"].squeeze()
            img = pkg["render"]
            weight = img[2].clamp(min=1e-6)
            mean = img[0] / weight
            var = img[1] / weight - mean * mean
            ok = alpha > be.min_alpha
            near = torch.round(mean).long()
            for i in ids:
                m = ok & (near == i)
                pure[i] += int((m & (var < 0.25)).sum())
                mixed[i] += int((m & (var >= 0.25)).sum())
    gm.alive = None
    res["pixels_pure"] = {str(i): pure[i] for i in ids}
    res["pixels_mixed"] = {str(i): mixed[i] for i in ids}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
