"""Per-state binding of Gaussians (backends/game/binding.py, S1), offline on a GaME checkpoint, and the re-emission
of a session's boundary and final snapshots with it.

  python -m update_layer.eval.state_binding CKPT_PREV CKPT_CUR TIMELINE OUT_TIMELINE SESSION_LETTER

The boundary snapshot (= CKPT_PREV's final map) and the final snapshot (CKPT_CUR) are re-bound and re-rendered with the
first-echo readout; every other snapshot of TIMELINE stays the run's own.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from update_layer.backends.game.binding import bind  # noqa: E402  (the same binding as the mapping run's S1)


def load_backend(path, info):
    from update_layer.backends.game.game import GameBackend
    ck = torch.load(path, map_location="cpu", weights_only=False)
    to_gpu = lambda x: (torch.nn.Parameter(x.detach().cuda(), requires_grad=x.requires_grad)
                        if isinstance(x, torch.nn.Parameter) else x.cuda() if torch.is_tensor(x)
                        else type(x)(to_gpu(v) for v in x) if isinstance(x, (tuple, list))
                        else {k: to_gpu(v) for k, v in x.items()} if isinstance(x, dict) else x)
    ck["backend"]["model"] = to_gpu(ck["backend"]["model"])
    be = GameBackend(info, own_update=False)
    be.game = be.prior_from_state(ck["backend"])
    return be, int(ck["prev_final"])


def main():
    prev, cur, tl_path, out, s = sys.argv[1:6]
    from update_layer.run import dataset_config
    from update_layer.eval.scenelist import SceneListTimeline, load_timeline
    _, info, specs = dataset_config("real")
    tl = load_timeline(tl_path)
    stamps = list(tl.stamps())
    scenes = [tl.scene(x) for x in stamps]
    for k, path in ((0, prev), (-1, cur)):
        be, t = load_backend(path, info)
        assert t == stamps[k], f"{path}: final stamp {t} != timeline stamp {stamps[k]}"
        print(path, bind(be, specs), flush=True)
        be.readout = "first_echo"
        scenes[k] = be._render_scene(t)
        del be
        torch.cuda.empty_cache()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(SceneListTimeline(stamps, scenes), f)
    print("wrote", out)


if __name__ == "__main__":
    main()
