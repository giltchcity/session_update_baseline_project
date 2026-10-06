"""Offline: a session's timeline with its final snapshot re-rendered from the session's checkpoint with another snapshot
readout (backends/game/game.py readout: 'first_echo' = readout.py), for scoring the readout without re-mapping.

  python -m update_layer.eval.reemit_final CHECKPOINT.pt TIMELINE.pkl OUT_TIMELINE.pkl real|synthetic [--readout first_echo]

Only the final snapshot (the map after the session-end steps, the one the checkpoint holds) changes; every earlier
snapshot is the run's own. The final stamp is the checkpoint's prev_final, which must equal the timeline's last stamp.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("timeline")
    ap.add_argument("out")
    ap.add_argument("dataset", choices=["real", "synthetic"])
    ap.add_argument("--readout", default="first_echo", choices=["first_echo", "e1"])
    ap.add_argument("--final-only", action="store_true",
                    help="TIMELINE is ignored: a one-snapshot timeline of the final map (for runs whose timelines are gone; "
                         "geometry only, no change/ghost/retention scores)")
    a = ap.parse_args()
    from update_layer.eval.tsdf_export import wait_for_gpu
    wait_for_gpu()
    from update_layer.run import dataset_config
    from update_layer.backends.game.game import GameBackend
    from update_layer.eval.scenelist import SceneListTimeline, load_timeline
    _, info, _ = dataset_config(a.dataset)
    tl = None if a.final_only else load_timeline(a.timeline)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    to_gpu = lambda x: (torch.nn.Parameter(x.detach().cuda(), requires_grad=x.requires_grad)
                        if isinstance(x, torch.nn.Parameter) else x.cuda() if torch.is_tensor(x)
                        else type(x)(to_gpu(v) for v in x) if isinstance(x, (tuple, list))
                        else {k: to_gpu(v) for k, v in x.items()} if isinstance(x, dict) else x)
    ck["backend"]["model"] = to_gpu(ck["backend"]["model"])
    be = GameBackend(info, own_update=False)
    be.game = be.prior_from_state(ck["backend"])
    be.readout = a.readout
    t = int(ck["prev_final"])
    final = be._render_scene(t)
    if tl is None:
        stamps, scenes = [t], [final]
    else:
        stamps = list(tl.stamps())
        assert stamps[-1] == t, f"timeline's last stamp {stamps[-1]} != checkpoint's final stamp {t}"
        scenes = [tl.scene(s) for s in stamps[:-1]] + [final]
    out = SceneListTimeline(stamps, scenes)
    with open(a.out, "wb") as f:
        pickle.dump(out, f)
    old_bg = len(tl.scene(t).background) if tl is not None else -1
    print(f"final snapshot {t}: background {old_bg} -> {len(final.background)}; objects "
          + ", ".join(f"{o.instance_id}: {len(o.points)}" for o in final.objects))


if __name__ == "__main__":
    main()
