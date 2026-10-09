"""The chain's final refinement (F2, post_ref) alone, from a saved pre_ref checkpoint: no mapping is repeated.

  python -m update_layer.eval.refine_only RUN_DIR {real|synthetic} STAGE

Loads RUN_DIR/checkpoint_<STAGE>.pt (keyframe images must be present: restore a stripped checkpoint first), runs
backend.finish_session (GaME's published refinement over all stored keyframes under T1, with the refinement guard),
re-renders the final snapshot and writes RUN_DIR/session_<STAGE>/timeline_post_ref.pkl and
RUN_DIR/checkpoint_<STAGE>_post_ref.pt exactly as run.py does after the chain's last session. 10-09: written so that a
refinement that failed in the driver can be re-run after a memory fix without the 70-minute mapping.
"""
import json, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from update_layer.backends.game.game import GameBackend  # noqa: E402
from update_layer.eval.scenelist import SceneListTimeline, TimelineTail  # noqa: E402
from update_layer.frames import FlatSession  # noqa: E402
from update_layer.run import dataset_config, save_checkpoint  # noqa: E402


def main() -> None:
    run, ds, s = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    cfg, info, specs = dataset_config(ds)
    spec = next(sp for sp in specs if sp.name.endswith("_" + s))
    ck = torch.load(run / f"checkpoint_{s}.pt", weights_only=False)
    be = GameBackend(info, own_update=False)
    be.game = be.prior_from_state(ck["backend"])
    be.session = FlatSession(spec, pixel_step=1)
    stamp = int(ck["prev_final"])
    d = run / f"session_{s}"
    tl_pre = SceneListTimeline.load(d / "timeline.pkl")
    st = tl_pre.stamps()
    be.stamps, be.scenes = [], []
    t1 = time.time()
    torch.cuda.reset_peak_memory_stats()
    be.finish_session(stamp)
    be.snapshot(stamp)
    tl = be.timeline()
    TimelineTail("timeline.pkl", len(st) - 1, tl.stamps()[-1:], tl.scenes[-1:]).save(d / "timeline_post_ref.pkl")
    save_checkpoint(run, s + "_post_ref", be, be.end_session(), ck.get("layer"), stamp, resume=False)
    info_ = json.loads((d / "run.json").read_text())
    info_["post_ref"] = {"step": "refine_only after a failed refinement", "status": "ok",
                         "seconds": round(time.time() - t1, 1), "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    (d / "run.json").write_text(json.dumps(info_, indent=2))
    print(f"== {spec.name}: post_ref ok (refine_only) {round(time.time() - t1)} s, peak {info_['post_ref']['peak_gpu_gb']} GB", flush=True)


if __name__ == "__main__":
    main()
