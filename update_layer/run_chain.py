"""Run a session chain (A -> B [-> C]) with the point backend and the update layer.

  python -m update_layer.run_chain synthetic OUT_DIR [--mode layer|naive|scratch]
  python -m update_layer.run_chain real OUT_DIR [--mode ...]

mode layer    the update layer decides (the method)
mode naive    memory without a decision layer: inherited content is kept, never retired
mode scratch  every session starts empty (the from-scratch control)

Each session writes OUT_DIR/session_X/{timeline.pkl, log.txt, sensor_statistics.txt, state.pkl}.
"""
from __future__ import annotations

import argparse
import json
import pickle
import resource
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from update_layer.evidence import SensorStatistics  # noqa: E402
from update_layer.frames import FlatSession, real_session, synthetic_session  # noqa: E402
from update_layer.layer import LayerConfig, UpdateLayerSession, export_state  # noqa: E402
from update_layer.timeline import LayerTimeline  # noqa: E402

CFG = Path("/home/jixian/Desktop/FT/wt_layer_43c/session_update_baseline/configs")
SYN = Path("/home/jixian/Desktop/FT/datasets/synthetic_ab/mapping_configs")


def dataset_config(name: str, mode: str) -> tuple:
    if name == "synthetic":
        ls = yaml.safe_load((SYN / "label_space.yaml").read_text())
        # t2 synthetic run: mapper_mechanism_10cm.yaml (voxel 0.10, change detection every 25
        # backend updates = 10.8 s between snapshots, mobility labels [6, 9, 15, 35, 39]).
        cfg = LayerConfig(round_s=10.8, map_resolution=0.10, high_mobility=[6, 9, 15, 35, 39],
                          object_semantics=ls["object_labels"],
                          dynamic_semantics=ls.get("dynamic_labels") or [], pixel_step=2)
        return cfg, [synthetic_session("a"), synthetic_session("b")]
    ls = yaml.safe_load((CFG / "nss_ade20k_room_label_space.yaml").read_text())
    # t2 real run: room18_instance_5cm.yaml (voxel 0.05, every 5 backend updates = ~2.1 s,
    # mobility labels [10, 15, 74, 75, 92, 115, 131, 139]); persons (12) are dynamic.
    cfg = LayerConfig(round_s=2.1, map_resolution=0.05,
                      high_mobility=[10, 15, 74, 75, 92, 115, 131, 139],
                      object_semantics=ls["object_labels"], dynamic_semantics=ls["dynamic_labels"],
                      pixel_step=4)
    return cfg, [real_session("a"), real_session("b"), real_session("c")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["synthetic", "real"])
    ap.add_argument("out")
    ap.add_argument("--mode", default="layer", choices=["layer", "naive", "scratch", "carve", "elements"])
    ap.add_argument("--sessions", default="")
    ap.add_argument("--segments", default="rounds", choices=["tracks", "rounds"])
    ap.add_argument("--decision", default="cusum", choices=["cusum", "single", "none"])
    ap.add_argument("--static-frames", action="store_true",
                    help="t2's static-surface frame selection for observations (LayerConfig.static_frames)")
    args = ap.parse_args()
    cfg, specs = dataset_config(args.dataset, args.mode)
    cfg.use_layer = args.mode not in ("naive", "carve", "elements")
    cfg.segments = args.segments
    cfg.objects_as_elements = args.mode in ("carve", "elements")
    cfg.single_look = args.mode == "carve"
    cfg.decision = args.decision
    cfg.static_frames = args.static_frames
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prior, stats = None, None
    prev_final = None
    for spec in specs:
        name = spec.name.split("_")[-1]
        if args.sessions and name not in args.sessions:
            continue
        d = out / f"session_{name}"
        d.mkdir(exist_ok=True)
        if args.mode == "scratch":
            prior, stats = None, None
        session = FlatSession(spec, pixel_step=cfg.pixel_step)
        run = UpdateLayerSession(cfg, prior=prior, stats=stats)
        if args.mode == "scratch" and prev_final is not None:
            run._snapshot(prev_final)       # the from-scratch map at the session boundary: empty
        t0 = time.time()
        res = run.run(session)
        LayerTimeline(res).save(d / "timeline.pkl")
        (d / "log.txt").write_text("\n".join(res.log) + "\n")
        res.stats.save(d / "sensor_statistics.txt")
        prior = export_state(res)
        prev_final = res.stamps[-1]
        with open(d / "state.pkl", "wb") as f:
            pickle.dump(prior, f, protocol=pickle.HIGHEST_PROTOCOL)
        stats = SensorStatistics()
        stats.load(d / "sensor_statistics.txt")
        (d / "run.json").write_text(json.dumps({
            "dataset": args.dataset, "mode": args.mode, "session": spec.name,
            "seconds": round(time.time() - t0, 1), "snapshots": len(res.stamps),
            "maxrss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
            "config": {k: (list(v) if isinstance(v, (list, tuple)) else v)
                       for k, v in cfg.__dict__.items()}}, indent=2))
        print(f"== {spec.name}: {len(res.stamps)} snapshots, {round(time.time() - t0)} s", flush=True)


if __name__ == "__main__":
    main()
