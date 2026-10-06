"""One entry point: a backend, a row, a dataset.

  python -m update_layer.run --backend points --row 4 --dataset synthetic --out FT/runs/NAME
         [--sessions ab] [--max-frames N] [--resume] [--core l2|t2]

Rows (interface.ROWS): 1 scratch, 2 naive, 3 own, 4 layer, 5 own+layer.
Inputs (since 2026-10-01 20:10, the same for every backend and row; INPUT_STEP): every frame of the
dataset (30 Hz) at real 960x540 (every 2nd pixel of 1920x1080) / synthetic 680x480 (native), depth
range and people masking from frames.py. The update layer reads its own evidence frames (its rate
evidence_hz and pixel_step, unchanged: that is the method). Round boundaries and snapshots fall on the
same stamps for every backend, so the rows of one backend differ only in what the table says and the
backends differ only in their representation. Before 20:10 points / wavemap / GaME integrated the
layer's frames (5 Hz, 480x270 / 340x240) and SurfelMeshing its own 30 Hz 960x540 / 680x480.
Per session: OUT/session_X/{timeline.pkl, run.json} (+ sensor_statistics.txt, layer_log.txt for
row 4). Run it from session_update_baseline_project/ (or put that folder on PYTHONPATH).

Checkpoint (backends with prior_state, e.g. GaME): after every carried session OUT/checkpoint.pt
holds what the next session starts from (backend prior, layer prior, the boundary stamp, RNG states);
--resume continues the chain after that session. It is removed when the chain is complete.

End-of-run step (F2, since 2026-10-06, user): the backend's finish_session (GaME: its published refinement) runs
once, after the last session of the chain (GaME run.py:36-37 refines once after all runs; run2 continues from run1's
unrefined map), not after every session. Every session's timeline.pkl / checkpoint_<s>.pt is the map without it
(pre_ref, the main protocol); after the last session also timeline_post_ref.pkl / checkpoint_<s>_post_ref.pt. A chain
split over several calls (--sessions, --resume) refines after the last session of each call.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import resource
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml

from .core.layer import LayerConfig, UpdateLayer
from .frames import FT, MOTION, FlatSession, real_session, synthetic_session
from .eval.scenelist import SceneListTimeline
from .interface import ROWS, Backend, DatasetInfo

PROJECT = Path(__file__).resolve().parents[1]
BACKENDS = {"points": "update_layer.backends.points:PointBackend",
            "game": "update_layer.backends.game.game:GameBackend",
            "surfelmeshing": "update_layer.backends.surfelmeshing.surfelmeshing:SurfelMeshingBackend",
            "wavemap": "update_layer.backends.wavemap.wavemap:WavemapBackend"}


INPUT_STEP = {"real": 2, "synthetic": 1}       # pixel step of the frames every backend integrates (30 Hz, all frames)


def dataset_config(name: str):
    """(LayerConfig, DatasetInfo, session specs) of a dataset, from the t2 mapper configs."""
    if name == "synthetic":
        ls = yaml.safe_load((FT / "datasets/synthetic_ab/mapping_configs/label_space.yaml").read_text())
        # t2 synthetic run: mapper_mechanism_10cm.yaml (voxel 0.10, max_range 15, change detection
        # every 25 backend updates = 10.8 s between snapshots, mobility labels [6, 9, 15, 35, 39]).
        cfg = LayerConfig(round_s=10.8, map_resolution=0.10, max_range=15.0,
                          high_mobility=[6, 9, 15, 35, 39], object_semantics=ls["object_labels"],
                          dynamic_semantics=ls.get("dynamic_labels") or [], pixel_step=2,
                          truncation=0.30,           # mapper_mechanism_10cm.yaml:28 truncation_distance
                          object_voxel=0.05)         # mapper_mechanism_10cm.yaml:77 object_reconstruction_resolution
        specs = [synthetic_session("a"), synthetic_session("b")]
    else:
        ls = yaml.safe_load((PROJECT / "session_update_baseline/configs/nss_ade20k_room_label_space.yaml")
                            .read_text())
        # t2 real run: room18_instance_5cm.yaml (voxel 0.05, max_range 5, every 5 backend updates
        # = ~2.1 s, mobility labels [10, 15, 74, 75, 92, 115, 131, 139]); persons (12) are dynamic.
        cfg = LayerConfig(round_s=2.1, map_resolution=0.05, max_range=5.0,
                          high_mobility=[10, 15, 74, 75, 92, 115, 131, 139],
                          object_semantics=ls["object_labels"], dynamic_semantics=ls["dynamic_labels"],
                          pixel_step=4, truncation=0.15,      # room18_instance_5cm.yaml:60 truncation_distance
                          object_voxel=0.02)                  # room18_instance_5cm.yaml:128 object_reconstruction_resolution
        specs = [real_session("a"), real_session("b"), real_session("c")]
    assert all(sp.depth_range[1] == cfg.max_range for sp in specs)
    info = DatasetInfo(name, list(cfg.dynamic_semantics), specs[0].depth_range, specs)
    return cfg, info, specs


def make_backend(name: str, info: DatasetInfo, own_update: bool, **kw):
    module, cls = BACKENDS[name].split(":")
    return getattr(importlib.import_module(module), cls)(info, own_update, **kw)


def save_checkpoint(out: Path, after: str, backend, final_map, l_prior, prev_final, resume: bool) -> None:
    """The map at the end of session `after`: OUT/checkpoint_<after>.pt for every session of every row (a final
    result: the median-depth TSDF export and every later evaluation read it); for a carried chain that goes
    on, also OUT/checkpoint.pt, what --resume continues from (a hard link to the same file)."""
    state = backend.prior_state(final_map)
    if state is None:
        return
    t0 = time.time()
    keep = out / f"checkpoint_{after}.pt"
    tmp = out / f"checkpoint_{after}.pt.tmp"
    torch.save({"after": after, "backend": state, "layer": l_prior, "prev_final": prev_final,
                "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                        "numpy": np.random.get_state(), "python": random.getstate()}}, tmp)
    tmp.replace(keep)
    if resume:
        (out / "checkpoint.pt").unlink(missing_ok=True)
        os.link(keep, out / "checkpoint.pt")
    print(f"checkpoint after session {after}: {keep.stat().st_size / 1e9:.1f} GB, "
          f"{time.time() - t0:.0f} s", flush=True)


def run_chain(backend_name: str, row: int, dataset: str, out: Path, sessions: str = "",
              max_frames: int = 0, verbose: bool = True, resume: bool = False, core: str = "l2",
              d1: bool = False, g5: bool = False, inside: bool = False, g5_reference: str = "tsdf",
              g5_dump: bool = False, g8_holdout: bool = False) -> None:
    cfg, info, specs = dataset_config(dataset)
    cfg.core = core
    cfg.d1 = d1 and core == "l2"
    cfg.g5 = g5 and core == "l2"
    cfg.inside = inside and cfg.g5
    cfg.g5_reference = g5_reference
    cfg.g5_dump = g5_dump and cfg.g5
    carry = row != 1                 # rows 2-5 start from the previous session's map
    own = row in (1, 3, 5)           # the backend's own change handling
    backend = make_backend(backend_name, info, own, work_dir=out)
    layer = UpdateLayer(cfg) if row in (4, 5) else None
    out.mkdir(parents=True, exist_ok=True)
    b_prior = l_prior = None
    prev_final = None
    done = ""                        # sessions already finished by the run being resumed
    if resume:
        ck = torch.load(out / "checkpoint.pt", weights_only=False)
        b_prior, l_prior, prev_final = backend.prior_from_state(ck["backend"]), ck["layer"], ck["prev_final"]
        torch.set_rng_state(ck["rng"]["torch"])
        torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        np.random.set_state(ck["rng"]["numpy"])
        random.setstate(ck["rng"]["python"])
        names = [sp.name.split("_")[-1] for sp in specs]
        done = "".join(names[:names.index(ck["after"]) + 1])
        print(f"resumed after session {ck['after']}", flush=True)
        del ck
    step = max(1, int(round(30.0 / cfg.evidence_hz)))
    # [F2] the chain's last session (the last of `sessions`, else the dataset's last): only after it the backend's
    # end-of-run step runs (GaME: its published refinement, once after all runs, GaME run.py:36-37)
    final = [sp for sp in specs if not sessions or sp.name.split("_")[-1] in sessions][-1]
    for spec in specs:
        name = spec.name.split("_")[-1]
        if (sessions and name not in sessions) or name in done:
            continue
        d = out / f"session_{name}"
        d.mkdir(exist_ok=True)
        t0 = time.time()
        session = FlatSession(spec, pixel_step=INPUT_STEP[dataset])          # what every backend integrates
        layer_session = FlatSession(spec, pixel_step=cfg.pixel_step)        # the layer's evidence frames
        MOTION.clear()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()          # run.json: this session's peak
        backend.start_session(spec, b_prior if carry else None)
        if layer is not None:
            layer.start_session(spec, l_prior)
        if prev_final is not None:
            backend.snapshot(prev_final)          # the map at the session boundary (row 1: empty)
        indices = list(range(len(session.ids)))
        if max_frames:
            indices = indices[:max_frames]
        round_start = session.stamp_ns(indices[0])
        retired = {}
        # [G8] GaME's test split (datasets.py:432, run2: every 10th frame except the first is held out): in the chain's
        # last session these frames are kept out of mapping (backend, layer evidence, D1 front end) and only rendered
        # for GaME's novel-view metrics (eval/game_render_metrics.py); round boundaries stay on the same stamps
        held = {n for n in range(len(indices)) if n % 10 == 0 and n != 0} if g8_holdout and spec is final else set()

        def load(i):
            lf = layer_session.load(i) if layer is not None and i % step == 0 else None
            return session.load(i), lf

        with ThreadPoolExecutor(max_workers=4) as pool:
            ahead = [pool.submit(load, i) for i in indices[:16]]
            for n in range(len(indices)):
                frame, layer_frame = ahead[n].result()
                ahead[n] = None
                if n + 16 < len(indices):
                    ahead.append(pool.submit(load, indices[n + 16]))
                if n not in held:
                    if layer is not None:
                        motion = layer.motion(frame)             # D1 front end, every frame (l2 core)
                        if motion is not None:
                            MOTION[frame.stamp_ns] = motion
                            while len(MOTION) > 64:
                                MOTION.pop(next(iter(MOTION)))
                    backend.integrate(frame)
                    if layer_frame is not None:
                        layer.observe(layer_frame)
                last = n == len(indices) - 1
                if frame.stamp_ns - round_start < cfg.round_s * 1e9 and not last:
                    continue
                stamp = frame.stamp_ns
                if layer is not None:
                    decided = layer.decide(stamp, last, backend.elements(),
                                           object_support=not backend.consumes_state_intervals)
                    for k, v in decided.items():
                        retired[k] = retired.get(k, 0) + len(v)
                    if backend.consumes_state_intervals:
                        backend.set_state_intervals(layer.state_intervals(), stamp)
                    ids = torch.unique(torch.cat(list(decided.values())))
                    backend.retire(ids, stamp)
                if last and layer is not None and cfg.g5:
                    # session-end memory test, then the backend's session-end step (GaME: refinement)
                    mem, start = layer.session_end_memory(backend.elements(), getattr(backend, "render_identity", None),
                                                          getattr(backend, "render_map", None))
                    if len(mem):
                        backend.retire(mem, start)
                    retired["memory"] = retired.get("memory", 0) + len(mem)
                backend.snapshot(stamp)
                round_start = stamp
                if verbose:
                    print(f"{spec.name} t={(stamp - session.stamp_ns(indices[0])) / 1e9:7.1f}s "
                          f"frames={n + 1}/{len(indices)} elements={len(backend.elements())} "
                          f"retired={retired} {time.time() - t0:6.1f}s", flush=True)
        backend.timeline().save(d / "timeline.pkl")
        final_map = backend.end_session()
        b_prior = final_map if carry else None
        if layer is not None:
            l_prior = layer.end_session(d)
        prev_final = stamp
        (d / "run.json").write_text(json.dumps({
            "backend": backend_name, "row": row, "row_name": ROWS[row], "dataset": dataset,
            "session": spec.name, "frames": len(indices), "seconds": round(time.time() - t0, 1),
            "input": {"pixel_step": INPUT_STEP[dataset], "every_frame": True,
                      "layer_pixel_step": cfg.pixel_step, "layer_frame_step": step},
            "retired_by_layer": retired, "own_update": own, "carried": carry,
            "g8_holdout": {"rule": "n % 10 == 0 and n != 0 (GaME datasets.py:432)", "frames": len(held),
                           "stamps": [session.stamp_ns(indices[n]) for n in sorted(held)]} if held else None,
            "backend_changes": list(backend.CHANGES),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0,
            "maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
            "config": {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in cfg.__dict__.items()}},
            indent=2))
        print(f"== {spec.name}: row {row} ({ROWS[row]}) {round(time.time() - t0)} s", flush=True)
        save_checkpoint(out, name, backend, final_map, l_prior, prev_final,
                        resume=carry and spec is not specs[-1])
        if spec is final and type(backend).finish_session is not Backend.finish_session:
            # [F2] the end-of-run step once, after the chain's last session. Everything above is the map before it
            # (pre_ref: timeline.pkl, checkpoint_<s>.pt, the main protocol); the map after it is saved next to it
            # (post_ref: timeline_post_ref.pkl = the same snapshots with the final one re-rendered,
            # checkpoint_<s>_post_ref.pt). Out of GPU memory there leaves pre_ref complete and is recorded.
            t1 = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            try:
                backend.finish_session(stamp)
                backend.snapshot(stamp)
                tl = backend.timeline()
                st = tl.stamps()
                SceneListTimeline(st[:-2] + st[-1:], tl.scenes[:-2] + tl.scenes[-1:]).save(d / "timeline_post_ref.pkl")
                save_checkpoint(out, name + "_post_ref", backend, backend.end_session(), l_prior, prev_final,
                                resume=False)
                status = "ok"
            except torch.cuda.OutOfMemoryError as e:
                status = "out of GPU memory: " + str(e).splitlines()[0][:200]
                torch.cuda.empty_cache()
            info_ = json.loads((d / "run.json").read_text())
            info_["post_ref"] = {"step": "finish_session after the chain's last session (F2)", "status": status,
                                 "seconds": round(time.time() - t1, 1),
                                 "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
                                 if torch.cuda.is_available() else 0}
            (d / "run.json").write_text(json.dumps(info_, indent=2))
            print(f"== {spec.name}: post_ref {status} {round(time.time() - t1)} s", flush=True)
    if (out / "checkpoint.pt").exists() and not sessions:
        (out / "checkpoint.pt").unlink(missing_ok=True)   # the chain is complete (checkpoint_<s>.pt stay)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    ap.add_argument("--row", type=int, choices=sorted(ROWS), required=True)
    ap.add_argument("--dataset", choices=["synthetic", "real"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sessions", default="", help="e.g. 'ab'; default all")
    ap.add_argument("--max-frames", type=int, default=0, help="frames per session (smoke tests)")
    ap.add_argument("--resume", action="store_true", help="continue after OUT/checkpoint.pt")
    ap.add_argument("--core", choices=["l2", "t2"], default="l2",
                    help="layer decision core: l2 = TSDF L2_FINAL2 (192c1cf), t2 = earlier port (control)")
    ap.add_argument("--d1", action="store_true", help="D1 front end (core/d1.py; l2 core only)")
    ap.add_argument("--g5", action="store_true", help="session-end memory test (core/session_end.py; l2 core)")
    ap.add_argument("--inside", action="store_true", help="with --g5: INSIDE (backends that render one identity)")
    ap.add_argument("--g5-reference", choices=["tsdf", "render"], default="tsdf",
                    help="step-2 reference surface of the noise table and 'displaced'")
    ap.add_argument("--g5-dump", action="store_true", help="with --g5: save the memory test's inputs and evidence")
    ap.add_argument("--g8-holdout", action="store_true",
                    help="hold every 10th frame of the chain's last session out of mapping (GaME's test split)")
    args = ap.parse_args()
    run_chain(args.backend, args.row, args.dataset, Path(args.out), args.sessions, args.max_frames,
              resume=args.resume, core=args.core, d1=args.d1, g5=args.g5, inside=args.inside,
              g5_reference=args.g5_reference, g5_dump=args.g5_dump, g8_holdout=args.g8_holdout)


if __name__ == "__main__":
    main()
