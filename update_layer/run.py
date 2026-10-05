"""One entry point: a backend, a row, a dataset.

  python -m update_layer.run --backend points --row 4 --dataset synthetic --out FT/runs/NAME
         [--sessions ab] [--max-frames N] [--resume]

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
"""
from __future__ import annotations

import argparse
import importlib
import json
import random
import resource
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml

from .core.layer import LayerConfig, UpdateLayer
from .frames import FT, FlatSession, real_session, synthetic_session
from .interface import ROWS, DatasetInfo

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
                          dynamic_semantics=ls.get("dynamic_labels") or [], pixel_step=2)
        specs = [synthetic_session("a"), synthetic_session("b")]
    else:
        ls = yaml.safe_load((PROJECT / "session_update_baseline/configs/nss_ade20k_room_label_space.yaml")
                            .read_text())
        # t2 real run: room18_instance_5cm.yaml (voxel 0.05, max_range 5, every 5 backend updates
        # = ~2.1 s, mobility labels [10, 15, 74, 75, 92, 115, 131, 139]); persons (12) are dynamic.
        cfg = LayerConfig(round_s=2.1, map_resolution=0.05, max_range=5.0,
                          high_mobility=[10, 15, 74, 75, 92, 115, 131, 139],
                          object_semantics=ls["object_labels"], dynamic_semantics=ls["dynamic_labels"],
                          pixel_step=4)
        specs = [real_session("a"), real_session("b"), real_session("c")]
    assert all(sp.depth_range[1] == cfg.max_range for sp in specs)
    info = DatasetInfo(name, list(cfg.dynamic_semantics), specs[0].depth_range, specs)
    return cfg, info, specs


def make_backend(name: str, info: DatasetInfo, own_update: bool, **kw):
    module, cls = BACKENDS[name].split(":")
    return getattr(importlib.import_module(module), cls)(info, own_update, **kw)


def save_checkpoint(out: Path, after: str, backend, b_prior, l_prior, prev_final) -> None:
    state = backend.prior_state(b_prior)
    if state is None:
        return
    t0 = time.time()
    tmp = out / "checkpoint.pt.tmp"
    torch.save({"after": after, "backend": state, "layer": l_prior, "prev_final": prev_final,
                "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                        "numpy": np.random.get_state(), "python": random.getstate()}}, tmp)
    tmp.replace(out / "checkpoint.pt")
    print(f"checkpoint after session {after}: {(out / 'checkpoint.pt').stat().st_size / 1e9:.1f} GB, "
          f"{time.time() - t0:.0f} s", flush=True)


def run_chain(backend_name: str, row: int, dataset: str, out: Path, sessions: str = "",
              max_frames: int = 0, verbose: bool = True, resume: bool = False) -> None:
    cfg, info, specs = dataset_config(dataset)
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
    for spec in specs:
        name = spec.name.split("_")[-1]
        if (sessions and name not in sessions) or name in done:
            continue
        d = out / f"session_{name}"
        d.mkdir(exist_ok=True)
        t0 = time.time()
        session = FlatSession(spec, pixel_step=INPUT_STEP[dataset])          # what every backend integrates
        layer_session = FlatSession(spec, pixel_step=cfg.pixel_step)        # the layer's evidence frames
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
                backend.integrate(frame)
                if layer_frame is not None:
                    layer.observe(layer_frame)
                last = n == len(indices) - 1
                if frame.stamp_ns - round_start < cfg.round_s * 1e9 and not last:
                    continue
                stamp = frame.stamp_ns
                if layer is not None:
                    decided = layer.decide(stamp, last, backend.elements())
                    for k, v in decided.items():
                        retired[k] = retired.get(k, 0) + len(v)
                    ids = torch.unique(torch.cat(list(decided.values())))
                    backend.retire(ids, stamp)
                backend.snapshot(stamp)
                round_start = stamp
                if verbose:
                    print(f"{spec.name} t={(stamp - session.stamp_ns(indices[0])) / 1e9:7.1f}s "
                          f"frames={n + 1}/{len(indices)} elements={len(backend.elements())} "
                          f"retired={retired} {time.time() - t0:6.1f}s", flush=True)
        backend.timeline().save(d / "timeline.pkl")
        b_prior = backend.end_session() if carry else None
        if layer is not None:
            l_prior = layer.end_session(d)
        prev_final = stamp
        (d / "run.json").write_text(json.dumps({
            "backend": backend_name, "row": row, "row_name": ROWS[row], "dataset": dataset,
            "session": spec.name, "frames": len(indices), "seconds": round(time.time() - t0, 1),
            "input": {"pixel_step": INPUT_STEP[dataset], "every_frame": True,
                      "layer_pixel_step": cfg.pixel_step, "layer_frame_step": step},
            "retired_by_layer": retired, "own_update": own, "carried": carry,
            "backend_changes": list(backend.CHANGES),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0,
            "maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
            "config": {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in cfg.__dict__.items()}},
            indent=2))
        print(f"== {spec.name}: row {row} ({ROWS[row]}) {round(time.time() - t0)} s", flush=True)
        if carry and spec is not specs[-1]:
            save_checkpoint(out, name, backend, b_prior, l_prior, prev_final)
    if (out / "checkpoint.pt").exists() and not sessions:
        (out / "checkpoint.pt").unlink()             # the chain is complete


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    ap.add_argument("--row", type=int, choices=sorted(ROWS), required=True)
    ap.add_argument("--dataset", choices=["synthetic", "real"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sessions", default="", help="e.g. 'ab'; default all")
    ap.add_argument("--max-frames", type=int, default=0, help="frames per session (smoke tests)")
    ap.add_argument("--resume", action="store_true", help="continue after OUT/checkpoint.pt")
    args = ap.parse_args()
    run_chain(args.backend, args.row, args.dataset, Path(args.out), args.sessions, args.max_frames,
              resume=args.resume)


if __name__ == "__main__":
    main()
