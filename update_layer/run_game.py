"""Session chain with GaME (3D Gaussian map) as the backend.

  python -m update_layer.run_game synthetic OUT_DIR --mode scratch  # GaME from scratch every session (row 1)
  python -m update_layer.run_game synthetic OUT_DIR --mode naive    # one map, nothing ever removed (row 2)
  python -m update_layer.run_game synthetic OUT_DIR --mode game     # GaME as published (row 3)
  python -m update_layer.run_game synthetic OUT_DIR --mode layer    # GaME + update layer (row 4)

GaME keeps one Gaussian model across the sessions (its own multi-session setting). In `layer`
mode GaME's change handling is off; the update layer runs in lockstep on the same frames
(the point frontend supplies its observations and evidence, see update_layer.layer) and every
retirement it decides is executed on the Gaussians:
  object state closed    -> Gaussians of that identity on the closed state's surface and not on
                            the identity's displayed surface (5 cm cells)
  background element     -> identity-0 Gaussians inside the retired element's 2 cm voxel
Snapshots: at every reconciliation round the map is rendered from fixed evaluation cameras
(all sessions' GT poses at 1 Hz, identical for every method) into EvaluationScenes.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from update_layer.backends.game_backend import (FLAT_CONFIG, GameFrames, SceneListTimeline,  # noqa: E402
                                                TrackedGaME)
from update_layer.evidence import DEV, SensorStatistics, voxel_keys  # noqa: E402
from update_layer.frames import FlatSession, real_session, synthetic_session  # noqa: E402
from update_layer.layer import OPEN, RoundBuffer, UpdateLayerSession, export_state  # noqa: E402
from update_layer.run_chain import dataset_config  # noqa: E402
from eval.scene.scene import EvaluationScene  # noqa: E402


def eval_cameras(specs, frames_by_session, hz: float = 1.0):
    """Fixed evaluation cameras: every session's GT pose at `hz`, with GaME's cropped intrinsics."""
    cams = []
    for spec, fr in zip(specs, frames_by_session):
        s = fr.session
        step = max(1, int(round(30.0 / hz)))
        h, w = fr.crop.rows, fr.crop.cols
        for i in range(0, len(s.ids), step):
            cams.append((s.pose(i), fr.K, h, w))
    return cams


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["synthetic", "real"])
    ap.add_argument("out")
    ap.add_argument("--mode", choices=["scratch", "naive", "game", "layer"], required=True)
    ap.add_argument("--hz", type=float, default=5.0, help="frame rate the update layer reads (evidence)")
    ap.add_argument("--game-hz", type=float, default=1.0, help="frame rate GaME maps (keyframe candidates)")
    ap.add_argument("--cam-hz", type=float, default=0.0, help="evaluation cameras per second of trajectory")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--static-frames", action="store_true",
                    help="t2's static-surface frame selection for observations (LayerConfig.static_frames)")
    args = ap.parse_args()
    cfg, specs = dataset_config(args.dataset, "layer")
    cfg.static_frames = args.static_frames
    step = 2 if args.dataset == "synthetic" else 4          # the resolution the layer reads
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sessions = [FlatSession(sp, pixel_step=1) for sp in specs]
    frames = [GameFrames(s, stride=int(round(30 / args.hz)), step=step, scale=FLAT_CONFIG["scale"]) for s in sessions]
    cams = eval_cameras(specs, frames, hz=args.cam_hz or (1.0 if args.dataset == "synthetic" else 0.5))
    game_every = max(1, int(round(args.hz / args.game_hz)))
    own_update = args.mode in ("game", "scratch")        # GaME's change handling (as published)
    game = TrackedGaME(dict(FLAT_CONFIG), change_handling=own_update)
    semantic_of = {}
    prior, stats = None, None
    frame_id = 0
    last_scene = None
    for spec, sess, fr in zip(specs, sessions, frames):
        name = spec.name.split("_")[-1]
        d = out / f"session_{name}"
        d.mkdir(exist_ok=True)
        t0 = time.time()
        if args.mode == "scratch" and last_scene is not None:   # every session starts empty
            del game
            gc.collect()                                      # the model and its patched methods form a cycle
            torch.cuda.empty_cache()
            game = TrackedGaME(dict(FLAT_CONFIG), change_handling=own_update)
            last_scene = EvaluationScene(last_scene.t_ns, np.zeros((0, 3), np.float32), np.zeros(0, np.uint32), [],
                                         np.zeros((0, 3), np.float32))   # the map at the boundary: empty
        layer = None
        if args.mode == "layer":
            # the layer sees the same frames as GaME (5 Hz); its point frontend supplies evidence
            cfg.evidence_hz = args.hz
            layer = UpdateLayerSession(cfg, prior=prior, stats=stats)
            layer.seed(sess.stamp_ns(0))
            ls = FlatSession(spec, pixel_step=cfg.pixel_step)
        stamps, scenes = [], []
        if last_scene is not None:                         # the inherited map is the first snapshot
            stamps.append(last_scene.t_ns)
            scenes.append(last_scene)
        buf = RoundBuffer()
        round_start = fr.stamp(0)
        retired_total = [0, 0]
        with ThreadPoolExecutor(max_workers=4) as pool:
            n_frames = min(len(fr), args.max_frames) if args.max_frames else len(fr)
            game_ks = [k for k in range(n_frames) if k % game_every == 0]
            futures = {k: pool.submit(fr.sample, k) for k in game_ks[:4]}
            queued = 4
            for k in range(n_frames):
                stamp = fr.stamp(k)
                if k in futures:
                    sample, f = futures.pop(k).result()
                    if queued < len(game_ks):
                        futures[game_ks[queued]] = pool.submit(fr.sample, game_ks[queued])
                        queued += 1
                    if f.semantic is not None:
                        ins, sem = f.instance, f.semantic
                        for i in np.unique(ins[ins > 0]).tolist():
                            if i not in semantic_of:
                                vals, cnt = np.unique(sem[ins == i], return_counts=True)
                                semantic_of[i] = int(vals[np.argmax(cnt)])
                    game.process(frame_id, sample, stamp)
                    frame_id += 1
                if layer is not None:
                    layer.integrate(ls.load(fr.indices[k]), buf)
                last = k == n_frames - 1
                if stamp - round_start >= cfg.round_s * 1e9 or last:
                    if layer is not None:
                        retired_total = [a + b for a, b in zip(retired_total, execute_round(layer, game, buf, stamp, last))]
                        buf = RoundBuffer()
                    scene = game.render_scene(cams, stamp, semantic_of)
                    stamps.append(stamp)
                    scenes.append(scene)
                    print(f"{name} t={(stamp - fr.stamp(0)) / 1e9:6.1f}s frames={k + 1}/{n_frames} "
                          f"gaussians={game.gaussian_model.get_xyz.shape[0]} keyframes={len(game.keyframes)} "
                          f"retired obj/bg gaussians={retired_total} {time.time() - t0:6.0f}s", flush=True)
                    round_start = stamp
        last_scene = scenes[-1]
        SceneListTimeline(stamps, scenes).save(d / "timeline.pkl")
        if layer is not None:
            res_stamps = layer.stamps
            (d / "log.txt").write_text("\n".join(layer.log) + "\n")
            layer.stats.save(d / "sensor_statistics.txt")
            from update_layer.layer import SessionResult
            res = SessionResult(layer.stamps, layer.display, layer.fragments, layer.fragment_identity,
                                layer.elements, layer.registry, layer.stats, layer.log, None)
            prior = export_state(res)
            stats = SensorStatistics()
            stats.load(d / "sensor_statistics.txt")
        (d / "run.json").write_text(json.dumps({
            "dataset": args.dataset, "mode": args.mode, "session": spec.name, "hz": args.hz,
            "game_hz": args.game_hz,
            "seconds": round(time.time() - t0, 1), "snapshots": len(stamps),
            "gaussians": int(game.gaussian_model.get_xyz.shape[0]), "keyframes": len(game.keyframes),
            "eval_cameras": len(cams), "retired_gaussians_obj_bg": retired_total,
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
            "maxrss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}, indent=2))
        print(f"== {spec.name}: {len(stamps)} snapshots, {round(time.time() - t0)} s", flush=True)


def _open_fragments(state) -> list:
    """Every not-yet-closed fragment of a physical state, including the in-session (B) state."""
    out = [f for f in state.fragments if f.death_time is None]
    if state.observed_new is not None:
        out.append(state.observed_new)
    if state.b_session is not None:
        out += _open_fragments(state.b_session)
    return out


def execute_round(layer: UpdateLayerSession, game: TrackedGaME, buf: RoundBuffer, stamp: int,
                  last: bool) -> list:
    """One reconciliation round of the layer, then its retirements executed on the Gaussians."""
    reg = layer.registry
    before = {i: _open_fragments(st) for i, st in reg.states.items()}
    dead_before = layer.elements.death[: layer.elements.n].clone()
    layer._verify(stamp)
    layer.closed_background.run(stamp)
    layer._element_rule(buf.stamps, stamp)
    layer._ingest(buf)
    if last:
        reg.finalize_pending_absences(stamp)
        layer._verify(stamp)
        reg.finalize_pending_absences(stamp)
        layer.closed_background.run(stamp)
    layer._snapshot(stamp)
    xyz = game.gaussian_model.get_xyz.detach()
    ident = game.identity
    retire = torch.zeros(len(xyz), dtype=torch.bool, device=xyz.device)
    # object states closed in this round
    n_obj = 0
    cell = 0.05
    for i, frags in before.items():
        closed = [f for f in frags if f.death_time is not None]
        if not closed:
            continue
        shown = reg.displayed(i)
        shown_keys = torch.unique(voxel_keys(shown[0], cell)) if shown is not None and len(shown[0]) else \
            torch.zeros(0, dtype=torch.int64, device=DEV)
        closed_keys = torch.unique(torch.cat([voxel_keys(f.points, cell) for f in closed]))
        mine = ident == i
        if not mine.any():
            continue
        gk = voxel_keys(xyz[mine], cell)
        hit = torch.isin(gk, closed_keys) & ~torch.isin(gk, shown_keys)
        idx = torch.nonzero(mine).squeeze(1)[hit]
        retire[idx] = True
        n_obj += int(hit.sum())
    # background elements retired in this round
    el = layer.elements
    newly = (el.death[: el.n] != OPEN) & (dead_before == OPEN)
    n_bg = 0
    if newly.any():
        dead_keys = torch.unique(el.encode(el.xyz(torch.nonzero(newly).squeeze(1))))
        bgm = ident == 0
        if bgm.any():
            gk = el.encode(xyz[bgm])
            hit = torch.isin(gk, dead_keys)
            retire[torch.nonzero(bgm).squeeze(1)[hit]] = True
            n_bg = int(hit.sum())
    game.retire_gaussians(retire)
    return [n_obj, n_bg]


if __name__ == "__main__":
    main()
