"""Lossless keyframe compression of a GaME checkpoint (disk, supervisor 2026-10-07): the stored keyframe images are the
dataset's own frames after GameBackend._sample's preprocessing, so they are rebuilt from the dataset instead of stored.

  python -m update_layer.eval.kf_strip strip   CKPT OUT real|synthetic    # writes OUT, then verifies OUT restores CKPT
  python -m update_layer.eval.kf_strip restore CKPT OUT real|synthetic    # a full checkpoint again

Per keyframe (stamp -> session and frame index): colour = crop(frame colour) / 255 as the backend stores it, rebuilt
from the dataset with the backend's own GPU operation (a CPU division differs in the last bit; needs a GPU, ~1 MB); depth = crop(frame depth) * scale with the pixels the backend zeroed (people and D1 motion
at mapping time, not recomputable offline) kept as a packed bit mask; masks (instance / semantic components) kept
bit-packed (np.packbits, exact). A keyframe whose rebuilt colour or depth is not bitwise equal to the stored tensor is
kept unchanged. Everything else (Gaussians, optimizer state, T1 / identity fields, poses, intrinsics, occlusion masks,
layer prior, RNG states) is copied as it is. 'strip' verifies: restore(OUT) == CKPT for every keyframe tensor, bitwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _sources(ds: str):
    from update_layer.backends.game.game import STEP, Crop
    from update_layer.frames import FlatSession
    from update_layer.run import dataset_config
    _, info, specs = dataset_config(ds)
    sess, by_stamp = {}, {}
    for sp in specs:
        fs = FlatSession(sp, pixel_step=1)
        sess[sp.name] = (fs, Crop(fs.K, STEP[info.name]))
        for i in range(len(fs.ids)):
            by_stamp[fs.stamp_ns(i)] = (sp.name, i)
    return sess, by_stamp


def _dataset_depth(sess, name, idx, frame=None) -> np.ndarray:
    """The frame's depth as GameBackend._sample crops it (metres, NaN -> 0): the one depth rebuild."""
    fs, c = sess[name]
    f = frame if frame is not None else fs.load(idx)
    return np.nan_to_num(c(f.depth), nan=0.0).astype(np.float32)


def _rebuild_color(sess, name, idx, frame=None) -> torch.Tensor:
    fs, c = sess[name]
    f = frame if frame is not None else fs.load(idx, color=True)
    from update_layer.backends.game.game import gu
    # exactly the backend's operation (GameBackend.integrate: np2torch on the GPU, permute, / 255): the division on the
    # CPU differs in the last bit
    return (gu.np2torch(np.ascontiguousarray(c(f.color)), device="cuda").permute(2, 0, 1) / 255.0).cpu()


def strip(ck: dict, ds: str) -> tuple:
    import os
    import yaml
    from update_layer.backends.game.game import CONFIGS
    b = ck["backend"]
    sess, by_stamp = _sources(ds)
    path = Path(os.environ["GAME_CONFIG"]) if os.environ.get("GAME_CONFIG") else CONFIGS[ds]   # the run's scale
    scale = float(yaml.safe_load(path.read_text())["game"].get("scale", 1.0))
    out_kf, kept, stripped = {}, 0, 0
    for kid, kf in b["keyframes"].items():
        t = b["kf_stamp"].get(kid)
        if isinstance(kf, dict) and kf.get("stripped"):          # already image-free ([S1] memory views)
            out_kf[kid] = kf
            continue
        if t is None or t not in by_stamp:
            out_kf[kid] = kf; kept += 1
            continue
        name, idx = by_stamp[t]
        f = sess[name][0].load(idx, color=True)
        color, depth_raw = _rebuild_color(sess, name, idx, f), _dataset_depth(sess, name, idx, f)
        stored_d = kf["depth"].numpy().reshape(depth_raw.shape)
        dscaled = depth_raw * scale
        zeroed = (stored_d == 0) & (dscaled != 0)
        rebuilt = dscaled.copy(); rebuilt[zeroed] = 0.0
        if not (torch.equal(color, kf["color"]) and np.array_equal(rebuilt, stored_d)):
            out_kf[kid] = kf; kept += 1
            continue
        m = kf["masks"].numpy().astype(bool)
        out_kf[kid] = {"stripped": True, "session": name, "index": idx, "scale": scale,
                       "depth_shape": tuple(kf["depth"].shape), "zeroed": np.packbits(zeroed, axis=None),
                       "masks_shape": tuple(m.shape), "masks": np.packbits(m, axis=None),
                       "pose": kf["pose"], "intrinsics": kf["intrinsics"]}
        stripped += 1
    new = dict(ck)
    new["backend"] = dict(b)
    new["backend"]["keyframes"] = out_kf
    new["kf_strip"] = {"tool": "update_layer/eval/kf_strip.py", "dataset": ds, "stripped": stripped, "kept": kept}
    return new, stripped, kept


def restore_depth(sess, kf: dict) -> torch.Tensor:
    """The depth tensor of one stripped keyframe (CPU only; the same rebuild as restore_one, without the colour)."""
    d = _dataset_depth(sess, kf["session"], kf["index"]) * kf["scale"]
    z = np.unpackbits(kf["zeroed"], count=d.size).astype(bool).reshape(d.shape)
    d[z] = 0.0
    return torch.from_numpy(d).reshape(kf["depth_shape"])


def restore_one(sess, kf: dict) -> dict:
    """One stripped keyframe -> the backend's keyframe dict (depth: restore_depth, the one implementation)."""
    color = _rebuild_color(sess, kf["session"], kf["index"])
    masks = np.unpackbits(kf["masks"], count=int(np.prod(kf["masks_shape"]))).astype(bool).reshape(kf["masks_shape"])
    return {"color": color, "depth": restore_depth(sess, kf),
            "masks": torch.from_numpy(masks), "pose": kf["pose"], "intrinsics": kf["intrinsics"]}


def restore(ck: dict, ds: str) -> dict:
    b = ck["backend"]
    sess, _ = _sources(ds)
    out_kf = {}
    for kid, kf in b["keyframes"].items():
        # [S1] memory views stay views: they never carried images or masks (the present's are never trained again)
        out_kf[kid] = restore_one(sess, kf) if isinstance(kf, dict) and kf.get("stripped") and not kf.get("view") else kf
    new = dict(ck)
    new["backend"] = dict(b)
    new["backend"]["keyframes"] = out_kf
    new.pop("kf_strip", None)
    return new


def _same(x, y) -> bool:
    ok = y is not None and set(x) == set(y)
    if ok:
        for k in x:
            xv, yv = x[k], y[k]
            if torch.is_tensor(xv):
                ok &= torch.is_tensor(yv) and xv.dtype == yv.dtype and xv.shape == yv.shape and torch.equal(xv, yv)
            elif isinstance(xv, np.ndarray):
                ok &= np.array_equal(xv, yv)
            else:
                ok &= xv == yv if not hasattr(xv, "__len__") else str(xv) == str(yv)
    return bool(ok)


def same_keyframes(a: dict, b: dict) -> tuple:
    bad = []
    for kid, x in a["backend"]["keyframes"].items():
        y = b["backend"]["keyframes"].get(kid)
        ok = y is not None and set(x) == set(y)
        if ok:
            for k in x:
                xv, yv = x[k], y[k]
                if torch.is_tensor(xv):
                    ok &= torch.is_tensor(yv) and xv.dtype == yv.dtype and xv.shape == yv.shape and torch.equal(xv, yv)
                elif isinstance(xv, np.ndarray):
                    ok &= np.array_equal(xv, yv)
                else:
                    ok &= xv == yv if not hasattr(xv, "__len__") else str(xv) == str(yv)
        if not ok:
            bad.append(kid)
    return len(a["backend"]["keyframes"]), bad


def main():
    mode, src, dst, ds = sys.argv[1:5]
    ck = torch.load(src, map_location="cpu", weights_only=False, mmap=True)      # memory-mapped: low RSS
    if mode == "strip":
        new, n_s, n_k = strip(ck, ds)
        tmp = Path(dst + ".tmp")
        torch.save(new, tmp)
        del new
        # verify keyframe by keyframe (no full restored copy in memory): restore(OUT) == CKPT, bitwise
        back = torch.load(tmp, map_location="cpu", weights_only=False, mmap=True)
        sess, _ = _sources(ds)
        bad = []
        for kid, x in ck["backend"]["keyframes"].items():
            y = back["backend"]["keyframes"].get(kid)
            if isinstance(y, dict) and y.get("stripped"):
                y = restore_one(sess, y)
            if not _same(x, y):
                bad.append(kid)
        n = len(ck["backend"]["keyframes"])
        if bad:
            tmp.unlink()
            sys.exit(f"verification FAILED: {len(bad)} of {n} keyframes differ after restore (e.g. {bad[:5]}); nothing written")
        tmp.replace(dst)
        print(f"{dst}: {n_s} keyframes stripped, {n_k} kept; restore verified bitwise for all {n} keyframes; "
              f"{Path(src).stat().st_size / 1e9:.2f} GB -> {Path(dst).stat().st_size / 1e9:.2f} GB")
    else:
        torch.save(restore(ck, ds), dst)
        print(f"{dst}: restored")


if __name__ == "__main__":
    main()
