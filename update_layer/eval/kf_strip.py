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


def _rebuild(sess, name, idx, scale):
    fs, c = sess[name]
    f = fs.load(idx, color=True)
    from update_layer.backends.game.game import gu
    # exactly the backend's operation (GameBackend.integrate: np2torch on the GPU, permute, / 255): the division on the
    # CPU differs in the last bit
    color = (gu.np2torch(np.ascontiguousarray(c(f.color)), device="cuda").permute(2, 0, 1) / 255.0).cpu()
    depth = np.nan_to_num(c(f.depth), nan=0.0).astype(np.float32)
    return color, depth


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
        if t is None or t not in by_stamp:
            out_kf[kid] = kf; kept += 1
            continue
        name, idx = by_stamp[t]
        color, depth_raw = _rebuild(sess, name, idx, scale)
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


def restore(ck: dict, ds: str) -> dict:
    b = ck["backend"]
    sess, _ = _sources(ds)
    out_kf = {}
    for kid, kf in b["keyframes"].items():
        if not (isinstance(kf, dict) and kf.get("stripped")):
            out_kf[kid] = kf
            continue
        color, depth_raw = _rebuild(sess, kf["session"], kf["index"], kf["scale"])
        d = depth_raw * kf["scale"]
        z = np.unpackbits(kf["zeroed"], count=d.size).astype(bool).reshape(d.shape)
        d[z] = 0.0
        masks = np.unpackbits(kf["masks"], count=int(np.prod(kf["masks_shape"]))).astype(bool).reshape(kf["masks_shape"])
        out_kf[kid] = {"color": color, "depth": torch.from_numpy(d).reshape(kf["depth_shape"]),
                       "masks": torch.from_numpy(masks), "pose": kf["pose"], "intrinsics": kf["intrinsics"]}
    new = dict(ck)
    new["backend"] = dict(b)
    new["backend"]["keyframes"] = out_kf
    new.pop("kf_strip", None)
    return new


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
    ck = torch.load(src, map_location="cpu", weights_only=False)
    if mode == "strip":
        new, n_s, n_k = strip(ck, ds)
        tmp = Path(dst + ".tmp")
        torch.save(new, tmp)
        back = restore(torch.load(tmp, map_location="cpu", weights_only=False), ds)
        n, bad = same_keyframes(ck, back)
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
