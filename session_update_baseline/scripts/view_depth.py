#!/usr/bin/env python3
"""View an RGB-D depth frame as an image (depth.tiff -> colored PNG).

Depth is float32 meters; invalid pixels (0) are shown in red, valid depth is
rendered with a turbo-like colormap over the given range.

Usage:
    python scripts/view_depth.py --rgbd <rgbd_dir> --frame 325 \
        [--max-depth 4.5] [--scale 0.5] [--save /tmp/depth_view.png]

    --scale 0.5 also shows what the mapper actually ingests (960x540,
    nearest-neighbor depth, same as the production player).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def turbo(n: int) -> np.ndarray:
    """Small turbo-like colormap (blue -> green -> yellow -> red)."""
    stops = np.array(
        [
            [0.0, 0.0, 0.4],
            [0.0, 0.5, 1.0],
            [0.0, 1.0, 0.6],
            [0.6, 1.0, 0.0],
            [1.0, 0.8, 0.0],
            [1.0, 0.4, 0.0],
            [0.8, 0.1, 0.0],
        ]
    )
    x = np.linspace(0.0, 1.0, n)
    return np.column_stack(
        [np.interp(x, np.linspace(0, 1, len(stops)), stops[:, i]) for i in range(3)]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgbd", type=Path, required=True, help="RGB-D session dir")
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--max-depth", type=float, default=4.5)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="downscale depth with nearest (1.0 = original)")
    ap.add_argument("--save", type=Path, default=Path("/tmp/depth_view.png"))
    args = ap.parse_args()

    depth = np.array(Image.open(args.rgbd / f"{args.frame:06d}_depth.tiff")).astype(float)
    color = np.array(Image.open(args.rgbd / f"{args.frame:06d}_color.png"))
    if args.scale != 1.0:
        w = max(1, int(round(depth.shape[1] * args.scale)))
        h = max(1, int(round(depth.shape[0] * args.scale)))
        depth = np.array(Image.fromarray(depth).resize((w, h), Image.NEAREST))
        color = np.array(Image.fromarray(color).resize((w, h), Image.BILINEAR))

    valid = depth > 0.05
    norm = np.clip(depth / args.max_depth, 0.0, 1.0)
    cmap = turbo(256)
    vis = np.zeros((*depth.shape, 3), dtype=np.uint8)
    idx = (norm[valid] * 255).astype(int)
    vis[valid] = (cmap[idx] * 255).astype(np.uint8)
    vis[~valid] = [255, 0, 0]  # invalid depth in red

    h0 = color.shape[0]
    canvas = np.zeros((h0, color.shape[1] * 2, 3), dtype=np.uint8)
    canvas[:h0, : color.shape[1]] = color
    canvas[:h0, color.shape[1]:] = vis
    args.save.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(args.save)
    print(f"saved {args.save}")
    print(f"  left = RGB | right = depth (valid in colormap, invalid in red)")
    print(f"  depth stats: valid={(valid.mean()):.1%}, "
          f"max={depth[valid].max() if valid.any() else 0:.2f}m")


if __name__ == "__main__":
    main()
