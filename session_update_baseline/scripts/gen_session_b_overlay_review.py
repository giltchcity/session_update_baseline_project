#!/usr/bin/env python3
"""Session B keyframe overlay review in the EXACT style of
runs/full_ab_instance_maps_20260813/session_a/previews/*_instances_overlay.jpg.

The A previews are rendered by generate_full_ab_instance_maps.py::render_preview:
per-instance deterministic color, additive-weighted overlay (0.55 rgb / 0.45
color), per-instance bounding box with label "I{id} {name} S{semantic}".  B never
got this review, so this script renders the reviewed B instance maps
(full_ab_instance_maps_physical_reviewed_20260814) the same way:
  - every 30 frames over all 4041 frames (135 images, like A's cadence)
  - every 5 frames inside the known problem window 3749-4040 (I7 labeled I9)

Output: runs/I7_inspect/session_b_overlay_review/<frame>_instances_overlay.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def color_for_instance(instance_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(instance_id * 104729)
    return tuple(int(value) for value in rng.integers(55, 256, size=3))


def render_preview(rgb, instance_map, instance_semantic, names) -> np.ndarray:
    # Byte-for-byte the same rendering as generate_full_ab_instance_maps.py.
    overlay = rgb.copy()
    for instance_id in (int(x) for x in np.unique(instance_map) if x):
        overlay[instance_map == instance_id] = color_for_instance(instance_id)
    preview = cv2.addWeighted(rgb, 0.55, overlay, 0.45, 0.0)
    for instance_id in (int(x) for x in np.unique(instance_map) if x):
        mask = instance_map == instance_id
        ys, xs = np.nonzero(mask)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        color = color_for_instance(instance_id)
        cv2.rectangle(preview, (x0, y0), (x1, y1), color, 3)
        semantic_id = instance_semantic[instance_id]
        cv2.putText(preview, f"I{instance_id} {names[semantic_id]} S{semantic_id}",
                    (x0, max(28, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
    return preview


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-dir", type=Path, required=True)
    ap.add_argument("--instance-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    instance_semantic = {}
    names = {}
    for ent in manifest["entity_catalog"]:
        instance_semantic[ent["instance_id"]] = ent["semantic_id"]
        names[ent["semantic_id"]] = ent["physical_name"]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[int] = []
    frames += list(range(0, 4041, 30))
    frames += list(range(3749, 4041, 5))
    frames = sorted(set(frames))

    for i, frame in enumerate(frames):
        stem = f"{frame:06d}"
        rgb = cv2.imread(str(args.rgb_dir / f"{stem}_color.png"))
        inst = cv2.imread(str(args.instance_dir / f"{stem}_instances.png"), cv2.IMREAD_UNCHANGED)
        if rgb is None or inst is None:
            print(f"frame {frame}: missing input, skipped")
            continue
        preview = render_preview(rgb, inst, instance_semantic, names)
        cv2.putText(preview, f"frame {frame}",
                    (preview.shape[1] - 210, preview.shape[0] - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(preview, f"frame {frame}",
                    (preview.shape[1] - 210, preview.shape[0] - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
        out = args.out_dir / f"{stem}_instances_overlay.jpg"
        cv2.imwrite(str(out), preview, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if i % 20 == 0:
            print(f"  {i}/{len(frames)}  frame {frame}")

    print(f"done: {len(frames)} overlay images -> {args.out_dir}")


if __name__ == "__main__":
    main()
