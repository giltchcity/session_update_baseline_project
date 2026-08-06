#!/usr/bin/env python3
"""Add conservative construction-site cues to ADE20K semantic predictions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--cone-label", type=int, default=43, help="ADE signboard proxy label")
    parser.add_argument("--min-saturation", type=int, default=115)
    parser.add_argument("--min-value", type=int, default=75)
    return parser.parse_args()


def link_or_copy(source: Path, target: Path) -> None:
    if source.resolve() == target.resolve():
        return
    if target.exists() or target.is_symlink():
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def read_ids(run_dir: Path) -> list[str]:
    with (run_dir / "timestamps.csv").open() as stream:
        return [row["ImageID"] for row in csv.DictReader(stream)]


def main() -> None:
    args = parse_args()
    args.output_run.mkdir(parents=True, exist_ok=True)
    ids = read_ids(args.input_run)
    frame_rows = []
    for index, image_id in enumerate(ids):
        color_path = args.input_run / f"{image_id}_color.png"
        depth_path = args.input_run / f"{image_id}_depth.tiff"
        pose_path = args.input_run / f"{image_id}_pose.txt"
        mask_path = args.input_run / f"{image_id}_segmentation.png"
        if not all(path.exists() for path in (color_path, depth_path, pose_path, mask_path)):
            raise FileNotFoundError(f"Incomplete frame {image_id} in {args.input_run}")

        rgb = Image.open(color_path).convert("RGB")
        hsv = np.asarray(rgb.convert("HSV"), dtype=np.uint8)
        depth = np.asarray(Image.open(depth_path), dtype=np.float32)
        labels = np.asarray(Image.open(mask_path), dtype=np.uint8).copy()
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        # PIL hue is [0, 255]. Orange construction cones/barriers are roughly
        # 8-18 degrees, while red safety paint wraps around zero.
        raw_orange = (
            (hue >= 4)
            & (hue <= 22)
            & (saturation >= args.min_saturation)
            & (value >= args.min_value)
            & np.isfinite(depth)
            & (depth > 0.0)
        )
        orange = np.zeros_like(raw_orange)
        components, count = ndimage.label(raw_orange, structure=np.ones((3, 3), dtype=np.uint8))
        height, width = raw_orange.shape
        for component_id, component_slice in enumerate(
            ndimage.find_objects(components), start=1
        ):
            if component_slice is None:
                continue
            y_slice, x_slice = component_slice
            box_height = y_slice.stop - y_slice.start
            box_width = x_slice.stop - x_slice.start
            component = components[component_slice] == component_id
            area = int(component.sum())
            is_cone_shaped = (
                area >= 15
                and y_slice.start >= int(0.20 * height)
                and y_slice.stop >= int(0.55 * height)
                and box_height <= int(0.50 * height)
                and box_width <= int(0.18 * width)
                and box_height >= 1.15 * box_width
                and area >= 0.08 * box_height * box_width
            )
            if is_cone_shaped:
                orange[component_slice] |= component
        labels[orange] = np.uint8(args.cone_label)
        Image.fromarray(labels, mode="L").save(
            args.output_run / f"{image_id}_segmentation.png"
        )
        for source in (color_path, depth_path, pose_path):
            link_or_copy(source, args.output_run / source.name)
        frame_rows.append({"image_id": image_id, "orange_candidate_pixels": int(orange.sum())})
        if (index + 1) % 25 == 0 or index + 1 == len(ids):
            print(f"CONSTRUCTION_REFINE {index + 1}/{len(ids)}")

    link_or_copy(args.input_run / "timestamps.csv", args.output_run / "timestamps.csv")
    source_intrinsics = next(
        path
        for path in (args.input_run / "Intrinsics.txt", args.input_run.parent / "Intrinsics.txt")
        if path.exists()
    )
    link_or_copy(source_intrinsics, args.output_run.parent / "Intrinsics.txt")
    manifest = {
        "input_run": str(args.input_run.resolve()),
        "output_run": str(args.output_run.resolve()),
        "cone_proxy_label": args.cone_label,
        "rule": "orange HSV + valid depth + lower-image compact vertical component",
        "frames": frame_rows,
        "total_orange_candidate_pixels": sum(row["orange_candidate_pixels"] for row in frame_rows),
    }
    (args.output_run / "construction_refinement_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(f"CONSTRUCTION_REFINE_COMPLETE output={args.output_run}")


if __name__ == "__main__":
    main()
