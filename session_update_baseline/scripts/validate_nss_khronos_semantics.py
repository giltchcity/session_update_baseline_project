#!/usr/bin/env python3
"""Validate NSS RGB-D semantics against Khronos/Hydra ADE20K conventions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--labelspace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-empty-depth", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.labelspace.read_text())
    total_labels = int(config["total_semantic_labels"])
    object_labels = np.asarray(config["object_labels"], dtype=np.uint8)
    with (args.run / "timestamps.csv").open() as stream:
        ids = [row["ImageID"] for row in csv.DictReader(stream)]

    rows = []
    for image_id in ids:
        paths = {
            "color": args.run / f"{image_id}_color.png",
            "depth": args.run / f"{image_id}_depth.tiff",
            "segmentation": args.run / f"{image_id}_segmentation.png",
            "pose": args.run / f"{image_id}_pose.txt",
        }
        missing = [name for name, path in paths.items() if not path.exists()]
        if missing:
            rows.append({"image_id": image_id, "ok": False, "missing": missing})
            continue
        color = np.asarray(Image.open(paths["color"]).convert("RGB"))
        depth = np.asarray(Image.open(paths["depth"]), dtype=np.float32)
        labels = np.asarray(Image.open(paths["segmentation"]), dtype=np.uint8)
        pose = np.loadtxt(paths["pose"], dtype=np.float64).reshape(4, 4)
        rotation = pose[:3, :3]
        valid_depth = np.isfinite(depth) & (depth > 0.0)
        valid_labels = bool(labels.size and int(labels.max()) < total_labels)
        shape_ok = color.shape[:2] == depth.shape == labels.shape
        pose_ok = bool(
            np.isfinite(pose).all()
            and np.linalg.norm(rotation.T @ rotation - np.eye(3)) < 1e-4
            and abs(np.linalg.det(rotation) - 1.0) < 1e-4
        )
        object_pixels = np.isin(labels, object_labels) & valid_depth
        has_observation = bool(valid_depth.any())
        rows.append(
            {
                "image_id": image_id,
                "ok": bool(shape_ok and pose_ok and valid_labels
                           and (has_observation or args.allow_empty_depth)),
                "observation_state": "observed" if has_observation else "unobserved",
                "valid_depth_ratio": float(valid_depth.mean()),
                "object_pixels_with_depth": int(object_pixels.sum()),
                "object_fraction_of_valid_depth": (
                    float(object_pixels.sum() / valid_depth.sum()) if has_observation else None
                ),
                "max_label": int(labels.max()),
            }
        )

    valid_ratios = [row["valid_depth_ratio"] for row in rows if "valid_depth_ratio" in row]
    object_fractions = [
        row["object_fraction_of_valid_depth"]
        for row in rows
        if row.get("object_fraction_of_valid_depth") is not None
    ]
    observed_ratios = [value for value in valid_ratios if value > 0.0]
    report = {
        "contract": "Khronos/Hydra ADE20K zero-indexed semantics; label 0 is wall, not invalid",
        "run": str(args.run.resolve()),
        "frame_count": len(ids),
        "all_frames_ok": all(row["ok"] for row in rows),
        "allow_empty_depth": args.allow_empty_depth,
        "observed_frame_count": len(observed_ratios),
        "unobserved_frame_count": len(valid_ratios) - len(observed_ratios),
        "valid_depth_ratio_min": float(np.min(valid_ratios)),
        "valid_depth_ratio_median": float(np.median(valid_ratios)),
        "valid_depth_ratio_max": float(np.max(valid_ratios)),
        "observed_depth_ratio_min": float(np.min(observed_ratios)),
        "observed_depth_ratio_median": float(np.median(observed_ratios)),
        "observed_depth_ratio_max": float(np.max(observed_ratios)),
        "object_fraction_min": float(np.min(object_fractions)),
        "object_fraction_median": float(np.median(object_fractions)),
        "object_fraction_max": float(np.max(object_fractions)),
        "bad_frames": [row for row in rows if not row["ok"]],
        "frames": rows,
    }
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "frames"}, indent=2))


if __name__ == "__main__":
    main()
