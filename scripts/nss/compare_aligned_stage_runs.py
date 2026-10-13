#!/usr/bin/env python3
"""Measure geometric change between aligned flat RGB-D stage runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read_ids(run: Path) -> list[str]:
    with (run / "timestamps.csv").open() as stream:
        return [row["ImageID"] for row in csv.DictReader(stream)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thresholds-m", nargs="+", type=float, default=[0.1, 0.2, 0.5])
    args = parser.parse_args()

    ids_a = read_ids(args.run_a)
    ids_b = read_ids(args.run_b)
    if ids_a != ids_b:
        raise RuntimeError("A/B ImageID sequences differ; comparison requires identical poses")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    totals = {
        "pixels": 0,
        "a_valid": 0,
        "b_valid": 0,
        "both_valid": 0,
        "appeared": 0,
        "disappeared": 0,
    }
    for threshold in args.thresholds_m:
        totals[f"changed_{threshold:g}"] = 0
        totals[f"b_closer_{threshold:g}"] = 0
        totals[f"b_farther_{threshold:g}"] = 0

    rows = []
    all_shared_deltas = []
    max_pose_error = 0.0
    for index, image_id in enumerate(ids_a):
        pose_a = np.loadtxt(args.run_a / f"{image_id}_pose.txt").reshape(4, 4)
        pose_b = np.loadtxt(args.run_b / f"{image_id}_pose.txt").reshape(4, 4)
        pose_error = float(np.max(np.abs(pose_a - pose_b)))
        max_pose_error = max(max_pose_error, pose_error)
        if pose_error > 1e-8:
            raise RuntimeError(f"A/B pose mismatch at {image_id}: max_abs={pose_error}")
        depth_a = np.asarray(Image.open(args.run_a / f"{image_id}_depth.tiff"), dtype=np.float32)
        depth_b = np.asarray(Image.open(args.run_b / f"{image_id}_depth.tiff"), dtype=np.float32)
        if depth_a.shape != depth_b.shape:
            raise RuntimeError(f"Depth shape mismatch at {image_id}")
        valid_a = np.isfinite(depth_a) & (depth_a > 0.0)
        valid_b = np.isfinite(depth_b) & (depth_b > 0.0)
        both = valid_a & valid_b
        appeared = ~valid_a & valid_b
        disappeared = valid_a & ~valid_b
        delta = depth_b[both] - depth_a[both]
        abs_delta = np.abs(delta)
        if abs_delta.size:
            all_shared_deltas.append(abs_delta)

        row = {
            "frame_index": index,
            "image_id": image_id,
            "a_valid_fraction": float(np.mean(valid_a)),
            "b_valid_fraction": float(np.mean(valid_b)),
            "both_valid_fraction": float(np.mean(both)),
            "appeared_fraction": float(np.mean(appeared)),
            "disappeared_fraction": float(np.mean(disappeared)),
            "shared_abs_depth_median_m": float(np.median(abs_delta)) if abs_delta.size else None,
            "shared_abs_depth_p90_m": float(np.percentile(abs_delta, 90)) if abs_delta.size else None,
        }
        totals["pixels"] += depth_a.size
        totals["a_valid"] += int(valid_a.sum())
        totals["b_valid"] += int(valid_b.sum())
        totals["both_valid"] += int(both.sum())
        totals["appeared"] += int(appeared.sum())
        totals["disappeared"] += int(disappeared.sum())
        for threshold in args.thresholds_m:
            changed = abs_delta > threshold
            closer = delta < -threshold
            farther = delta > threshold
            key = f"{threshold:g}"
            row[f"changed_shared_fraction_{key}m"] = float(np.mean(changed)) if delta.size else 0.0
            row[f"b_closer_shared_fraction_{key}m"] = float(np.mean(closer)) if delta.size else 0.0
            row[f"b_farther_shared_fraction_{key}m"] = float(np.mean(farther)) if delta.size else 0.0
            totals[f"changed_{key}"] += int(changed.sum())
            totals[f"b_closer_{key}"] += int(closer.sum())
            totals[f"b_farther_{key}"] += int(farther.sum())
        rows.append(row)

    with (args.output_dir / "frame_change_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    shared_delta = np.concatenate(all_shared_deltas) if all_shared_deltas else np.empty(0)
    summary = {
        "contract": "same ImageID and camera pose; Stage2 mesh transformed into Stage1 coordinates",
        "run_a": str(args.run_a.resolve()),
        "run_b": str(args.run_b.resolve()),
        "frame_count": len(rows),
        "max_pose_matrix_abs_difference": max_pose_error,
        "pixel_count": totals["pixels"],
        "a_valid_fraction": totals["a_valid"] / totals["pixels"],
        "b_valid_fraction": totals["b_valid"] / totals["pixels"],
        "both_valid_fraction": totals["both_valid"] / totals["pixels"],
        "appeared_fraction_all_pixels": totals["appeared"] / totals["pixels"],
        "disappeared_fraction_all_pixels": totals["disappeared"] / totals["pixels"],
        "shared_abs_depth_median_m": float(np.median(shared_delta)) if shared_delta.size else None,
        "shared_abs_depth_p90_m": float(np.percentile(shared_delta, 90)) if shared_delta.size else None,
        "thresholds": {},
    }
    for threshold in args.thresholds_m:
        key = f"{threshold:g}"
        denominator = max(totals["both_valid"], 1)
        summary["thresholds"][key] = {
            "changed_shared_fraction": totals[f"changed_{key}"] / denominator,
            "b_closer_shared_fraction": totals[f"b_closer_{key}"] / denominator,
            "b_farther_shared_fraction": totals[f"b_farther_{key}"] / denominator,
        }
    (args.output_dir / "change_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
