#!/usr/bin/env python3
"""Validate a Panoptic flat-style virtual RGB-D dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path, help="Dataset root containing groundtruth_labels.csv and run dir.")
    parser.add_argument("--run-name", default="run_virtual")
    parser.add_argument("--report", type=Path, help="Optional JSON report path.")
    parser.add_argument("--min-valid-ratio", type=float, default=0.01)
    parser.add_argument("--max-valid-ratio", type=float, default=0.95)
    return parser.parse_args()


def read_timestamps(path: Path) -> list[str]:
    with path.open() as f:
        reader = csv.DictReader(f)
        return [row["ImageID"] for row in reader]


def check_pose(path: Path) -> dict:
    pose = np.loadtxt(path)
    if pose.shape != (4, 4):
        return {"ok": False, "error": f"pose shape {pose.shape}"}
    rot = pose[:3, :3]
    orth_error = float(np.linalg.norm(rot.T @ rot - np.eye(3)))
    det = float(np.linalg.det(rot))
    finite = bool(np.isfinite(pose).all())
    return {
        "ok": bool(finite and orth_error < 1e-4 and abs(det - 1.0) < 1e-4),
        "finite": finite,
        "orth_error": orth_error,
        "det": det,
        "position": pose[:3, 3].tolist(),
    }


def check_frame(run_dir: Path, image_id: str, min_valid_ratio: float, max_valid_ratio: float) -> dict:
    paths = {
        "color": run_dir / f"{image_id}_color.png",
        "depth": run_dir / f"{image_id}_depth.tiff",
        "seg": run_dir / f"{image_id}_segmentation.png",
        "pose": run_dir / f"{image_id}_pose.txt",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        return {"image_id": image_id, "ok": False, "missing": missing}

    color = np.asarray(Image.open(paths["color"]))
    depth = np.asarray(Image.open(paths["depth"]))
    seg = np.asarray(Image.open(paths["seg"]))
    valid = np.isfinite(depth) & (depth > 0.0)
    seg_nonzero = seg > 0
    valid_ratio = float(np.mean(valid))
    depth_on_seg_mismatch = int(np.count_nonzero(valid & ~seg_nonzero))
    seg_without_depth = int(np.count_nonzero(seg_nonzero & ~valid))

    if np.any(valid):
        valid_depth = depth[valid]
        depth_stats = {
            "min": float(np.min(valid_depth)),
            "median": float(np.median(valid_depth)),
            "max": float(np.max(valid_depth)),
        }
    else:
        depth_stats = {"min": None, "median": None, "max": None}

    pose_check = check_pose(paths["pose"])
    ok = (
        color.ndim == 3
        and depth.ndim == 2
        and seg.ndim == 2
        and color.shape[:2] == depth.shape == seg.shape
        and min_valid_ratio <= valid_ratio <= max_valid_ratio
        and depth_on_seg_mismatch == 0
        and seg_without_depth == 0
        and pose_check["ok"]
    )
    return {
        "image_id": image_id,
        "ok": bool(ok),
        "shape": list(depth.shape),
        "color_dtype": str(color.dtype),
        "depth_dtype": str(depth.dtype),
        "seg_dtype": str(seg.dtype),
        "valid_ratio": valid_ratio,
        "depth": depth_stats,
        "seg_ids": sorted(int(x) for x in np.unique(seg)),
        "depth_on_seg_mismatch": depth_on_seg_mismatch,
        "seg_without_depth": seg_without_depth,
        "pose": pose_check,
    }


def main() -> None:
    args = parse_args()
    run_dir = args.root / args.run_name
    timestamps_path = run_dir / "timestamps.csv"
    labels_path = args.root / "groundtruth_labels.csv"
    if not timestamps_path.exists():
        raise SystemExit(f"missing {timestamps_path}")
    if not labels_path.exists():
        raise SystemExit(f"missing {labels_path}")

    ids = read_timestamps(timestamps_path)
    frames = [check_frame(run_dir, image_id, args.min_valid_ratio, args.max_valid_ratio) for image_id in ids]
    valid_ratios = [f["valid_ratio"] for f in frames if "valid_ratio" in f]
    positions = np.asarray([f["pose"]["position"] for f in frames if f.get("pose", {}).get("ok")], dtype=float)
    report = {
        "root": str(args.root),
        "run_dir": str(run_dir),
        "frame_count": len(ids),
        "all_frames_ok": all(f["ok"] for f in frames),
        "valid_ratio_min": float(np.min(valid_ratios)) if valid_ratios else None,
        "valid_ratio_median": float(np.median(valid_ratios)) if valid_ratios else None,
        "valid_ratio_max": float(np.max(valid_ratios)) if valid_ratios else None,
        "camera_bbox_min": positions.min(axis=0).tolist() if len(positions) else None,
        "camera_bbox_max": positions.max(axis=0).tolist() if len(positions) else None,
        "bad_frames": [f for f in frames if not f["ok"]],
        "frames": frames,
    }

    print(json.dumps({k: v for k, v in report.items() if k != "frames"}, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        print(f"WROTE {args.report}")


if __name__ == "__main__":
    main()
