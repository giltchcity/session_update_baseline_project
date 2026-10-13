#!/usr/bin/env python3
"""Stabilize predicted NSS semantic masks through world-space voting.

NSS does not provide per-pixel object semantics. The textured-mesh adapter uses
an ADE20K model, whose class predictions can flicker between adjacent rendered
views. This adapter backprojects predicted object pixels with the rendered depth
and world_T_camera pose. A world voxel becomes a coarse object voxel only after
it receives object evidence from multiple distinct frames.

This is input adaptation, not ground-truth annotation. The output deliberately
uses one coarse object label so Khronos tests temporal object consistency rather
than ADE20K fine-class stability on synthetic renders.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import yaml
from PIL import Image


KEY_BITS = 21
KEY_BIAS = 1 << (KEY_BITS - 1)
KEY_MASK = (1 << KEY_BITS) - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-run", required=True, type=Path)
    parser.add_argument("--output-run", required=True, type=Path)
    parser.add_argument("--labelspace", required=True, type=Path)
    parser.add_argument("--voxel-size", type=float, default=0.20)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--min-object-frames", type=int, default=2)
    parser.add_argument("--canonical-object-label", type=int, default=19)
    return parser.parse_args()


def read_ids(run_dir: Path) -> List[str]:
    with (run_dir / "timestamps.csv").open() as stream:
        return [row["ImageID"] for row in csv.DictReader(stream)]


def load_intrinsics(run_dir: Path) -> Tuple[float, float, float, float]:
    candidates = [run_dir / "Intrinsics.txt", run_dir.parent / "Intrinsics.txt"]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError("Intrinsics.txt not found beside run directory")
    parsed: Dict[str, float] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        try:
            parsed[key.strip()] = float(value.strip())
        except ValueError:
            continue
    cx = parsed.get("c_x", parsed.get("u"))
    cy = parsed.get("c_y", parsed.get("v"))
    if cx is None or cy is None:
        raise KeyError("Intrinsics.txt must define c_x/c_y or u/v")
    return parsed["f_x"], parsed["f_y"], cx, cy


def pack_voxels(points: np.ndarray, voxel_size: float) -> np.ndarray:
    indices = np.floor(points / voxel_size).astype(np.int64)
    shifted = indices + KEY_BIAS
    if np.any(shifted < 0) or np.any(shifted > KEY_MASK):
        raise ValueError("voxel coordinate exceeds packed-key range")
    return (shifted[:, 0] << (2 * KEY_BITS)) | (shifted[:, 1] << KEY_BITS) | shifted[:, 2]


def backproject_codes(
    depth: np.ndarray,
    pose: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    voxel_size: float,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sampled = depth[::stride, ::stride]
    rows, cols = np.nonzero(np.isfinite(sampled) & (sampled > 0.0))
    if len(rows) == 0:
        return rows, cols, np.empty(0, dtype=np.int64)
    v = rows * stride
    u = cols * stride
    z = sampled[rows, cols].astype(np.float64)
    camera = np.stack(((u - cx) * z / fx, (v - cy) * z / fy, z, np.ones_like(z)), axis=1)
    world = (pose @ camera.T).T[:, :3]
    return rows, cols, pack_voxels(world, voxel_size)


def link_or_copy(source: Path, target: Path) -> None:
    if source.resolve() == target.resolve():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> None:
    args = parse_args()
    if args.sample_stride < 1 or args.min_object_frames < 1:
        raise ValueError("sample-stride and min-object-frames must be positive")

    labelspace = yaml.safe_load(args.labelspace.read_text())
    object_labels = np.asarray(sorted(set(labelspace["object_labels"])), dtype=np.uint8)
    if args.canonical_object_label not in set(int(value) for value in object_labels):
        raise ValueError("canonical object label is not an object in the label space")

    ids = read_ids(args.input_run)
    fx, fy, cx, cy = load_intrinsics(args.input_run)
    frame_object_codes: List[np.ndarray] = []

    for index, image_id in enumerate(ids):
        depth = np.asarray(Image.open(args.input_run / f"{image_id}_depth.tiff"), dtype=np.float32)
        labels = np.asarray(Image.open(args.input_run / f"{image_id}_segmentation.png"), dtype=np.uint8)
        pose = np.loadtxt(args.input_run / f"{image_id}_pose.txt", dtype=np.float64).reshape(4, 4)
        rows, cols, codes = backproject_codes(
            depth, pose, fx, fy, cx, cy, args.voxel_size, args.sample_stride
        )
        sampled_labels = labels[:: args.sample_stride, :: args.sample_stride]
        is_object = np.isin(sampled_labels[rows, cols], object_labels)
        frame_object_codes.append(np.unique(codes[is_object]))
        if (index + 1) % 25 == 0 or index + 1 == len(ids):
            print(f"VOTE_PASS {index + 1}/{len(ids)}")

    nonempty = [codes for codes in frame_object_codes if len(codes)]
    if nonempty:
        candidates, counts = np.unique(np.concatenate(nonempty), return_counts=True)
        stable_codes = candidates[counts >= args.min_object_frames]
    else:
        stable_codes = np.empty(0, dtype=np.int64)

    args.output_run.mkdir(parents=True, exist_ok=True)
    output_object_pixels = 0
    per_frame = []
    for index, image_id in enumerate(ids):
        depth = np.asarray(Image.open(args.input_run / f"{image_id}_depth.tiff"), dtype=np.float32)
        labels = np.asarray(Image.open(args.input_run / f"{image_id}_segmentation.png"), dtype=np.uint8)
        pose = np.loadtxt(args.input_run / f"{image_id}_pose.txt", dtype=np.float64).reshape(4, 4)
        rows, cols, codes = backproject_codes(
            depth, pose, fx, fy, cx, cy, args.voxel_size, args.sample_stride
        )
        coarse = labels[:: args.sample_stride, :: args.sample_stride].copy()
        coarse[np.isin(coarse, object_labels)] = 0
        stable = np.isin(codes, stable_codes)
        coarse[rows[stable], cols[stable]] = args.canonical_object_label
        output = np.repeat(np.repeat(coarse, args.sample_stride, axis=0), args.sample_stride, axis=1)
        output = output[: labels.shape[0], : labels.shape[1]]
        object_pixels = int(np.count_nonzero(output == args.canonical_object_label))
        output_object_pixels += object_pixels
        Image.fromarray(output.astype(np.uint8), mode="L").save(
            args.output_run / f"{image_id}_segmentation.png"
        )
        for suffix in ("_color.png", "_depth.tiff", "_pose.txt"):
            link_or_copy(args.input_run / f"{image_id}{suffix}", args.output_run / f"{image_id}{suffix}")
        per_frame.append({"image_id": image_id, "object_pixels": object_pixels})
        if (index + 1) % 25 == 0 or index + 1 == len(ids):
            print(f"WRITE_PASS {index + 1}/{len(ids)}")

    link_or_copy(args.input_run / "timestamps.csv", args.output_run / "timestamps.csv")
    for candidate in (args.input_run.parent / "Intrinsics.txt", args.input_run / "Intrinsics.txt"):
        if candidate.exists():
            link_or_copy(candidate, args.output_run.parent / "Intrinsics.txt")
            break

    manifest = {
        "input_run": str(args.input_run.resolve()),
        "output_run": str(args.output_run.resolve()),
        "method": "world_voxel_multiframe_coarse_object_voting",
        "semantic_source": "SegFormer ADE20K prediction; not NSS ground truth",
        "voxel_size": args.voxel_size,
        "sample_stride": args.sample_stride,
        "min_object_frames": args.min_object_frames,
        "canonical_object_label": args.canonical_object_label,
        "input_object_labels": [int(value) for value in object_labels],
        "stable_object_voxels": int(len(stable_codes)),
        "output_object_pixels": output_object_pixels,
        "frames": per_frame,
    }
    (args.output_run / "semantic_stabilization_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(
        "STABILIZATION_COMPLETE "
        f"frames={len(ids)} stable_object_voxels={len(stable_codes)} "
        f"object_pixels={output_object_pixels}"
    )


if __name__ == "__main__":
    main()
