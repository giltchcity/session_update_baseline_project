#!/usr/bin/env python3
"""Propagate validated sparse semantic masks onto a denser RGB-D trajectory."""

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
    parser.add_argument("--dense-run", type=Path, required=True)
    parser.add_argument("--sparse-semantic-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--depth-tolerance-m", type=float, default=0.20)
    parser.add_argument("--max-fill-px", type=float, default=20.0)
    parser.add_argument("--frame-limit", type=int, default=0)
    parser.add_argument("--reuse-existing", action="store_true")
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


def read_intrinsics(run: Path) -> tuple[int, int, float, float, float, float]:
    path = next(path for path in (run / "Intrinsics.txt", run.parent / "Intrinsics.txt")
                if path.exists())
    values = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = float(value.strip())
    return (int(values.get("width", values.get("Res_x", 640))),
            int(values.get("height", values.get("Res_y", 480))),
            values["f_x"], values["f_y"],
            values.get("c_x", values.get("u")),
            values.get("c_y", values.get("v")))


def project_mask(source_run: Path, source_id: str, target_pose: np.ndarray,
                 target_depth: np.ndarray, intrinsics: tuple[int, int, float, float, float, float]
                 ) -> tuple[np.ndarray, np.ndarray]:
    width, height, fx, fy, cx, cy = intrinsics
    source_depth = np.asarray(Image.open(source_run / f"{source_id}_depth.tiff"),
                              dtype=np.float32)
    source_mask = np.asarray(Image.open(source_run / f"{source_id}_segmentation.png"),
                             dtype=np.uint8)
    source_pose = np.loadtxt(source_run / f"{source_id}_pose.txt").reshape(4, 4)

    v, u = np.indices(source_depth.shape)
    valid = np.isfinite(source_depth) & (source_depth > 0.0)
    z = source_depth[valid]
    points = np.stack(((u[valid] - cx) * z / fx,
                       (v[valid] - cy) * z / fy, z, np.ones_like(z)), axis=0)
    target_points = np.linalg.inv(target_pose) @ (source_pose @ points)
    z_target = target_points[2]
    valid_z = z_target > 0.0
    x = np.rint(fx * target_points[0] / z_target + cx).astype(np.int32)
    y = np.rint(fy * target_points[1] / z_target + cy).astype(np.int32)
    inside = valid_z & (x >= 0) & (x < width) & (y >= 0) & (y < height)

    x, y, z_target = x[inside], y[inside], z_target[inside]
    labels = source_mask[valid][inside]
    if len(z_target) == 0:
        return (
            np.zeros((height, width), dtype=np.uint8),
            np.full((height, width), np.inf, dtype=np.float32),
        )
    pixels = y * width + x
    order = np.lexsort((z_target, pixels))
    pixels, z_target, labels = pixels[order], z_target[order], labels[order]
    first = np.r_[True, pixels[1:] != pixels[:-1]]
    pixels, z_target, labels = pixels[first], z_target[first], labels[first]

    reference = target_depth.reshape(-1)[pixels]
    consistent = ((reference > 0.0) & np.isfinite(reference)
                  & (np.abs(z_target - reference) <= args_global.depth_tolerance_m))
    output = np.zeros(height * width, dtype=np.uint8)
    error = np.full(height * width, np.inf, dtype=np.float32)
    output[pixels[consistent]] = labels[consistent]
    error[pixels[consistent]] = np.abs(z_target[consistent] - reference[consistent])
    return output.reshape(height, width), error.reshape(height, width)


def nearest_dense_indices(dense_poses: list[np.ndarray], sparse_poses: list[np.ndarray]) -> list[int]:
    dense_xyz = np.stack([pose[:3, 3] for pose in dense_poses])
    dense_forward = np.stack([pose[:3, 2] for pose in dense_poses])
    result = []
    for pose in sparse_poses:
        translation_error = np.linalg.norm(dense_xyz - pose[:3, 3], axis=1)
        forward_dot = np.clip(dense_forward @ pose[:3, 2], -1.0, 1.0)
        rotation_error = np.arccos(forward_dot)
        # Exact samples score zero. The orientation term disambiguates an
        # out-and-back route where the same location is viewed in both directions.
        score = translation_error + 0.25 * rotation_error
        result.append(int(np.argmin(score)))
    if any(b <= a for a, b in zip(result, result[1:])):
        raise RuntimeError("Sparse poses do not map monotonically onto the dense trajectory")
    return result


def main() -> None:
    global args_global
    args_global = parse_args()
    args_global.output_run.mkdir(parents=True, exist_ok=True)
    intrinsics = read_intrinsics(args_global.dense_run)
    dense_pose_files = sorted(args_global.dense_run.glob("*_pose.txt"))
    sparse_pose_files = sorted(args_global.sparse_semantic_run.glob("*_pose.txt"))
    dense_ids = [path.name[:-len("_pose.txt")] for path in dense_pose_files]
    sparse_ids = [path.name[:-len("_pose.txt")] for path in sparse_pose_files]
    dense_poses = [np.loadtxt(path).reshape(4, 4) for path in dense_pose_files]
    sparse_poses = [np.loadtxt(path).reshape(4, 4) for path in sparse_pose_files]
    anchors = nearest_dense_indices(dense_poses, sparse_poses)
    if max(np.linalg.norm(dense_poses[index][:3, 3] - sparse_poses[k][:3, 3])
           for k, index in enumerate(anchors)) > 1e-4:
        raise RuntimeError("Sparse semantic poses are not exact samples of the dense trajectory")

    process_count = (min(args_global.frame_limit, len(dense_ids))
                     if args_global.frame_limit > 0 else len(dense_ids))
    frame_rows = []
    for dense_index, (image_id, target_pose) in enumerate(
            zip(dense_ids[:process_count], dense_poses[:process_count])):
        output_mask = args_global.output_run / f"{image_id}_segmentation.png"
        if output_mask.exists() and args_global.reuse_existing:
            for suffix in ("_color.png", "_depth.tiff", "_pose.txt"):
                link_or_copy(args_global.dense_run / f"{image_id}{suffix}",
                             args_global.output_run / f"{image_id}{suffix}")
            frame_rows.append({"image_id": image_id, "projected_fraction": None,
                               "reused_mask": True})
            print(f"PROPAGATED_REUSED {dense_index + 1}/{process_count} id={image_id}")
            continue
        nearest = int(np.argmin(np.abs(np.asarray(anchors) - dense_index)))
        target_depth = np.asarray(
            Image.open(args_global.dense_run / f"{image_id}_depth.tiff"), dtype=np.float32)
        if anchors[nearest] == dense_index:
            mask = np.asarray(Image.open(
                args_global.sparse_semantic_run / f"{sparse_ids[nearest]}_segmentation.png"),
                dtype=np.uint8)
            projected_fraction = 1.0
        else:
            candidates = {nearest}
            if anchors[nearest] < dense_index and nearest + 1 < len(anchors):
                candidates.add(nearest + 1)
            if anchors[nearest] > dense_index and nearest > 0:
                candidates.add(nearest - 1)
            mask = np.zeros(target_depth.shape, dtype=np.uint8)
            best_error = np.full(target_depth.shape, np.inf, dtype=np.float32)
            for sparse_index in sorted(candidates):
                projected, error = project_mask(
                    args_global.sparse_semantic_run, sparse_ids[sparse_index], target_pose,
                    target_depth, intrinsics)
                better = error < best_error
                mask[better] = projected[better]
                best_error[better] = error[better]
            observed = np.isfinite(best_error)
            projected_fraction = float(np.mean(observed[target_depth > 0.0]))
            if np.any(observed):
                distance, indices = ndimage.distance_transform_edt(
                    ~observed, return_distances=True, return_indices=True)
                fill = (~observed) & (target_depth > 0.0) & (distance <= args_global.max_fill_px)
                mask[fill] = mask[indices[0][fill], indices[1][fill]]

        Image.fromarray(mask, mode="L").save(output_mask)
        for suffix in ("_color.png", "_depth.tiff", "_pose.txt"):
            link_or_copy(args_global.dense_run / f"{image_id}{suffix}",
                         args_global.output_run / f"{image_id}{suffix}")
        frame_rows.append({"image_id": image_id, "projected_fraction": projected_fraction})
        print(f"PROPAGATED_FRAME {dense_index + 1}/{process_count} id={image_id}")

    link_or_copy(args_global.dense_run / "timestamps.csv",
                 args_global.output_run / "timestamps.csv")
    intrinsics_path = next(path for path in (
        args_global.dense_run / "Intrinsics.txt", args_global.dense_run.parent / "Intrinsics.txt")
                           if path.exists())
    link_or_copy(intrinsics_path, args_global.output_run.parent / "Intrinsics.txt")
    for name in ("groundtruth_labels.csv", "groundtruth_labels_classes.csv"):
        source = args_global.sparse_semantic_run.parent / name
        if source.exists():
            link_or_copy(source, args_global.output_run.parent / name)

    manifest = {
        "method": "depth_pose_reprojection_from_validated_sparse_semantics",
        "dense_run": str(args_global.dense_run.resolve()),
        "sparse_semantic_run": str(args_global.sparse_semantic_run.resolve()),
        "dense_frames_available": len(dense_ids),
        "dense_frames_processed": process_count,
        "sparse_frames": len(sparse_ids),
        "sparse_dense_anchor_indices": anchors,
        "depth_tolerance_m": args_global.depth_tolerance_m,
        "max_fill_px": args_global.max_fill_px,
        "frames": frame_rows,
    }
    (args_global.output_run / "semantic_propagation_manifest.json").write_text(
        json.dumps(manifest, indent=2))


if __name__ == "__main__":
    args_global: argparse.Namespace
    main()
