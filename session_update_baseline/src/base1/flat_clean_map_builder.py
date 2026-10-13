#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def parse_intrinsics(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        match = re.search(r"[-+]?[0-9]*\.?[0-9]+", value)
        if match:
            values[key.strip()] = float(match.group(0))
    return {
        "width": values.get("Res_x", 640.0),
        "height": values.get("Res_y", 480.0),
        "cx": values.get("u", 320.0),
        "cy": values.get("v", 240.0),
        "fx": values.get("f_x", 320.0),
        "fy": values.get("f_y", 320.0),
    }


def load_timestamps(run_dir: Path) -> list[str]:
    rows: list[tuple[int, str]] = []
    with (run_dir / "timestamps.csv").open(newline="") as fin:
        for row in csv.DictReader(fin):
            rows.append((int(row["TimeStamp"]), row["ImageID"]))
    rows.sort()
    return [image_id for _, image_id in rows]


def load_label_table(flat_root: Path) -> tuple[dict[int, dict[str, object]], dict[int, str]]:
    labels: dict[int, dict[str, object]] = {}
    with (flat_root / "groundtruth_labels.csv").open(newline="") as fin:
        for row in csv.DictReader(fin):
            instance_id = int(row["InstanceID"])
            labels[instance_id] = {
                "label_id": instance_id,
                "class_id": int(row["ClassID"]),
                "panoptic_id": int(row["PanopticID"]),
                "mesh_id": int(row["MeshID"]),
                "name": row["Name"],
                "size": row.get("Size", ""),
            }

    classes: dict[int, str] = {}
    with (flat_root / "groundtruth_labels_classes.csv").open(newline="") as fin:
        for row in csv.DictReader(fin):
            classes[int(row["ClassID"])] = row["Name"]
    return labels, classes


def read_pose(path: Path) -> np.ndarray:
    return np.loadtxt(str(path), dtype=np.float32).reshape(4, 4)


def voxel_downsample(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    if len(points) == 0 or voxel_size_m <= 0.0:
        return points.astype(np.float32, copy=False)
    cells = np.floor(points[:, :3] / voxel_size_m).astype(np.int64)
    _, indices = np.unique(cells, axis=0, return_index=True)
    indices.sort()
    return points[indices].astype(np.float32, copy=False)


def voxel_downsample_min_support(
    points: np.ndarray,
    voxel_size_m: float,
    min_support: int,
) -> np.ndarray:
    if len(points) == 0 or voxel_size_m <= 0.0 or min_support <= 1:
        return voxel_downsample(points, voxel_size_m)
    cells = np.floor(points[:, :3] / voxel_size_m).astype(np.int64)
    _, indices, counts = np.unique(cells, axis=0, return_index=True, return_counts=True)
    supported = indices[counts >= min_support]
    supported.sort()
    return points[supported].astype(np.float32, copy=False)


def free_cell_key(point: np.ndarray, cell_size_m: float) -> tuple[int, int, int]:
    cell = np.floor(point[:3] / cell_size_m).astype(np.int64)
    return int(cell[0]), int(cell[1]), int(cell[2])


def accumulate_free_space_counts(
    free_counts: Counter[tuple[int, int, int]],
    origins: np.ndarray,
    surfaces: np.ndarray,
    cell_size_m: float,
    surface_clearance_m: float,
    ray_stride: int,
) -> None:
    if len(surfaces) == 0:
        return
    step_m = max(0.5 * cell_size_m, 0.02)
    stride = max(1, ray_stride)
    for origin, surface in zip(origins[::stride], surfaces[::stride]):
        ray = surface[:3] - origin[:3]
        depth = float(np.linalg.norm(ray))
        if not math.isfinite(depth) or depth <= surface_clearance_m + step_m:
            continue
        direction = ray / depth
        max_t = depth - surface_clearance_m
        t = step_m
        seen_on_ray: set[tuple[int, int, int]] = set()
        while t < max_t:
            key = free_cell_key(origin + direction * t, cell_size_m)
            if key not in seen_on_ray:
                free_counts[key] += 1
                seen_on_ray.add(key)
            t += step_m


def filter_points_by_free_space_consistency(
    points: np.ndarray,
    free_counts: Counter[tuple[int, int, int]],
    cell_size_m: float,
    min_free_observations: int,
) -> np.ndarray:
    if len(points) == 0 or min_free_observations <= 0 or not free_counts:
        return points
    keep = np.ones(len(points), dtype=bool)
    for idx, point in enumerate(points[:, :3]):
        if free_counts.get(free_cell_key(point, cell_size_m), 0) >= min_free_observations:
            keep[idx] = False
    return points[keep]


def point_stats(points: np.ndarray) -> dict[str, object]:
    if len(points) == 0:
        nan = float("nan")
        return {
            "points": 0,
            "bbox_min": [nan, nan, nan],
            "bbox_max": [nan, nan, nan],
            "centroid": [nan, nan, nan],
            "extent": [nan, nan, nan],
        }
    xyz = points[:, :3]
    mn = xyz.min(axis=0)
    mx = xyz.max(axis=0)
    centroid = xyz.mean(axis=0)
    return {
        "points": int(len(points)),
        "bbox_min": [float(x) for x in mn],
        "bbox_max": [float(x) for x in mx],
        "centroid": [float(x) for x in centroid],
        "extent": [float(x) for x in (mx - mn)],
    }


def snap_background_plane(points: np.ndarray, name: str, class_name: str) -> np.ndarray:
    if len(points) == 0:
        return points
    snapped = points.copy()
    xyz = snapped[:, :3]
    text = f"{name} {class_name}".lower()
    if "floor" in text or "ceiling" in text:
        xyz[:, 2] = float(np.median(xyz[:, 2]))
        return snapped

    if "wall" in text and "tv_wall" not in text:
        x_lo, x_hi = np.quantile(xyz[:, 0], [0.02, 0.98])
        y_lo, y_hi = np.quantile(xyz[:, 1], [0.02, 0.98])
        distances = np.column_stack(
            [
                np.abs(xyz[:, 0] - x_lo),
                np.abs(xyz[:, 0] - x_hi),
                np.abs(xyz[:, 1] - y_lo),
                np.abs(xyz[:, 1] - y_hi),
            ]
        )
        nearest = np.argmin(distances, axis=1)
        xyz[nearest == 0, 0] = float(x_lo)
        xyz[nearest == 1, 0] = float(x_hi)
        xyz[nearest == 2, 1] = float(y_lo)
        xyz[nearest == 3, 1] = float(y_hi)
        return snapped

    variances = np.var(xyz, axis=0)
    axis = int(np.argmin(variances))
    extents = xyz.max(axis=0) - xyz.min(axis=0)
    if extents[axis] <= 0.35:
        xyz[:, axis] = float(np.median(xyz[:, axis]))
    return snapped


def write_binary_ply(
    path: Path,
    points: np.ndarray,
    has_ray_origins: bool = False,
    has_colors: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(points)
    expected_width = 3 + (3 if has_ray_origins else 0) + (3 if has_colors else 0)
    if pts.ndim != 2 or pts.shape[1] != expected_width:
        raise ValueError(f"expected Nx{expected_width} points for {path}, got {pts.shape}")
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
    ).encode("ascii")
    if has_ray_origins:
        header += (
            "property float origin_x\n"
            "property float origin_y\n"
            "property float origin_z\n"
        ).encode("ascii")
    if has_colors:
        header += (
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
        ).encode("ascii")
    header += "end_header\n".encode("ascii")
    with path.open("wb") as fout:
        fout.write(header)
        if not has_colors:
            np.asarray(pts, dtype="<f4").tofile(fout)
            return

        fields: list[tuple[str, str]] = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
        if has_ray_origins:
            fields += [("origin_x", "<f4"), ("origin_y", "<f4"), ("origin_z", "<f4")]
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        structured = np.empty(len(pts), dtype=np.dtype(fields))
        structured["x"] = pts[:, 0].astype("<f4")
        structured["y"] = pts[:, 1].astype("<f4")
        structured["z"] = pts[:, 2].astype("<f4")
        offset = 3
        if has_ray_origins:
            structured["origin_x"] = pts[:, offset + 0].astype("<f4")
            structured["origin_y"] = pts[:, offset + 1].astype("<f4")
            structured["origin_z"] = pts[:, offset + 2].astype("<f4")
            offset += 3
        colors = np.clip(np.rint(pts[:, offset : offset + 3]), 0, 255).astype(np.uint8)
        structured["red"] = colors[:, 0]
        structured["green"] = colors[:, 1]
        structured["blue"] = colors[:, 2]
        structured.tofile(fout)


def project_flat_run(
    flat_root: Path,
    run_name: str,
    segmentation_source: str,
    frame_stride: int,
    frame_limit: int,
    pixel_stride: int,
    max_depth_m: float,
    object_voxel_size_m: float,
    write_ray_origins: bool,
    write_colors: bool,
    background_label_ids: set[int],
    background_min_observations: int,
    instance_min_observations: int,
    free_space_filter_background: bool,
    free_space_cell_size_m: float,
    free_space_surface_clearance_m: float,
    free_space_min_observations: int,
    free_space_ray_stride: int,
) -> tuple[dict[int, np.ndarray], dict[str, object]]:
    run_dir = flat_root / run_name
    intr = parse_intrinsics(flat_root / "Intrinsics.txt")
    image_ids = load_timestamps(run_dir)
    selected_ids = image_ids[::frame_stride]
    if frame_limit > 0:
        selected_ids = selected_ids[:frame_limit]

    width = int(intr["width"])
    height = int(intr["height"])
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    us = np.arange(0, width, pixel_stride, dtype=np.float32)
    vs = np.arange(0, height, pixel_stride, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)
    u_flat = uu.reshape(-1)
    v_flat = vv.reshape(-1)
    u_idx = u_flat.astype(np.int32)
    v_idx = v_flat.astype(np.int32)

    by_label: dict[int, list[np.ndarray]] = {}
    missing = Counter()
    frames_used = 0
    sampled_pixels = 0
    valid_projected_points = 0
    free_counts: Counter[tuple[int, int, int]] = Counter()

    for image_id in selected_ids:
        depth_path = run_dir / f"{image_id}_depth.tiff"
        pose_path = run_dir / f"{image_id}_pose.txt"
        if segmentation_source == "groundtruth":
            seg_path = run_dir / f"{image_id}_segmentation.png"
        else:
            seg_path = run_dir / f"{image_id}_predicted.png"
        color_path = run_dir / f"{image_id}_color.png"
        if not depth_path.exists():
            missing["depth"] += 1
            continue
        if not seg_path.exists():
            missing["segmentation"] += 1
            continue
        if not pose_path.exists():
            missing["pose"] += 1
            continue
        if write_colors and not color_path.exists():
            missing["color"] += 1
            continue

        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        segmentation = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)
        color = cv2.imread(str(color_path), cv2.IMREAD_COLOR) if write_colors else None
        if depth is None:
            missing["depth_read"] += 1
            continue
        if segmentation is None:
            missing["segmentation_read"] += 1
            continue
        if write_colors and color is None:
            missing["color_read"] += 1
            continue
        if segmentation.ndim == 3:
            segmentation = segmentation[:, :, 0]
        pose = read_pose(pose_path)

        z = depth[v_idx, u_idx].astype(np.float32)
        labels = segmentation[v_idx, u_idx].astype(np.int32)
        valid = np.isfinite(z) & (z > 0.05) & (z < max_depth_m)
        if not np.any(valid):
            continue

        sampled_pixels += int(len(z))
        u = u_flat[valid]
        v = v_flat[valid]
        z = z[valid]
        labels = labels[valid]

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        camera = np.stack([x, y, z, np.ones_like(z)], axis=0)
        world = (pose @ camera)[:3, :].T.astype(np.float32)
        origin = pose[:3, 3].astype(np.float32)
        if free_space_filter_background:
            origins_for_free = np.repeat(origin.reshape(1, 3), world.shape[0], axis=0)
            accumulate_free_space_counts(
                free_counts,
                origins_for_free,
                world,
                free_space_cell_size_m,
                free_space_surface_clearance_m,
                free_space_ray_stride,
            )
        if write_ray_origins:
            origins = np.repeat(origin.reshape(1, 3), world.shape[0], axis=0)
            samples = np.concatenate([world, origins], axis=1).astype(np.float32)
        else:
            samples = world
        if write_colors:
            assert color is not None
            rgb = color[v_idx, u_idx][valid][:, ::-1].astype(np.float32)
            samples = np.concatenate([samples, rgb], axis=1).astype(np.float32)

        for label_id in np.unique(labels):
            mask = labels == label_id
            if np.any(mask):
                by_label.setdefault(int(label_id), []).append(samples[mask])
                valid_projected_points += int(mask.sum())
        frames_used += 1

    objects: dict[int, np.ndarray] = {}
    for label_id, chunks in by_label.items():
        points = np.concatenate(chunks, axis=0)
        if free_space_filter_background and label_id in background_label_ids:
            points = filter_points_by_free_space_consistency(
                points,
                free_counts,
                free_space_cell_size_m,
                free_space_min_observations,
            )
        min_support = (
            background_min_observations
            if label_id in background_label_ids
            else instance_min_observations
        )
        objects[label_id] = voxel_downsample_min_support(
            points,
            object_voxel_size_m,
            min_support,
        )

    metadata = {
        "flat_root": str(flat_root),
        "run": run_name,
        "segmentation_source": segmentation_source,
        "frames_total": len(image_ids),
        "frames_selected": len(selected_ids),
        "frames_used": frames_used,
        "frame_stride": frame_stride,
        "frame_limit": frame_limit,
        "pixel_stride": pixel_stride,
        "sampled_pixels": sampled_pixels,
        "valid_projected_points": valid_projected_points,
        "objects_projected": len(objects),
        "missing_inputs": dict(missing),
        "max_depth_m": max_depth_m,
        "object_voxel_size_m": object_voxel_size_m,
        "write_ray_origins": write_ray_origins,
        "write_colors": write_colors,
        "background_min_observations": background_min_observations,
        "instance_min_observations": instance_min_observations,
        "free_space_filter_background": free_space_filter_background,
        "free_space_cell_size_m": free_space_cell_size_m,
        "free_space_surface_clearance_m": free_space_surface_clearance_m,
        "free_space_min_observations": free_space_min_observations,
        "free_space_ray_stride": free_space_ray_stride,
        "free_space_cells": len(free_counts),
        "intrinsics": intr,
    }
    return objects, metadata


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Base1 clean object map directly from one Panoptic flat run."
    )
    parser.add_argument("--flat-root", type=Path, required=True)
    parser.add_argument("--run", choices=["run1", "run2"], required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--segmentation-source",
        choices=["groundtruth", "detectron"],
        default="groundtruth",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--frame-limit", type=int, default=0)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--max-depth-m", type=float, default=10.0)
    parser.add_argument("--object-voxel-size-m", type=float, default=0.03)
    parser.add_argument("--map-voxel-size-m", type=float, default=0.03)
    parser.add_argument("--background-min-observations", type=int, default=1)
    parser.add_argument("--instance-min-observations", type=int, default=1)
    parser.add_argument("--snap-background-planes", action="store_true")
    parser.add_argument("--free-space-filter-background", action="store_true")
    parser.add_argument("--free-space-cell-size-m", type=float, default=0.05)
    parser.add_argument("--free-space-surface-clearance-m", type=float, default=0.05)
    parser.add_argument("--free-space-min-observations", type=int, default=2)
    parser.add_argument("--free-space-ray-stride", type=int, default=4)
    parser.add_argument(
        "--write-ray-origins",
        action="store_true",
        help="Store per-point camera ray origins for Panoptic .panmap official evaluation backends.",
    )
    parser.add_argument(
        "--write-colors",
        action="store_true",
        help="Store RGB sampled from flat *_color.png files in each object PLY.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    objects_dir = args.out_dir / "objects"
    objects_dir.mkdir(parents=True, exist_ok=True)
    labels, classes = load_label_table(args.flat_root)
    background_label_ids = {
        label_id
        for label_id, meta in labels.items()
        if int(meta.get("panoptic_id", -1)) == 0
    }
    objects, projection = project_flat_run(
        args.flat_root,
        args.run,
        args.segmentation_source,
        args.frame_stride,
        args.frame_limit,
        args.pixel_stride,
        args.max_depth_m,
        args.object_voxel_size_m,
        args.write_ray_origins,
        args.write_colors,
        background_label_ids,
        args.background_min_observations,
        args.instance_min_observations,
        args.free_space_filter_background,
        args.free_space_cell_size_m,
        args.free_space_surface_clearance_m,
        args.free_space_min_observations,
        args.free_space_ray_stride,
    )

    object_rows: list[dict[str, object]] = []
    memory_objects: list[dict[str, object]] = []
    selected_points: list[np.ndarray] = []
    for label_id in sorted(objects):
        points = objects[label_id]
        label_meta = labels.get(
            label_id,
            {
                "label_id": label_id,
                "class_id": -1,
                "panoptic_id": -1,
                "mesh_id": -1,
                "name": f"ID_{label_id}",
                "size": "",
            },
        )
        class_id = int(label_meta["class_id"])
        object_name = str(label_meta["name"])
        class_name = classes.get(class_id, "unknown")
        if args.snap_background_planes and label_id in background_label_ids:
            points = snap_background_plane(points, object_name, class_name)
            objects[label_id] = points
        stats = point_stats(points)
        object_file = objects_dir / f"{label_id:03d}_{safe_name(object_name)}.ply"
        write_binary_ply(object_file, points, args.write_ray_origins, args.write_colors)
        if len(points) > 0:
            selected_points.append(points)

        row = {
            "label_id": label_id,
            "name": object_name,
            "class_id": class_id,
            "class_name": class_name,
            "panoptic_id": int(label_meta.get("panoptic_id", -1)),
            "mesh_id": int(label_meta.get("mesh_id", -1)),
            "size": label_meta.get("size", ""),
            "points": stats["points"],
            "bbox_min_x": stats["bbox_min"][0],
            "bbox_min_y": stats["bbox_min"][1],
            "bbox_min_z": stats["bbox_min"][2],
            "bbox_max_x": stats["bbox_max"][0],
            "bbox_max_y": stats["bbox_max"][1],
            "bbox_max_z": stats["bbox_max"][2],
            "centroid_x": stats["centroid"][0],
            "centroid_y": stats["centroid"][1],
            "centroid_z": stats["centroid"][2],
            "extent_x": stats["extent"][0],
            "extent_y": stats["extent"][1],
            "extent_z": stats["extent"][2],
            "points_file": str(object_file.relative_to(args.out_dir)),
        }
        object_rows.append(row)
        memory_obj = {
            "label_id": label_id,
            "name": object_name,
            "class_id": class_id,
            "class_name": class_name,
            "points": int(stats["points"]),
            "bbox_min": stats["bbox_min"],
            "bbox_max": stats["bbox_max"],
            "centroid": stats["centroid"],
            "extent": stats["extent"],
            "points_file": str(object_file.relative_to(args.out_dir)),
            "stationarity_alpha": 2.0,
            "stationarity_beta": 1.0,
            "session_state": "present",
        }
        memory_objects.append(memory_obj)

    map_points = (
        np.concatenate(selected_points, axis=0)
        if selected_points
        else np.empty((0, 3), dtype=np.float32)
    )
    map_points = voxel_downsample(map_points, args.map_voxel_size_m)
    write_binary_ply(args.out_dir / "map.ply", map_points, args.write_ray_origins, args.write_colors)

    decision_counts = {"present": len(memory_objects)}
    output_map = {
        "format": "base1_flat_clean_map_v1",
        "map_type": "object_point_map",
        "flat_dataset_root": str(args.flat_root),
        "run": args.run,
        "map_ply": "map.ply",
        "objects_dir": "objects",
        "config": {
            "segmentation_source": args.segmentation_source,
            "frame_stride": args.frame_stride,
            "frame_limit": args.frame_limit,
            "pixel_stride": args.pixel_stride,
            "max_depth_m": args.max_depth_m,
            "object_voxel_size_m": args.object_voxel_size_m,
            "map_voxel_size_m": args.map_voxel_size_m,
            "write_ray_origins": args.write_ray_origins,
            "write_colors": args.write_colors,
            "background_min_observations": args.background_min_observations,
            "instance_min_observations": args.instance_min_observations,
            "snap_background_planes": args.snap_background_planes,
            "free_space_filter_background": args.free_space_filter_background,
            "free_space_cell_size_m": args.free_space_cell_size_m,
            "free_space_surface_clearance_m": args.free_space_surface_clearance_m,
            "free_space_min_observations": args.free_space_min_observations,
            "free_space_ray_stride": args.free_space_ray_stride,
        },
        "projection": projection,
        "summary": {
            "objects_total": len(memory_objects),
            "map_points": int(len(map_points)),
            "decision_counts": decision_counts,
        },
        "objects": memory_objects,
    }

    (args.out_dir / "map.base1map.json").write_text(
        json.dumps(output_map, indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "object_memory.json").write_text(
        json.dumps(
            {
                "format": "base1_flat_object_memory_v1",
                "source_map": "map.base1map.json",
                "flat_dataset_root": str(args.flat_root),
                "run": args.run,
                "objects": memory_objects,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_csv(
        args.out_dir / "object_summary.csv",
        object_rows,
        [
            "label_id",
            "name",
            "class_id",
            "class_name",
            "panoptic_id",
            "mesh_id",
            "size",
            "points",
            "bbox_min_x",
            "bbox_min_y",
            "bbox_min_z",
            "bbox_max_x",
            "bbox_max_y",
            "bbox_max_z",
            "centroid_x",
            "centroid_y",
            "centroid_z",
            "extent_x",
            "extent_y",
            "extent_z",
            "points_file",
        ],
    )
    evidence = {
        "method": "base1_flat_clean_map_builder",
        "inputs": {
            "flat_root": str(args.flat_root),
            "run": args.run,
            "segmentation_source": args.segmentation_source,
        },
        "outputs": {
            "map": str(args.out_dir / "map.base1map.json"),
            "map_ply": str(args.out_dir / "map.ply"),
            "object_memory": str(args.out_dir / "object_memory.json"),
            "object_summary": str(args.out_dir / "object_summary.csv"),
        },
        "summary": output_map["summary"],
        "projection": projection,
    }
    (args.out_dir / "evidence_summary.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )

    print(f"WROTE {args.out_dir / 'map.base1map.json'}")
    print(f"WROTE {args.out_dir / 'map.ply'} points={len(map_points)}")
    print(f"WROTE {args.out_dir / 'object_memory.json'} objects={len(memory_objects)}")


if __name__ == "__main__":
    main()
