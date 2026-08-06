#!/usr/bin/env python3
"""Render real NSS OBJ textures at the poses of an existing flat RGB-D run."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument(
        "--mesh-transform",
        type=Path,
        help="Optional 4x4 transform applied to the textured mesh before rendering.",
    )
    parser.add_argument("--input-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--frames", nargs="+", type=int)
    parser.add_argument("--overwrite", action="store_true")
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


def read_intrinsics(run_dir: Path) -> tuple[int, int, float, float, float, float]:
    candidates = (run_dir / "Intrinsics.txt", run_dir.parent / "Intrinsics.txt")
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError("Intrinsics.txt not found")
    values: dict[str, float] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = float(value.strip())
    width = int(values.get("width", values.get("w", 640)))
    height = int(values.get("height", values.get("h", 480)))
    cx = values.get("c_x", values.get("u"))
    cy = values.get("c_y", values.get("v"))
    if cx is None or cy is None:
        raise KeyError("Intrinsics must contain c_x/c_y or u/v")
    return width, height, values["f_x"], values["f_y"], cx, cy


def load_timestamp_rows(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "timestamps.csv").open() as stream:
        return list(csv.DictReader(stream))


def add_textured_model(
    renderer: rendering.OffscreenRenderer,
    path: Path,
    mesh_transform: np.ndarray | None,
) -> dict:
    model = o3d.io.read_triangle_model(str(path))
    if not model.meshes:
        raise RuntimeError(f"Could not load textured mesh: {path}")
    for index, part in enumerate(model.meshes):
        if mesh_transform is not None:
            part.mesh.transform(mesh_transform)
        material = (
            model.materials[part.material_idx]
            if 0 <= part.material_idx < len(model.materials)
            else rendering.MaterialRecord()
        )
        material.shader = "defaultUnlit"
        renderer.scene.add_geometry(f"part_{index:04d}", part.mesh, material)
    return {"mesh_parts": len(model.meshes), "materials": len(model.materials)}


def write_contact_sheet(paths: list[Path], output: Path, max_images: int = 12) -> None:
    if len(paths) > max_images:
        indices = np.linspace(0, len(paths) - 1, max_images, dtype=int)
        paths = [paths[index] for index in indices]
    images = [Image.open(path).convert("RGB") for path in paths]
    if not images:
        return
    columns = 3
    width, height = images[0].size
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * height), "white")
    for index, image in enumerate(images):
        sheet.paste(image, ((index % columns) * width, (index // columns) * height))
    sheet.save(output)


def main() -> None:
    args = parse_args()
    args.output_run.mkdir(parents=True, exist_ok=True)
    width, height, fx, fy, cx, cy = read_intrinsics(args.input_run)
    timestamp_rows = load_timestamp_rows(args.input_run)
    all_ids = [row["ImageID"] for row in timestamp_rows]
    selected = args.frames if args.frames else list(range(len(all_ids)))
    invalid = [index for index in selected if index < 0 or index >= len(all_ids)]
    if invalid:
        raise IndexError(f"Frame indices out of range: {invalid}")

    renderer = rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    renderer.scene.set_lighting(rendering.Open3DScene.LightingProfile.NO_SHADOWS,
                                [0.0, -1.0, -1.0])
    mesh_transform = None
    if args.mesh_transform is not None:
        mesh_transform = np.loadtxt(args.mesh_transform, dtype=np.float64).reshape(4, 4)
        if not np.all(np.isfinite(mesh_transform)) or not np.allclose(
            mesh_transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8
        ):
            raise SystemExit(f"Invalid mesh transform: {args.mesh_transform}")
    model_stats = add_textured_model(renderer, args.mesh, mesh_transform)
    intrinsic = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    rows = []
    color_paths = []
    for order, frame_index in enumerate(selected):
        image_id = all_ids[frame_index]
        source_depth = args.input_run / f"{image_id}_depth.tiff"
        source_pose = args.input_run / f"{image_id}_pose.txt"
        output_color = args.output_run / f"{image_id}_color.png"
        link_or_copy(source_depth, args.output_run / source_depth.name)
        link_or_copy(source_pose, args.output_run / source_pose.name)

        if output_color.exists() and not args.overwrite:
            color_paths.append(output_color)
            rows.append(
                {
                    "frame_index": frame_index,
                    "image_id": image_id,
                    "depth_overlap": None,
                    "depth_delta_median_m": None,
                    "depth_delta_p90_m": None,
                    "reused_render": True,
                }
            )
            print(f"TEXTURED_REUSED {order + 1}/{len(selected)} id={image_id}")
            continue

        pose = np.loadtxt(source_pose).reshape(4, 4)
        renderer.setup_camera(intrinsic, np.linalg.inv(pose), width, height)
        color = np.asarray(renderer.render_to_image())[:, :, :3]
        rendered_depth = np.asarray(
            renderer.render_to_depth_image(z_in_view_space=True), dtype=np.float32
        )
        Image.fromarray(color).save(output_color)
        color_paths.append(output_color)

        reference_depth = np.asarray(Image.open(source_depth), dtype=np.float32)
        valid = (
            np.isfinite(rendered_depth)
            & (rendered_depth > 0.0)
            & np.isfinite(reference_depth)
            & (reference_depth > 0.0)
        )
        delta = np.abs(rendered_depth[valid] - reference_depth[valid])
        rows.append(
            {
                "frame_index": frame_index,
                "image_id": image_id,
                "depth_overlap": float(np.mean(valid)),
                "depth_delta_median_m": float(np.median(delta)) if delta.size else None,
                "depth_delta_p90_m": float(np.percentile(delta, 90)) if delta.size else None,
            }
        )
        print(f"TEXTURED_FRAME {order + 1}/{len(selected)} id={image_id}")

    output_timestamps = args.output_run / "timestamps.csv"
    if output_timestamps.exists() or output_timestamps.is_symlink():
        output_timestamps.unlink()
    with output_timestamps.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=timestamp_rows[0].keys())
        writer.writeheader()
        writer.writerows(timestamp_rows[index] for index in selected)
    source_intrinsics = next(
        path
        for path in (args.input_run / "Intrinsics.txt", args.input_run.parent / "Intrinsics.txt")
        if path.exists()
    )
    link_or_copy(source_intrinsics, args.output_run.parent / "Intrinsics.txt")
    for name in ("groundtruth_labels.csv", "groundtruth_labels_classes.csv"):
        source = args.input_run.parent / name
        if source.exists():
            link_or_copy(source, args.output_run.parent / name)
    write_contact_sheet(color_paths, args.output_run / "textured_rgb_contact_sheet.png")
    manifest = {
        "mesh": str(args.mesh.resolve()),
        "mesh_transform_file": str(args.mesh_transform.resolve()) if args.mesh_transform else None,
        "mesh_transform": mesh_transform.tolist() if mesh_transform is not None else None,
        "input_run": str(args.input_run.resolve()),
        "output_run": str(args.output_run.resolve()),
        "model": model_stats,
        "intrinsics": {"width": width, "height": height, "fx": fx, "fy": fy,
                       "cx": cx, "cy": cy},
        "frames": rows,
    }
    (args.output_run / "textured_rgb_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"TEXTURED_COMPLETE frames={len(rows)} output={args.output_run}")


if __name__ == "__main__":
    main()
