#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a textured NSS OBJ along an OVI-MAP/Replica-style trajectory."
    )
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--cam-params", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means render all remaining frames.")
    parser.add_argument(
        "--frames",
        nargs="+",
        type=int,
        help="Explicit frame IDs to render; overrides --start and --limit.",
    )
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--background", choices=("white", "black"), default="white")
    parser.add_argument("--shader", choices=("lit", "unlit"), default="lit")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_camera(path: Path) -> dict:
    data = json.loads(path.read_text())["camera"]
    required = ("w", "h", "fx", "fy", "cx", "cy", "scale")
    missing = [key for key in required if key not in data]
    if missing:
        raise SystemExit(f"{path}: missing camera keys {missing}")
    return data


def load_poses(scene_dir: Path) -> np.ndarray:
    traj = np.loadtxt(scene_dir / "traj.txt")
    poses = traj.reshape(-1, 4, 4)
    if len(poses) == 0:
        raise SystemExit(f"{scene_dir / 'traj.txt'}: empty trajectory")
    return poses


def add_model(renderer: rendering.OffscreenRenderer, mesh_path: Path, shader: str) -> dict:
    model = o3d.io.read_triangle_model(str(mesh_path))
    if not model.meshes:
        raise SystemExit(f"Failed to load triangle model: {mesh_path}")

    material_count = len(model.materials)
    for i, part in enumerate(model.meshes):
        if 0 <= part.material_idx < material_count:
            material = model.materials[part.material_idx]
        else:
            material = rendering.MaterialRecord()
        material.shader = "defaultUnlit" if shader == "unlit" else (material.shader or "defaultLit")
        renderer.scene.add_geometry(f"mesh_{i:04d}", part.mesh, material)
    return {"mesh_parts": len(model.meshes), "materials": material_count}


def render_frame(
    renderer: rendering.OffscreenRenderer,
    pose: np.ndarray,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float,
    rgb_path: Path,
    depth_path: Path,
    quality: int,
) -> dict:
    intrinsic = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    extrinsic = np.linalg.inv(pose)
    renderer.setup_camera(intrinsic, extrinsic, width, height)

    rgb = np.asarray(renderer.render_to_image())
    depth = np.asarray(renderer.render_to_depth_image(z_in_view_space=True)).astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    depth_mm = np.where(valid, depth * depth_scale, 0.0)
    depth_mm = np.clip(depth_mm, 0.0, np.iinfo(np.uint16).max).astype(np.uint16)

    Image.fromarray(rgb).save(rgb_path, quality=quality)
    Image.fromarray(depth_mm).save(depth_path)

    values = depth[valid]
    return {
        "hit_pixels": int(np.count_nonzero(valid)),
        "hit_ratio": float(np.mean(valid)),
        "depth_min": float(values.min()) if values.size else None,
        "depth_median": float(np.median(values)) if values.size else None,
        "depth_max": float(values.max()) if values.size else None,
    }


def write_contact_sheet(paths: list[Path], out_path: Path, columns: int = 4, thumb_w: int = 320) -> None:
    if not paths:
        return
    images = [Image.open(path).convert("RGB") for path in paths]
    thumbs = []
    for image in images:
        scale = thumb_w / image.width
        thumb_h = int(round(image.height * scale))
        thumbs.append(image.resize((thumb_w, thumb_h)))
    rows = int(np.ceil(len(thumbs) / columns))
    sheet = Image.new("RGB", (columns * thumb_w, rows * thumbs[0].height), "white")
    for i, thumb in enumerate(thumbs):
        x = (i % columns) * thumb_w
        y = (i // columns) * thumb.height
        sheet.paste(thumb, (x, y))
    sheet.save(out_path)


def main() -> None:
    args = parse_args()
    camera = load_camera(args.cam_params)
    poses = load_poses(args.scene_dir)

    width = int(camera["w"])
    height = int(camera["h"])
    fx = float(camera["fx"])
    fy = float(camera["fy"])
    cx = float(camera["cx"])
    cy = float(camera["cy"])
    depth_scale = float(camera["scale"])

    out_dir = args.out_dir or (args.scene_dir / "results")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.frames:
        frame_ids = list(dict.fromkeys(args.frames))
        invalid = [frame_id for frame_id in frame_ids if frame_id < 0 or frame_id >= len(poses)]
        if invalid:
            raise SystemExit(f"Frame IDs outside [0, {len(poses) - 1}]: {invalid}")
        frame_start = min(frame_ids)
        frame_end = max(frame_ids) + 1
    else:
        frame_end = len(poses) if args.limit <= 0 else min(len(poses), args.start + args.limit)
        frame_ids = list(range(args.start, frame_end))
        frame_start = args.start
    if not frame_ids:
        raise SystemExit("No frames selected.")

    renderer = rendering.OffscreenRenderer(width, height)
    bg = [1.0, 1.0, 1.0, 1.0] if args.background == "white" else [0.0, 0.0, 0.0, 1.0]
    renderer.scene.set_background(bg)
    renderer.scene.set_lighting(rendering.Open3DScene.LightingProfile.NO_SHADOWS, [0.0, -1.0, -1.0])
    model_stats = add_model(renderer, args.mesh, args.shader)

    frames = []
    preview_paths = []
    preview_stride = max(1, len(frame_ids) // 12)
    for local_i, frame_id in enumerate(frame_ids):
        rgb_path = out_dir / f"frame{frame_id:06d}.jpg"
        depth_path = out_dir / f"depth{frame_id:06d}.png"
        if not args.overwrite and rgb_path.exists() and depth_path.exists():
            print(f"SKIP {frame_id:06d} existing")
            continue

        stats = render_frame(
            renderer,
            poses[frame_id],
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            depth_scale,
            rgb_path,
            depth_path,
            args.quality,
        )
        frames.append({"frame_id": frame_id, **stats})
        if local_i % preview_stride == 0 or local_i == len(frame_ids) - 1:
            preview_paths.append(rgb_path)
        print(
            f"FRAME {frame_id:06d} hit={stats['hit_ratio']:.3f} "
            f"depth_med={stats['depth_median']}"
        )

    if preview_paths:
        write_contact_sheet(preview_paths, args.scene_dir / "render_preview_contact_sheet.jpg")

    hit_ratios = np.asarray([row["hit_ratio"] for row in frames], dtype=np.float64)
    med_depths = np.asarray(
        [row["depth_median"] if row["depth_median"] is not None else np.nan for row in frames],
        dtype=np.float64,
    )
    manifest = {
        "mesh": str(args.mesh.resolve()),
        "scene_dir": str(args.scene_dir.resolve()),
        "results_dir": str(out_dir.resolve()),
        "cam_params": str(args.cam_params.resolve()),
        "trajectory": str((args.scene_dir / "traj.txt").resolve()),
        "frame_start": frame_start,
        "frame_end_exclusive": frame_end,
        "frame_ids": frame_ids,
        "rendered_frames": len(frames),
        "camera": camera,
        "shader": args.shader,
        "background": args.background,
        "model": model_stats,
        "preview_contact_sheet": str((args.scene_dir / "render_preview_contact_sheet.jpg").resolve()),
        "hit_ratio_min": float(hit_ratios.min()) if hit_ratios.size else None,
        "hit_ratio_median": float(np.median(hit_ratios)) if hit_ratios.size else None,
        "hit_ratio_max": float(hit_ratios.max()) if hit_ratios.size else None,
        "depth_median_min": float(np.nanmin(med_depths)) if med_depths.size else None,
        "depth_median_median": float(np.nanmedian(med_depths)) if med_depths.size else None,
        "depth_median_max": float(np.nanmax(med_depths)) if med_depths.size else None,
        "frames": frames,
    }
    manifest_path = args.scene_dir / "render_textured_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "frames"}, indent=2))


if __name__ == "__main__":
    main()
