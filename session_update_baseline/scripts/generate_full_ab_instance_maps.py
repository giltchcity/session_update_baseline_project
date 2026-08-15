#!/usr/bin/env python3
"""Generate full-rate physical-instance maps for the local A/B RGB-D sessions.

The full-rate semantic masks already contain the per-pixel object silhouettes.
This program assigns those silhouettes to the small, reviewed set of physical
room entities.  Static single-object classes always have one ID.  Multi-object
classes are assigned using depth, camera pose, and room-space centers.  Session
A's reviewed instance maps are immutable anchors.  Session B centers are
calibrated in A coordinates and matched one-to-one to A's physical IDs.
"""

from __future__ import annotations

import argparse
import bisect
from collections import defaultdict
import json
import math
from pathlib import Path
import re

import cv2
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment


MULTI_SEMANTIC_IDS = {74, 92, 115}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", choices=("a", "b"), required=True)
    parser.add_argument("--semantic-dir", type=Path, required=True)
    parser.add_argument("--geometry-dir", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--anchor-dir", type=Path)
    parser.add_argument("--world-transform", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-step", type=int, default=30)
    parser.add_argument("--preview-every", type=int, default=150)
    parser.add_argument("--min-component-pixels", type=int, default=100)
    parser.add_argument("--max-center-distance-m", type=float, default=1.35)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    return parser.parse_args()


def require_image(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(path)
    return image


def load_intrinsics(path: Path) -> tuple[int, int, float, float, float, float]:
    text = path.read_text(encoding="utf-8")

    def value(key: str) -> float:
        match = re.search(rf"^{re.escape(key)}:\s*([-+0-9.eE]+)\s*$", text, re.MULTILINE)
        if not match:
            raise ValueError(f"Missing {key} in {path}")
        return float(match.group(1))

    return (
        int(value("Res_x")), int(value("Res_y")), value("f_x"),
        value("f_y"), value("u"), value("v"),
    )


def load_catalog(path: Path) -> tuple[dict[int, int], dict[int, str], dict[int, np.ndarray]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    instance_semantic: dict[int, int] = {}
    names: dict[int, str] = {}
    centers: dict[int, np.ndarray] = {}
    for cls in manifest["canonicalization"]["classes"]:
        semantic_id = int(cls["semantic_id"])
        names[semantic_id] = str(cls["semantic_name"])
        for instance_id, center in zip(cls["canonical_instance_ids"], cls["world_centers_m"]):
            instance_id = int(instance_id)
            instance_semantic[instance_id] = semantic_id
            if center is not None:
                centers[instance_id] = np.asarray(center, dtype=np.float64)
    return instance_semantic, names, centers


def frame_stems(semantic_dir: Path, start: int | None, end: int | None) -> list[str]:
    suffix = "_segmentation.png"
    stems = sorted(p.name[:-len(suffix)] for p in semantic_dir.glob(f"*{suffix}"))
    if start is not None:
        stems = [stem for stem in stems if int(stem) >= start]
    if end is not None:
        stems = [stem for stem in stems if int(stem) <= end]
    if not stems:
        raise ValueError("No semantic frames selected")
    return stems


def components(mask: np.ndarray, min_pixels: int) -> list[np.ndarray]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    result = []
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= min_pixels:
            result.append(labels == index)
    return result


def bbox_gap(first: np.ndarray, second: np.ndarray) -> float:
    ax, ay, aw, ah = (int(value) for value in first[:4])
    bx, by, bw, bh = (int(value) for value in second[:4])
    gap_x = max(ax - (bx + bw), bx - (ax + aw), 0)
    gap_y = max(ay - (by + bh), by - (ay + ah), 0)
    return math.hypot(gap_x, gap_y)


def cleaned_singleton_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Keep one physical object's body while deleting only tiny remote islands."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 2:
        return mask, 0
    main = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    main_area = int(stats[main, cv2.CC_STAT_AREA])
    keep = mask.copy()
    removed = 0
    for index in range(1, count):
        if index == main:
            continue
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < max(500, 0.02 * main_area) and bbox_gap(stats[main], stats[index]) > 50.0:
            keep[labels == index] = False
            removed += 1
    return keep, removed


def load_anchor_visibility(anchor_dir: Path | None) -> tuple[list[int], dict[int, set[int]]]:
    if anchor_dir is None:
        return [], {}
    visibility: dict[int, set[int]] = {}
    for path in sorted(anchor_dir.glob("*_instances.png")):
        frame = int(path.name[:6])
        visibility[frame] = {int(value) for value in np.unique(require_image(path)) if value}
    return sorted(visibility), visibility


def instance_visible_near_frame(
    instance_id: int,
    frame_number: int,
    anchor_frames: list[int],
    anchor_visibility: dict[int, set[int]],
) -> bool:
    if not anchor_frames:
        return True
    index = bisect.bisect_left(anchor_frames, frame_number)
    neighbors = []
    if index > 0:
        neighbors.append(anchor_frames[index - 1])
    if index < len(anchor_frames):
        neighbors.append(anchor_frames[index])
    return any(instance_id in anchor_visibility[frame] for frame in neighbors)


def world_centroid(
    mask: np.ndarray,
    depth: np.ndarray,
    pose: np.ndarray,
    intrinsics: tuple[int, int, float, float, float, float],
    stride: int = 5,
) -> np.ndarray | None:
    width, height, fx, fy, cx, cy = intrinsics
    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    sampled = np.zeros_like(mask)
    sampled[::stride, ::stride] = mask[::stride, ::stride]
    valid = sampled & np.isfinite(depth) & (depth > 0.05)
    v, u = np.nonzero(valid)
    if len(u) < 10:
        return None
    z = depth[v, u].astype(np.float64)
    camera = np.stack(((u - cx) * z / fx, (v - cy) * z / fy, z, np.ones_like(z)), axis=1)
    return np.median((pose @ camera.T).T[:, :3], axis=0)


def projected_center(
    center: np.ndarray,
    pose: np.ndarray,
    intrinsics: tuple[int, int, float, float, float, float],
) -> np.ndarray | None:
    _, _, fx, fy, cx, cy = intrinsics
    camera = np.linalg.inv(pose) @ np.r_[center, 1.0]
    if camera[2] <= 0.05:
        return None
    return np.array((fx * camera[0] / camera[2] + cx, fy * camera[1] / camera[2] + cy))


def robust_kmeans(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) < count:
        raise ValueError(f"Need at least {count} calibration points, got {len(points)}")
    centers, labels = kmeans2(points, count, iter=60, minit="++", seed=104729)
    for _ in range(3):
        refined = []
        for index in range(count):
            group = points[labels == index]
            refined.append(centers[index] if len(group) == 0 else np.median(group, axis=0))
        centers = np.asarray(refined)
        labels = np.argmin(np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2), axis=1)
    return centers


def calibrate_session_b_centers(
    stems: list[str],
    semantic_dir: Path,
    geometry_dir: Path,
    transform: np.ndarray,
    intrinsics: tuple[int, int, float, float, float, float],
    canonical: dict[int, np.ndarray],
    instance_semantic: dict[int, int],
    step: int,
    min_pixels: int,
) -> tuple[dict[int, np.ndarray], dict[str, object]]:
    points: dict[int, list[np.ndarray]] = defaultdict(list)
    sampled = [stem for stem in stems if int(stem) % step == 0]
    if stems[-1] not in sampled:
        sampled.append(stems[-1])
    semantic_ids = sorted(set(instance_semantic.values()))
    for stem in sampled:
        semantic = require_image(semantic_dir / f"{stem}_segmentation.png")
        relevant = any(np.any(semantic == semantic_id) for semantic_id in semantic_ids)
        if not relevant:
            continue
        depth = require_image(geometry_dir / f"{stem}_depth.tiff")
        pose = transform @ np.loadtxt(geometry_dir / f"{stem}_pose.txt", dtype=np.float64)
        for semantic_id in semantic_ids:
            for mask in components(semantic == semantic_id, max(250, min_pixels)):
                center = world_centroid(mask, depth, pose, intrinsics)
                if center is not None:
                    points[semantic_id].append(center)

    result = dict(canonical)
    audit: dict[str, object] = {}
    for semantic_id in semantic_ids:
        instance_ids = sorted(i for i, sid in instance_semantic.items() if sid == semantic_id)
        values = np.asarray(points[semantic_id], dtype=np.float64)
        if len(values) == 0:
            audit[str(semantic_id)] = {
                "sample_count": 0,
                "one_to_one_assignments": [],
                "fallback": "A canonical centers; class was not observed in B calibration frames",
            }
            continue
        # Reject only points that cannot plausibly belong to any room object.
        old = np.asarray([canonical[i] for i in instance_ids])
        plausible = np.min(np.linalg.norm(values[:, None, :] - old[None, :, :], axis=2), axis=1) < 3.0
        values = values[plausible]
        if len(instance_ids) == 1:
            new = np.median(values, axis=0, keepdims=True)
        else:
            new = robust_kmeans(values, len(instance_ids))
        costs = np.linalg.norm(old[:, None, :] - new[None, :, :], axis=2)
        old_rows, new_cols = linear_sum_assignment(costs)
        assignments = []
        for old_index, new_index in zip(old_rows, new_cols):
            instance_id = instance_ids[int(old_index)]
            result[instance_id] = new[int(new_index)]
            assignments.append({
                "instance_id": instance_id,
                "a_center_m": old[int(old_index)].round(6).tolist(),
                "b_center_m": new[int(new_index)].round(6).tolist(),
                "cross_session_shift_m": round(float(costs[old_index, new_index]), 6),
            })
        audit[str(semantic_id)] = {
            "sample_count": int(len(values)),
            "one_to_one_assignments": assignments,
        }
    return result, audit


def choose_instance(
    mask: np.ndarray,
    point: np.ndarray | None,
    pose: np.ndarray,
    candidate_ids: list[int],
    centers: dict[int, np.ndarray],
    intrinsics: tuple[int, int, float, float, float, float],
    maximum_distance: float,
) -> tuple[int, str, float]:
    if point is not None:
        distances = {i: float(np.linalg.norm(point - centers[i])) for i in candidate_ids}
        instance_id = min(distances, key=distances.get)
        distance = distances[instance_id]
        return (instance_id if distance <= maximum_distance else 0, "world", distance)
    ys, xs = np.nonzero(mask)
    image_center = np.array((xs.mean(), ys.mean()))
    distances = {}
    for instance_id in candidate_ids:
        projection = projected_center(centers[instance_id], pose, intrinsics)
        distances[instance_id] = np.inf if projection is None else float(np.linalg.norm(image_center - projection))
    instance_id = min(distances, key=distances.get)
    distance = distances[instance_id]
    return (instance_id if distance <= 350.0 else 0, "projection", distance)


def temporal_instance(
    mask: np.ndarray,
    previous_map: np.ndarray | None,
    candidate_ids: list[int],
    minimum_overlap_pixels: int = 100,
    minimum_overlap_ratio: float = 0.20,
) -> tuple[int, int, float] | None:
    """Keep identity through adjacent frames before falling back to 3D re-ID."""
    if previous_map is None:
        return None
    values, counts = np.unique(previous_map[mask], return_counts=True)
    valid = [
        (int(count), int(value))
        for value, count in zip(values, counts)
        if int(value) in candidate_ids
    ]
    if not valid:
        return None
    overlap, instance_id = max(valid)
    ratio = overlap / max(1, int(mask.sum()))
    if overlap < minimum_overlap_pixels or ratio < minimum_overlap_ratio:
        return None
    return instance_id, overlap, ratio


def color_for_instance(instance_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(instance_id * 104729)
    return tuple(int(value) for value in rng.integers(55, 256, size=3))


def render_preview(rgb: np.ndarray, instance_map: np.ndarray, instance_semantic: dict[int, int], names: dict[int, str]) -> np.ndarray:
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


def refine_from_future_anchors(
    stems: list[str],
    semantic_dir: Path,
    output_maps: Path,
    anchor_frames: list[int],
    by_semantic: dict[int, list[int]],
    min_pixels: int,
) -> dict[str, int]:
    """Run backward overlap propagation inside each reviewed A interval."""
    selected = {int(stem) for stem in stems}
    anchors = [frame for frame in anchor_frames if frame in selected]
    changed_frames = 0
    changed_components = 0
    changed_pixels = 0
    for left, right in zip(anchors, anchors[1:]):
        next_map = require_image(output_maps / f"{right:06d}_instances.png")
        # The left anchor owns the first half of the interval and the right
        # anchor owns the second half.  Letting the right anchor propagate all
        # the way to left+1 can overwrite a perfectly stable forward track.
        midpoint = (left + right) // 2
        for frame_number in range(right - 1, midpoint, -1):
            if frame_number not in selected:
                continue
            stem = f"{frame_number:06d}"
            path = output_maps / f"{stem}_instances.png"
            current = require_image(path)
            corrected = current.copy()
            semantic = require_image(semantic_dir / f"{stem}_segmentation.png")
            for semantic_id in sorted(MULTI_SEMANTIC_IDS):
                candidate_ids = sorted(by_semantic[semantic_id])
                for mask in components(semantic == semantic_id, min_pixels):
                    temporal = temporal_instance(mask, next_map, candidate_ids)
                    if temporal is None:
                        continue
                    instance_id, _, _ = temporal
                    different = mask & (corrected != instance_id)
                    count = int(different.sum())
                    if not count:
                        continue
                    corrected[mask] = instance_id
                    changed_components += 1
                    changed_pixels += count
            if not np.array_equal(current, corrected):
                if not cv2.imwrite(str(path), corrected, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                    raise OSError(path)
                changed_frames += 1
            next_map = corrected
    return {
        "changed_frame_count": changed_frames,
        "changed_component_count": changed_components,
        "changed_pixel_count": changed_pixels,
    }


def main() -> None:
    args = parse_args()
    stems = frame_stems(args.semantic_dir, args.start_frame, args.end_frame)
    calibration_stems = frame_stems(args.semantic_dir, None, None)
    intrinsics = load_intrinsics(args.geometry_dir / "Intrinsics.txt")
    instance_semantic, names, canonical_centers = load_catalog(args.canonical_manifest)
    transform = np.eye(4) if args.world_transform is None else np.loadtxt(args.world_transform, dtype=np.float64)
    session_centers = dict(canonical_centers)
    calibration: dict[str, object] = {}
    if args.session == "b":
        session_centers, calibration = calibrate_session_b_centers(
            calibration_stems, args.semantic_dir, args.geometry_dir, transform, intrinsics,
            canonical_centers, instance_semantic, args.calibration_step,
            args.min_component_pixels,
        )

    output_maps = args.output_dir / "instance_maps"
    output_previews = args.output_dir / "previews"
    output_maps.mkdir(parents=True, exist_ok=True)
    output_previews.mkdir(parents=True, exist_ok=True)
    anchor_frames, anchor_visibility = load_anchor_visibility(args.anchor_dir)
    by_semantic: dict[int, list[int]] = defaultdict(list)
    for instance_id, semantic_id in instance_semantic.items():
        by_semantic[semantic_id].append(instance_id)
    anchor_count = 0
    dropped_components = 0
    temporal_assignment_count = 0
    previous_instance_map: np.ndarray | None = None
    audit_path = args.output_dir / "frame_audit.jsonl"
    with audit_path.open("w", encoding="utf-8") as audit_file:
        for frame_index, stem in enumerate(stems):
            anchor_path = None if args.anchor_dir is None else args.anchor_dir / f"{stem}_instances.png"
            frame_audit: dict[str, object] = {"frame": stem, "anchor": False, "assignments": []}
            if anchor_path is not None and anchor_path.is_file():
                instance_map = require_image(anchor_path).astype(np.uint16)
                anchor_count += 1
                frame_audit["anchor"] = True
            else:
                semantic = require_image(args.semantic_dir / f"{stem}_segmentation.png")
                instance_map = np.zeros(semantic.shape, dtype=np.uint16)
                frame_number = int(stem)
                for semantic_id, candidate_ids in by_semantic.items():
                    if semantic_id in MULTI_SEMANTIC_IDS:
                        continue
                    instance_id = candidate_ids[0]
                    if not instance_visible_near_frame(
                        instance_id, frame_number, anchor_frames, anchor_visibility
                    ):
                        frame_audit["assignments"].append({
                            "semantic_id": semantic_id,
                            "instance_id": 0,
                            "pixel_count": int(np.sum(semantic == semantic_id)),
                            "evidence": "absent_in_adjacent_reviewed_anchors",
                            "distance": None,
                        })
                        continue
                    mask, removed_islands = cleaned_singleton_mask(semantic == semantic_id)
                    instance_map[mask] = instance_id
                    frame_audit["assignments"].append({
                        "semantic_id": semantic_id,
                        "instance_id": instance_id,
                        "pixel_count": int(mask.sum()),
                        "evidence": "reviewed_single_physical_entity",
                        "distance": None,
                        "removed_remote_island_count": removed_islands,
                    })
                if any(np.any(semantic == sid) for sid in by_semantic):
                    depth = require_image(args.geometry_dir / f"{stem}_depth.tiff")
                    pose = transform @ np.loadtxt(args.geometry_dir / f"{stem}_pose.txt", dtype=np.float64)
                    for semantic_id in sorted(MULTI_SEMANTIC_IDS):
                        candidate_ids = sorted(by_semantic[semantic_id])
                        for mask in components(semantic == semantic_id, args.min_component_pixels):
                            point = world_centroid(mask, depth, pose, intrinsics)
                            spatial_id, spatial_evidence, spatial_distance = choose_instance(
                                mask, point, pose, candidate_ids, session_centers,
                                intrinsics, args.max_center_distance_m,
                            )
                            temporal = temporal_instance(mask, previous_instance_map, candidate_ids)
                            # Adjacent-frame overlap is stronger than a noisy RGB-D
                            # centroid, but do not carry an identity through an
                            # implausible room-space jump when reliable depth exists.
                            spatially_plausible = (
                                point is None
                                or spatial_distance <= 2.0 * args.max_center_distance_m
                            )
                            if temporal is not None and spatially_plausible:
                                instance_id, overlap_pixels, overlap_ratio = temporal
                                evidence = "temporal_overlap"
                                distance = spatial_distance
                                temporal_assignment_count += 1
                            else:
                                instance_id = spatial_id
                                evidence = spatial_evidence
                                distance = spatial_distance
                                overlap_pixels = 0
                                overlap_ratio = 0.0
                            if instance_id:
                                instance_map[mask] = instance_id
                            else:
                                dropped_components += 1
                            frame_audit["assignments"].append({
                                "semantic_id": semantic_id,
                                "instance_id": instance_id,
                                "pixel_count": int(mask.sum()),
                                "evidence": evidence,
                                "distance": round(float(distance), 6),
                                "temporal_overlap_pixels": overlap_pixels,
                                "temporal_overlap_ratio": round(float(overlap_ratio), 6),
                            })
            map_path = output_maps / f"{stem}_instances.png"
            if not cv2.imwrite(str(map_path), instance_map, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise OSError(map_path)
            if frame_index % args.preview_every == 0 or frame_index == len(stems) - 1:
                rgb = require_image(args.geometry_dir / f"{stem}_color.png", cv2.IMREAD_COLOR)
                preview_path = output_previews / f"{stem}_instances_overlay.jpg"
                if not cv2.imwrite(str(preview_path), render_preview(rgb, instance_map, instance_semantic, names), [cv2.IMWRITE_JPEG_QUALITY, 90]):
                    raise OSError(preview_path)
            audit_file.write(json.dumps(frame_audit, ensure_ascii=False) + "\n")
            previous_instance_map = instance_map
            if (frame_index + 1) % 250 == 0 or frame_index + 1 == len(stems):
                print(f"{args.session.upper()}: {frame_index + 1}/{len(stems)}", flush=True)

    future_anchor_refinement = {
        "changed_frame_count": 0,
        "changed_component_count": 0,
        "changed_pixel_count": 0,
    }
    if args.session == "a" and anchor_frames:
        future_anchor_refinement = refine_from_future_anchors(
            stems, args.semantic_dir, output_maps, anchor_frames,
            by_semantic, args.min_component_pixels,
        )
        # Previews must reflect the final, bidirectionally constrained maps.
        for frame_index, stem in enumerate(stems):
            if frame_index % args.preview_every != 0 and frame_index != len(stems) - 1:
                continue
            rgb = require_image(args.geometry_dir / f"{stem}_color.png", cv2.IMREAD_COLOR)
            instance_map = require_image(output_maps / f"{stem}_instances.png")
            preview_path = output_previews / f"{stem}_instances_overlay.jpg"
            if not cv2.imwrite(
                str(preview_path),
                render_preview(rgb, instance_map, instance_semantic, names),
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            ):
                raise OSError(preview_path)

    manifest = {
        "schema": "full_ab_physical_instance_maps/v1",
        "status": "generated_not_yet_connected_to_khronos",
        "session": args.session.upper(),
        "frame_count": len(stems),
        "first_frame": stems[0],
        "last_frame": stems[-1],
        "instance_dtype": "uint16",
        "background_id": 0,
        "active_physical_instance_ids": sorted(instance_semantic),
        "instance_to_semantic": {str(k): v for k, v in sorted(instance_semantic.items())},
        "semantic_dir": str(args.semantic_dir.resolve()),
        "geometry_dir": str(args.geometry_dir.resolve()),
        "canonical_manifest": str(args.canonical_manifest.resolve()),
        "anchor_dir": None if args.anchor_dir is None else str(args.anchor_dir.resolve()),
        "anchor_frame_count": anchor_count,
        "world_transform": transform.tolist(),
        "session_centers_m": {str(k): v.round(6).tolist() for k, v in sorted(session_centers.items())},
        "session_b_calibration": calibration,
        "dropped_unmatched_component_count": dropped_components,
        "temporal_assignment_count": temporal_assignment_count,
        "future_anchor_refinement": future_anchor_refinement,
        "frame_audit": str(audit_path.resolve()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("session", "frame_count", "anchor_frame_count", "dropped_unmatched_component_count")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
