#!/usr/bin/env python3
"""Build a small, reviewable instance-map prototype from semantic masks.

This tool intentionally does not alter the source semantic masks.  Pixels from
configured object classes are grouped in 2D first, then matched between frames
using world-space centroids computed from depth and camera poses.  Classes known
to contain exactly one physical object can be forced to one stable instance.

The output is an instance-id PNG plus JSON metadata retaining the semantic id;
the two identities are never conflated.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml


@dataclass
class Detection:
    semantic_id: int
    mask: np.ndarray
    pixel_count: int
    bbox_xywh: tuple[int, int, int, int]
    image_centroid: np.ndarray
    world_centroid: np.ndarray | None = None
    instance_id: int = 0


@dataclass
class Track:
    instance_id: int
    semantic_id: int
    world_centroid: np.ndarray | None
    image_centroid: np.ndarray
    last_frame_number: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-dir", type=Path, required=True)
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--geometry-dir", type=Path, required=True)
    parser.add_argument(
        "--intrinsics",
        "--camera-yaml",
        dest="intrinsics",
        type=Path,
        required=True,
        help="NSS Intrinsics.txt or an ORB-SLAM OpenCV camera YAML.",
    )
    parser.add_argument("--label-space", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    frame_selection = parser.add_mutually_exclusive_group(required=True)
    frame_selection.add_argument(
        "--frames",
        nargs="+",
        help="Frame stems, for example 000360 000390 000420.",
    )
    frame_selection.add_argument(
        "--all-frames",
        action="store_true",
        help="Process every *_segmentation.png file in --semantic-dir.",
    )
    parser.add_argument(
        "--single-instance-label",
        type=int,
        action="append",
        default=[],
        help="Semantic label known to represent one physical object (repeatable).",
    )
    parser.add_argument("--min-component-pixels", type=int, default=500)
    parser.add_argument("--max-match-distance-m", type=float, default=1.25)
    parser.add_argument(
        "--max-image-match-distance",
        type=float,
        default=0.22,
        help="Fallback normalized-image centroid distance when usable depth is absent.",
    )
    parser.add_argument(
        "--max-image-match-gap-frames",
        type=int,
        default=120,
        help="Maximum source-frame gap for image-only matching.",
    )
    parser.add_argument("--centroid-sample-stride", type=int, default=3)
    parser.add_argument(
        "--canonicalize-world-threshold-m",
        type=float,
        default=0.0,
        help="If positive, merge temporal tracks into world-space canonical instances.",
    )
    parser.add_argument("--canonical-min-observations", type=int, default=2)
    parser.add_argument("--canonical-assignment-distance-m", type=float, default=1.25)
    parser.add_argument(
        "--canonical-class-threshold",
        action="append",
        default=[],
        metavar="SEMANTIC_ID=METERS",
        help="Override the canonical clustering threshold for one semantic class.",
    )
    return parser.parse_args()


def load_label_space(path: Path) -> tuple[list[int], dict[int, str]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    object_labels = [int(value) for value in data["object_labels"]]
    names = {
        int(item["label"]): str(item["name"])
        for item in data.get("label_names", [])
    }
    return object_labels, names


def load_intrinsics(path: Path) -> tuple[int, int, float, float, float, float]:
    text = path.read_text(encoding="utf-8")

    def number(*keys: str) -> float:
        for key in keys:
            match = re.search(rf"^{re.escape(key)}:\s*([-+0-9.eE]+)\s*$", text, re.MULTILINE)
            if match:
                return float(match.group(1))
        raise ValueError(f"Missing all of {keys!r} in {path}")

    return (
        int(number("Res_x", "width", "Camera.width")),
        int(number("Res_y", "height", "Camera.height")),
        number("f_x", "Camera1.fx"),
        number("f_y", "Camera1.fy"),
        number("u", "c_x", "Camera1.cx"),
        number("v", "c_y", "Camera1.cy"),
    )


def require_image(path: Path, flags: int) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def components_for_label(
    semantic: np.ndarray,
    semantic_id: int,
    min_pixels: int,
    force_single: bool,
) -> list[Detection]:
    binary = (semantic == semantic_id).astype(np.uint8)
    total_pixels = int(binary.sum())
    if total_pixels == 0:
        return []

    if force_single:
        ys, xs = np.nonzero(binary)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
        image_centroid = np.array([xs.mean() / semantic.shape[1], ys.mean() / semantic.shape[0]])
        return [Detection(semantic_id, binary.astype(bool), total_pixels, bbox, image_centroid)]

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    major = [
        index
        for index in range(1, count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= min_pixels
    ]
    if not major:
        major = [int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1]

    grouped: dict[int, list[int]] = {index: [index] for index in major}
    for index in range(1, count):
        if index in grouped:
            continue
        nearest = min(
            major,
            key=lambda candidate: float(np.linalg.norm(centroids[index] - centroids[candidate])),
        )
        grouped[nearest].append(index)

    detections: list[Detection] = []
    for component_ids in grouped.values():
        mask = np.isin(labels, component_ids)
        ys, xs = np.nonzero(mask)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
        image_centroid = np.array([xs.mean() / semantic.shape[1], ys.mean() / semantic.shape[0]])
        detections.append(Detection(semantic_id, mask, int(mask.sum()), bbox, image_centroid))
    detections.sort(key=lambda detection: detection.pixel_count, reverse=True)
    return detections


def calculate_world_centroid(
    mask: np.ndarray,
    depth: np.ndarray,
    pose: np.ndarray,
    intrinsics: tuple[int, int, float, float, float, float],
    sample_stride: int,
) -> np.ndarray | None:
    width, height, fx, fy, cx, cy = intrinsics
    if depth.shape != (height, width):
        raise ValueError(f"Depth shape {depth.shape} does not match intrinsics {(height, width)}")
    depth_mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    depth_mask[::sample_stride, :] &= True
    sampled = np.zeros_like(depth_mask)
    sampled[::sample_stride, ::sample_stride] = depth_mask[::sample_stride, ::sample_stride]
    valid = sampled & np.isfinite(depth) & (depth > 0.05)
    v, u = np.nonzero(valid)
    if len(u) < 10:
        return None
    z = depth[v, u].astype(np.float64)
    camera = np.stack(((u - cx) * z / fx, (v - cy) * z / fy, z, np.ones_like(z)), axis=1)
    world = (pose @ camera.T).T[:, :3]
    return np.median(world, axis=0)


def assign_instances(
    detections: list[Detection],
    tracks: list[Track],
    singleton_ids: dict[int, int],
    singleton_labels: set[int],
    frame_number: int,
    next_instance_id: int,
    max_distance: float,
    max_image_distance: float,
    max_image_gap_frames: int,
) -> int:
    for detection in detections:
        if detection.semantic_id not in singleton_labels:
            continue
        instance_id = singleton_ids.get(detection.semantic_id)
        if instance_id is None:
            instance_id = next_instance_id
            next_instance_id += 1
            singleton_ids[detection.semantic_id] = instance_id
            tracks.append(
                Track(
                    instance_id,
                    detection.semantic_id,
                    detection.world_centroid,
                    detection.image_centroid,
                    frame_number,
                )
            )
        detection.instance_id = instance_id
        for track in tracks:
            if track.instance_id == instance_id:
                if detection.world_centroid is not None:
                    track.world_centroid = detection.world_centroid
                track.image_centroid = detection.image_centroid
                track.last_frame_number = frame_number
                break

    candidates: list[tuple[float, int, int]] = []
    for detection_index, detection in enumerate(detections):
        if detection.instance_id or detection.world_centroid is None:
            continue
        for track_index, track in enumerate(tracks):
            if track.semantic_id != detection.semantic_id or track.world_centroid is None:
                continue
            distance = float(np.linalg.norm(detection.world_centroid - track.world_centroid))
            if distance <= max_distance:
                candidates.append((distance, detection_index, track_index))

    used_detections: set[int] = set()
    used_tracks: set[int] = set()
    for _, detection_index, track_index in sorted(candidates):
        if detection_index in used_detections or track_index in used_tracks:
            continue
        detection = detections[detection_index]
        track = tracks[track_index]
        detection.instance_id = track.instance_id
        if detection.world_centroid is not None:
            track.world_centroid = detection.world_centroid
        track.image_centroid = detection.image_centroid
        track.last_frame_number = frame_number
        used_detections.add(detection_index)
        used_tracks.add(track_index)

    # RGB-D sensors commonly return invalid depth on screens and reflective
    # surfaces.  Preserve identity there with a conservative image-space
    # fallback after all stronger world-space matches have been consumed.
    assigned_ids = {detection.instance_id for detection in detections if detection.instance_id}
    image_candidates: list[tuple[float, int, int]] = []
    for detection_index, detection in enumerate(detections):
        if detection.instance_id:
            continue
        for track_index, track in enumerate(tracks):
            if track.instance_id in assigned_ids or track.semantic_id != detection.semantic_id:
                continue
            if frame_number - track.last_frame_number > max_image_gap_frames:
                continue
            distance = float(np.linalg.norm(detection.image_centroid - track.image_centroid))
            if distance <= max_image_distance:
                image_candidates.append((distance, detection_index, track_index))
    used_detections.clear()
    used_tracks.clear()
    for _, detection_index, track_index in sorted(image_candidates):
        if detection_index in used_detections or track_index in used_tracks:
            continue
        detection = detections[detection_index]
        track = tracks[track_index]
        detection.instance_id = track.instance_id
        if detection.world_centroid is not None:
            track.world_centroid = detection.world_centroid
        track.image_centroid = detection.image_centroid
        track.last_frame_number = frame_number
        used_detections.add(detection_index)
        used_tracks.add(track_index)

    for detection in detections:
        if detection.instance_id:
            continue
        detection.instance_id = next_instance_id
        next_instance_id += 1
        tracks.append(
            Track(
                detection.instance_id,
                detection.semantic_id,
                detection.world_centroid,
                detection.image_centroid,
                frame_number,
            )
        )
    return next_instance_id


def color_for_instance(instance_id: int) -> tuple[int, int, int]:
    # Deterministic, high-contrast BGR color.
    rng = np.random.default_rng(instance_id * 104729)
    return tuple(int(value) for value in rng.integers(55, 256, size=3))


def render_preview(rgb: np.ndarray, instance_map: np.ndarray, detections: Iterable[Detection], names: dict[int, str]) -> np.ndarray:
    overlay = rgb.copy()
    for detection in detections:
        color = color_for_instance(detection.instance_id)
        overlay[detection.mask] = color
    preview = cv2.addWeighted(rgb, 0.55, overlay, 0.45, 0.0)
    for detection in detections:
        x, y, width, height = detection.bbox_xywh
        color = color_for_instance(detection.instance_id)
        cv2.rectangle(preview, (x, y), (x + width - 1, y + height - 1), color, 3)
        text = f"I{detection.instance_id} {names.get(detection.semantic_id, str(detection.semantic_id))} S{detection.semantic_id}"
        cv2.putText(preview, text, (x, max(28, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    return preview


def canonicalize_world_tracks(
    frame_records: list[dict[str, object]],
    instance_dir: Path,
    preview_dir: Path,
    rgb_dir: Path,
    names: dict[int, str],
    singleton_labels: set[int],
    cluster_threshold_m: float,
    min_observations: int,
    assignment_distance_m: float,
    class_thresholds: dict[int, float],
) -> dict[str, object]:
    """Collapse temporary tracks into a small set of room-level instances."""
    from scipy.cluster.hierarchy import fcluster, linkage

    points_by_semantic: dict[int, list[np.ndarray]] = defaultdict(list)
    observed_semantics: set[int] = set()
    for frame in frame_records:
        for item in frame["instances"]:
            semantic_id = int(item["semantic_id"])
            observed_semantics.add(semantic_id)
            if item["world_centroid_m"] is not None:
                points_by_semantic[semantic_id].append(
                    np.asarray(item["world_centroid_m"], dtype=np.float64)
                )

    centers_by_semantic: dict[int, list[np.ndarray | None]] = {}
    cluster_sizes: dict[int, list[int]] = {}
    for semantic_id in sorted(observed_semantics):
        points = np.asarray(points_by_semantic.get(semantic_id, []), dtype=np.float64)
        if semantic_id in singleton_labels or len(points) < 2:
            center = None if len(points) == 0 else np.median(points, axis=0)
            centers_by_semantic[semantic_id] = [center]
            cluster_sizes[semantic_id] = [len(points)]
            continue

        semantic_threshold = class_thresholds.get(semantic_id, cluster_threshold_m)
        labels = fcluster(
            linkage(points, method="complete"),
            semantic_threshold,
            criterion="distance",
        )
        groups = []
        for cluster_id in sorted(set(int(value) for value in labels)):
            cluster_points = points[labels == cluster_id]
            groups.append((len(cluster_points), np.median(cluster_points, axis=0)))
        major = [group for group in groups if group[0] >= min_observations]
        if not major:
            major = [max(groups, key=lambda group: group[0])]
        major.sort(key=lambda group: tuple(float(value) for value in group[1]))
        centers_by_semantic[semantic_id] = [group[1] for group in major]
        cluster_sizes[semantic_id] = [group[0] for group in major]

    canonical_ids: dict[tuple[int, int], int] = {}
    next_id = 1
    for semantic_id in sorted(centers_by_semantic):
        for cluster_index in range(len(centers_by_semantic[semantic_id])):
            canonical_ids[(semantic_id, cluster_index)] = next_id
            next_id += 1

    def nearest_cluster(semantic_id: int, point: list[float] | None) -> int | None:
        centers = centers_by_semantic[semantic_id]
        if len(centers) == 1:
            return 0
        if point is None:
            return None
        vector = np.asarray(point, dtype=np.float64)
        distances = [
            np.inf if center is None else float(np.linalg.norm(vector - center))
            for center in centers
        ]
        index = int(np.argmin(distances))
        return index if distances[index] <= assignment_distance_m else None

    # A temporary track can bridge depth dropouts. Use its reliable observations
    # to vote for a canonical room instance, but never let a lone bad 3D point
    # override temporal image evidence.
    track_votes: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    for frame in frame_records:
        for item in frame["instances"]:
            semantic_id = int(item["semantic_id"])
            cluster_index = nearest_cluster(semantic_id, item["world_centroid_m"])
            if cluster_index is not None:
                track_votes[(semantic_id, int(item["instance_id"]))][cluster_index] += 1
    track_alias = {
        key: votes.most_common(1)[0][0]
        for key, votes in track_votes.items()
        if votes
    }

    last_image_centroid: dict[tuple[int, int], np.ndarray] = {}
    for frame in frame_records:
        old_map = require_image(Path(frame["instance_map"]), cv2.IMREAD_UNCHANGED)
        height, width = old_map.shape
        assignments: list[tuple[dict[str, object], int]] = []
        occupied: dict[int, set[int]] = defaultdict(set)
        deferred: list[dict[str, object]] = []

        for item in frame["instances"]:
            semantic_id = int(item["semantic_id"])
            cluster_index = nearest_cluster(semantic_id, item["world_centroid_m"])
            if cluster_index is None:
                cluster_index = track_alias.get((semantic_id, int(item["instance_id"])))
            if cluster_index is None:
                deferred.append(item)
                continue
            assignments.append((item, cluster_index))
            occupied[semantic_id].add(cluster_index)

        for item in deferred:
            semantic_id = int(item["semantic_id"])
            x, y, box_width, box_height = (int(value) for value in item["bbox_xywh"])
            image_center = np.array(
                [(x + box_width / 2.0) / width, (y + box_height / 2.0) / height]
            )
            available = [
                index
                for index in range(len(centers_by_semantic[semantic_id]))
                if index not in occupied[semantic_id]
            ]
            candidates = available or list(range(len(centers_by_semantic[semantic_id])))
            with_history = [
                index
                for index in candidates
                if (semantic_id, index) in last_image_centroid
            ]
            if with_history:
                cluster_index = min(
                    with_history,
                    key=lambda index: float(
                        np.linalg.norm(
                            image_center - last_image_centroid[(semantic_id, index)]
                        )
                    ),
                )
            else:
                cluster_index = candidates[0]
            assignments.append((item, cluster_index))
            occupied[semantic_id].add(cluster_index)

        new_map = np.zeros_like(old_map, dtype=np.uint16)
        assigned_items: dict[int, list[dict[str, object]]] = defaultdict(list)
        for item, cluster_index in assignments:
            semantic_id = int(item["semantic_id"])
            canonical_id = canonical_ids[(semantic_id, cluster_index)]
            new_map[old_map == int(item["instance_id"])] = canonical_id
            assigned_items[canonical_id].append(item)

        canonical_instances: list[dict[str, object]] = []
        preview_detections: list[Detection] = []
        for canonical_id in sorted(assigned_items):
            items = assigned_items[canonical_id]
            semantic_id = int(items[0]["semantic_id"])
            mask = new_map == canonical_id
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                continue
            bbox = (
                int(xs.min()),
                int(ys.min()),
                int(xs.max() - xs.min() + 1),
                int(ys.max() - ys.min() + 1),
            )
            image_center = np.array([xs.mean() / width, ys.mean() / height])
            cluster_index = next(
                index
                for (sid, index), value in canonical_ids.items()
                if sid == semantic_id and value == canonical_id
            )
            last_image_centroid[(semantic_id, cluster_index)] = image_center
            world_points = [
                np.asarray(item["world_centroid_m"], dtype=np.float64)
                for item in items
                if item["world_centroid_m"] is not None
            ]
            world_centroid = (
                None if not world_points else np.median(np.asarray(world_points), axis=0)
            )
            canonical_instances.append(
                {
                    "instance_id": canonical_id,
                    "semantic_id": semantic_id,
                    "semantic_name": names.get(semantic_id, str(semantic_id)),
                    "pixel_count": int(mask.sum()),
                    "bbox_xywh": list(bbox),
                    "world_centroid_m": None
                    if world_centroid is None
                    else [round(float(value), 6) for value in world_centroid],
                    "forced_single_instance": semantic_id in singleton_labels,
                    "canonical_cluster": cluster_index,
                }
            )
            preview_detections.append(
                Detection(
                    semantic_id,
                    mask,
                    int(mask.sum()),
                    bbox,
                    image_center,
                    world_centroid,
                    canonical_id,
                )
            )

        if not cv2.imwrite(str(frame["instance_map"]), new_map):
            raise OSError(f"Failed to rewrite {frame['instance_map']}")
        rgb = require_image(rgb_dir / f"{frame['frame']}_color.png", cv2.IMREAD_COLOR)
        if not cv2.imwrite(
            str(frame["preview"]),
            render_preview(rgb, new_map, preview_detections, names),
        ):
            raise OSError(f"Failed to rewrite {frame['preview']}")
        frame["instances"] = canonical_instances
        frame["instance_count"] = len(canonical_instances)

    summary = []
    for semantic_id in sorted(centers_by_semantic):
        summary.append(
            {
                "semantic_id": semantic_id,
                "semantic_name": names.get(semantic_id, str(semantic_id)),
                "canonical_instance_count": len(centers_by_semantic[semantic_id]),
                "canonical_instance_ids": [
                    canonical_ids[(semantic_id, index)]
                    for index in range(len(centers_by_semantic[semantic_id]))
                ],
                "support_observations": cluster_sizes[semantic_id],
                "cluster_threshold_m": class_thresholds.get(
                    semantic_id, cluster_threshold_m
                ),
                "world_centers_m": [
                    None
                    if center is None
                    else [round(float(value), 6) for value in center]
                    for center in centers_by_semantic[semantic_id]
                ],
            }
        )
    return {
        "method": "complete_linkage_world_centroids_with_temporal_fallback",
        "cluster_threshold_m": cluster_threshold_m,
        "minimum_observations": min_observations,
        "assignment_distance_m": assignment_distance_m,
        "global_instance_count": next_id - 1,
        "classes": summary,
    }


def main() -> None:
    args = parse_args()
    canonical_class_thresholds: dict[int, float] = {}
    for value in args.canonical_class_threshold:
        semantic_text, separator, threshold_text = value.partition("=")
        if not separator:
            raise ValueError(
                f"Invalid --canonical-class-threshold {value!r}; expected ID=METERS"
            )
        canonical_class_thresholds[int(semantic_text)] = float(threshold_text)
    frame_stems = args.frames
    if args.all_frames:
        suffix = "_segmentation.png"
        frame_stems = [
            path.name[: -len(suffix)]
            for path in args.semantic_dir.glob(f"*{suffix}")
        ]
    if not frame_stems:
        raise ValueError("No semantic frames selected")
    object_labels, names = load_label_space(args.label_space)
    singleton_labels = set(args.single_instance_label)
    invalid_singletons = singleton_labels.difference(object_labels)
    if invalid_singletons:
        raise ValueError(f"Singleton labels are not configured object labels: {sorted(invalid_singletons)}")

    intrinsics = load_intrinsics(args.intrinsics)
    output_instances = args.output_dir / "instance_maps"
    output_previews = args.output_dir / "previews"
    output_instances.mkdir(parents=True, exist_ok=True)
    output_previews.mkdir(parents=True, exist_ok=True)

    tracks: list[Track] = []
    singleton_ids: dict[int, int] = {}
    next_instance_id = 1
    frame_records: list[dict[str, object]] = []

    for stem in sorted(frame_stems, key=int):
        semantic_path = args.semantic_dir / f"{stem}_segmentation.png"
        rgb_path = args.rgb_dir / f"{stem}_color.png"
        depth_path = args.geometry_dir / f"{stem}_depth.tiff"
        pose_path = args.geometry_dir / f"{stem}_pose.txt"
        semantic = require_image(semantic_path, cv2.IMREAD_UNCHANGED)
        rgb = require_image(rgb_path, cv2.IMREAD_COLOR)
        depth = require_image(depth_path, cv2.IMREAD_UNCHANGED)
        pose = np.loadtxt(pose_path, dtype=np.float64)
        if semantic.ndim != 2 or semantic.dtype != np.uint8:
            raise ValueError(f"Expected uint8 semantic image, got {semantic.dtype} {semantic.shape}: {semantic_path}")
        if rgb.shape[:2] != semantic.shape:
            raise ValueError(f"RGB and semantic dimensions differ for frame {stem}")
        if pose.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 pose in {pose_path}")

        detections: list[Detection] = []
        for semantic_id in object_labels:
            detections.extend(
                components_for_label(
                    semantic,
                    semantic_id,
                    args.min_component_pixels,
                    semantic_id in singleton_labels,
                )
            )
        for detection in detections:
            detection.world_centroid = calculate_world_centroid(
                detection.mask,
                depth,
                pose,
                intrinsics,
                args.centroid_sample_stride,
            )

        next_instance_id = assign_instances(
            detections,
            tracks,
            singleton_ids,
            singleton_labels,
            int(stem),
            next_instance_id,
            args.max_match_distance_m,
            args.max_image_match_distance,
            args.max_image_match_gap_frames,
        )

        instance_map = np.zeros(semantic.shape, dtype=np.uint16)
        for detection in detections:
            if detection.instance_id > np.iinfo(np.uint16).max:
                raise OverflowError("Instance id does not fit uint16")
            instance_map[detection.mask] = detection.instance_id

        instance_path = output_instances / f"{stem}_instances.png"
        preview_path = output_previews / f"{stem}_instances_overlay.png"
        if not cv2.imwrite(str(instance_path), instance_map):
            raise OSError(f"Failed to write {instance_path}")
        if not cv2.imwrite(str(preview_path), render_preview(rgb, instance_map, detections, names)):
            raise OSError(f"Failed to write {preview_path}")

        instances = []
        for detection in sorted(detections, key=lambda value: value.instance_id):
            instances.append(
                {
                    "instance_id": detection.instance_id,
                    "semantic_id": detection.semantic_id,
                    "semantic_name": names.get(detection.semantic_id, str(detection.semantic_id)),
                    "pixel_count": detection.pixel_count,
                    "bbox_xywh": list(detection.bbox_xywh),
                    "world_centroid_m": None
                    if detection.world_centroid is None
                    else [round(float(value), 6) for value in detection.world_centroid],
                    "forced_single_instance": detection.semantic_id in singleton_labels,
                }
            )
        frame_records.append(
            {
                "frame": stem,
                "semantic_source": str(semantic_path.resolve()),
                "instance_map": str(instance_path.resolve()),
                "preview": str(preview_path.resolve()),
                "instance_count": len(instances),
                "instances": instances,
            }
        )

    canonicalization = None
    if args.canonicalize_world_threshold_m > 0.0:
        canonicalization = canonicalize_world_tracks(
            frame_records,
            output_instances,
            output_previews,
            args.rgb_dir,
            names,
            singleton_labels,
            args.canonicalize_world_threshold_m,
            args.canonical_min_observations,
            args.canonical_assignment_distance_m,
            canonical_class_thresholds,
        )

    manifest = {
        "schema": "semantic_instance_prototype/v1",
        "status": "prototype_not_connected_to_khronos",
        "inputs": {
            "semantic_dir": str(args.semantic_dir.resolve()),
            "rgb_dir": str(args.rgb_dir.resolve()),
            "geometry_dir": str(args.geometry_dir.resolve()),
            "intrinsics": str(args.intrinsics.resolve()),
            "label_space": str(args.label_space.resolve()),
        },
        "instance_image": {
            "dtype": "uint16",
            "background_id": 0,
            "meaning": "nonzero pixels are physical instance ids; semantic ids are in this manifest",
        },
        "semantic_image": {
            "dtype": "uint8",
            "meaning": "unchanged ADE20K semantic ids",
        },
        "object_labels": object_labels,
        "single_instance_labels": sorted(singleton_labels),
        "canonicalization": canonicalization,
        "parameters": {
            "min_component_pixels": args.min_component_pixels,
            "max_match_distance_m": args.max_match_distance_m,
            "max_image_match_distance": args.max_image_match_distance,
            "max_image_match_gap_frames": args.max_image_match_gap_frames,
            "centroid_sample_stride": args.centroid_sample_stride,
        },
        "frames": frame_records,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(frame_records)} frames to {args.output_dir}")
    print(f"Manifest: {manifest_path}")
    for frame in frame_records:
        chair = [item for item in frame["instances"] if item["semantic_id"] == 75]
        print(
            f"{frame['frame']}: {frame['instance_count']} instances; "
            f"chair={[(item['instance_id'], item['pixel_count']) for item in chair]}"
        )


if __name__ == "__main__":
    main()
