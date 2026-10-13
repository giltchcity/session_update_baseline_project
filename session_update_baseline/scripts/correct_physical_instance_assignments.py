#!/usr/bin/env python3
"""Apply reviewed room-level instance corrections to cleaned instance maps."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np


COMPUTER_ID = 74
COMPUTER_CENTERS = {
    7: np.array([3.357538, -0.433612, 0.790512]),  # white monitor
    8: np.array([4.003528, -0.193254, 1.080408]),  # left black device
    9: np.array([4.315913, -0.129828, 1.474723]),  # rear desk device
    20: np.array([3.160000, -0.090000, 1.100000]),  # pink laptop
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--geometry-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_image(path: Path, flags: int) -> np.ndarray:
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
        int(value("Res_x")),
        int(value("Res_y")),
        value("f_x"),
        value("f_y"),
        value("u"),
        value("v"),
    )


def world_centroid(
    mask: np.ndarray,
    depth: np.ndarray,
    pose: np.ndarray,
    intrinsics: tuple[int, int, float, float, float, float],
) -> np.ndarray | None:
    width, height, fx, fy, cx, cy = intrinsics
    reduced = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    sampled = np.zeros_like(reduced)
    sampled[::2, ::2] = reduced[::2, ::2]
    valid = sampled & np.isfinite(depth) & (depth > 0.05)
    v, u = np.nonzero(valid)
    if len(u) < 10:
        return None
    z = depth[v, u].astype(np.float64)
    camera = np.stack(
        ((u - cx) * z / fx, (v - cy) * z / fy, z, np.ones_like(z)), axis=1
    )
    return np.median((pose @ camera.T).T[:, :3], axis=0)


def projected_centers(
    pose: np.ndarray,
    intrinsics: tuple[int, int, float, float, float, float],
) -> dict[int, np.ndarray | None]:
    _, _, fx, fy, cx, cy = intrinsics
    inverse = np.linalg.inv(pose)
    result: dict[int, np.ndarray | None] = {}
    for instance_id, center in COMPUTER_CENTERS.items():
        camera = inverse @ np.r_[center, 1.0]
        if camera[2] <= 0:
            result[instance_id] = None
        else:
            result[instance_id] = np.array(
                [fx * camera[0] / camera[2] + cx, fy * camera[1] / camera[2] + cy]
            )
    return result


def assign_computer_components(
    instance_map: np.ndarray,
    depth: np.ndarray,
    pose: np.ndarray,
    intrinsics: tuple[int, int, float, float, float, float],
) -> list[dict[str, object]]:
    computer_mask = np.isin(instance_map, [7, 8, 9, 20])
    instance_map[computer_mask] = 0
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        computer_mask.astype(np.uint8), 8
    )
    projections = projected_centers(pose, intrinsics)
    assignments: list[dict[str, object]] = []
    for component in range(1, count):
        mask = labels == component
        area = int(stats[component, cv2.CC_STAT_AREA])
        point = world_centroid(mask, depth, pose, intrinsics)
        if point is not None:
            distances = {
                instance_id: float(np.linalg.norm(point - center))
                for instance_id, center in COMPUTER_CENTERS.items()
            }
            evidence = "world_centroid"
            evidence_value = [round(float(value), 6) for value in point]
        else:
            image_point = centroids[component]
            distances = {
                instance_id: np.inf
                if projected is None
                else float(np.linalg.norm(image_point - projected))
                for instance_id, projected in projections.items()
            }
            evidence = "projected_room_center"
            evidence_value = [round(float(value), 3) for value in image_point]
        instance_id = min(distances, key=distances.get)
        instance_map[mask] = instance_id
        assignments.append(
            {
                "instance_id": instance_id,
                "pixel_count": area,
                "bbox_xywh": [int(value) for value in stats[component, :4]],
                "evidence": evidence,
                "evidence_value": evidence_value,
                "assignment_distance": round(float(distances[instance_id]), 6),
            }
        )
    return assignments


def apply_reviewed_computer_overrides(
    stem: str, instance_map: np.ndarray
) -> list[dict[str, object]]:
    """Resolve views where bad/missing depth beats the room-center assignment."""
    changes: list[dict[str, object]] = []
    rules = {
        # Foreground open laptop is I20; distant white monitor stays I7.
        "003840": [(7, 1000, 20)],
        # Closed pink laptop lies below the white desktop monitor.
        "002490": [(7, 800, 20)],
        # White monitor is at the extreme right; black device remains I8.
        "002430": [(8, 1750, 7)],
    }
    for source_id, coordinate_threshold, destination_id in rules.get(stem, []):
        binary = (instance_map == source_id).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        for component in range(1, count):
            x = int(stats[component, cv2.CC_STAT_LEFT])
            y = int(stats[component, cv2.CC_STAT_TOP])
            compare = y if stem == "002490" else x
            if compare < coordinate_threshold:
                continue
            area = int(stats[component, cv2.CC_STAT_AREA])
            instance_map[labels == component] = destination_id
            changes.append(
                {
                    "source_instance_id": source_id,
                    "destination_instance_id": destination_id,
                    "pixel_count": area,
                    "bbox_xywh": [int(value) for value in stats[component, :4]],
                }
            )

    # Remove only microscopic computer remnants after physical reassignment.
    for instance_id in COMPUTER_CENTERS:
        binary = (instance_map == instance_id).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        if count <= 2:
            continue
        largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        for component in range(1, count):
            area = int(stats[component, cv2.CC_STAT_AREA])
            if component == largest or area >= 50:
                continue
            instance_map[labels == component] = 0
            changes.append(
                {
                    "source_instance_id": instance_id,
                    "destination_instance_id": 0,
                    "pixel_count": area,
                    "bbox_xywh": [int(value) for value in stats[component, :4]],
                }
            )
    return changes


def color_for_instance(instance_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(instance_id * 104729)
    return tuple(int(value) for value in rng.integers(55, 256, size=3))


def metadata_for_frame(
    instance_map: np.ndarray, old_instances: list[dict[str, object]]
) -> list[dict[str, object]]:
    old = {int(item["instance_id"]): item for item in old_instances}
    for cluster, instance_id in enumerate([7, 8, 9, 20]):
        old.setdefault(
            instance_id,
            {
                "instance_id": instance_id,
                "semantic_id": COMPUTER_ID,
                "semantic_name": "computer",
                "world_centroid_m": [
                    round(float(value), 6)
                    for value in COMPUTER_CENTERS[instance_id]
                ],
                "forced_single_instance": False,
                "canonical_cluster": cluster,
            },
        )
    result: list[dict[str, object]] = []
    for instance_id in (int(value) for value in np.unique(instance_map) if value):
        mask = instance_map == instance_id
        ys, xs = np.nonzero(mask)
        item = dict(old[instance_id])
        item["pixel_count"] = int(mask.sum())
        item["bbox_xywh"] = [
            int(xs.min()),
            int(ys.min()),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        ]
        if instance_id in COMPUTER_CENTERS:
            item["semantic_id"] = COMPUTER_ID
            item["semantic_name"] = "computer"
            item["canonical_cluster"] = [7, 8, 9, 20].index(instance_id)
        result.append(item)
    return result


def render_preview(
    rgb: np.ndarray, instance_map: np.ndarray, instances: list[dict[str, object]]
) -> np.ndarray:
    overlay = rgb.copy()
    for item in instances:
        instance_id = int(item["instance_id"])
        overlay[instance_map == instance_id] = color_for_instance(instance_id)
    preview = cv2.addWeighted(rgb, 0.55, overlay, 0.45, 0.0)
    for item in instances:
        instance_id = int(item["instance_id"])
        x, y, width, height = (int(value) for value in item["bbox_xywh"])
        color = color_for_instance(instance_id)
        cv2.rectangle(preview, (x, y), (x + width - 1, y + height - 1), color, 3)
        label = f"I{instance_id} {item['semantic_name']} S{item['semantic_id']}"
        cv2.putText(
            preview,
            label,
            (x, max(28, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )
    return preview


def update_manifest_summary(manifest: dict[str, object]) -> None:
    canonical = manifest.get("canonicalization")
    if not canonical:
        return
    for item in canonical["classes"]:
        semantic_id = int(item["semantic_id"])
        if semantic_id == COMPUTER_ID:
            item["canonical_instance_count"] = 4
            item["canonical_instance_ids"] = [7, 8, 9, 20]
            item["world_centers_m"] = [
                [round(float(value), 6) for value in COMPUTER_CENTERS[instance_id]]
                for instance_id in [7, 8, 9, 20]
            ]
        elif semantic_id == 115:
            ids = [int(value) for value in item["canonical_instance_ids"]]
            if 15 in ids:
                index = ids.index(15)
                for key in ("canonical_instance_ids", "world_centers_m", "support_observations"):
                    if key in item:
                        item[key].pop(index)
                item["canonical_instance_count"] = len(item["canonical_instance_ids"])
    active_ids = {
        int(item["instance_id"])
        for frame in manifest["frames"]
        for item in frame["instances"]
    }
    canonical["global_instance_count"] = len(active_ids)
    canonical["active_instance_ids"] = sorted(active_ids)
    canonical["max_instance_id"] = max(active_ids)


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.input_dir / "manifest.json").read_text(encoding="utf-8"))
    intrinsics = load_intrinsics(args.geometry_dir / "Intrinsics.txt")
    output_maps = args.output_dir / "instance_maps"
    output_previews = args.output_dir / "previews"
    output_maps.mkdir(parents=True, exist_ok=True)
    output_previews.mkdir(parents=True, exist_ok=True)
    audit: list[dict[str, object]] = []

    for frame in manifest["frames"]:
        stem = str(frame["frame"])
        instance_map = read_image(
            args.input_dir / "instance_maps" / f"{stem}_instances.png",
            cv2.IMREAD_UNCHANGED,
        )
        rgb = read_image(args.rgb_dir / f"{stem}_color.png", cv2.IMREAD_COLOR)
        depth = read_image(args.geometry_dir / f"{stem}_depth.tiff", cv2.IMREAD_UNCHANGED)
        pose = np.loadtxt(args.geometry_dir / f"{stem}_pose.txt", dtype=np.float64)

        assignments = assign_computer_components(
            instance_map, depth, pose, intrinsics
        )
        reviewed_overrides = apply_reviewed_computer_overrides(stem, instance_map)
        deletions: list[dict[str, object]] = []
        deletion_rules = [(15, "false bag annotation on ceiling/light")]
        if stem == "002670":
            deletion_rules.append(
                (1, "false bed annotation; no bed visible in frame 002670")
            )
        for instance_id, reason in deletion_rules:
            pixels = int(np.sum(instance_map == instance_id))
            if pixels:
                instance_map[instance_map == instance_id] = 0
                deletions.append(
                    {"instance_id": instance_id, "pixel_count": pixels, "reason": reason}
                )

        instances = metadata_for_frame(instance_map, frame["instances"])
        map_path = output_maps / f"{stem}_instances.png"
        preview_path = output_previews / f"{stem}_instances_overlay.png"
        if not cv2.imwrite(str(map_path), instance_map):
            raise OSError(map_path)
        if not cv2.imwrite(str(preview_path), render_preview(rgb, instance_map, instances)):
            raise OSError(preview_path)
        frame["instance_map"] = str(map_path.resolve())
        frame["preview"] = str(preview_path.resolve())
        frame["instances"] = instances
        frame["instance_count"] = len(instances)
        audit.append(
            {
                "frame": stem,
                "computer_component_assignments": assignments,
                "reviewed_computer_overrides": reviewed_overrides,
                "deleted_false_instances": deletions,
            }
        )

    manifest["schema"] = "semantic_instance_prototype/v1-physical-corrected"
    manifest["status"] = "corrected_candidate_not_connected_to_khronos"
    manifest["physical_corrections"] = {
        "computer_instances": {
            "semantic_id": COMPUTER_ID,
            "instance_ids": [7, 8, 9, 20],
            "method": "component_world_centroid_with_projected_room_center_fallback",
        },
        "removed_false_global_instance_ids": [15],
        "removed_false_frame_instances": [{"frame": "002670", "instance_id": 1}],
        "audit_file": str((args.output_dir / "physical_corrections.json").resolve()),
    }
    update_manifest_summary(manifest)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "physical_corrections.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(manifest['frames'])} corrected frames to {args.output_dir}")


if __name__ == "__main__":
    main()
