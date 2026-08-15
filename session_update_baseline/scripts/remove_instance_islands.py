#!/usr/bin/env python3
"""Remove small, spatially isolated islands from physical-instance maps."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--absolute-pixels", type=int, default=500)
    parser.add_argument("--relative-area", type=float, default=0.02)
    parser.add_argument("--minimum-gap-pixels", type=float, default=50.0)
    return parser.parse_args()


def color_for_instance(instance_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(instance_id * 104729)
    return tuple(int(value) for value in rng.integers(55, 256, size=3))


def bbox_gap(first: np.ndarray, second: np.ndarray) -> float:
    ax, ay, aw, ah = (int(value) for value in first[:4])
    bx, by, bw, bh = (int(value) for value in second[:4])
    gap_x = max(ax - (bx + bw), bx - (ax + aw), 0)
    gap_y = max(ay - (by + bh), by - (ay + ah), 0)
    return math.hypot(gap_x, gap_y)


def clean_map(
    instance_map: np.ndarray,
    absolute_pixels: int,
    relative_area: float,
    minimum_gap_pixels: float,
) -> list[dict[str, object]]:
    removals: list[dict[str, object]] = []
    for instance_id in (int(value) for value in np.unique(instance_map) if value):
        binary = (instance_map == instance_id).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        if count <= 2:
            continue
        components = sorted(
            range(1, count),
            key=lambda index: int(stats[index, cv2.CC_STAT_AREA]),
            reverse=True,
        )
        main = components[0]
        main_area = int(stats[main, cv2.CC_STAT_AREA])
        area_limit = max(absolute_pixels, relative_area * main_area)
        for component in components[1:]:
            area = int(stats[component, cv2.CC_STAT_AREA])
            gap = bbox_gap(stats[main], stats[component])
            if area >= area_limit or gap <= minimum_gap_pixels:
                continue
            bbox = [int(value) for value in stats[component, :4]]
            instance_map[(labels == component) & (instance_map == instance_id)] = 0
            removals.append(
                {
                    "instance_id": instance_id,
                    "pixel_count": area,
                    "bbox_xywh": bbox,
                    "gap_from_main_pixels": round(gap, 3),
                }
            )
    return removals


def refresh_instances(
    instance_map: np.ndarray, instances: list[dict[str, object]]
) -> list[dict[str, object]]:
    metadata = {int(item["instance_id"]): item for item in instances}
    refreshed: list[dict[str, object]] = []
    for instance_id in (int(value) for value in np.unique(instance_map) if value):
        mask = instance_map == instance_id
        ys, xs = np.nonzero(mask)
        item = dict(metadata[instance_id])
        item["pixel_count"] = int(mask.sum())
        item["bbox_xywh"] = [
            int(xs.min()),
            int(ys.min()),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        ]
        refreshed.append(item)
    return refreshed


def render_preview(
    rgb: np.ndarray, instance_map: np.ndarray, instances: list[dict[str, object]]
) -> np.ndarray:
    metadata = {int(item["instance_id"]): item for item in instances}
    overlay = rgb.copy()
    for instance_id in metadata:
        overlay[instance_map == instance_id] = color_for_instance(instance_id)
    preview = cv2.addWeighted(rgb, 0.55, overlay, 0.45, 0.0)
    for instance_id, item in metadata.items():
        x, y, width, height = (int(value) for value in item["bbox_xywh"])
        color = color_for_instance(instance_id)
        cv2.rectangle(preview, (x, y), (x + width - 1, y + height - 1), color, 3)
        text = f"I{instance_id} {item['semantic_name']} S{item['semantic_id']}"
        cv2.putText(
            preview,
            text,
            (x, max(28, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )
    return preview


def main() -> None:
    args = parse_args()
    manifest_path = args.input_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    instance_dir = args.output_dir / "instance_maps"
    preview_dir = args.output_dir / "previews"
    instance_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    audit: list[dict[str, object]] = []
    for frame in manifest["frames"]:
        stem = str(frame["frame"])
        source_map = args.input_dir / "instance_maps" / f"{stem}_instances.png"
        rgb_path = args.rgb_dir / f"{stem}_color.png"
        instance_map = cv2.imread(str(source_map), cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if instance_map is None or rgb is None:
            raise FileNotFoundError(f"Missing input for frame {stem}")

        removals = clean_map(
            instance_map,
            args.absolute_pixels,
            args.relative_area,
            args.minimum_gap_pixels,
        )
        refreshed = refresh_instances(instance_map, frame["instances"])
        output_map = instance_dir / f"{stem}_instances.png"
        output_preview = preview_dir / f"{stem}_instances_overlay.png"
        if not cv2.imwrite(str(output_map), instance_map):
            raise OSError(f"Failed to write {output_map}")
        if not cv2.imwrite(str(output_preview), render_preview(rgb, instance_map, refreshed)):
            raise OSError(f"Failed to write {output_preview}")
        frame["instance_map"] = str(output_map.resolve())
        frame["preview"] = str(output_preview.resolve())
        frame["instances"] = refreshed
        frame["instance_count"] = len(refreshed)
        if removals:
            audit.append({"frame": stem, "removed_components": removals})

    manifest["schema"] = "semantic_instance_prototype/v1-cleaned-islands"
    manifest["status"] = "cleaned_candidate_not_connected_to_khronos"
    manifest["cleanup"] = {
        "policy": "remove_only_small_components_spatially_isolated_from_the_instance_main_component",
        "absolute_pixels": args.absolute_pixels,
        "relative_area": args.relative_area,
        "minimum_gap_pixels": args.minimum_gap_pixels,
        "removed_frame_count": len(audit),
        "removed_component_count": sum(len(item["removed_components"]) for item in audit),
        "removed_pixel_count": sum(
            int(component["pixel_count"])
            for item in audit
            for component in item["removed_components"]
        ),
        "audit_file": str((args.output_dir / "removed_islands.json").resolve()),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "removed_islands.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["cleanup"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
