#!/usr/bin/env python3
"""Finish a reviewed local bag split without changing the source sequence.

This small post-process is intentionally explicit: in the reviewed A interval,
I12 and the upper I14 component are the same wardrobe-top backpack (canonical
I14), while lower I14 pixels belong to the two SAM2-tracked floor bags I16/I17.
Residual floor pixels are assigned to the nearest propagated I16/I17 mask.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--semantic-dir", type=Path, required=True)
    parser.add_argument("--source-instance-dir", type=Path, required=True)
    parser.add_argument("--refined-instance-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--semantic-id", type=int, default=115)
    parser.add_argument("--canonical-top-id", type=int, default=14)
    parser.add_argument("--legacy-top-id", type=int, default=12)
    parser.add_argument("--floor-instance-ids", type=int, nargs=2, default=[16, 17])
    parser.add_argument("--floor-y", type=int, default=400)
    parser.add_argument("--preview-frames", type=int, nargs="*", default=[])
    return parser.parse_args()


def read_image(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(path)
    return image


def color_for(instance_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(instance_id * 104729)
    return tuple(int(value) for value in rng.integers(55, 256, size=3))


def selected_overlay(
    rgb: np.ndarray, instance_map: np.ndarray, selected_ids: set[int]
) -> np.ndarray:
    result = rgb.copy()
    painted = rgb.copy()
    for instance_id in sorted(selected_ids):
        mask = instance_map == instance_id
        if not np.any(mask):
            continue
        color = color_for(instance_id)
        painted[mask] = color
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        if count <= 1:
            continue
        component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, width, height, _ = stats[component]
        cv2.rectangle(result, (x, y), (x + width - 1, y + height - 1), color, 3)
        cv2.putText(
            result,
            f"I{instance_id}",
            (x, max(28, y - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
    blended = cv2.addWeighted(result, 0.55, painted, 0.45, 0.0)
    return blended


def main() -> None:
    args = parse_args()
    maps_dir = args.output_dir / "instance_maps"
    previews_dir = args.output_dir / "previews"
    maps_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    floor_ids = tuple(int(value) for value in args.floor_instance_ids)
    reviewed_ids = {
        args.legacy_top_id,
        args.canonical_top_id,
        floor_ids[0],
        floor_ids[1],
    }
    preview_frames = set(args.preview_frames)
    audit: list[dict[str, object]] = []

    for frame in range(args.start_frame, args.end_frame + 1):
        filename = f"{frame:06d}_instances.png"
        original = read_image(args.source_instance_dir / filename)
        refined = read_image(args.refined_instance_dir / filename)
        semantic = read_image(
            args.semantic_dir / f"{frame:06d}_segmentation.png"
        )
        if original.shape != refined.shape or original.shape != semantic.shape:
            raise ValueError(f"shape mismatch at frame {frame:06d}")

        corrected = refined.copy()
        height, _ = corrected.shape
        lower = np.arange(height)[:, None] >= args.floor_y

        # The top object was inconsistently called apparel I12 or bag I14.
        # It is one physical backpack, so canonicalize both upper masks to I14.
        corrected[(original == args.legacy_top_id) & ~lower] = args.canonical_top_id
        corrected[(corrected == args.legacy_top_id) & ~lower] = args.canonical_top_id
        corrected[(corrected == args.canonical_top_id) & lower] = 0
        corrected[(corrected == args.legacy_top_id) & lower] = 0

        propagated_floor = np.isin(refined, floor_ids) & lower
        source_floor = np.isin(
            original,
            [args.canonical_top_id, floor_ids[0], floor_ids[1]],
        ) & lower
        # Only fill reviewed pixels.  In particular, I13 (the colourful bag)
        # is excluded even though it shares semantic class 115.
        floor_support = (
            (source_floor | propagated_floor)
            & ((semantic == args.semantic_id) | source_floor)
            & (original != 13)
        )
        corrected[floor_support] = 0

        if np.any(propagated_floor):
            _, indices = ndimage.distance_transform_edt(
                ~propagated_floor, return_indices=True
            )
            nearest = refined[indices[0], indices[1]]
            corrected[floor_support] = nearest[floor_support]
        else:
            # SAM2 can legitimately lose an almost fully occluded object near
            # the interval boundary.  Preserve an already canonical I16/I17
            # source label there; never resurrect the erroneous lower I14.
            canonical_source = np.isin(original, floor_ids) & lower
            corrected[canonical_source] = original[canonical_source]

        must_assign = np.isin(original, floor_ids) & lower
        invalid = must_assign & ~np.isin(corrected, floor_ids)
        if np.any(invalid):
            raise RuntimeError(
                f"{frame:06d}: {int(invalid.sum())} reviewed floor pixels unassigned"
            )

        output_path = maps_dir / filename
        if not cv2.imwrite(
            str(output_path), corrected, [cv2.IMWRITE_PNG_COMPRESSION, 3]
        ):
            raise OSError(output_path)

        counts = {
            str(instance_id): int(np.sum(corrected == instance_id))
            for instance_id in sorted(reviewed_ids)
        }
        audit.append({"frame": frame, "instance_pixel_counts": counts})

        if frame in preview_frames:
            rgb = read_image(
                args.rgb_dir / f"{frame:06d}_color.png", cv2.IMREAD_COLOR
            )
            before = selected_overlay(rgb, original, reviewed_ids)
            after = selected_overlay(rgb, corrected, reviewed_ids)
            cv2.putText(
                before, "BEFORE", (25, 45), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (255, 255, 255), 3, cv2.LINE_AA
            )
            cv2.putText(
                after, "CORRECTED", (25, 45), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (255, 255, 255), 3, cv2.LINE_AA
            )
            comparison = np.hstack([before, after])
            preview_path = previews_dir / f"{frame:06d}_before_after.jpg"
            if not cv2.imwrite(
                str(preview_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 92]
            ):
                raise OSError(preview_path)

    manifest = {
        "schema": "reviewed_local_bag_cleanup/v1",
        "status": "local_review_candidate_not_connected_to_khronos",
        "source_instance_dir": str(args.source_instance_dir.resolve()),
        "refined_instance_dir": str(args.refined_instance_dir.resolve()),
        "semantic_dir": str(args.semantic_dir.resolve()),
        "frame_range": [args.start_frame, args.end_frame],
        "manual_identity_rules": {
            "upper_I12_to_I14": "same wardrobe-top physical backpack",
            "lower_I14_to_nearest_I16_or_I17": "remove merged floor-bag identity",
            "I13": "preserved colourful bag",
        },
        "floor_y": args.floor_y,
        "frame_count": args.end_frame - args.start_frame + 1,
        "audit": audit,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "frame_range": manifest["frame_range"],
                "frame_count": manifest["frame_count"],
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
