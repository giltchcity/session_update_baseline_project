#!/usr/bin/env python3
"""Assemble reviewed A/B instance maps with one room-wide physical ID catalog.

The source sequence is never modified.  This script layers manually reviewed
SAM2 repairs over it, then applies simultaneous (non-chaining) ID corrections
for objects whose old IDs changed between videos or between anchor intervals.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


ENTITY_CATALOG = {
    1: (7, "bed", "床"),
    2: (10, "cabinet", "柜子"),
    3: (15, "table", "桌子"),
    4: (24, "shelf", "书架"),
    5: (33, "desk", "书桌"),
    6: (35, "wardrobe", "衣柜"),
    7: (74, "white desktop monitor (incl. black gaming laptop)", "白色桌面显示器(含黑色游戏本)"),
    10: (75, "single swivel chair", "唯一一把转椅"),
    11: (92, "hanging apparel", "挂着的衣物"),
    12: (115, "maroon backpack", "酒红色背包"),
    13: (115, "colorful checkered shopping bag", "彩色格纹袋"),
    14: (115, "blue backpack on wardrobe top", "衣柜顶蓝色背包"),
    16: (115, "small patterned floor bag", "地面小花纹袋"),
    17: (115, "large white floor shopping bag", "地面大白袋"),
    18: (131, "blanket", "毯子"),
    19: (139, "fan", "风扇"),
    20: (74, "silver MacBook laptop", "银色 MacBook"),
}

OVERRIDE_NAMES = {
    # 2026-08-15: the 7 local_sam2_refinement_* dirs were removed during disk
    # cleanup; raw instance_labels already separate the bag instances (verified),
    # so the SAM2 split repairs are no longer needed. Only the i7 monitor fix is
    # rebuilt and applied. Raw labels are staged as _instances.png by the caller.
    "session_a": [],
    "session_b": [
        "i7_monitor_override_1968_1976_20260814",
    ],
}

# These maps were reviewed after the catalog-normalization pass and therefore
# already use final room-wide IDs.  They must be applied after corrected_map()
# rather than sent through the legacy-ID remapper a second time.
CANONICAL_OVERRIDE_NAMES = {
    "session_a": [],
    "session_b": [],
}

PREVIEW_FRAMES = {
    "session_a": [404, 595, 1361, 1778, 2099, 2717, 3510, 3751, 3803, 3935],
    "session_b": [200, 959, 1062, 1117, 1417, 1503, 2821, 3756],
}

SWITCH_AUDIT_GROUPS = [
    {7, 20},
    {12, 13, 14, 16, 17},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--session-a-rgb", type=Path, required=True)
    parser.add_argument("--session-b-rgb", type=Path, required=True)
    parser.add_argument(
        "--update-existing",
        action="store_true",
        help="Re-audit an existing candidate and rewrite only maps changed by reviewed rules.",
    )
    return parser.parse_args()


def read_image(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(path)
    return image


def frame_number(path: Path) -> int:
    return int(path.name.split("_", 1)[0])


def in_any(frame: int, intervals: list[tuple[int, int]]) -> bool:
    return any(start <= frame <= end for start, end in intervals)


def destination_for_a(frame: int, source_id: int) -> int:
    if source_id == 15:
        return 0
    # I12 (maroon backpack) is only labeled in B.  In A the bag lies on the bed
    # and the manual annotation policy keeps it with the bed scene rather than a
    # finer I12 instance (user decision).  The old remap absorbed the stray A
    # tracklet into apparel (11) / wardrobe-top bag (14); keep that policy.
    if source_id == 12:
        return 11 if 1268 <= frame <= 1290 else 14
    if source_id == 14 and 420 <= frame <= 475:
        return 13
    if source_id == 17 and in_any(frame, [(2803, 2820), (3269, 3270)]):
        return 13
    if source_id == 17 and 3328 <= frame <= 3330:
        return 16
    if source_id == 8:
        if in_any(frame, [(1635, 1650), (2161, 2198), (3751, 3949)]):
            return 7
        if in_any(
            frame,
            [
                (450, 1112),
                (1740, 1748),
                (1964, 1980),
                (2399, 2541),
                (2686, 2708),
            ],
        ):
            return 7
        return 0
    # I9 (black gaming laptop) is merged into I7 (white desktop monitor): the two
    # sit at nearly the same spot, are hard to keep apart, and the user decided
    # one instance is enough.  The only other S74 entity in A is the MacBook,
    # source 9 -> 20 on the frames where the reviewer identified it as MacBook.
    if source_id == 9 and 1962 <= frame <= 1980:
        return 20
    if source_id == 9:
        return 7
    # This is the white monitor entering at the upper image boundary; the old
    # room-centroid assignment called it I20 until the view crossed frame 2670.
    if source_id == 20 and in_any(frame, [(2659, 2659), (2661, 2670)]):
        return 7
    return source_id


def destination_for_b(frame: int, source_id: int) -> int:
    if source_id == 15:
        return 0
    if 1640 <= frame <= 1641 and source_id in (7, 8, 9, 20):
        return 0
    if source_id == 8:
        return 0
    # I9 (black gaming laptop) is merged into I7 (white desktop monitor), so
    # every S74 tracklet lands on 7; the MacBook never appears in B.
    if source_id == 7:
        return 7
    if source_id == 20:
        return 7
    if source_id == 9:
        # 3749-4040: the white desktop monitor re-enters from the right edge at
        # 3749 (SAM2 re-identified it as a new tracklet after losing it at 3664);
        # 3D extent/color analysis (2026-08-14) confirms it is I7, not I9.
        return 7
    # B inherited a rotated bag/apparel numbering from the old calibration.
    if source_id == 12:
        return 14
    if source_id == 13:
        return 12
    if source_id == 14:
        return 13
    return source_id


def build_override_index(
    runs_root: Path, names: list[str], session: str
) -> tuple[dict[int, Path], list[str]]:
    index: dict[int, Path] = {}
    resolved: list[str] = []
    for name in names:
        root = runs_root / name
        manifest = root / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        resolved.append(str(root.resolve()))
        for path in (root / "instance_maps").glob("*_instances.png"):
            frame = frame_number(path)
            if frame in index:
                raise ValueError(f"overlapping local overrides for {session} frame {frame}")
            index[frame] = path
    return index, resolved


def corrected_map(session: str, frame: int, source: np.ndarray) -> tuple[np.ndarray, Counter]:
    destination = destination_for_a if session == "session_a" else destination_for_b
    result = source.copy()
    changes: Counter = Counter()
    # Read every mask from the untouched snapshot so swaps such as B I13->I12,
    # I14->I13 cannot cascade through one another.
    for source_id in (int(value) for value in np.unique(source) if value):
        destination_id = destination(frame, source_id)
        if destination_id == source_id:
            continue
        mask = source == source_id
        pixels = int(mask.sum())
        result[mask] = destination_id
        changes[(source_id, destination_id)] += pixels
    return result, changes


def color_for_instance(instance_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(instance_id * 104729)
    return tuple(int(value) for value in rng.integers(55, 256, size=3))


def render_preview(rgb: np.ndarray, instance_map: np.ndarray, title: str) -> np.ndarray:
    painted = rgb.copy()
    for instance_id in (int(value) for value in np.unique(instance_map) if value):
        painted[instance_map == instance_id] = color_for_instance(instance_id)
    image = cv2.addWeighted(rgb, 0.58, painted, 0.42, 0.0)
    for instance_id in (int(value) for value in np.unique(instance_map) if value):
        mask = instance_map == instance_id
        ys, xs = np.nonzero(mask)
        color = color_for_instance(instance_id)
        cv2.rectangle(
            image,
            (int(xs.min()), int(ys.min())),
            (int(xs.max()), int(ys.max())),
            color,
            3,
        )
        label = f"I{instance_id} {ENTITY_CATALOG[instance_id][1]}"
        cv2.putText(
            image,
            label,
            (int(xs.min()), max(30, int(ys.min()) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.rectangle(image, (0, 0), (image.shape[1], 45), (20, 20, 20), -1)
    cv2.putText(image, title, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return image


def build_contact_sheet(images: list[np.ndarray], output_path: Path) -> None:
    thumb_width, thumb_height = 640, 360
    thumbs = [cv2.resize(image, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA) for image in images]
    rows = []
    for start in range(0, len(thumbs), 2):
        pair = thumbs[start : start + 2]
        if len(pair) == 1:
            pair.append(np.zeros_like(pair[0]))
        rows.append(np.hstack(pair))
    sheet = np.vstack(rows)
    if not cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(output_path)


def process_session(
    args: argparse.Namespace,
    session: str,
    rgb_root: Path,
) -> dict[str, object]:
    source_dir = args.source_root / session / "instance_maps"
    source_paths = sorted(source_dir.glob("*_instances.png"))
    if not source_paths:
        raise FileNotFoundError(source_dir)
    expected = list(range(len(source_paths)))
    actual = [frame_number(path) for path in source_paths]
    if actual != expected:
        raise ValueError(f"{session}: expected contiguous frames 0..{len(source_paths) - 1}")

    output_dir = args.output_root / session / "instance_maps"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.update_existing:
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = args.output_root / "previews" / session
    preview_dir.mkdir(parents=True, exist_ok=True)
    override_index, override_roots = build_override_index(
        args.runs_root, OVERRIDE_NAMES[session], session
    )
    canonical_override_index, canonical_override_roots = build_override_index(
        args.runs_root, CANONICAL_OVERRIDE_NAMES[session], session
    )

    total_changes: Counter = Counter()
    active_ids: set[int] = set()
    preview_images: list[np.ndarray] = []
    updated_existing_frames: list[int] = []
    temporal_cross_id_alerts: list[dict[str, object]] = []
    previous_result: np.ndarray | None = None
    audit_path = args.output_root / session / "frame_audit.jsonl"
    with audit_path.open("w", encoding="utf-8") as audit_file:
        for source_path in source_paths:
            frame = frame_number(source_path)
            input_path = override_index.get(frame, source_path)
            source = read_image(input_path)
            result, changes = corrected_map(session, frame, source)
            if frame in canonical_override_index:
                canonical = read_image(canonical_override_index[frame])
                if canonical.shape != result.shape:
                    raise ValueError(
                        f"{session} frame {frame}: canonical override shape "
                        f"{canonical.shape} != {result.shape}"
                    )
                result = canonical
            unknown = set(int(value) for value in np.unique(result) if value) - set(ENTITY_CATALOG)
            if unknown:
                raise ValueError(f"{session} frame {frame}: unknown IDs {sorted(unknown)}")
            active_ids.update(int(value) for value in np.unique(result) if value)
            total_changes.update(changes)
            if previous_result is not None:
                for group in SWITCH_AUDIT_GROUPS:
                    for old_id in group:
                        old_area = int(np.sum(previous_result == old_id))
                        if old_area < 500:
                            continue
                        for new_id in group - {old_id}:
                            new_area = int(np.sum(result == new_id))
                            if new_area < 500:
                                continue
                            overlap = int(
                                np.sum((previous_result == old_id) & (result == new_id))
                            )
                            old_ratio = overlap / old_area
                            new_ratio = overlap / new_area
                            if overlap >= 500 and old_ratio >= 0.5 and new_ratio >= 0.5:
                                temporal_cross_id_alerts.append(
                                    {
                                        "previous_frame": frame - 1,
                                        "frame": frame,
                                        "old_id": old_id,
                                        "new_id": new_id,
                                        "overlap_pixels": overlap,
                                        "old_overlap_ratio": round(old_ratio, 6),
                                        "new_overlap_ratio": round(new_ratio, 6),
                                    }
                                )
            previous_result = result
            output_path = output_dir / f"{frame:06d}_instances.png"
            needs_write = True
            if args.update_existing and output_path.is_file():
                needs_write = not np.array_equal(read_image(output_path), result)
            if needs_write:
                if not cv2.imwrite(str(output_path), result, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                    raise OSError(output_path)
                if args.update_existing:
                    updated_existing_frames.append(frame)
            record = {
                "frame": frame,
                "input": (
                    "canonical_review_override"
                    if frame in canonical_override_index
                    else "local_review_override"
                    if frame in override_index
                    else "source"
                ),
                "relabels": [
                    {"source_id": old, "destination_id": new, "pixels": pixels}
                    for (old, new), pixels in sorted(changes.items())
                ],
            }
            audit_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            if frame in PREVIEW_FRAMES[session]:
                rgb = read_image(rgb_root / f"{frame:06d}_color.png", cv2.IMREAD_COLOR)
                preview = render_preview(rgb, result, f"{session} frame {frame:06d}")
                preview_path = preview_dir / f"{frame:06d}_reviewed_overlay.jpg"
                if not cv2.imwrite(str(preview_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    raise OSError(preview_path)
                preview_images.append(preview)

    build_contact_sheet(preview_images, args.output_root / "previews" / f"{session}_contact_sheet.jpg")
    return {
        "source_instance_dir": str(source_dir.resolve()),
        "local_override_roots": override_roots,
        "canonical_override_roots": canonical_override_roots,
        "local_override_frame_count": len(override_index),
        "canonical_override_frame_count": len(canonical_override_index),
        "frame_count": len(source_paths),
        "frame_range": [0, len(source_paths) - 1],
        "updated_existing_frames": updated_existing_frames,
        "active_instance_ids": sorted(active_ids),
        "temporal_cross_id_alerts": temporal_cross_id_alerts,
        "relabel_pixel_totals": [
            {"source_id": old, "destination_id": new, "pixels": pixels}
            for (old, new), pixels in sorted(total_changes.items())
        ],
    }


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.update_existing:
        raise FileExistsError(f"refusing to overwrite non-empty {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    sessions = {
        "session_a": process_session(args, "session_a", args.session_a_rgb),
        "session_b": process_session(args, "session_b", args.session_b_rgb),
    }
    expected_ids = sorted(ENTITY_CATALOG)
    validation = {
        "frame_counts_match_source": sessions["session_a"]["frame_count"] == 4003
        and sessions["session_b"]["frame_count"] == 4041,
        "forbidden_instance_ids_absent": all(
            instance_id not in session_data["active_instance_ids"]
            for instance_id in (8, 15)
            for session_data in sessions.values()
        ),
        "no_high_confidence_within_class_id_switches": all(
            not session_data["temporal_cross_id_alerts"]
            for session_data in sessions.values()
        ),
        "session_a_has_all_except_bed_absorbed_i12": sessions["session_a"]["active_instance_ids"]
        == [instance_id for instance_id in expected_ids if instance_id != 12],
        "session_b_has_all_except_mac_and_merged_laptop": sessions["session_b"]["active_instance_ids"]
        == [instance_id for instance_id in expected_ids if instance_id not in (9, 20)],
        "computer_ids_are_7_and_20_with_mac_only_in_a": all(
            instance_id in sessions["session_a"]["active_instance_ids"]
            for instance_id in (7, 20)
        )
        and 9 not in sessions["session_a"]["active_instance_ids"]
        and sessions["session_b"]["active_instance_ids"]
        == [instance_id for instance_id in expected_ids if instance_id not in (9, 20)],
        "catalog_instance_ids": expected_ids,
    }
    manifest = {
        "schema": "reviewed_ab_physical_instances/v1",
        "status": "complete_review_candidate_not_connected_to_khronos",
        "principle": "one physical room object has one ID across both A and B videos",
        "entity_count": len(ENTITY_CATALOG),
        "entity_catalog": [
            {
                "instance_id": instance_id,
                "semantic_id": semantic_id,
                "physical_name": name,
                "physical_name_zh": name_zh,
            }
            for instance_id, (semantic_id, name, name_zh) in ENTITY_CATALOG.items()
        ],
        "reviewed_corrections": {
            "inactive_ids_removed": [8, 15],
            "single_chair_id": 10,
            "apparel_id": 11,
            "bag_ids": [12, 13, 14, 16, 17],
            "computer_ids": [7, 20],
            "notes": [
                "I12 is the maroon backpack, not a second apparel instance.",
                "In A, the maroon bag lies on the bed and follows the manual annotation policy: it remains bed I1 rather than a finer I12 instance.",
                "I16 and I17 were locally split with reviewed SAM2 seeds where old masks merged them.",
                "B bag and computer IDs were normalized to the same physical entities used in A.",
                "2026-08-14 user decision: I9 (black gaming laptop) is merged into I7 (white desktop monitor) across A and B -- the two sit at nearly the same spot and are not reliably separable, so one instance is enough.  Remap: source 9 -> 7 in both sessions (A keeps source 9 -> 20 on MacBook frames 1962-1980), source 8 -> 7 on laptop-labeled A intervals, source 7 -> 7 in B.",
                "All relabels are simultaneous and cannot cascade through swapped IDs.",
            ],
        },
        "sessions": sessions,
        "validation": validation,
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not all(
        validation[key]
        for key in (
            "frame_counts_match_source",
            "forbidden_instance_ids_absent",
            "no_high_confidence_within_class_id_switches",
            "session_a_has_all_except_bed_absorbed_i12",
            "session_b_has_all_except_mac_and_merged_laptop",
            "computer_ids_are_7_and_20_with_mac_only_in_a",
        )
    ):
        raise RuntimeError(f"validation failed; inspect {manifest_path}")
    print(
        json.dumps(
            {
                "output_root": str(args.output_root.resolve()),
                "entity_count": len(ENTITY_CATALOG),
                "sessions": {
                    key: {"frame_count": value["frame_count"], "active_instance_ids": value["active_instance_ids"]}
                    for key, value in sessions.items()
                },
                "validation": validation,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
