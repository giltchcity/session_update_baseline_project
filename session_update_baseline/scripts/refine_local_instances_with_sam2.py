#!/usr/bin/env python3
"""Split locally merged physical instances with SAM2 video propagation.

This is deliberately a local repair tool.  It reads an existing full instance
sequence, prompts SAM2 with reviewed instance masks on one seed frame, and
writes a separate corrected interval.  Source maps are never overwritten.
Predictions are clipped to the selected semantic class (or the same physical
IDs in the source map), so the video model cannot spill onto unrelated pixels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage

from sam2.build_sam import build_sam2_video_predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--semantic-dir", type=Path, required=True)
    parser.add_argument("--source-instance-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-config",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
    )
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--seed-frame", type=int, required=True)
    parser.add_argument("--semantic-id", type=int, required=True)
    parser.add_argument("--instance-ids", type=int, nargs="+", required=True)
    parser.add_argument("--replaceable-instance-ids", type=int, nargs="+", default=[])
    parser.add_argument(
        "--support-instance-ids",
        type=int,
        nargs="+",
        default=[],
        help="Restrict predictions to these source-map IDs instead of every pixel in the semantic class.",
    )
    parser.add_argument(
        "--seed-box-mask",
        type=int,
        nargs=6,
        action="append",
        metavar=("DEST_ID", "SOURCE_ID", "X0", "Y0", "X1", "Y1"),
        default=[],
        help="Override a seed mask with SOURCE_ID pixels inside a reviewed box; repeat for each destination ID.",
    )
    parser.add_argument(
        "--seed-box",
        type=int,
        nargs=5,
        action="append",
        metavar=("DEST_ID", "X0", "Y0", "X1", "Y1"),
        default=[],
        help="Prompt DEST_ID directly from a reviewed RGB bounding box; repeat for each destination ID.",
    )
    parser.add_argument(
        "--fill-support-nearest",
        action="store_true",
        help="Fill reviewed source support with its nearest propagated instance label.",
    )
    parser.add_argument("--preview-frames", type=int, nargs="*", default=[])
    return parser.parse_args()


def require_image(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(path)
    return image


def color_for_instance(instance_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(instance_id * 104729)
    return tuple(int(value) for value in rng.integers(55, 256, size=3))


def preview(rgb: np.ndarray, instance_map: np.ndarray) -> np.ndarray:
    painted = rgb.copy()
    for instance_id in (int(value) for value in np.unique(instance_map) if value):
        painted[instance_map == instance_id] = color_for_instance(instance_id)
    result = cv2.addWeighted(rgb, 0.55, painted, 0.45, 0.0)
    for instance_id in (int(value) for value in np.unique(instance_map) if value):
        mask = instance_map == instance_id
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        color = color_for_instance(instance_id)
        cv2.rectangle(
            result,
            (int(xs.min()), int(ys.min())),
            (int(xs.max()), int(ys.max())),
            color,
            3,
        )
        cv2.putText(
            result,
            f"I{instance_id}",
            (int(xs.min()), max(30, int(ys.min()) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SAM2 local refinement requires a visible CUDA device")
    if not (args.start_frame <= args.seed_frame <= args.end_frame):
        raise ValueError("seed frame must lie inside the selected interval")

    frame_numbers = list(range(args.start_frame, args.end_frame + 1))
    frame_dir = args.output_dir / "sam2_input_frames"
    output_maps = args.output_dir / "instance_maps"
    output_previews = args.output_dir / "previews"
    frame_dir.mkdir(parents=True, exist_ok=True)
    output_maps.mkdir(parents=True, exist_ok=True)
    output_previews.mkdir(parents=True, exist_ok=True)

    # SAM2 accepts a numerically named JPEG directory.  PIL detects image
    # contents, so lightweight .jpg symlinks may point to the lossless PNG RGB.
    for index, frame in enumerate(frame_numbers):
        target = args.rgb_dir / f"{frame:06d}_color.png"
        if not target.is_file():
            raise FileNotFoundError(target)
        link = frame_dir / f"{index:06d}.jpg"
        if not link.exists():
            link.symlink_to(target.resolve())

    predictor = build_sam2_video_predictor(
        args.model_config,
        str(args.checkpoint),
        device="cuda",
        apply_postprocessing=False,
    )
    state = predictor.init_state(
        str(frame_dir),
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=False,
    )
    seed_index = args.seed_frame - args.start_frame
    seed_map = require_image(
        args.source_instance_dir / f"{args.seed_frame:06d}_instances.png"
    )
    box_masks: dict[int, np.ndarray] = {}
    for destination_id, source_id, x0, y0, x1, y1 in args.seed_box_mask:
        if destination_id in box_masks:
            raise ValueError(f"duplicate seed box for I{destination_id}")
        if not (0 <= x0 < x1 <= seed_map.shape[1] and 0 <= y0 < y1 <= seed_map.shape[0]):
            raise ValueError(f"invalid seed box for I{destination_id}: {(x0, y0, x1, y1)}")
        mask = seed_map == source_id
        box = np.zeros_like(mask)
        box[y0:y1, x0:x1] = True
        box_masks[destination_id] = mask & box
    prompt_boxes: dict[int, np.ndarray] = {}
    for destination_id, x0, y0, x1, y1 in args.seed_box:
        if destination_id in prompt_boxes or destination_id in box_masks:
            raise ValueError(f"duplicate seed prompt for I{destination_id}")
        if not (0 <= x0 < x1 <= seed_map.shape[1] and 0 <= y0 < y1 <= seed_map.shape[0]):
            raise ValueError(f"invalid seed box for I{destination_id}: {(x0, y0, x1, y1)}")
        prompt_boxes[destination_id] = np.asarray([x0, y0, x1, y1], dtype=np.float32)
    for instance_id in args.instance_ids:
        if instance_id in prompt_boxes:
            predictor.add_new_points_or_box(
                state,
                seed_index,
                instance_id,
                box=prompt_boxes[instance_id],
            )
            continue
        mask = box_masks.get(instance_id, seed_map == instance_id)
        if int(mask.sum()) == 0:
            raise ValueError(f"I{instance_id} is absent from seed {args.seed_frame:06d}")
        predictor.add_new_mask(state, seed_index, instance_id, mask)

    predictions: dict[int, tuple[list[int], np.ndarray]] = {}

    def collect(reverse: bool, maximum: int) -> None:
        for index, object_ids, logits in predictor.propagate_in_video(
            state,
            start_frame_idx=seed_index,
            max_frame_num_to_track=maximum,
            reverse=reverse,
        ):
            scores = logits[:, 0].float().cpu().numpy()
            predictions[int(index)] = ([int(value) for value in object_ids], scores)

    collect(False, args.end_frame - args.seed_frame + 1)
    collect(True, args.seed_frame - args.start_frame + 1)

    preview_set = set(args.preview_frames)
    replaceable = set(args.replaceable_instance_ids) | set(args.instance_ids)
    audit = []
    for index, frame in enumerate(frame_numbers):
        object_ids, scores = predictions[index]
        source_path = args.source_instance_dir / f"{frame:06d}_instances.png"
        original = require_image(source_path)
        semantic = require_image(args.semantic_dir / f"{frame:06d}_segmentation.png")
        if args.support_instance_ids:
            allowed = np.isin(original, args.support_instance_ids)
        else:
            allowed = (semantic == args.semantic_id) | np.isin(original, sorted(replaceable))
        winner = np.argmax(scores, axis=0)
        confidence = np.max(scores, axis=0)
        assigned = allowed & (confidence > 0.0)

        corrected = original.copy()
        # Clear only the reviewed identities; another physical object sharing
        # the same semantic class remains untouched unless SAM2 claims its pixels.
        corrected[np.isin(corrected, args.instance_ids)] = 0
        corrected[assigned & np.isin(original, sorted(replaceable))] = 0
        for object_index, instance_id in enumerate(object_ids):
            corrected[assigned & (winner == object_index)] = instance_id

        if args.fill_support_nearest and args.support_instance_ids:
            support = np.isin(original, args.support_instance_ids)
            propagated = support & np.isin(corrected, args.instance_ids)
            if np.any(propagated):
                _, indices = ndimage.distance_transform_edt(
                    ~propagated, return_indices=True
                )
                nearest = corrected[indices[0], indices[1]]
                corrected[support] = nearest[support]
            else:
                # At a fully occluded interval boundary, retaining an existing
                # canonical source ID is safer than inventing one.
                canonical = support & np.isin(original, args.instance_ids)
                corrected[canonical] = original[canonical]

        path = output_maps / f"{frame:06d}_instances.png"
        if not cv2.imwrite(str(path), corrected, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise OSError(path)
        counts = {str(i): int(np.sum(corrected == i)) for i in args.instance_ids}
        audit.append({"frame": frame, "instance_pixel_counts": counts})
        if frame in preview_set:
            rgb = require_image(args.rgb_dir / f"{frame:06d}_color.png", cv2.IMREAD_COLOR)
            image = preview(rgb, corrected)
            preview_path = output_previews / f"{frame:06d}_instances_overlay.jpg"
            if not cv2.imwrite(str(preview_path), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise OSError(preview_path)

    manifest = {
        "schema": "sam2_local_physical_instance_refinement/v1",
        "status": "local_review_candidate_not_connected_to_khronos",
        "cuda_device": torch.cuda.get_device_name(0),
        "model_config": args.model_config,
        "checkpoint": str(args.checkpoint.resolve()),
        "source_instance_dir": str(args.source_instance_dir.resolve()),
        "semantic_dir": str(args.semantic_dir.resolve()),
        "frame_range": [args.start_frame, args.end_frame],
        "seed_frame": args.seed_frame,
        "semantic_id": args.semantic_id,
        "instance_ids": args.instance_ids,
        "replaceable_instance_ids": sorted(replaceable),
        "support_instance_ids": args.support_instance_ids,
        "seed_box_masks": args.seed_box_mask,
        "seed_boxes": args.seed_box,
        "fill_support_nearest": args.fill_support_nearest,
        "frame_count": len(frame_numbers),
        "audit": audit,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: manifest[key] for key in ("cuda_device", "frame_range", "seed_frame", "frame_count")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
