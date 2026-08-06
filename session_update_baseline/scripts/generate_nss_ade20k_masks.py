#!/usr/bin/env python3
"""Generate ADE20K semantic masks for an NSS virtual-flat run."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--frame-limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def prepare_run(input_run: Path, output_run: Path) -> list[Path]:
    output_run.mkdir(parents=True, exist_ok=True)
    color_files = sorted(input_run.glob("*_color.png"))
    if not color_files:
        raise RuntimeError(f"No color frames found in {input_run}")

    for color_file in color_files:
        image_id = color_file.name[: -len("_color.png")]
        for suffix in ("_color.png", "_depth.tiff", "_pose.txt"):
            source = input_run / f"{image_id}{suffix}"
            if not source.exists():
                raise RuntimeError(f"Missing frame input: {source}")
            link_or_copy(source, output_run / source.name)

    timestamps = input_run / "timestamps.csv"
    if not timestamps.exists():
        raise RuntimeError(f"Missing {timestamps}")
    link_or_copy(timestamps, output_run / timestamps.name)

    intrinsics_candidates = [input_run / "Intrinsics.txt", input_run.parent / "Intrinsics.txt"]
    intrinsics = next((path for path in intrinsics_candidates if path.exists()), None)
    if intrinsics is None:
        raise RuntimeError(f"No Intrinsics.txt found for {input_run}")
    link_or_copy(intrinsics, output_run.parent / "Intrinsics.txt")
    return color_files


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    color_files = prepare_run(args.input_run, args.output_run)
    if args.frame_limit > 0:
        color_files = color_files[: args.frame_limit]

    processor = SegformerFeatureExtractor.from_pretrained(
        str(args.model_dir), local_files_only=True
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        str(args.model_dir), local_files_only=True
    ).eval()

    total_histogram: Counter[int] = Counter()
    frame_rows = []
    for frame_index, color_file in enumerate(color_files):
        image_id = color_file.name[: -len("_color.png")]
        output_mask = args.output_run / f"{image_id}_segmentation.png"
        if output_mask.exists() and not args.overwrite:
            segmentation = np.asarray(Image.open(output_mask), dtype=np.uint8)
        else:
            image = Image.open(color_file).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits
            logits = torch.nn.functional.interpolate(
                logits,
                size=(image.height, image.width),
                mode="bilinear",
                align_corners=False,
            )
            segmentation = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            Image.fromarray(segmentation, mode="L").save(output_mask)

        labels, counts = np.unique(segmentation, return_counts=True)
        frame_histogram = {int(label): int(count) for label, count in zip(labels, counts)}
        total_histogram.update(frame_histogram)
        frame_rows.append(
            {
                "image_id": image_id,
                "num_labels": len(frame_histogram),
                "dominant_label": max(frame_histogram, key=frame_histogram.get),
                "dominant_fraction": max(frame_histogram.values()) / segmentation.size,
            }
        )
        print(
            f"SEMANTIC_FRAME {frame_index + 1}/{len(color_files)} "
            f"id={image_id} labels={len(frame_histogram)}"
        )

    label_names = {int(key): value for key, value in model.config.id2label.items()}
    histogram_rows = [
        {
            "label": label,
            "name": label_names.get(label, "unknown"),
            "pixels": pixels,
            "fraction": pixels / sum(total_histogram.values()),
        }
        for label, pixels in total_histogram.most_common()
    ]
    with (args.output_run / "semantic_histogram.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("label", "name", "pixels", "fraction"))
        writer.writeheader()
        writer.writerows(histogram_rows)

    manifest = {
        "input_run": str(args.input_run.resolve()),
        "output_run": str(args.output_run.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "model_architecture": model.config.architectures,
        "label_space": "ADE20K (zero-indexed, 150 classes)",
        "frames_processed": len(frame_rows),
        "frames": frame_rows,
    }
    (args.output_run / "semantic_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"SEMANTIC_COMPLETE frames={len(frame_rows)} output={args.output_run}")


if __name__ == "__main__":
    main()
