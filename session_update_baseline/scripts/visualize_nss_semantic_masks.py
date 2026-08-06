#!/usr/bin/env python3
"""Create full-resolution RGB/semantic/object-mask review sheets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--labelspace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", nargs="+", type=int)
    return parser.parse_args()


def colorize(labels: np.ndarray) -> np.ndarray:
    ids = np.arange(256, dtype=np.uint32)
    palette = np.stack(((ids * 37 + 53) % 255,
                        (ids * 67 + 97) % 255,
                        (ids * 101 + 193) % 255), axis=1).astype(np.uint8)
    palette[0] = [110, 110, 110]
    return palette[labels]


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.labelspace.read_text())
    object_labels = np.asarray(config["object_labels"], dtype=np.uint8)
    names = {int(row["label"]): row["name"] for row in config["label_names"]}
    color_files = sorted(args.run.glob("*_color.png"))
    if not color_files:
        raise RuntimeError(f"No RGB files in {args.run}")
    if args.frames:
        invalid = [index for index in args.frames if index < 0 or index >= len(color_files)]
        if invalid:
            raise IndexError(f"Frame indices out of range: {invalid}")
        color_files = [color_files[index] for index in args.frames]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    histogram: dict[int, int] = {}
    for color_path in color_files:
        image_id = color_path.name[:-10]
        rgb = np.asarray(Image.open(color_path).convert("RGB"))
        labels = np.asarray(Image.open(args.run / f"{image_id}_segmentation.png"), dtype=np.uint8)
        semantic = colorize(labels)
        object_mask = np.isin(labels, object_labels)
        binary = rgb.copy()
        binary[~object_mask] = (binary[~object_mask].astype(np.float32) * 0.25).astype(np.uint8)
        binary[object_mask] = (
            0.35 * binary[object_mask].astype(np.float32) + 0.65 * np.array([0, 255, 80])
        ).astype(np.uint8)
        panel = np.concatenate((rgb, semantic, binary), axis=1)
        canvas = Image.fromarray(panel)
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, panel.shape[1], 28), fill=(0, 0, 0))
        draw.text((8, 7), f"{image_id}: RGB | ADE20K prediction | broad movable candidates",
                  fill=(255, 255, 255))
        rows.append(canvas)
        labels_present, counts = np.unique(labels, return_counts=True)
        for label, count in zip(labels_present, counts):
            histogram[int(label)] = histogram.get(int(label), 0) + int(count)

    width = rows[0].width
    sheet = Image.new("RGB", (width, sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet.save(args.out)

    report = args.out.with_suffix(".csv")
    total = sum(histogram.values())
    with report.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("label", "name", "pixels", "fraction", "object_candidate"))
        for label, count in sorted(histogram.items(), key=lambda item: item[1], reverse=True):
            writer.writerow((label, names.get(label, "unknown"), count, count / total,
                             bool(label in set(int(value) for value in object_labels))))
    print(f"SEMANTIC_REVIEW {args.out}")


if __name__ == "__main__":
    main()
