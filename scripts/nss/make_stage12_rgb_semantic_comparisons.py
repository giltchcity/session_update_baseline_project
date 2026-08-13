#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
CELL_WIDTH = OUTPUT_WIDTH // 2
CELL_HEIGHT = OUTPUT_HEIGHT // 2


def colorize(labels: np.ndarray) -> np.ndarray:
    ids = np.arange(256, dtype=np.uint32)
    palette = np.stack(
        (
            (ids * 37 + 53) % 255,
            (ids * 67 + 97) % 255,
            (ids * 101 + 193) % 255,
        ),
        axis=1,
    ).astype(np.uint8)
    palette[0] = [110, 110, 110]
    return palette[labels]


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def crop_16_9(image: Image.Image) -> Image.Image:
    target_ratio = 16.0 / 9.0
    width, height = image.size
    current_ratio = width / height
    if current_ratio < target_ratio:
        crop_height = int(round(width / target_ratio))
        top = max(0, (height - crop_height) // 2)
        return image.crop((0, top, width, top + crop_height))
    crop_width = int(round(height * target_ratio))
    left = max(0, (width - crop_width) // 2)
    return image.crop((left, 0, left + crop_width, height))


def make_cell(image: Image.Image, title: str, accent: tuple[int, int, int]) -> Image.Image:
    image = crop_16_9(image.convert("RGB")).resize((CELL_WIDTH, CELL_HEIGHT), Image.Resampling.LANCZOS)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, 0, CELL_WIDTH, 58), fill=(0, 0, 0, 190))
    draw.rectangle((0, 0, 10, 58), fill=accent + (255,))
    draw.text((28, 12), title, font=font(30, bold=True), fill=(255, 255, 255, 255))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def load_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def select_frames(
    rows: list[dict[str, str]],
    count: int,
    selection_mode: str,
) -> list[dict[str, str]]:
    selected = []
    segment_edges = np.linspace(0, len(rows), count + 1, dtype=int)
    for start, end in zip(segment_edges[:-1], segment_edges[1:]):
        candidates = rows[start:end]
        candidates = [
            row
            for row in candidates
            if float(row["a_valid_fraction"]) >= 0.55
            and float(row["b_valid_fraction"]) >= 0.55
        ]
        if not candidates:
            candidates = rows[start:end]

        def change_score(row: dict[str, str]) -> float:
            return (
                float(row["changed_shared_fraction_0.2m"])
                * float(row["both_valid_fraction"])
                + float(row["appeared_fraction"])
                + float(row["disappeared_fraction"])
            )

        def stable_score(row: dict[str, str]) -> float:
            return (
                float(row["shared_abs_depth_median_m"])
                + 2.0 * float(row["appeared_fraction"])
                + 2.0 * float(row["disappeared_fraction"])
                - 0.5 * float(row["both_valid_fraction"])
            )

        if selection_mode == "changed":
            selected.append(max(candidates, key=change_score))
        else:
            selected.append(min(candidates, key=stable_score))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--change-metrics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--selection-mode", choices=("changed", "stable"), default="changed")
    args = parser.parse_args()

    rows = load_metrics(args.change_metrics)
    selected = select_frames(rows, args.count, args.selection_mode)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for output_index, row in enumerate(selected, start=1):
        image_id = row["image_id"]
        rgb_a = Image.open(args.run_a / f"{image_id}_color.png").convert("RGB")
        rgb_b = Image.open(args.run_b / f"{image_id}_color.png").convert("RGB")
        labels_a = np.asarray(
            Image.open(args.run_a / f"{image_id}_segmentation.png"),
            dtype=np.uint8,
        )
        labels_b = np.asarray(
            Image.open(args.run_b / f"{image_id}_segmentation.png"),
            dtype=np.uint8,
        )
        semantic_a = Image.fromarray(colorize(labels_a), mode="RGB")
        semantic_b = Image.fromarray(colorize(labels_b), mode="RGB")

        canvas = Image.new("RGB", (OUTPUT_WIDTH, OUTPUT_HEIGHT), (12, 14, 17))
        cells = (
            (0, 0, make_cell(rgb_a, f"STAGE 1  |  RGB  |  frame {image_id}", (215, 25, 28))),
            (
                0,
                CELL_HEIGHT,
                make_cell(semantic_a, f"STAGE 1  |  SEMANTIC  |  frame {image_id}", (215, 25, 28)),
            ),
            (
                CELL_WIDTH,
                0,
                make_cell(rgb_b, f"STAGE 2  |  RGB  |  same pose {image_id}", (0, 166, 202)),
            ),
            (
                CELL_WIDTH,
                CELL_HEIGHT,
                make_cell(
                    semantic_b,
                    f"STAGE 2  |  SEMANTIC  |  same pose {image_id}",
                    (0, 166, 202),
                ),
            ),
        )
        for x, y, cell in cells:
            canvas.paste(cell, (x, y))

        draw = ImageDraw.Draw(canvas)
        draw.rectangle((CELL_WIDTH - 2, 0, CELL_WIDTH + 2, OUTPUT_HEIGHT), fill=(255, 255, 255))
        draw.rectangle((0, CELL_HEIGHT - 2, OUTPUT_WIDTH, CELL_HEIGHT + 2), fill=(255, 255, 255))
        output_path = args.out_dir / f"stage12_same_pose_{output_index:02d}_frame_{image_id}.png"
        canvas.save(output_path, optimize=True)
        manifest.append(
            {
                "output": str(output_path),
                "image_id": image_id,
                "frame_index": int(row["frame_index"]),
                "appeared_fraction": float(row["appeared_fraction"]),
                "disappeared_fraction": float(row["disappeared_fraction"]),
                "changed_shared_fraction_0.2m": float(row["changed_shared_fraction_0.2m"]),
                "selection_mode": args.selection_mode,
            }
        )
        print(f"WROTE {output_path}")

    (args.out_dir / "selection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"STAGE12_COMPARISONS_READY count={len(manifest)} out={args.out_dir}")


if __name__ == "__main__":
    main()
