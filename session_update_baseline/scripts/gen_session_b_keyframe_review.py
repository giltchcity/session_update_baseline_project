#!/usr/bin/env python3
"""Generate per-keyframe review images for session B with I7/I9/S74 boxes.

The A video got a full 134-keyframe manual review (manual134 workflow), but B
never did -- only sparse samples.  This script gives B the equivalent: for the
whole session every 30 frames, plus densely every 5 frames inside the known
problem window 3749-4040 (where the white monitor is labeled I9), a downscaled
RGB frame with:

  - red box + label    I7   white desktop monitor pixels (physical id 7)
  - yellow box + label I9   black gaming laptop pixels (physical id 9)
  - white box + label  S74-only  semantic S74 pixels not under I7/I9

A frame with a big white-box (S74-only) region next to a red I7 box is a gap
that needs filling; a frame where the big white monitor region is boxed yellow
(I9) is a mislabel that needs the 9->7 remap.

Output: runs/I7_inspect/session_b_keyframe_review/<frame>_review.jpg (960x540)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/usr/lib/python3/dist-packages")  # system cv2/numpy if needed

ID7_COLOR = (40, 40, 235)     # red-ish  (BGR) -- I7 white monitor
ID9_COLOR = (0, 180, 255)     # yellow  (BGR) -- I9 black laptop
S74_COLOR = (255, 255, 255)   # white   (BGR) -- S74 not under I7/I9

STEP_SWEEP = 30
DENSE_START, DENSE_END, DENSE_STEP = 3749, 4040, 5


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    return int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)


def label(canvas: np.ndarray, text: str, color: tuple[int, int, int], y: int) -> int:
    """Draw text with a black halo so it stays readable on any background."""
    cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return y + 20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-dir", type=Path, required=True)
    ap.add_argument("--instance-dir", type=Path, required=True)
    ap.add_argument("--semantic-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[int] = []
    frames += list(range(0, 4041, STEP_SWEEP))
    frames += list(range(DENSE_START, DENSE_END + 1, DENSE_STEP))
    frames = sorted(set(frames))

    for i, frame in enumerate(frames):
        fname = f"{frame:06d}"
        rgb = cv2.imread(str(args.rgb_dir / f"{fname}_color.png"))
        inst = cv2.imread(str(args.instance_dir / f"{fname}_instances.png"), cv2.IMREAD_UNCHANGED)
        sem = cv2.imread(str(args.semantic_dir / f"{fname}_segmentation.png"), cv2.IMREAD_UNCHANGED)
        if rgb is None or inst is None or sem is None:
            print(f"frame {frame}: missing input, skipped")
            continue

        canvas = cv2.resize(rgb, (960, 540), interpolation=cv2.INTER_AREA)
        scale = 0.5

        def draw(mask: np.ndarray, color, name: str, y: int) -> int:
            b = bbox(mask)
            if b is None:
                return y
            x0, y0, w, h = b
            p = int(mask.sum())
            cv2.rectangle(canvas, (int(x0 * scale), int(y0 * scale)),
                          (int((x0 + w) * scale), int((y0 + h) * scale)),
                          color, 2)
            return label(canvas, f"{name} {p}px", color, y)

        y = 20
        y = draw(inst == 7, ID7_COLOR, "I7 white-monitor", y)
        y = draw(inst == 9, ID9_COLOR, "I9 black-laptop", y)
        y = draw((sem == 74) & (inst != 7) & (inst != 9), S74_COLOR, "S74-only", y)
        cv2.putText(canvas, f"frame {frame}", (960 - 130, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, f"frame {frame}", (960 - 130, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        out = args.out_dir / f"{fname}_review.jpg"
        cv2.imwrite(str(out), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if i % 20 == 0:
            print(f"  {i}/{len(frames)}  frame {frame}")

    (args.out_dir / "00_README.txt").write_text(
        "图例:\n"
        "  红框+红字 I7   = 白色显示器 (physical id 7) 像素数\n"
        "  黄框+黄字 I9   = 黑色游戏本 (physical id 9) 像素数\n"
        "  白框+白字 S74-only = 语义 S74 但既不在 I7 也不在 I9 的像素\n"
        "问题特征:\n"
        "  A. 大片白色显示器区域被黄框(I9)框住 -> 需要 9->7 重标\n"
        "  B. 大片 S74-only 白框紧挨着 I7 -> 需要补画成 I7\n"
        "  看到问题就记下帧号告诉我,例如: '003850 显示器被标成 I9'\n",
        encoding="utf-8",
    )
    print(f"done: {len(frames)} frames -> {args.out_dir}")


if __name__ == "__main__":
    main()
