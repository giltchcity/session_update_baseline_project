#!/usr/bin/env python3
"""Generate a local instance-map override that fills I7 (white monitor) gaps.

The reviewed A/B instance maps are built by build_reviewed_ab_physical_instances.py
layering: source -> local SAM2 overrides -> room-wide ID remap.  The source tracker
never labeled the monitor in some windows where the S74 semantic channel still sees
it (the pixels under S74 are background id 0 in the source).  The review layer only
remaps IDs, it cannot invent pixels, so those frames stayed without I7.

This script creates a NEW override directory (same layout as the existing
local_sam2_refinement_* dirs) for exactly the frames listed as CONFIRMED_I7_GAPS.
For each such frame it paints pixels whose semantic label is S74 (74) as physical
id 7, but only where the pre-review source map currently has no object pixel of a
different entity (source id 0), so it never steals pixels from I9 or other entities.

Default is --dry-run (print per-frame painted-pixel counts, write nothing).
Run with --write to materialize the override, then register its dir name in
OVERRIDE_NAMES["session_b"] of build_reviewed_ab_physical_instances.py and
re-run that script with --update-existing so only affected frames are rewritten.

Sources of truth used here are the SAME inputs as the reviewed build:
  --semantic-root   datasets/local_ab/semantics/session_b
  --source-root     runs/full_ab_instance_maps_20260813 (pre-review source)
  --output-root     runs/<new override name>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# Frames where the semantic channel sees the white monitor (S74) with no I7 in
# the reviewed map and no I9/other entity at those pixels in the source.
# Verified 2026-08-14: S74 cluster grows from the left image edge 4301 -> 88099
# px on frames 1968..1976; the reviewed I7 run starts at frame 1977 at the same
# left-edge location (97943 px) -- i.e. the reviewer/source started one frame late.
CONFIRMED_I7_GAPS = list(range(1968, 1977))

# Candidate frames needing visual confirmation (see runs/I7_inspect/annotation_candidates/):
#   1177-1199  mid-screen S74 cluster ~40x30 px, laptop not visible in those frames
#   1642       right-top sliver right after the deliberate 1640/1641 semantic burst zeroing
#   3406       bottom band 420x150 px, 46 frames before I7 re-enters at 3452
#   132        single-frame S74 in the middle of the 131/133 flicker
# Uncomment below only after looking at the crops.
# CONFIRMED_I7_GAPS += list(range(1177, 1200))
# CONFIRMED_I7_GAPS += [3406, 132, 1642]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--write", action="store_true", help="actually write the override maps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_maps = args.output_root / "instance_maps"
    out_maps.mkdir(parents=True, exist_ok=True)

    total = 0
    for frame in sorted(CONFIRMED_I7_GAPS):
        sem = cv2.imread(str(args.semantic_root / f"{frame:06d}_segmentation.png"), cv2.IMREAD_UNCHANGED)
        src = cv2.imread(str(args.source_root / f"{frame:06d}_instances.png"), cv2.IMREAD_UNCHANGED)
        if sem is None or src is None:
            raise FileNotFoundError(f"frame {frame}: semantic/source map missing")
        if sem.shape != src.shape:
            raise ValueError(f"frame {frame}: shape mismatch {sem.shape} vs {src.shape}")

        # S74 pixels that carry no other physical entity in the pre-review source.
        paintable = (sem == 74) & (src == 0)
        n = int(paintable.sum())
        total += n
        if n == 0:
            print(f"frame {frame:6d}: nothing paintable, skipped")
            continue
        print(f"frame {frame:6d}: painting {n} px as I7")
        if args.write:
            fixed = src.copy()
            # Paint in SOURCE-ID space: build_reviewed_ab_physical_instances.py
            # sends override maps through destination_for_b(), where source 20
            # is the monitor tracklet and remaps to final I7.  Painting final
            # id 7 directly would be remapped by the source_id==7 -> 9 rule.
            fixed[paintable] = 20
            cv2.imwrite(str(out_maps / f"{frame:06d}_instances.png"), fixed)

    print(f"total paintable pixels: {total}")
    if not args.write:
        print("dry-run: nothing written. Re-run with --write to materialize.")
    else:
        print(f"override maps written to {out_maps}")
        print(
            "next: add this dir name to OVERRIDE_NAMES['session_b'] in "
            "build_reviewed_ab_physical_instances.py and re-run it with --update-existing"
        )


if __name__ == "__main__":
    main()
