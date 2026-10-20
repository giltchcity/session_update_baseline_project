"""Flat (Schmid et al., ICRA 2022: panoptic_mapping's flat_dataset) as an A/B dataset of the update layer: run1 / run2
-> FT/datasets/flat_ab/session_{a,b}. GaME evaluates on the same 762 frames (HF voviktyl/GaME_Flat: colour, depth and
pose identical frame by frame, checked 2026-10-07), without the ground-truth labels this copy has.

  python scripts/prepare_flat_ab.py [SRC] [DST]   (default FT/datasets/panoptic_mapping/flat_dataset -> FT/datasets/flat_ab)

Per session, the layout frames.py reads:
  rgbd/<id>_color.png, <id>_pose.txt   links to the original files (pose: camera-to-world, metres)
  rgbd/<id>_depth.tiff                 the depth as GaME's FlatDataset reads it (src/entities/datasets.py, 'fix window'):
                                       readings beyond 10 m are the windows; when there are fewer than 6, each takes
                                       the reading below it, otherwise every one takes its column's farthest reading
  rgbd/timestamps.csv                  ImageID,TimeStamp in ImageID order (GaME's frame order, its sorted files), the
                                       stamps relative to the session's first frame (the original lists absolute
                                       stamps, not in frame order)
  rgbd/Intrinsics.txt                  the original's values (640 x 480, f 320, principal point 320, 240)
  instance_labels/<id>_segmentation.png   the ground-truth instance of every thing (PanopticID 1), 0 elsewhere
  semantics/<id>_segmentation.png      the ground-truth class (ClassID) of every pixel
mapping_configs/label_space.yaml       object_labels: the classes of the things; no dynamic classes; the class names
session.json                           each session's run and the absolute stamp of its first frame
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

FT = Path("/home/jixian/Desktop/FT")
WINDOW_M = 10.0          # GaME FlatDataset: depth_data > 10
RUNS = {"a": "run1", "b": "run2"}


def game_depth(depth: np.ndarray) -> np.ndarray:
    """GaME's FlatDataset.__getitem__ 'fix window', line for line (metres; GaME then multiplies by its scale 10)."""
    depth = depth.copy()
    window = depth > WINDOW_M
    if window.any():
        if window.sum() < 6:
            for r, c in np.argwhere(window):
                depth[r, c] = depth[r + 1, c]
        else:
            depth[window] = 0
            depth = depth + window * depth.max(0)
    return depth.astype(np.float32)


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else FT / "datasets/panoptic_mapping/flat_dataset"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else FT / "datasets/flat_ab"
    instances = {int(r["InstanceID"]): r for r in csv.DictReader((src / "groundtruth_labels.csv").open())}
    classes = {int(r["ClassID"]): r["Name"] for r in csv.DictReader((src / "groundtruth_labels_classes.csv").open())}
    lut_instance = np.zeros(256, np.uint16)
    lut_class = np.zeros(256, np.uint16)
    for i, r in instances.items():
        lut_instance[i] = i if r["PanopticID"] == "1" else 0
        lut_class[i] = int(r["ClassID"])
    # the original's values without their units ('f_x: 320 (px)') and without its comment line
    values = dict(line.split(":", 1) for line in (src / "Intrinsics.txt").read_text().splitlines() if ":" in line)
    intrinsics = "".join(f"{k.strip()}: {float(v.split()[0]):g}\n" for k, v in values.items())
    sessions = {}
    for stage, run in RUNS.items():
        rows = sorted(csv.DictReader((src / run / "timestamps.csv").open()), key=lambda r: r["ImageID"])
        stamps = [int(r["TimeStamp"]) for r in rows]
        assert all(b > a for a, b in zip(stamps, stamps[1:])), f"{run}: stamps not increasing in ImageID order"
        out = dst / f"session_{stage}"
        rgbd, inst, sem = out / "rgbd", out / "instance_labels", out / "semantics"
        for d in (rgbd, inst, sem):
            d.mkdir(parents=True, exist_ok=True)
        (rgbd / "Intrinsics.txt").write_text(intrinsics)
        with (rgbd / "timestamps.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ImageID", "TimeStamp"])
            w.writerows([r["ImageID"], t - stamps[0]] for r, t in zip(rows, stamps))
        filled = left = 0
        for r in rows:
            fid = r["ImageID"]
            for kind in ("color.png", "pose.txt"):
                link = rgbd / f"{fid}_{kind}"
                if not link.is_symlink():
                    os.symlink(src / run / f"{fid}_{kind}", link)
            depth = np.asarray(Image.open(src / run / f"{fid}_depth.tiff"), dtype=np.float32)
            fixed = game_depth(depth)
            filled += int((depth > WINDOW_M).sum())
            left += int((fixed > WINDOW_M).sum())             # a window pixel whose lower neighbour is a window too
            Image.fromarray(fixed, mode="F").save(rgbd / f"{fid}_depth.tiff", compression="tiff_adobe_deflate")
            assert np.array_equal(np.asarray(Image.open(rgbd / f"{fid}_depth.tiff")), fixed)
            seg = np.asarray(Image.open(src / run / f"{fid}_segmentation.png"))
            assert seg.dtype == np.uint8 and np.isin(seg, list(instances)).all(), f"{run}/{fid}: unknown instance"
            Image.fromarray(lut_instance[seg]).save(inst / f"{fid}_segmentation.png")
            Image.fromarray(lut_class[seg]).save(sem / f"{fid}_segmentation.png")
        sessions[stage] = {"run": run, "frames": len(rows), "start_ns": stamps[0], "window_pixels_filled": filled,
                           "window_pixels_left": left}
        print(f"session {stage} ({run}): {len(rows)} frames, {filled} window pixels filled, {left} left", flush=True)
    things = sorted({int(r["ClassID"]) for r in instances.values() if r["PanopticID"] == "1"})
    cfg = dst / "mapping_configs"
    cfg.mkdir(exist_ok=True)
    lines = ["# Flat ground truth (groundtruth_labels.csv, groundtruth_labels_classes.csv): object classes are the",
             "# classes of the things (PanopticID 1); ceiling, floor, wall and window are stuff. No dynamic classes.",
             f"total_semantic_labels: {len(classes)}", "invalid_labels: []", "dynamic_labels: []",
             "object_labels: [" + ", ".join(map(str, things)) + "]", "label_names:"]
    lines += [f"- label: {c}\n  name: {json.dumps(n)}" for c, n in sorted(classes.items())]
    (cfg / "label_space.yaml").write_text("\n".join(lines) + "\n")
    (dst / "session.json").write_text(json.dumps({"source": str(src), **sessions}, indent=1) + "\n")


if __name__ == "__main__":
    main()
