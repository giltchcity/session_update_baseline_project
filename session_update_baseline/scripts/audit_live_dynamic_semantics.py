#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


def stamp_ns(msg: Image) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def image_array(msg: Image) -> np.ndarray:
    channels = 3 if msg.encoding in ("rgb8", "bgr8") else 1
    row = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    pixels = row[:, : msg.width * channels]
    if channels == 1:
        return pixels.reshape(msg.height, msg.width)
    return pixels.reshape(msg.height, msg.width, channels)


class DynamicSemanticAudit(Node):
    def __init__(self, target_frames: int, output: str) -> None:
        super().__init__("dynamic_semantic_audit")
        self.target_frames = target_frames
        self.output = output
        self.rgb: dict[int, np.ndarray] = {}
        self.labels: dict[int, np.ndarray] = {}
        self.counts: Counter[int] = Counter()
        self.total_labels: Counter[int] = Counter()
        self.frames = 0
        self.dynamic_pixels = 0
        self.create_subscription(Image, "/nss/rgb/image_raw", self.on_rgb, 30)
        self.create_subscription(Image, "/nss/semantic/image_raw", self.on_labels, 30)
        self.create_subscription(
            Image, "/khronos_node/visualization/dynamic_image", self.on_dynamic, 30
        )

    def trim(self) -> None:
        for cache in (self.rgb, self.labels):
            while len(cache) > 90:
                cache.pop(next(iter(cache)))

    def on_rgb(self, msg: Image) -> None:
        self.rgb[stamp_ns(msg)] = image_array(msg).copy()
        self.trim()

    def on_labels(self, msg: Image) -> None:
        self.labels[stamp_ns(msg)] = image_array(msg).copy()
        self.trim()

    def on_dynamic(self, msg: Image) -> None:
        key = stamp_ns(msg)
        rgb = self.rgb.pop(key, None)
        labels = self.labels.pop(key, None)
        if rgb is None or labels is None:
            return
        rendered = image_array(msg)
        dynamic = np.any(rendered != rgb, axis=2)
        valid_labels = labels.reshape(-1)
        ids, counts = np.unique(valid_labels, return_counts=True)
        self.total_labels.update(dict(zip(ids.tolist(), counts.tolist())))
        dynamic_labels = labels[dynamic]
        ids, counts = np.unique(dynamic_labels, return_counts=True)
        self.counts.update(dict(zip(ids.tolist(), counts.tolist())))
        self.dynamic_pixels += int(dynamic.sum())
        self.frames += 1
        if self.frames >= self.target_frames:
            result = {
                "matched_frames": self.frames,
                "dynamic_pixels": self.dynamic_pixels,
                "dynamic_pixels_by_semantic_id": dict(sorted(self.counts.items())),
                "observed_pixels_by_semantic_id": dict(sorted(self.total_labels.items())),
                "dynamic_fraction_by_semantic_id": {
                    str(label): self.counts[label] / total
                    for label, total in sorted(self.total_labels.items())
                    if total
                },
            }
            with open(self.output, "w", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2)
            print(json.dumps(result, indent=2), flush=True)
            rclpy.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rclpy.init()
    node = DynamicSemanticAudit(args.frames, args.output)
    deadline = time.monotonic() + args.timeout_s
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if rclpy.ok():
            raise RuntimeError(f"timed out after {node.frames} matched frames")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
