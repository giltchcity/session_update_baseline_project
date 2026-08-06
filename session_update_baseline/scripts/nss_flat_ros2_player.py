#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster


def read_frames(run_dir: Path) -> list[tuple[str, int]]:
    with (run_dir / "timestamps.csv").open(newline="") as fin:
        rows = list(csv.DictReader(fin))
    frames = []
    for row in rows:
        image_id = row.get("ImageID", row.get("image_id", ""))
        stamp = row.get("TimeStamp", row.get("timestamp", ""))
        if not image_id or not stamp:
            raise ValueError("timestamps.csv requires ImageID and TimeStamp columns")
        frames.append((image_id, int(stamp)))
    if not frames:
        raise ValueError(f"no frames in {run_dir / 'timestamps.csv'}")
    return frames


def load_intrinsics(run_dir: Path) -> tuple[int, int, float, float, float, float]:
    path = run_dir / "Intrinsics.txt"
    if not path.exists():
        path = run_dir.parent / "Intrinsics.txt"
    if not path.exists():
        raise ValueError(f"Intrinsics.txt not found in {run_dir} or {run_dir.parent}")
    text = path.read_text().strip()
    if ":" in text:
        parsed = {}
        for line in text.splitlines():
            key, value = line.split(":", 1)
            parsed[key.strip()] = float(value.strip())
        width = parsed["Res_x"]
        height = parsed["Res_y"]
        fx = parsed["f_x"]
        fy = parsed["f_y"]
        cx = parsed["u"]
        cy = parsed["v"]
        return int(width), int(height), fx, fy, cx, cy
    values = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if len(values) == 6:
        width, height, fx, fy, cx, cy = values
    elif len(values) == 9:
        matrix = values.reshape(3, 3)
        fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
        sample = cv2.imread(str(run_dir / "000000_color.png"), cv2.IMREAD_COLOR)
        if sample is None:
            raise ValueError("cannot infer image size from 000000_color.png")
        height, width = sample.shape[:2]
    else:
        raise ValueError(f"unsupported Intrinsics.txt format with {len(values)} values")
    return int(width), int(height), float(fx), float(fy), float(cx), float(cy)


def transform_message(parent: str, child: str, matrix: np.ndarray, stamp) -> TransformStamped:
    msg = TransformStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = parent
    msg.child_frame_id = child
    msg.transform.translation.x = float(matrix[0, 3])
    msg.transform.translation.y = float(matrix[1, 3])
    msg.transform.translation.z = float(matrix[2, 3])
    quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    msg.transform.rotation.x = float(quat[0])
    msg.transform.rotation.y = float(quat[1])
    msg.transform.rotation.z = float(quat[2])
    msg.transform.rotation.w = float(quat[3])
    return msg


class NssFlatPlayer(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("nss_flat_ros2_player")
        self.args = args
        self.bridge = CvBridge()
        self.color_pub = self.create_publisher(Image, args.rgb_topic, 10)
        self.depth_pub = self.create_publisher(Image, args.depth_topic, 10)
        self.label_pub = self.create_publisher(Image, args.label_topic, 10)
        self.info_pub = self.create_publisher(CameraInfo, args.camera_info_topic, 10)
        self.tf_pub = TransformBroadcaster(self)
        self.static_tf_pub = StaticTransformBroadcaster(self)

    def wait_for_consumers(
        self,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> None:
        deadline = time.monotonic() + self.args.discovery_timeout_s
        while time.monotonic() < deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            counts = (
                self.color_pub.get_subscription_count(),
                self.depth_pub.get_subscription_count(),
                self.label_pub.get_subscription_count(),
                self.info_pub.get_subscription_count(),
            )
            if counts[3] > 0:
                stamp = self.get_clock().now().to_msg()
                self.info_pub.publish(self.camera_info(stamp, width, height, fx, fy, cx, cy))
            if min(counts[:3]) > 0:
                self.get_logger().info(f"Khronos subscribers ready: {counts}")
                return
        counts = (
            self.color_pub.get_subscription_count(),
            self.depth_pub.get_subscription_count(),
            self.label_pub.get_subscription_count(),
            self.info_pub.get_subscription_count(),
        )
        raise RuntimeError(f"Khronos subscribers not ready before timeout: {counts}")

    def publish_static_tree(self, stamp) -> None:
        identity = np.eye(4, dtype=np.float64)
        self.static_tf_pub.sendTransform(
            [
                transform_message(self.args.world_frame, self.args.map_frame, identity, stamp),
                transform_message(self.args.map_frame, self.args.odom_frame, identity, stamp),
                transform_message(self.args.robot_frame, self.args.sensor_frame, identity, stamp),
            ]
        )

    def camera_info(self, stamp, width: int, height: int, fx: float, fy: float, cx: float, cy: float) -> CameraInfo:
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = self.args.sensor_frame
        msg.width = width
        msg.height = height
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0] * 5
        msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
        msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return msg

    def play(self) -> dict[str, object]:
        frames = read_frames(self.args.run_dir)
        if self.args.frame_limit > 0:
            frames = frames[: self.args.frame_limit]
        width, height, fx, fy, cx, cy = load_intrinsics(self.args.run_dir)
        base_dataset_ns = frames[0][1]
        base_wall_ns = self.get_clock().now().nanoseconds + 1_000_000_000
        self.publish_static_tree(rclpy.time.Time(nanoseconds=base_wall_ns).to_msg())
        self.wait_for_consumers(width, height, fx, fy, cx, cy)
        previous_target = time.monotonic()
        published = 0
        skipped_empty_depth = []

        for index, (image_id, dataset_ns) in enumerate(frames):
            if index > 0:
                delta_dataset_s = (dataset_ns - frames[index - 1][1]) * 1.0e-9
                previous_target += max(0.0, delta_dataset_s / self.args.play_rate)
                while rclpy.ok() and time.monotonic() < previous_target:
                    rclpy.spin_once(self, timeout_sec=min(0.01, previous_target - time.monotonic()))

            stamp_ns = base_wall_ns + int((dataset_ns - base_dataset_ns) / self.args.play_rate)
            stamp = rclpy.time.Time(nanoseconds=stamp_ns).to_msg()
            color = cv2.imread(str(self.args.run_dir / f"{image_id}_color.png"), cv2.IMREAD_COLOR)
            depth = cv2.imread(str(self.args.run_dir / f"{image_id}_depth.tiff"), cv2.IMREAD_UNCHANGED)
            labels = cv2.imread(str(self.args.run_dir / f"{image_id}_segmentation.png"), cv2.IMREAD_UNCHANGED)
            pose = np.loadtxt(self.args.run_dir / f"{image_id}_pose.txt", dtype=np.float64).reshape(4, 4)
            if color is None or depth is None or labels is None:
                raise RuntimeError(f"missing image data for frame {image_id}")
            if color.shape[:2] != (height, width) or depth.shape != (height, width) or labels.shape != (height, width):
                raise RuntimeError(f"shape mismatch for frame {image_id}")

            self.tf_pub.sendTransform(
                transform_message(self.args.odom_frame, self.args.robot_frame, pose, stamp)
            )
            # TF and image messages use different DDS topics. Give the listener a
            # short head start so the image callback cannot query this stamp first.
            settle_deadline = time.monotonic() + self.args.tf_settle_s
            while rclpy.ok() and time.monotonic() < settle_deadline:
                rclpy.spin_once(
                    self,
                    timeout_sec=min(0.005, settle_deadline - time.monotonic()),
                )
            if not np.any(np.isfinite(depth) & (depth > 0.0)):
                skipped_empty_depth.append(image_id)
                rclpy.spin_once(self, timeout_sec=0.01)
                continue
            info = self.camera_info(stamp, width, height, fx, fy, cx, cy)
            color_msg = self.bridge.cv2_to_imgmsg(cv2.cvtColor(color, cv2.COLOR_BGR2RGB), encoding="rgb8")
            depth_msg = self.bridge.cv2_to_imgmsg(depth.astype(np.float32, copy=False), encoding="32FC1")
            label_msg = self.bridge.cv2_to_imgmsg(labels, encoding="mono8")
            for msg in (color_msg, depth_msg, label_msg):
                msg.header.stamp = stamp
                msg.header.frame_id = self.args.sensor_frame
            self.info_pub.publish(info)
            self.color_pub.publish(color_msg)
            self.depth_pub.publish(depth_msg)
            self.label_pub.publish(label_msg)
            published += 1
            rclpy.spin_once(self, timeout_sec=0.01)

        end = time.monotonic() + self.args.post_wait_s
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        return {
            "run_dir": str(self.args.run_dir),
            "frames_available": len(read_frames(self.args.run_dir)),
            "frames_encountered": len(frames),
            "frames_published": published,
            "frames_skipped_empty_depth": len(skipped_empty_depth),
            "skipped_empty_depth_ids": skipped_empty_depth,
            "play_rate": self.args.play_rate,
            "intrinsics": {"width": width, "height": height, "fx": fx, "fy": fy, "cx": cx, "cy": cy},
            "frames": {
                "world": self.args.world_frame,
                "map": self.args.map_frame,
                "odom": self.args.odom_frame,
                "robot": self.args.robot_frame,
                "sensor": self.args.sensor_frame,
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish an NSS virtual-flat run to Khronos ROS2 inputs.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--play-rate", type=float, default=5.0)
    parser.add_argument("--post-wait-s", type=float, default=3.0)
    parser.add_argument(
        "--tf-settle-s",
        type=float,
        default=0.02,
        help="Delay images briefly after each dynamic TF publication.",
    )
    parser.add_argument("--discovery-timeout-s", type=float, default=30.0)
    parser.add_argument("--frame-limit", type=int, default=0)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--rgb-topic", default="/nss/rgb/image_raw")
    parser.add_argument("--depth-topic", default="/nss/depth/image_raw")
    parser.add_argument("--label-topic", default="/nss/semantic/image_raw")
    parser.add_argument("--camera-info-topic", default="/nss/rgb/camera_info")
    parser.add_argument("--world-frame", default="world")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--odom-frame", default="odom")
    parser.add_argument("--robot-frame", default="robot_0")
    parser.add_argument("--sensor-frame", default="left_cam")
    args = parser.parse_args()
    if args.play_rate <= 0.0:
        parser.error("--play-rate must be positive")
    if args.tf_settle_s < 0.0:
        parser.error("--tf-settle-s must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = NssFlatPlayer(args)
    try:
        summary = node.play()
        if args.manifest:
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(json.dumps(summary, indent=2) + "\n")
        print("NSS_KHRONOS_PLAYBACK_COMPLETE " + json.dumps(summary, sort_keys=True))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
