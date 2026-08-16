#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import datetime
import json
import math
import os
import re
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from std_msgs.msg import Empty, UInt64


SESSION_TIMEZONE = "Asia/Shanghai"
ROS_TIME_MAX_NS = (2**31 - 1) * 1_000_000_000 + 999_999_999
_ACQUISITION_STAMP_RE = re.compile(
    r"(?<!\d)(?P<date>20\d{6})_(?P<time>\d{6})_(?P<millis>\d{3})(?!\d)"
)


def resolve_session_time_contract(
    run_dir: Path,
    frames: list[tuple[str, int]],
    explicit_session_start_ns: int | None = None,
) -> dict[str, object]:
    """Resolve reproducible ROS stamps for one independently executed session.

    The session start denotes the first selected observation. By default it is
    parsed from the acquisition directory name, e.g.
    ``session_a_20260809_204010_201_*``. An explicit start exists for controlled
    diagnostics such as reusing one recording as a synthetic Session C.
    """
    if not frames:
        raise ValueError("cannot resolve session time without frames")
    if explicit_session_start_ns is not None:
        start_ns = int(explicit_session_start_ns)
        source = "explicit_cli"
        acquisition_token = None
        acquisition_local = None
    else:
        match = _ACQUISITION_STAMP_RE.search(run_dir.name)
        if match is None:
            raise ValueError(
                "cannot infer acquisition start from run directory name; "
                "expected YYYYMMDD_HHMMSS_mmm or pass --session-start-ns: "
                f"{run_dir}"
            )
        acquisition_token = match.group(0)
        local_naive = datetime.datetime.strptime(
            match.group("date") + match.group("time"), "%Y%m%d%H%M%S"
        ).replace(microsecond=int(match.group("millis")) * 1000)
        local_time = local_naive.replace(tzinfo=ZoneInfo(SESSION_TIMEZONE))
        utc_time = local_time.astimezone(datetime.timezone.utc)
        epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
        delta = utc_time - epoch
        start_ns = (
            delta.days * 86_400 * 1_000_000_000
            + delta.seconds * 1_000_000_000
            + delta.microseconds * 1000
        )
        source = "acquisition_directory_name"
        acquisition_local = local_time.isoformat()

    base_dataset_ns = int(frames[0][1])
    last_dataset_ns = int(frames[-1][1])
    if start_ns < 0:
        raise ValueError(f"session start must be non-negative: {start_ns}")
    if last_dataset_ns < base_dataset_ns:
        raise ValueError("dataset timestamps must not go backwards")
    last_stamp_ns = start_ns + (last_dataset_ns - base_dataset_ns)
    if start_ns > ROS_TIME_MAX_NS or last_stamp_ns > ROS_TIME_MAX_NS:
        raise ValueError(
            "session timestamps exceed ROS2 builtin_interfaces/Time signed-int32 "
            f"seconds range: first={start_ns} last={last_stamp_ns} "
            f"max={ROS_TIME_MAX_NS}"
        )
    return {
        "policy": "acquisition_start_plus_relative_dataset_time",
        "session_start_ns": start_ns,
        "session_end_ns": last_stamp_ns,
        "session_start_source": source,
        "timezone": SESSION_TIMEZONE,
        "acquisition_token": acquisition_token,
        "acquisition_local": acquisition_local,
        "base_dataset_ns": base_dataset_ns,
        "dataset_end_ns": last_dataset_ns,
    }


def validate_recurrent_session_start(
    time_contract: dict[str, object], input_latest_stamp_ns: int
) -> None:
    new_start = int(time_contract["session_start_ns"])
    prior_latest = int(input_latest_stamp_ns)
    if new_start <= prior_latest:
        raise ValueError(
            "recurrent observations do not start after input state: "
            f"input_latest={prior_latest} new_start={new_start}; "
            "use the real later acquisition or an explicit --session-start-ns"
        )


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
    image_ids = [image_id for image_id, _ in frames]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError(f"duplicate ImageID in {run_dir / 'timestamps.csv'}")
    for index, ((_, previous), (_, current)) in enumerate(
        zip(frames, frames[1:]), start=1
    ):
        if current <= previous:
            raise ValueError(
                f"timestamps must be strictly increasing at rows {index}/{index + 1}: "
                f"{previous} -> {current}"
            )
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


def load_world_transform(path: Path | None) -> np.ndarray:
    if path is None:
        return np.eye(4, dtype=np.float64)
    matrix = np.loadtxt(path, dtype=np.float64).reshape(4, 4)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"world transform contains non-finite values: {path}")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-8):
        raise ValueError(f"world transform has an invalid last row: {path}")
    if not np.isclose(np.linalg.det(matrix[:3, :3]), 1.0, atol=1.0e-3):
        raise ValueError(f"world transform rotation is invalid: {path}")
    return matrix


def resolve_frame_image(directory: Path, image_id: str, role: str) -> Path:
    """Resolve one authoritative per-frame label without creating a staging copy."""
    suffixes = (
        ("_segmentation.png", "_instances.png")
        if role == "instance"
        else ("_segmentation.png",)
    )
    matches = [directory / f"{image_id}{suffix}" for suffix in suffixes]
    existing = [path for path in matches if path.is_file()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise RuntimeError(
            f"ambiguous {role} maps for frame {image_id}: "
            + ", ".join(str(path) for path in existing)
        )
    raise RuntimeError(
        f"missing {role} map for frame {image_id}; tried "
        + ", ".join(str(path) for path in matches)
    )


def load_physical_catalog(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "session_update_physical_catalog/v1":
        raise ValueError(f"unsupported physical catalog schema: {path}")
    mapping: dict[int, int] = {}
    names: dict[int, str] = {}
    for item in payload.get("entities", []):
        instance_id = int(item["physical_instance_id"])
        semantic_id = int(item["semantic_id"])
        if instance_id <= 0 or instance_id > 0xFFFF:
            raise ValueError(f"invalid physical ID I{instance_id} in {path}")
        if instance_id in mapping:
            raise ValueError(f"duplicate physical ID I{instance_id} in {path}")
        mapping[instance_id] = semantic_id
        names[instance_id] = str(item["name"])
    if not mapping:
        raise ValueError(f"physical catalog is empty: {path}")
    semantic_bounds = payload.get("semantic_id_range", [0, 0xFFFF])
    if any(not semantic_bounds[0] <= value <= semantic_bounds[1] for value in mapping.values()):
        raise ValueError(f"catalog semantic ID outside declared range: {path}")
    payload["instance_to_semantic"] = mapping
    payload["instance_names"] = names
    return payload


def find_reviewed_instance_manifest(instance_dir: Path) -> Path | None:
    for parent in (instance_dir, *instance_dir.parents[:3]):
        candidate = parent / "manifest.json"
        if candidate.is_file():
            return candidate
    return None


def preflight_session_inputs(
    frames: list[tuple[str, int]],
    run_dir: Path,
    semantic_dir: Path,
    instance_dir: Path,
    catalog_path: Path,
) -> dict[str, object]:
    """Validate the complete file ledger and sampled label content before ROS starts.

    The reviewed instance manifest is the full-sequence ledger.  We do not
    decompress 8 billion label pixels twice; instead every required file is
    checked and evenly spaced frames are decoded.  Streaming still hard-fails
    any malformed frame, while the outer runner keeps such output transactional.
    """
    if semantic_dir.resolve() == instance_dir.resolve():
        raise ValueError("semantic and physical-instance directories must be different")
    catalog = load_physical_catalog(catalog_path)
    assert catalog is not None
    mapping = catalog["instance_to_semantic"]
    semantic_min, semantic_max = map(int, catalog["semantic_id_range"])
    resolved = validate_frame_inputs(frames, semantic_dir, instance_dir)

    missing: list[str] = []
    for image_id, _ in frames:
        for suffix in ("_color.png", "_depth.tiff", "_pose.txt"):
            path = run_dir / f"{image_id}{suffix}"
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"missing/empty RGB-D-pose inputs ({len(missing)}): {preview}")

    # First, last, and roughly one frame per second at 30 FPS.  This catches
    # shape/type/protocol drift without doubling full RGB-D decoding time.
    sample_indices = sorted(set([0, len(frames) - 1, *range(0, len(frames), 30)]))
    observed_ids: set[int] = set()
    raw_semantic_hist: dict[int, dict[int, int]] = {}
    for index in sample_indices:
        image_id = frames[index][0]
        semantic_path = resolved["semantic_files"][image_id]
        instance_path = resolved["instance_files"][image_id]
        if semantic_path.resolve() == instance_path.resolve():
            raise ValueError(f"frame {image_id} uses one file as semantic and instance input")
        semantic = cv2.imread(str(semantic_path), cv2.IMREAD_UNCHANGED)
        instance = cv2.imread(str(instance_path), cv2.IMREAD_UNCHANGED)
        if semantic is None or instance is None:
            raise ValueError(f"cannot decode semantic/instance frame {image_id}")
        if semantic.ndim != 2 or instance.ndim != 2 or semantic.shape != instance.shape:
            raise ValueError(
                f"semantic/instance shape mismatch for {image_id}: "
                f"{semantic.shape} vs {instance.shape}"
            )
        if not np.issubdtype(semantic.dtype, np.integer) or not np.issubdtype(
            instance.dtype, np.integer
        ):
            raise ValueError(f"semantic/instance images must be integer for {image_id}")
        if int(semantic.min()) < semantic_min or int(semantic.max()) > semantic_max:
            raise ValueError(f"semantic IDs outside catalog range for {image_id}")
        ids = set(map(int, np.unique(instance))) - {0}
        unknown = ids - set(mapping)
        if unknown:
            raise ValueError(f"unknown physical IDs in {image_id}: {sorted(unknown)}")
        observed_ids.update(ids)
        for instance_id in ids:
            values, counts = np.unique(semantic[instance == instance_id], return_counts=True)
            histogram = raw_semantic_hist.setdefault(instance_id, {})
            for value, count in zip(values, counts):
                semantic_id = int(value)
                histogram[semantic_id] = histogram.get(semantic_id, 0) + int(count)

    manifest_path = find_reviewed_instance_manifest(instance_dir)
    manifest_record = None
    ledger_ids = None
    if manifest_path is not None:
        reviewed = json.loads(manifest_path.read_text(encoding="utf-8"))
        if reviewed.get("schema") == "reviewed_ab_physical_instances/v1":
            session_key = next(
                (key for key in ("session_a", "session_b") if run_dir.name.startswith(key)),
                None,
            )
            session_record = reviewed.get("sessions", {}).get(session_key or "", {})
            if session_key and int(session_record.get("frame_count", -1)) != len(frames):
                raise ValueError(
                    f"reviewed ledger frame count differs for {session_key}: "
                    f"{session_record.get('frame_count')} != {len(frames)}"
                )
            ledger_ids = sorted(map(int, session_record.get("active_instance_ids", [])))
            expected = catalog.get("known_complete_sessions", {}).get(session_key or "", {})
            expected_ids = sorted(map(int, expected.get("active_physical_instance_ids", [])))
            if expected_ids and ledger_ids != expected_ids:
                raise ValueError(
                    f"reviewed ledger physical IDs differ from catalog for {session_key}: "
                    f"{ledger_ids} != {expected_ids}"
                )
            manifest_record = str(manifest_path.resolve())

    histogram_json = {
        str(instance_id): {
            "catalog_semantic_id": int(mapping[instance_id]),
            "raw_sample_semantic_counts": {
                str(label): count for label, count in sorted(histogram.items())
            },
        }
        for instance_id, histogram in sorted(raw_semantic_hist.items())
    }
    return {
        "frame_count": len(frames),
        "sampled_frame_count": len(sample_indices),
        "sampled_physical_ids": sorted(observed_ids),
        "reviewed_ledger_physical_ids": ledger_ids,
        "reviewed_manifest": manifest_record,
        "instance_suffixes": resolved["instance_suffixes"],
        "physical_semantic_contract": histogram_json,
    }


def validate_frame_inputs(
    frames: list[tuple[str, int]], semantic_dir: Path, instance_dir: Path | None
) -> dict[str, object]:
    """Fail before ROS startup if either authoritative label stream is incomplete."""
    semantic_files: dict[str, Path] = {}
    instance_files: dict[str, Path] = {}
    for image_id, _ in frames:
        semantic_files[image_id] = resolve_frame_image(semantic_dir, image_id, "semantic")
        if instance_dir is not None:
            instance_files[image_id] = resolve_frame_image(instance_dir, image_id, "instance")
    instance_suffixes = sorted({path.name[len(image_id):] for image_id, path in instance_files.items()})
    return {
        "semantic_files": semantic_files,
        "instance_files": instance_files,
        "instance_suffixes": instance_suffixes,
    }


def apply_physical_semantic_contract(
    labels: np.ndarray,
    instances: np.ndarray,
    instance_to_semantic: dict[int, int],
    image_id: str,
) -> np.ndarray:
    """Apply the authoritative physical-ID catalog without guessing categories."""
    observed = set(map(int, np.unique(instances))) - {0}
    unknown = observed - set(instance_to_semantic)
    if unknown:
        raise RuntimeError(
            f"unknown physical instance IDs for frame {image_id}: {sorted(unknown)}"
        )
    contracted = labels.copy()
    for instance_id in observed:
        contracted[instances == instance_id] = instance_to_semantic[instance_id]
    return contracted


def require_valid_depth(depth: np.ndarray, image_id: str) -> None:
    """Reject an observation that contains no finite positive depth sample."""
    if not np.any(np.isfinite(depth) & (depth > 0.0)):
        raise RuntimeError(f"depth contains no valid measurements for frame {image_id}")


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
        self.last_processed_stamp_ns = None
        self.finish_acknowledged = False
        self.ack_sub = None
        if args.flow_control == "ack":
            self.ack_sub = self.create_subscription(
                UInt64, args.ack_topic, self.ack_callback, 10
            )
        # Playback-complete signal. A topic is used instead of the official
        # finish_mapping_and_save service because the ROS2 CLI cannot open
        # sockets in this sandbox, while topics work (the ACK loop proves it).
        self.finish_pub = self.create_publisher(
            Empty, args.finish_topic, rclpy.qos.QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        )
        self.finish_ack_sub = self.create_subscription(
            Empty, args.finish_ack_topic, self.finish_ack_callback, 1
        )

    def ack_callback(self, message: UInt64) -> None:
        self.last_processed_stamp_ns = int(message.data)

    def finish_ack_callback(self, _message: Empty) -> None:
        self.finish_acknowledged = True

    def wait_for_processed(self, stamp_ns: int) -> None:
        deadline = time.monotonic() + self.args.ack_timeout_s
        while rclpy.ok() and self.last_processed_stamp_ns != stamp_ns:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Khronos did not acknowledge frame {stamp_ns} within "
                    f"{self.args.ack_timeout_s:g}s; last ACK={self.last_processed_stamp_ns}"
                )
            # 1 ms rather than 10: this is the hot loop of ack flow control, and a
            # coarser timeout adds up to that much idle wait to every single frame
            # after the mapper has already answered.
            rclpy.spin_once(self, timeout_sec=0.001)

    def wait_for_consumers(
        self,
        deterministic_stamp,
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
                self.info_pub.publish(
                    self.camera_info(
                        deterministic_stamp, width, height, fx, fy, cx, cy
                    )
                )
            finish_count = self.finish_pub.get_subscription_count()
            if min(counts[:3]) > 0 and finish_count > 0:
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
        all_frames = read_frames(self.args.run_dir)
        input_preflight = None
        if self.args.instance_dir is not None:
            input_preflight = preflight_session_inputs(
                all_frames,
                self.args.run_dir,
                self.args.label_dir,
                self.args.instance_dir,
                self.args.physical_catalog,
            )
        frames = all_frames
        if self.args.frame_start > 0:
            frames = frames[self.args.frame_start:]
        if self.args.frame_limit > 0:
            frames = frames[: self.args.frame_limit]
        frame_inputs = validate_frame_inputs(frames, self.args.label_dir, self.args.instance_dir)
        catalog = load_physical_catalog(self.args.physical_catalog)
        if self.args.instance_dir is not None and catalog is None:
            raise RuntimeError("packed instance input requires --physical-catalog")
        instance_to_semantic = catalog["instance_to_semantic"] if catalog else {}
        semantic_id_bounds = (
            tuple(map(int, catalog["semantic_id_range"])) if catalog else (0, 0xFFFF)
        )
        semantic_files = frame_inputs["semantic_files"]
        instance_files = frame_inputs["instance_files"]
        width, height, fx, fy, cx, cy = load_intrinsics(self.args.run_dir)
        source_width, source_height = width, height
        if self.args.image_scale != 1.0:
            width = max(1, int(round(width * self.args.image_scale)))
            height = max(1, int(round(height * self.args.image_scale)))
            fx *= self.args.image_scale
            fy *= self.args.image_scale
            cx *= self.args.image_scale
            cy *= self.args.image_scale
        world_transform = load_world_transform(self.args.world_transform)
        time_contract = resolve_session_time_contract(
            self.args.run_dir, frames, self.args.session_start_ns
        )
        base_dataset_ns = int(time_contract["base_dataset_ns"])
        session_start_ns = int(time_contract["session_start_ns"])
        deterministic_start_stamp = rclpy.time.Time(
            nanoseconds=session_start_ns
        ).to_msg()
        self.publish_static_tree(deterministic_start_stamp)
        self.wait_for_consumers(
            deterministic_start_stamp, width, height, fx, fy, cx, cy
        )
        previous_target = time.monotonic()
        published = 0
        first_published_stamp_ns = None
        last_published_stamp_ns = None
        skipped_empty_depth = []
        label_protocol = (
            "packed_semantic_instance_32sc1"
            if self.args.instance_dir is not None
            else "semantic_only"
        )
        print(f"PLAYBACK_LABEL_PROTOCOL {label_protocol}", flush=True)

        # Decoding a frame costs about as much as the mapper spends on it, and in
        # ack mode the two used to run strictly one after the other. Reading the
        # next frame on a worker thread overlaps that cost with the mapper's, so
        # each cycle waits for whichever of the two is slower instead of both.
        loader = ThreadPoolExecutor(max_workers=1)
        _ack_wait_total = 0.0
        _loader_wait_total = 0.0
        _prepare_total = 0.0
        _tf_settle_total = 0.0
        _publish_total = 0.0
        _loop_start = time.perf_counter()

        def read_frame(image_id: str):
            color = cv2.imread(str(self.args.run_dir / f"{image_id}_color.png"), cv2.IMREAD_COLOR)
            depth = cv2.imread(
                str(self.args.run_dir / f"{image_id}_depth.tiff"), cv2.IMREAD_UNCHANGED
            )
            labels = cv2.imread(
                str(semantic_files[image_id]), cv2.IMREAD_UNCHANGED
            )
            instances = None
            if self.args.instance_dir is not None:
                instances = cv2.imread(
                    str(instance_files[image_id]),
                    cv2.IMREAD_UNCHANGED,
                )
                if instances is None:
                    raise RuntimeError(f"missing instance map for frame {image_id}")
            pose = np.loadtxt(
                self.args.run_dir / f"{image_id}_pose.txt", dtype=np.float64
            ).reshape(4, 4)
            if not np.all(np.isfinite(pose)):
                raise RuntimeError(f"pose contains non-finite values for frame {image_id}")
            if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-8):
                raise RuntimeError(f"pose has an invalid last row for frame {image_id}")
            if not np.isclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1.0e-3):
                raise RuntimeError(f"pose rotation is invalid for frame {image_id}")
            return color, depth, labels, instances, pose

        pending = loader.submit(read_frame, frames[0][0]) if frames else None

        for index, (image_id, dataset_ns) in enumerate(frames):
            if index > 0:
                delta_dataset_s = (dataset_ns - frames[index - 1][1]) * 1.0e-9
                previous_target += max(0.0, delta_dataset_s / self.args.play_rate)
                while rclpy.ok() and time.monotonic() < previous_target:
                    rclpy.spin_once(
                        self, timeout_sec=min(0.01, previous_target - time.monotonic())
                    )

            stamp_ns = session_start_ns + (dataset_ns - base_dataset_ns)
            stamp = rclpy.time.Time(nanoseconds=stamp_ns).to_msg()
            _t_loader = time.perf_counter()
            color, depth, labels, instances, pose = pending.result()
            _loader_wait_total += time.perf_counter() - _t_loader
            if index + 1 < len(frames):
                pending = loader.submit(read_frame, frames[index + 1][0])
            _t_prepare = time.perf_counter()
            pose = world_transform @ pose
            if color is None or depth is None or labels is None:
                raise RuntimeError(f"missing image data for frame {image_id}")
            if color.ndim != 3 or color.shape[2] != 3:
                raise RuntimeError(f"color must be HxWx3 for frame {image_id}: {color.shape}")
            if depth.ndim != 2 or labels.ndim != 2 or (
                instances is not None and instances.ndim != 2
            ):
                raise RuntimeError(
                    f"depth/semantic/instance must be single-channel for frame {image_id}"
                )
            if not np.issubdtype(labels.dtype, np.integer):
                raise RuntimeError(f"semantic map is not integer-valued for frame {image_id}")
            if instances is not None and not np.issubdtype(instances.dtype, np.integer):
                raise RuntimeError(f"instance map is not integer-valued for frame {image_id}")

            image_shapes = [color.shape[:2], depth.shape, labels.shape]
            if instances is not None:
                image_shapes.append(instances.shape)
            if any(shape != (height, width) for shape in image_shapes):
                if any(shape != (source_height, source_width) for shape in image_shapes):
                    raise RuntimeError(
                        f"shape mismatch for frame {image_id}: "
                        f"color={color.shape[:2]} depth={depth.shape} "
                        f"semantic={labels.shape} "
                        f"instance={None if instances is None else instances.shape}"
                    )
                color = cv2.resize(color, (width, height), interpolation=cv2.INTER_AREA)
                depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
                labels = cv2.resize(labels, (width, height), interpolation=cv2.INTER_NEAREST)
                if instances is not None:
                    instances = cv2.resize(
                        instances, (width, height), interpolation=cv2.INTER_NEAREST
                    )

            if instances is not None:
                # The reviewed physical catalog is authoritative for object
                # category. This preserves raw ADE semantics for background and
                # person, while enforcing e.g. I10->S75 and the user's I1 bed
                # surface policy for every nonzero physical pixel.
                labels = apply_physical_semantic_contract(
                    labels, instances, instance_to_semantic, image_id
                )

            semantic_min = int(labels.min())
            semantic_max = int(labels.max())
            if semantic_min < semantic_id_bounds[0] or semantic_max > semantic_id_bounds[1]:
                raise RuntimeError(
                    f"semantic IDs outside allowed range {semantic_id_bounds} for frame {image_id}: "
                    f"[{semantic_min}, {semantic_max}]"
                )
            if instances is not None:
                instance_min = int(instances.min())
                instance_max = int(instances.max())
                if instance_min < 0 or instance_max > 0xFFFF:
                    raise RuntimeError(
                        f"physical instance IDs outside uint16 for frame {image_id}: "
                        f"[{instance_min}, {instance_max}]"
                    )

            if os.environ.get("NSS_REJECT_INVALID_DEPTH", "0") == "1":
                # DIAGNOSTIC ABLATION ONLY (default off): reject invalid depth
                # before it reaches the mapper. depth<=0 or non-finite becomes
                # NaN, which Hydra's ProjectiveIntegrator already treats as an
                # invalid measurement (non-finite sdf is rejected upstream of
                # any TSDF update). This isolates the zero-depth leakage path.
                depth = depth.astype(np.float32, copy=True)
                depth[~np.isfinite(depth) | (depth <= 0.0)] = np.nan

            _prepare_total += time.perf_counter() - _t_prepare
            _t_tf = time.perf_counter()
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
            _tf_settle_total += time.perf_counter() - _t_tf
            require_valid_depth(depth, image_id)
            _t_publish = time.perf_counter()
            info = self.camera_info(stamp, width, height, fx, fy, cx, cy)
            color_msg = self.bridge.cv2_to_imgmsg(cv2.cvtColor(color, cv2.COLOR_BGR2RGB), encoding="rgb8")
            depth_msg = self.bridge.cv2_to_imgmsg(depth.astype(np.float32, copy=False), encoding="32FC1")
            if instances is None:
                label_msg = self.bridge.cv2_to_imgmsg(labels, encoding="mono8")
            else:
                # The mapper configuration must explicitly declare this wire
                # protocol. CV_32SC1 alone is not a discriminator because Hydra
                # normalizes ordinary semantic-only images to the same type.
                packed = (labels.astype(np.int32) << 16) | instances.astype(np.int32)
                label_msg = self.bridge.cv2_to_imgmsg(packed, encoding="32SC1")
            for msg in (color_msg, depth_msg, label_msg):
                msg.header.stamp = stamp
                msg.header.frame_id = self.args.sensor_frame
            self.info_pub.publish(info)
            self.color_pub.publish(color_msg)
            self.depth_pub.publish(depth_msg)
            self.label_pub.publish(label_msg)
            _publish_total += time.perf_counter() - _t_publish
            published += 1
            if first_published_stamp_ns is None:
                first_published_stamp_ns = stamp_ns
            last_published_stamp_ns = stamp_ns
            _t_pub = time.perf_counter()
            if self.args.flow_control == "ack":
                self.wait_for_processed(stamp_ns)
            _ack_wait_total += time.perf_counter() - _t_pub
            if published == 1 or published % 100 == 0 or published == len(frames):
                print(
                    f"PLAYBACK_PROGRESS published={published} total={len(frames)} "
                    f"remaining={len(frames) - published}",
                    flush=True,
                )
            # ACK waiting above already services subscriptions. Do not inject a
            # guaranteed 10 ms idle sleep after every acknowledged frame.
            rclpy.spin_once(
                self, timeout_sec=0.0 if self.args.flow_control == "ack" else 0.01
            )

        end = time.monotonic() + self.args.post_wait_s
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

        # Active-window ACK proves every input reached D1 integration. The
        # explicit finish ACK below separately proves the downstream queues were
        # drained and the terminal shared Update was serialized.
        loader.shutdown(wait=False)
        _elapsed = time.perf_counter() - _loop_start
        if published:
            _known = (
                _loader_wait_total
                + _prepare_total
                + _tf_settle_total
                + _publish_total
                + _ack_wait_total
            )
            print(
                "TIMING_PROBE "
                f"frames={published} "
                f"loop_ms_per_frame={_elapsed/published*1000:.2f} "
                f"loader_wait_ms={_loader_wait_total/published*1000:.2f} "
                f"prepare_ms={_prepare_total/published*1000:.2f} "
                f"tf_settle_ms={_tf_settle_total/published*1000:.2f} "
                f"encode_publish_ms={_publish_total/published*1000:.2f} "
                f"publish_to_ack_ms={_ack_wait_total/published*1000:.2f} "
                f"other_ms={(_elapsed-_known)/published*1000:.2f}",
                flush=True,
            )
        print(f"PLAYBACK_FINISH_SIGNAL topic={self.args.finish_topic}", flush=True)
        self.finish_pub.publish(Empty())
        finish_deadline = time.monotonic() + self.args.finish_timeout_s
        while rclpy.ok() and not self.finish_acknowledged:
            if time.monotonic() >= finish_deadline:
                raise RuntimeError(
                    "mapper did not acknowledge terminal save within "
                    f"{self.args.finish_timeout_s:g}s"
                )
            rclpy.spin_once(self, timeout_sec=0.05)

        return {
            "run_dir": str(self.args.run_dir),
            "label_dir": str(self.args.label_dir),
            "instance_dir": str(self.args.instance_dir) if self.args.instance_dir else None,
            "physical_catalog": str(self.args.physical_catalog) if self.args.physical_catalog else None,
            "input_preflight": input_preflight,
            "label_protocol": label_protocol,
            "instance_filename_suffixes": frame_inputs["instance_suffixes"],
            "world_transform": str(self.args.world_transform) if self.args.world_transform else None,
            "world_transform_matrix": world_transform.tolist(),
            "frames_available": len(read_frames(self.args.run_dir)),
            "frames_encountered": len(frames),
            "frames_published": published,
            "frames_skipped_empty_depth": len(skipped_empty_depth),
            "skipped_empty_depth_ids": skipped_empty_depth,
            "play_rate": self.args.play_rate,
            "flow_control": self.args.flow_control,
            "finish_timeout_s": self.args.finish_timeout_s,
            "image_scale": self.args.image_scale,
            "timestamp_policy": time_contract["policy"],
            "timestamp_scale": 1.0,
            "base_dataset_ns": base_dataset_ns,
            "session_start_ns": session_start_ns,
            "timestamp_provenance": time_contract,
            "dataset_bounds_ns": [frames[0][1], frames[-1][1]],
            "published_bounds_ns": [first_published_stamp_ns, last_published_stamp_ns],
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
    parser.add_argument(
        "--label-dir",
        type=Path,
        help="Directory containing the authoritative <ImageID>_segmentation.png files. Defaults to --run-dir.",
    )
    parser.add_argument(
        "--physical-catalog",
        type=Path,
        help="JSON mapping stable physical IDs to their authoritative semantic classes.",
    )
    parser.add_argument(
        "--world-transform",
        type=Path,
        help="Optional 4x4 transform left-multiplied onto every input pose.",
    )
    parser.add_argument("--post-wait-s", type=float, default=3.0)
    parser.add_argument("--play-rate", type=float, default=1.0)
    parser.add_argument("--image-scale", type=float, default=1.0)
    parser.add_argument("--flow-control", choices=("realtime", "ack"), default="realtime")
    parser.add_argument("--ack-topic", default="/session_update/frame_processed")
    parser.add_argument("--finish-topic", default="/session_update/finish_and_save")
    parser.add_argument("--finish-ack-topic", default="/session_update/finish_saved")
    parser.add_argument(
        "--instance-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing <ImageID>_segmentation.png (canonical) or "
            "legacy <ImageID>_instances.png physical-instance maps."
        ),
    )
    parser.add_argument("--ack-timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--finish-timeout-s",
        type=float,
        default=1800.0,
        help="Bounded wait for terminal change detection and final state serialization.",
    )
    parser.add_argument(
        "--tf-settle-s",
        type=float,
        default=0.02,
        help="Delay images briefly after each dynamic TF publication.",
    )
    parser.add_argument("--discovery-timeout-s", type=float, default=30.0)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-limit", type=int, default=0)
    parser.add_argument(
        "--session-start-ns",
        type=int,
        default=None,
        help=(
            "Deterministic ROS timestamp of the first selected observation. "
            "Defaults to acquisition time parsed from --run-dir in Asia/Shanghai."
        ),
    )
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
    args.run_dir = args.run_dir.resolve()
    args.label_dir = (args.label_dir or args.run_dir).resolve()
    if not args.run_dir.is_dir():
        parser.error(f"--run-dir is not a directory: {args.run_dir}")
    if not args.label_dir.is_dir():
        parser.error(f"--label-dir is not a directory: {args.label_dir}")
    if args.instance_dir is not None:
        args.instance_dir = args.instance_dir.resolve()
        if not args.instance_dir.is_dir():
            parser.error(f"--instance-dir is not a directory: {args.instance_dir}")
        if args.physical_catalog is None:
            parser.error("--physical-catalog is required with --instance-dir")
    if args.physical_catalog is not None:
        args.physical_catalog = args.physical_catalog.resolve()
        if not args.physical_catalog.is_file():
            parser.error(f"--physical-catalog is not a file: {args.physical_catalog}")
    if args.world_transform is not None:
        args.world_transform = args.world_transform.resolve()
        if not args.world_transform.is_file():
            parser.error(f"--world-transform is not a file: {args.world_transform}")
    if args.play_rate <= 0.0:
        parser.error("--play-rate must be positive")
    if not 0.0 < args.image_scale <= 1.0:
        parser.error("--image-scale must be in (0, 1]")
    if args.tf_settle_s < 0.0:
        parser.error("--tf-settle-s must be non-negative")
    if not math.isfinite(args.ack_timeout_s) or args.ack_timeout_s <= 0.0:
        parser.error("--ack-timeout-s must be finite and positive")
    if not math.isfinite(args.finish_timeout_s) or args.finish_timeout_s <= 0.0:
        parser.error("--finish-timeout-s must be finite and positive")
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
