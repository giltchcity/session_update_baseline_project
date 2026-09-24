"""Frame access for the flat RGB-D datasets (real room-18 A/B/C, synthetic A/B).

A frame is what the update layer consumes: depth, per-pixel physical identity,
semantic label, camera-to-world pose (already in the common world frame) and
intrinsics. Nothing here depends on how a map is represented.

Layout (see datasets/local_ab and datasets/synthetic_ab/README.md):
  rgbd/NNNNNN_depth.tiff   float32 optical-Z depth in metres (0 or NaN invalid)
  rgbd/NNNNNN_pose.txt     4x4 camera-to-world, session frame
  rgbd/Intrinsics.txt      Res_x, Res_y, f_x, f_y, u, v
  rgbd/timestamps.csv      ImageID,TimeStamp (session-relative ns)
  instance_labels/...      physical identity per pixel, 0 = none (annotation input)
  semantics/...            semantic class per pixel
  alignment                session-to-common-world 4x4 (real B/C only)
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import numpy as np
from PIL import Image

FT = Path("/home/jixian/Desktop/FT")


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    # Continuous image coordinate of pixel (0, 0)'s sample point: 0 when integer pixel
    # coordinates are sample centres (real Azure Kinect), 0.5 when rays pass through
    # (u + 0.5, v + 0.5) (synthetic renderer, datasets/synthetic_ab/README.md).
    offset: float = 0.0

    @staticmethod
    def load(path: Path, offset: float = 0.0) -> "Intrinsics":
        values = {}
        for line in path.read_text().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                values[key.strip()] = float(value)
        return Intrinsics(int(values["Res_x"]), int(values["Res_y"]), values["f_x"],
                          values["f_y"], values["u"], values["v"], offset)

    def scaled(self, factor: int) -> "Intrinsics":
        """Intrinsics of the image subsampled by `factor` (pixel (i, j) -> (i*factor, j*factor))."""
        return Intrinsics(self.width // factor, self.height // factor, self.fx / factor,
                          self.fy / factor, self.cx / factor, self.cy / factor,
                          self.offset / factor)

    def project(self, cam: np.ndarray) -> tuple:
        """Pixel indices (u, v) and depth z of camera-frame points; u = -1 when not in view."""
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.floor(self.fx * cam[:, 0] / z + self.cx - self.offset + 0.5)
            v = np.floor(self.fy * cam[:, 1] / z + self.cy - self.offset + 0.5)
        ok = (z > 0) & np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < self.width) & \
            (v >= 0) & (v < self.height)
        u = np.where(ok, u, -1).astype(np.int64)
        v = np.where(ok, v, -1).astype(np.int64)
        return u, v, z


@dataclass
class Frame:
    index: int
    stamp_ns: int                 # absolute sensor time
    depth: np.ndarray             # HxW float32, NaN where invalid
    instance: np.ndarray          # HxW int32 physical identity, 0 = none
    semantic: Optional[np.ndarray]
    T_world_cam: np.ndarray       # 4x4 camera-to-world in the common world frame
    K: Intrinsics
    color: Optional[np.ndarray] = None   # HxWx3 uint8 RGB, loaded on request


@dataclass
class SessionSpec:
    name: str
    rgbd: Path
    instance: Path
    semantic: Optional[Path]
    start_ns: int                 # absolute time of frame 0
    T_world_session: np.ndarray   # session frame -> common world frame
    pixel_offset: float = 0.0
    depth_range: tuple = (0.1, 5.0)


def _identity() -> np.ndarray:
    return np.eye(4)


def real_session(stage: str) -> SessionSpec:
    """Real room-18 sessions; world = session A frame (alignment files)."""
    root = FT / "datasets/local_ab"
    rgbd = sorted((root / "rgbd").glob(f"session_{stage}_*"))[0]
    start = {"a": 1786279210201000000, "b": 1786302302620000000, "c": 1787394423387000000}[stage]
    T = _identity()
    if stage != "a":
        T = np.loadtxt(root / f"alignment/session_{stage}_to_session_a.txt")
    return SessionSpec(f"real_{stage}", rgbd, root / f"instance_labels/session_{stage}",
                       root / f"semantics/session_{stage}", start, T)


def synthetic_session(stage: str) -> SessionSpec:
    """Synthetic A/B; both sessions already share world coordinates.

    Playback offsets A = 1 s, B = 141 s (datasets/synthetic_ab/README.md, 'Mapping').
    """
    root = FT / "datasets/synthetic_ab" / f"session_{stage}"
    start = {"a": 1_000_000_000, "b": 141_000_000_000}[stage]
    return SessionSpec(f"synthetic_{stage}", root / "rgbd", root / "instance_labels",
                       root / "semantics", start, _identity(), pixel_offset=0.5)


class FlatSession:
    """Iterates the frames of one session, optionally subsampled in time and space."""

    def __init__(self, spec: SessionSpec, stride: int = 1, pixel_step: int = 1):
        self.spec = spec
        self.stride = stride
        self.pixel_step = pixel_step
        self.K_full = Intrinsics.load(spec.rgbd / "Intrinsics.txt", spec.pixel_offset)
        self.K = self.K_full.scaled(pixel_step) if pixel_step > 1 else self.K_full
        self.ids: List[str] = []
        self.rel_ns: List[int] = []
        with (spec.rgbd / "timestamps.csv").open() as f:
            for row in csv.DictReader(f):
                self.ids.append(row["ImageID"])
                self.rel_ns.append(int(row["TimeStamp"]))

    def indices(self) -> Sequence[int]:
        return range(0, len(self.ids), self.stride)

    def stamp_ns(self, i: int) -> int:
        return self.spec.start_ns + self.rel_ns[i]

    def pose(self, i: int) -> np.ndarray:
        T = np.loadtxt(self.spec.rgbd / f"{self.ids[i]}_pose.txt")
        return self.spec.T_world_session @ T

    def load(self, i: int, color: bool = False) -> Frame:
        fid = self.ids[i]
        s = self.pixel_step
        depth = np.asarray(Image.open(self.spec.rgbd / f"{fid}_depth.tiff"), dtype=np.float32)
        instance = np.asarray(Image.open(self.spec.instance / f"{fid}_segmentation.png")).astype(np.int32)
        semantic = None
        if self.spec.semantic is not None:
            p = self.spec.semantic / f"{fid}_segmentation.png"
            if p.exists():
                semantic = np.asarray(Image.open(p)).astype(np.int32)
        if s > 1:
            # Nearest subsampling at the pixel centres the scaled intrinsics describe.
            depth, instance = depth[::s, ::s], instance[::s, ::s]
            semantic = semantic[::s, ::s] if semantic is not None else None
        depth = depth.copy()
        lo, hi = self.spec.depth_range
        depth[~np.isfinite(depth) | (depth < lo) | (depth > hi)] = np.nan
        rgb = None
        if color:
            rgb = np.asarray(Image.open(self.spec.rgbd / f"{fid}_color.png").convert("RGB"))
            if s > 1:
                rgb = rgb[::s, ::s]
            rgb = np.ascontiguousarray(rgb)
        return Frame(i, self.stamp_ns(i), depth, instance, semantic, self.pose(i), self.K, rgb)

    def __iter__(self) -> Iterator[Frame]:
        for i in self.indices():
            yield self.load(i)


def backproject(frame: Frame, mask: Optional[np.ndarray] = None) -> tuple:
    """World points and pixel indices of the valid (and masked) depth pixels."""
    valid = np.isfinite(frame.depth)
    if mask is not None:
        valid &= mask
    v, u = np.nonzero(valid)
    z = frame.depth[v, u]
    K = frame.K
    x = (u + K.offset - K.cx) / K.fx * z
    y = (v + K.offset - K.cy) / K.fy * z
    cam = np.stack([x, y, z], axis=1)
    R, t = frame.T_world_cam[:3, :3], frame.T_world_cam[:3, 3]
    return cam @ R.T + t, v, u
