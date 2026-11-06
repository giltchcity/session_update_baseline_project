"""Frame access for the flat RGB-D datasets (real room-18 A/B/C, synthetic A/B, Flat A/B).

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
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

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
    # [3RScan] the full-resolution intrinsics of every visit of the chain, when its visits' cameras differ (rooms
    # 0cac766c, 4a9a43d6): game.Crop cuts them all to one image size (GaME reprojects between keyframes of one size)
    crop_group: tuple = ()

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
    # Sensor range of the reference mapper: real room18_instance_5cm.yaml max_range 5 m,
    # synthetic mapper_mechanism_10cm.yaml max_range 15 m; min 0.1 m (khronos.launch default).
    depth_range: tuple = (0.1, 5.0)
    # [3RScan] Intrinsics.txt of every visit of the chain (Intrinsics.crop_group)
    crop_group: tuple = ()


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

    Playback offsets A = 1 s, B = 141 s (datasets/synthetic_ab/README.md, 'Mapping'). The label images carry the
    evaluation units (the paper scope's EVAL_SCOPE_UNITS.csv, 29 units), in place: every member of a unit reads as the
    unit's first core member (instance labels, since 10-21) and as that member's class (semantic labels, since 10-22;
    a mirror on a cabinet is the cabinet). Isaac's per-member labels: session_<s>/{instance_labels,semantics}_per_member.
    """
    root = FT / "datasets/synthetic_ab" / f"session_{stage}"
    start = {"a": 1_000_000_000, "b": 141_000_000_000}[stage]
    return SessionSpec(f"synthetic_{stage}", root / "rgbd", root / "instance_labels",
                       root / "semantics", start, _identity(), pixel_offset=0.5,
                       depth_range=(0.1, 15.0))


def flat_session(stage: str) -> SessionSpec:
    """Flat (panoptic_mapping's flat_dataset, the scene GaME evaluates on) run1 / run2 as sessions a / b, as
    scripts/prepare_flat_ab.py writes them (datasets/flat_ab: GaME's depth, the ground-truth panoptic labels); both runs
    share world coordinates. Rendered like synthetic A/B (Unreal): rays through (u + 0.5, v + 0.5), so GaME's centred
    principal point (cx 320 of 640) takes the whole image; the synthetic sensor range. Stamps: the recording's own
    (run1 2021-07-26, run2 2021-08-13)."""
    root = FT / "datasets/flat_ab"
    start = json.loads((root / "session.json").read_text())[stage]["start_ns"]
    s = root / f"session_{stage}"
    return SessionSpec(f"flat_{stage}", s / "rgbd", s / "instance_labels", s / "semantics", start, _identity(),
                       pixel_offset=0.5, depth_range=(0.1, 15.0))


def aria_session(room: str, stage: str) -> SessionSpec:
    """GaME's Aria clips (its processed Aria Digital Twin recordings, HF voviktyl/GaME_Aria) as
    analysis/aria_20261101/make_aria_ab.py writes them (datasets/aria_ab/<room>: run1 / run2 as sessions a / b, shared
    world coordinates, 30 Hz). The labels are GaME's own class-agnostic SAM masks (ids per frame; label_space.yaml names no
    object class: segments, not object instances). Principal point 255.5 of 512 in the pixel-centre convention (offset 0); range 0.1-10 m: the clips' depth reaches 8 m
    (99th percentile 7.9 m) and GaME integrates all of it."""
    root = FT / "datasets/aria_ab" / room
    start = json.loads((root / "session.json").read_text())[stage]["start_ns"]
    s, lab = root / f"session_{stage}", root / "sam_labels" / f"session_{stage}"
    return SessionSpec(f"aria-{room}_{stage}", s / "rgbd", lab / "instance_labels", lab / "semantics", start, _identity(),
                       pixel_offset=0.0, depth_range=(0.1, 10.0))


def tesse_session(stage: str) -> SessionSpec:
    """TESSE office (Khronos's tesse_cd, the uHumans2 office scene), one session of 4346 frames at 20 Hz, as the takeover
    conversion writes it (wt_takeover_runs/tesse/data/session_a: the rosbag's depth, the scene's instance and
    semantic labels in the tesse_cd_office label space). Rendered (Unity) with the principal point at the image's centre
    (cx 360 of 720): rays through (u + 0.5, v + 0.5), as synthetic A/B and Flat. Sensor range 0.1-5 m, Khronos's uHumans2
    setup (as the other methods' TESSE runs). Stamps: the recording's own."""
    s = FT / "wt_takeover_runs/tesse/data" / f"session_{stage}"
    start = json.loads((s / "session.json").read_text())["start_ns"]
    return SessionSpec(f"tesse_{stage}", s / "rgbd", s / "instance_labels", s / "semantics", start, _identity(),
                       pixel_offset=0.5, depth_range=(0.1, 5.0))


def rscan_session(room: str, k: int) -> SessionSpec:
    """3RScan (the P3 multi-revisit chains, datasets/3rscan): visit k of a room's chain (mapping_configs/<room>/chain.json:
    the reference, then its rescans in 3RScan.json order), as the TSDF and Khronos rows run it
    (wt_takeover_runs/analysis/p3_3rscan_ours.sh): the scan's frames (datasets/3rscan/sessions/<scan>: RGB-D at 960 x 540,
    10 Hz, the 3RScan instances, their classes contracted by the room's physical catalog); a rescan's poses taken to the
    reference's frame by world_transform_<scan>_to_<reference>.txt; visit k starts at 1 + 200 k s; a real camera
    (integer pixel centres); sensor range 0.1-5 m. The visits are sessions a, b, c, d in order, named
    3rscan-<room>_<letter>: the room in the name, as a session's name gives back its dataset (kf_strip). room: the room's
    id with or without the p3_ of its mapping_configs folder."""
    rid = room[3:] if room.startswith("p3_") else room
    root = FT / "datasets/3rscan"
    C = root / "mapping_configs" / f"p3_{rid}"
    visits = json.loads((C / "chain.json").read_text())["visits"]
    scan, s = visits[k], root / "sessions" / visits[k]
    T = _identity() if k == 0 else np.loadtxt(C / f"world_transform_{scan}_to_{visits[0]}.txt")
    return SessionSpec(f"3rscan-{rid}_{'abcd'[k]}", s / "rgbd", s / "instance_labels", s / "semantics",
                       1_000_000_000 + 200_000_000_000 * k, T, pixel_offset=0.0, depth_range=(0.1, 5.0),
                       crop_group=tuple(root / "sessions" / v / "rgbd" / "Intrinsics.txt" for v in visits))


class FlatSession:
    """Iterates the frames of one session, optionally subsampled in time and space."""

    def __init__(self, spec: SessionSpec, stride: int = 1, pixel_step: int = 1):
        self.spec = spec
        self.stride = stride
        self.pixel_step = pixel_step
        self.K_full = Intrinsics.load(spec.rgbd / "Intrinsics.txt", spec.pixel_offset)
        group = tuple(Intrinsics.load(p, spec.pixel_offset) for p in spec.crop_group)
        if len(set(group)) > 1:
            self.K_full = dataclasses.replace(self.K_full, crop_group=group)
        self.K = self.K_full.scaled(pixel_step) if pixel_step > 1 else self.K_full
        self.ids: List[str] = []
        self.rel_ns: List[int] = []
        with (spec.rgbd / "timestamps.csv").open() as f:
            for row in csv.DictReader(f):
                self.ids.append(row["ImageID"])
                self.rel_ns.append(int(row["TimeStamp"]))

    def indices(self) -> Sequence[int]:
        return range(0, len(self.ids), self.stride)

    def evidence_step(self, hz: float) -> int:
        """Input frames per frame of a stream at `hz` (the layer's evidence frames, run.py and core/layer.py): the
        session's frame rate (its median frame interval) over hz, every frame when the input is slower (real and
        synthetic: 30 Hz, every 6th frame at 5 Hz; Flat: 2-3 Hz, every frame)."""
        dt = float(np.median(np.diff(self.rel_ns))) if len(self.rel_ns) > 1 else 0.0
        return max(1, int(round(1e9 / dt / hz))) if dt > 0 else 1

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


# D1 motion pixels of the layer's front end (core/d1.py) by frame stamp: the dynamic clusters of a
# frame, which no backend integrates besides the dynamic classes (Khronos ActiveWindow::updateMap
# integration mask). Filled per frame by run.py for the rows that run the layer; a bounded window.
MOTION: Dict[int, np.ndarray] = {}


def motion_mask(frame: Frame) -> Optional[np.ndarray]:
    """The frame's D1 motion pixels at its resolution (nearest, integer resolution ratios only)."""
    m = MOTION.get(int(frame.stamp_ns))
    if m is None:
        return None
    H, W = frame.depth.shape
    if m.shape == (H, W):
        return m
    mh, mw = m.shape
    if (H % mh == 0 and W % mw == 0 and H // mh == W // mw) or (mh % H == 0 and mw % W == 0 and mh // H == mw // W):
        return m[(np.arange(H) * mh) // H][:, (np.arange(W) * mw) // W]
    return None


def dynamic_mask(frame: Frame, dynamic_semantics: Sequence[int]) -> np.ndarray:
    """Pixels no backend integrates and the layer codes as unidentified: no physical identity
    and a dynamic class (people), and the frame's D1 motion pixels when the layer runs."""
    if frame.semantic is None or not len(dynamic_semantics):
        out = np.zeros(frame.depth.shape, dtype=bool)
    else:
        out = (frame.instance <= 0) & np.isin(frame.semantic, list(dynamic_semantics))
    m = motion_mask(frame)
    return out if m is None else (out | m)


# -- the same pixel geometry on the GPU (torch is imported on use: the scoring environments
#    import this module without torch) ------------------------------------------------------------
def torch_pixel_normals(depth, K: Intrinsics, R_world_cam):
    """Per-pixel world normals from depth differences (NaN where undefined), towards the camera."""
    import torch
    h, w = depth.shape
    dev = depth.device
    u = (torch.arange(w, device=dev, dtype=torch.float32) + K.offset - K.cx) / K.fx
    v = (torch.arange(h, device=dev, dtype=torch.float32) + K.offset - K.cy) / K.fy
    P = torch.stack([u[None, :] * depth, v[:, None] * depth, depth], dim=-1)
    dx = torch.full_like(P, float("nan"))
    dy = torch.full_like(P, float("nan"))
    dx[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dy[1:-1, :] = P[2:, :] - P[:-2, :]
    n = torch.cross(dx, dy, dim=-1)
    n = n / torch.linalg.norm(n, dim=-1, keepdim=True)
    flip = (n * P).sum(-1) > 0
    n = torch.where(flip[..., None], -n, n)
    return n @ R_world_cam.T


def torch_backproject(depth, K: Intrinsics, T_world_cam, mask):
    """World points and pixel indices (v, u) of the valid masked depth pixels."""
    import torch
    valid = torch.isfinite(depth) & mask
    v, u = torch.nonzero(valid, as_tuple=True)
    z = depth[v, u]
    x = (u.to(torch.float32) + K.offset - K.cx) / K.fx * z
    y = (v.to(torch.float32) + K.offset - K.cy) / K.fy * z
    cam = torch.stack([x, y, z], dim=1)
    return cam @ T_world_cam[:3, :3].T + T_world_cam[:3, 3], v, u

