"""MeshObjectExtractor's object reconstruction confidence (Khronos @192c1cf, ports/mapping_core/khronos/src/
active_window/object_extraction/mesh_object_extractor.cpp:351-427, 513-526; integration/object_integrator.cpp:59-80;
hydra semantic_integrator.cpp:119-130) on the layer's stored frames.

Khronos reconstructs each object segment in its own TSDF (object_reconstruction_resolution: real 0.02, synthetic 0.05;
truncation 2 voxels) from the segment's static frames; every voxel inside a frame's truncation band counts one
observation, labelled 1 when the frame's object image at its pixel is the segment, else 0 (BinarySemanticIntegrator:
counts; pixels of the integration mask are skipped). computeConfidence = count(1) / count(0 + 1) (an unobserved voxel:
0); voxels below min_object_reconstruction_confidence (0.5: room18_instance_5cm.yaml:126, mapper_mechanism_10cm.yaml:75)
are erased before the mesh is extracted, and the mesh's bounding box takes the max / min volume gates (:427-442;
only_extract_reconstructed_objects: no mesh, no object). Here the voxels are those of the segment's own points (its
measured surface), the frame's measurement at a voxel is the stored range at its nearest pixel, and the integration
mask is the frame's dynamic mask.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch


def reconstruction_confidence(store, rejected: Optional[Sequence[Optional[torch.Tensor]]], identity: int,
                              points: torch.Tensor, frames: Sequence[int], voxel: float) -> torch.Tensor:
    """Per point: the confidence of its voxel (count of object-labelled observations / all observations, 0 when
    unobserved) over the stored frames `frames` (evidence-store indices of the segment)."""
    dev = points.device
    keys = torch.floor(points / voxel).to(torch.int64)
    uk, inv = torch.unique(keys, dim=0, return_inverse=True)
    centre = (uk.to(torch.float32) + 0.5) * voxel
    n1 = torch.zeros(len(uk), dtype=torch.int64, device=dev)
    n0 = torch.zeros_like(n1)
    trunc = 2.0 * voxel
    K = store.K
    H, W = store.rng.shape[1:]
    for f in sorted(set(int(x) for x in frames)):
        T = store.T[f]
        cam = centre @ T[:3, :3].T + T[:3, 3]
        z = cam[:, 2]
        ok = z > 0
        zs = torch.where(ok, z, torch.ones_like(z))
        u = torch.floor(cam[:, 0] / zs * K.fx + K.cx - K.offset + 0.5).to(torch.int64)
        v = torch.floor(cam[:, 1] / zs * K.fy + K.cy - K.offset + 0.5).to(torch.int64)
        ok &= (u >= 0) & (v >= 0) & (u < W) & (v < H)
        idx = torch.nonzero(ok).squeeze(1)
        if not len(idx):
            continue
        u, v = u[idx], v[idx]
        r = store.rng[f, v, u].to(torch.float32) * 1e-3
        good = r > 0
        if rejected is not None and f < len(rejected) and rejected[f] is not None:
            good &= ~rejected[f][v, u]
        sdf = r - torch.linalg.norm(cam[idx], dim=1)
        band = good & (sdf.abs() < trunc)
        label = store.code[f, v, u] == identity
        n1.index_add_(0, idx[band & label], torch.ones(int((band & label).sum()), dtype=torch.int64, device=dev))
        n0.index_add_(0, idx[band & ~label], torch.ones(int((band & ~label).sum()), dtype=torch.int64, device=dev))
    total = n0 + n1
    conf = torch.where(total > 0, n1.to(torch.float32) / total.clamp(min=1).to(torch.float32),
                       torch.zeros(len(uk), device=dev))
    return conf[inv]


def box_volume(points: torch.Tensor) -> float:
    if not len(points):
        return 0.0
    return float(torch.prod(points.max(dim=0).values - points.min(dim=0).values))


def measurement_in_range(depth: torch.Tensor, K, min_range: float, max_range: float) -> torch.Tensor:
    """InstanceForwarding isValidObjectMeasurementPixel (instance_forwarding.cpp:55-70 @192c1cf): an object pixel is
    forwarded only when its range (distance along the ray) is finite, positive, inside the sensor's range and not
    beyond the detector's max_range (room18_instance_5cm.yaml:79 5 m, mapper_mechanism_10cm.yaml:40 15 m)."""
    H, W = depth.shape
    u = (torch.arange(W, device=depth.device, dtype=torch.float32) + K.offset - K.cx) / K.fx
    v = (torch.arange(H, device=depth.device, dtype=torch.float32) + K.offset - K.cy) / K.fy
    rng = depth * torch.sqrt(u[None, :] ** 2 + v[:, None] ** 2 + 1.0)
    return torch.isfinite(rng) & (rng > 0) & (rng >= min_range) & (rng <= max_range)


def vote_frames(observed: Sequence[int], end: int, window: int, start: int = -1) -> list:
    """The frames an extraction's reconstruction votes over (active_window.cpp:342-386, mesh_object_extractor.cpp:
    473-492 @192c1cf): the track's observations still in the frame buffer -- the trailing `window` frames up to the
    extraction (FrameDataBuffer max_buffer_size 100 x store_every_n_frames 3 = 300 input frames: room18_instance_5cm
    .yaml:54-55, mapper_mechanism_10cm.yaml:24-25) -- from the latest static start / motion cut `start` on
    (selectStaticFrames / after_stamp). Indices are evidence-store frame indices."""
    lo = max(end - window + 1, start)
    return sorted({int(f) for f in observed if lo <= f <= end})
