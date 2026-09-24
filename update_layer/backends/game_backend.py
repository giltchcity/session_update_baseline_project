"""GaME (CVPR 2026, Gaussian mapping of evolving scenes) as a map backend.

GaME's code is used unmodified (FT/baselines/GaME @1c971d6; built for sm_120 with CUDA 12.8).
This adapter
  * feeds our flat sessions in GaME's Flat conventions: depth and translation x10 (config
    `scale`), world-to-camera pose, object masks = our GT physical-instance masks (the same
    identity input the update layer gets), invalid depth 0;
  * crops each frame so the principal point is the image centre (GaME's cameras are FoV-based);
  * tracks every Gaussian's uid, identity (the instance id of the pixel it was seeded from) and
    lifetime by wrapping GaussianModel.densification_postfix / prune_points on the instance;
  * runs GaME's own change handling (detect_additions / detect_removals) or none, in which case
    the update layer decides and `retire_gaussians` executes (prune + keyframe occlusion masks);
  * renders fixed evaluation cameras (depth + identity) into EvaluationScenes at snapshot times.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np
import torch

GAME = Path("/home/jixian/Desktop/FT/baselines/GaME")
sys.path.insert(0, str(GAME))
from src.entities.game import GaME  # noqa: E402
from src.entities.losses import l1_loss  # noqa: E402
from src.flashsplat.gaussian_renderer import flashsplat_render  # noqa: E402
from src.utils import utils as gu  # noqa: E402

sys.path.insert(0, "/home/jixian/Desktop/FT")
from eval.scene.scene import EvaluationScene, SceneObject  # noqa: E402

from ..frames import Frame, FlatSession  # noqa: E402

UINT64_MAX = 18446744073709551615

# GaME's Flat configuration (configs/flat/flat.yaml), used as published for our indoor data.
FLAT_CONFIG = dict(
    first_keyframe_iters=200, keyframe_iters=50, refinement_only=False, refinement_iters=30000,
    depth_change_threshold=3.0, min_opacity=0.3, color_error_threshold=0.3,
    removal_coverage_threshold=0.25, addition_coverage_threshold=0.4, occlusion_ignore_threshold=0.9,
    gaussian_seed_threshold=0.2, covis_ignore_threshold=1.0, keyframe_translation_diff=1.25,
    opacity_reset_interval=6000, densify_from_iter=1, densification_interval=5000,
    densify_grad_threshold=0.0002, min_grad=0.01, scale=10.0, num_label_channels=256,
    isotropic_reg_weight=0.5)


@dataclass
class Crop:
    top: int
    left: int
    rows: int
    cols: int
    step: int

    @staticmethod
    def centred(K, step: int, height_full: int, width_full: int) -> "Crop":
        """Crop + subsample so that the principal point is the exact image centre.

        Pixel-centre convention: a sample at integer (u, v) has continuous coordinate u + offset;
        a centred principal point satisfies (c - offset - first) / step == (n - 1) / 2.
        """
        def axis(c, offset, full):
            best = None
            for first in range(0, 64):
                cs = (c - offset - first) / step
                n = int(round(2 * cs + 1))
                if n < 2 or first + step * (n - 1) > full - 1:
                    continue
                err = abs((n - 1) / 2 - cs)
                if best is None or err < best[0] - 1e-9 or (abs(err - best[0]) < 1e-9 and n > best[2]):
                    best = (err, first, n)
            return best[1], best[2]
        top, rows = axis(K.cy, K.offset, height_full)
        left, cols = axis(K.cx, K.offset, width_full)
        return Crop(top, left, rows, cols, step)

    def apply(self, img: np.ndarray) -> np.ndarray:
        s = self.step
        return img[self.top:self.top + s * (self.rows - 1) + 1:s, self.left:self.left + s * (self.cols - 1) + 1:s]

    def intrinsics(self, K) -> np.ndarray:
        s = self.step
        return np.array([[K.fx / s, 0, (self.cols - 1) / 2.0], [0, K.fy / s, (self.rows - 1) / 2.0], [0, 0, 1]])


class GameFrames:
    """Our flat session in GaME's sample format (every `stride`-th 30 Hz frame)."""

    def __init__(self, session: FlatSession, stride: int, step: int, scale: float, min_mask_px: int = 50):
        assert session.pixel_step == 1, "load full-resolution frames; the crop subsamples"
        self.session = session
        self.indices = list(range(0, len(session.ids), stride))
        self.scale = scale
        K = session.K
        self.crop = Crop.centred(K, step, K.height, K.width)
        self.K = self.crop.intrinsics(K)
        self.min_mask_px = max(1, min_mask_px // (step * step))

    def __len__(self) -> int:
        return len(self.indices)

    def stamp(self, k: int) -> int:
        return self.session.stamp_ns(self.indices[k])

    def sample(self, k: int) -> Tuple[dict, Frame]:
        f = self.session.load(self.indices[k], color=True)
        c = self.crop
        color = c.apply(f.color)
        depth = np.nan_to_num(c.apply(f.depth), nan=0.0).astype(np.float32)
        instance = c.apply(f.instance).astype(np.int64)
        ids, counts = np.unique(instance[instance > 0], return_counts=True)
        masks = [instance == i for i, n in zip(ids, counts) if n >= self.min_mask_px]
        h, w = depth.shape
        masks = torch.from_numpy(np.stack(masks)) if masks else torch.zeros((0, h, w), dtype=torch.bool)
        pose = np.linalg.inv(f.T_world_cam)                 # world-to-camera, as GaME's Flat loader
        pose[:3, 3] *= self.scale
        sample = dict(frame_id=k, color=np.ascontiguousarray(color), depth=depth * self.scale,
                      masks=masks, pose=pose.astype(np.float32), intrinsics=self.K,
                      instance=np.ascontiguousarray(instance),
                      semantic=np.ascontiguousarray(c.apply(f.semantic)) if f.semantic is not None else None)
        return sample, f


class TrackedGaME(GaME):
    """GaME with per-Gaussian uid / identity / lifetime bookkeeping and optional change handling."""

    def __init__(self, config: dict, change_handling: bool):
        super().__init__(config, wandb_online=False)
        self.change_handling = change_handling
        self.now = 0
        self.next_uid = 0
        self.uid = torch.zeros(0, dtype=torch.int64, device="cuda")
        self.identity = torch.zeros(0, dtype=torch.int64, device="cuda")
        self.birth = torch.zeros(0, dtype=torch.int64, device="cuda")
        self.deaths: Dict[int, int] = {}
        self.pending_identity: Optional[torch.Tensor] = None
        self.current_instance: Optional[np.ndarray] = None
        gm = self.gaussian_model
        post, prune = gm.densification_postfix, gm.prune_points

        def densification_postfix(new_xyz, *rest):
            n = new_xyz.shape[0]
            post(new_xyz, *rest)
            ids = self.pending_identity
            if ids is None or len(ids) != n:
                ids = torch.zeros(n, dtype=torch.int64, device="cuda")
            self.uid = torch.cat([self.uid, torch.arange(self.next_uid, self.next_uid + n, device="cuda")])
            self.identity = torch.cat([self.identity, ids.to("cuda")])
            self.birth = torch.cat([self.birth, torch.full((n,), self.now, dtype=torch.int64, device="cuda")])
            self.next_uid += n
            self.pending_identity = None

        def prune_points(mask):
            mask = mask.to("cuda").bool()
            for u in self.uid[mask].tolist():
                self.deaths[u] = self.now
            keep = ~mask
            prune(mask)
            self.uid, self.identity, self.birth = self.uid[keep], self.identity[keep], self.birth[keep]

        gm.densification_postfix = densification_postfix
        gm.prune_points = prune_points

    @torch.no_grad()
    def _add_gaussians(self, color, depth, segmentation, pose, intrinsics):
        """GaME._add_gaussians (game.py:545-597) verbatim, plus the seeded points' identities."""
        if self.gaussian_model.get_xyz.shape[0] == 0:
            seeding_mask = torch.ones_like(depth)[None]
        else:
            pose = pose.clone().detach().cpu()
            flashsplat_view = gu.flashsplat_cam(color, depth, segmentation, intrinsics, pose, None)
            render_pkg = flashsplat_render(flashsplat_view, self.gaussian_model, gu.flashsplat_pipe(),
                                           torch.ones(3).cuda(), obj_num=self.num_label_channels)
            rendered_depth, rendered_alpha, rendered_color = (
                render_pkg["depth"].clone(), render_pkg["alpha"].clone(), render_pkg["render"].clone())
            depth_error = torch.abs(depth - rendered_depth)
            depth_error_mask = (rendered_depth > depth) * (depth_error > 40 * depth_error.median())
            alpha_mask = rendered_alpha < self.config["min_opacity"]
            seeding_mask = alpha_mask | depth_error_mask
            og_seeding_img = torch.stack([seeding_mask[0]] * 3, dim=-1)
            l1_color_loss = l1_loss(rendered_color, color, agg='none').permute((1, 2, 0)).mean(-1)
            threshed_color = torch.stack(
                [(torch.ones_like(l1_color_loss) * self.config["gaussian_seed_threshold"]) < l1_color_loss] * 3,
                -1).bool()
            newly_added = threshed_color > og_seeding_img
            seeding_mask = (newly_added[:, :, 0].bool() | seeding_mask[0])[None]
        pose = gu.torch2np(pose)
        seeding_mask = gu.torch2np(seeding_mask[0] if seeding_mask.dim() == 3 else seeding_mask).astype(np.uint8)
        color = color.clone().permute(1, 2, 0) * 255
        filtered_color = gu.torch2np(color).astype(np.uint8)
        filtered_color[seeding_mask == 0] = 0
        filtered_depth = gu.torch2np(depth.clone())
        filtered_depth[seeding_mask == 0] = 0
        cloud_to_add = gu.rgbd2ptcloud(filtered_color, filtered_depth, intrinsics, pose)
        # Open3D back-projects row-major the pixels with 0 < depth <= depth_trunc (100)
        valid = (filtered_depth > 0) & (filtered_depth <= 100)
        ids = self.current_instance[valid] if self.current_instance is not None \
            else np.zeros(int(valid.sum()), np.int64)
        cloud_to_add = cloud_to_add.uniform_down_sample(2)
        ids = ids[::2]
        if len(ids) == len(cloud_to_add.points):
            self.pending_identity = torch.as_tensor(ids, dtype=torch.int64)
        gu.add_points(self.gaussian_model, cloud_to_add)

    def process(self, frame_id: int, sample: dict, stamp: int) -> bool:
        """One frame of GaME.train (game.py:616-645); returns True if it became a keyframe."""
        self.now = stamp
        pose, intrinsics = sample["pose"], sample["intrinsics"]
        self.estimated_poses[frame_id] = pose
        if not self.is_keyframe(pose):
            return False
        keyframe_data = {
            "color": gu.np2torch(sample["color"], device="cuda").permute(2, 0, 1) / 255.0,
            "depth": gu.np2torch(sample["depth"], device="cuda"),
            "masks": sample["masks"].cuda(),
            "pose": gu.np2torch(pose, device="cuda"),
            "intrinsics": intrinsics,
        }
        first = not self.keyframes
        if self.change_handling and len(self.keyframes) > 2:
            self.detect_additions(keyframe_data)
        self.keyframes[frame_id] = gu.dict2device(keyframe_data, "cpu")
        self._last_keyframe_id = frame_id
        self.current_instance = sample["instance"]
        self._add_gaussians(keyframe_data["color"], keyframe_data["depth"], None,
                            keyframe_data["pose"], keyframe_data["intrinsics"])
        self.current_instance = None
        num_iters = self.config["first_keyframe_iters"] if first else self.config["keyframe_iters"]
        self.optimize_model(50, only_frame_id=frame_id)
        if self.change_handling:
            self.detect_removals(frame_id, keyframe_data)
        self.optimize_model(num_iters)
        return True

    # -- execution of layer decisions -------------------------------------------------------
    @torch.no_grad()
    def retire_gaussians(self, mask: torch.Tensor) -> int:
        """Prune the given Gaussians and mask where they were seen in stored keyframes, so that
        optimisation on old keyframes does not re-grow them (GaME's occlusion masks)."""
        mask = mask.to("cuda").bool()
        n = int(mask.sum())
        if n == 0:
            return 0
        pipe, bg = gu.flashsplat_pipe(), torch.zeros(3).cuda()
        for kid, kf in self.keyframes.items():
            view = gu.flashsplat_cam(kf["color"].cuda(), kf["depth"].cuda(), None, kf["intrinsics"],
                                     kf["pose"], kid)
            pkg = flashsplat_render(view, self.gaussian_model, pipe, bg, obj_num=1, used_mask=mask)
            seen = (pkg["alpha"] > self.config["min_opacity"]).squeeze()
            if seen.any():
                prev = self.occlusion_masks.get(kid)
                self.occlusion_masks[kid] = seen if prev is None else (prev.to(seen.device) | seen)
        self.gaussian_model.prune_points(mask)
        return n

    # -- evaluation rendering --------------------------------------------------------------
    @torch.no_grad()
    def render_scene(self, cameras: List[Tuple[np.ndarray, np.ndarray, int, int]], t_ns: int,
                     semantic_of: Dict[int, int], bg_voxel: float = 0.02, obj_voxel: float = 0.01,
                     min_alpha: float = 0.5) -> EvaluationScene:
        """Depth + identity from fixed cameras, back-projected into world samples (1 cm union).

        Identity per pixel: each Gaussian carries (id, id^2, 1) as colour; a pixel whose blended
        variance of id is ~0 is covered by one identity only; mixed boundary pixels are dropped
        from the objects (kept as background when their mean id is ~0).
        """
        gm = self.gaussian_model
        scale = self.config.get("scale", 1.0)
        n = gm.get_xyz.shape[0]
        idf = self.identity.to(torch.float32)
        codes = torch.stack([idf, idf * idf, torch.ones_like(idf)], dim=1)
        pipe, bg = gu.flashsplat_pipe(), torch.zeros(3).cuda()
        pts_all, id_all, nrm_all = [], [], []
        for T_world_cam, K, h, w in cameras:
            pose = np.linalg.inv(T_world_cam)
            pose[:3, 3] *= scale
            dummy = torch.zeros((3, h, w), device="cuda")
            view = gu.flashsplat_cam(dummy, torch.zeros((h, w), device="cuda"), None, K,
                                     torch.as_tensor(pose, dtype=torch.float32), None)
            pkg = flashsplat_render(view, gm, pipe, bg, override_color=codes, obj_num=1)
            alpha = pkg["alpha"].squeeze()
            depth = pkg["depth"].squeeze() / scale
            img = pkg["render"]
            weight = img[2].clamp(min=1e-6)
            mean = img[0] / weight
            var = img[1] / weight - mean * mean
            ok = (alpha > min_alpha) & (depth > 0) & torch.isfinite(depth)
            v, u = torch.nonzero(ok, as_tuple=True)
            z = depth[v, u]
            x = (u.float() - K[0, 2]) / K[0, 0] * z
            y = (v.float() - K[1, 2]) / K[1, 1] * z
            Tw = torch.as_tensor(T_world_cam, dtype=torch.float32, device="cuda")
            p = torch.stack([x, y, z], 1) @ Tw[:3, :3].T + Tw[:3, 3]
            pure = var[v, u] < 0.25
            ident = torch.where(pure, torch.round(mean[v, u]).long(), torch.full_like(v, -1))
            # normals from rendered-depth differences (as update_layer.layer.pixel_normals)
            uu = (torch.arange(w, device="cuda", dtype=torch.float32) - K[0, 2]) / K[0, 0]
            vv = (torch.arange(h, device="cuda", dtype=torch.float32) - K[1, 2]) / K[1, 1]
            dd = torch.where(ok, depth, torch.full_like(depth, float("nan")))
            Pc = torch.stack([uu[None, :] * dd, vv[:, None] * dd, dd], dim=-1)
            dx = torch.full_like(Pc, float("nan"))
            dy = torch.full_like(Pc, float("nan"))
            dx[:, 1:-1] = Pc[:, 2:] - Pc[:, :-2]
            dy[1:-1, :] = Pc[2:, :] - Pc[:-2, :]
            nc = torch.cross(dx, dy, dim=-1)
            nc = nc / torch.linalg.norm(nc, dim=-1, keepdim=True)
            nc = torch.where(((nc * Pc).sum(-1) > 0)[..., None], -nc, nc)
            pts_all.append(p)
            id_all.append(ident)
            nrm_all.append(nc[v, u] @ Tw[:3, :3].T)
        if not pts_all or n == 0:
            return EvaluationScene(t_ns, np.zeros((0, 3), np.float32), np.zeros(0, np.uint32), [])
        P, I, N = torch.cat(pts_all), torch.cat(id_all), torch.cat(nrm_all)

        def union(sel, voxel, with_id):
            """One sample per voxel (and per identity for objects), the first in render order."""
            q, ids, nn = P[sel], I[sel], N[sel]
            if len(q) == 0:
                return q, ids, nn
            cols = torch.floor(q / voxel).long()
            if with_id:
                cols = torch.cat([cols, ids[:, None]], dim=1)
            _, inv = torch.unique(cols, dim=0, return_inverse=True)
            first = torch.full((int(inv.max()) + 1,), len(q), dtype=torch.int64, device="cuda")
            first.scatter_reduce_(0, inv, torch.arange(len(q), device="cuda"), reduce="amin")
            return q[first], ids[first], nn[first]
        bp, bi, bn = union(I <= 0, bg_voxel, False)
        op, oi, on = union(I > 0, obj_voxel, True)
        P = torch.cat([bp, op]).cpu().numpy()
        I = torch.cat([bi, oi]).cpu().numpy()
        NR = torch.cat([bn, on]).cpu().numpy().astype(np.float32)
        bg_mask = (I <= 0)
        objects = []
        for i in np.unique(I[I > 0]):
            m = I == i
            objects.append(SceneObject(id=str(int(i)), instance_id=int(i), semantic=int(semantic_of.get(int(i), -1)),
                                       points=P[m].astype(np.float32), present=True,
                                       first_observed_ns=[0], last_observed_ns=[UINT64_MAX], normals=NR[m]))
        return EvaluationScene(t_ns, P[bg_mask].astype(np.float32),
                               np.zeros(int(bg_mask.sum()), np.uint32), objects, NR[bg_mask])


from ..scenelist import SceneListTimeline  # noqa: E402,F401  (re-export)
