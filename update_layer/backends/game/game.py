"""GaME (CVPR 2026, Gaussian mapping of evolving scenes) as a backend.

=====================================================================================================
CHANGES TO PUBLISHED GaME (FT/baselines/GaME @1c971d6) -- everything that behaves differently.
GaME's source files are not edited (except GaME_1c971d6.patch: a missing #include, build only); the
changes are overrides in this file, marked [CHANGED] where they are, and their labels are written
into every run's run.json ("backend_changes").
  [C1] keyframe test: relative pose (TrackedGaME.is_keyframe; since 2026-09-30).
       published  delta = inv(last_w2c) @ now_w2c. Its translation is c_last - R_last^T R_now c_now:
                  a turn of the camera counts as a move of (turn angle x distance of the camera from
                  the world origin), so the test depends on where a dataset puts its origin.
       fixed      delta = last_w2c @ inv(now_w2c): the current camera in the last keyframe's frame
                  (the true camera motion).
       measured   keyframes at 5 Hz, synthetic (cameras 7-15 m from the origin, flat config):
                  A 680 -> 323, B 559 -> 451; real (origin = first camera of A): aria config
                  627/648/1148 -> 629/634/1126, tum config 477/514/916 -> 391/372/752.
  [C2] keyframe test: rotation threshold (same method; since 2026-09-30).
       published  np.any(as_euler("xyz") > 50) with the angles in radians -- never true.
       fixed      the Euler angles in degrees, > 50 deg (what the number means).
  [C3] keyframe sampling without GaME's endless loop (TrackedGaME._sample_valid_keyframe and
       optimize_model): published loops forever when every remaining keyframe is covered by
       occlusion masks; here covered frames leave the draw and, when none is left, the step is
       skipped.
  [C4] training loss only on pixels with measured depth (TrackedGaME.optimize_model; since
       2026-10-01 16:30). Published: mask = rendered depth not NaN, so a pixel without depth (stored
       as 0) trains the rendered depth towards 0 and the colour with no geometry behind it. GaME's
       datasets (Replica, Aria) have dense depth; our Kinect frames have ~49% pixels without depth
       (narrower depth FoV, beyond 5 m): the real map grew false surface in free space (real row 4
       of 2026-10-01: G1 precision 21%, ~1300 m2 of surface in a ~100-240 m2 room). Synthetic
       frames have 0% such pixels, so synthetic runs are unaffected.
  [C5] addition test only on pixels with measured depth (TrackedGaME._find_added_geometry_masks;
       since 2026-10-01 16:30). Published: depth 0 - rendered depth < -eps counts as new geometry
       in front, i.e. every pixel without depth looks like an addition.
  Runs before 2026-09-30 17:00 used C3 only (GaME synthetic rows 1-4 and their row-4 reruns).
  GaME real rows 3 / 4 run before 2026-10-01 16:30 lack C4 / C5 and are invalid.
Input differences (not code changes): segment masks = GT instances + connected regions of one
semantic class (GaME expects SAM automatic masks); people are never integrated; frames are cropped
so the principal point is centred (GaME's cameras are FoV-based).
Config for real data since 2026-10-01: configs/kinect_real.yaml next to this file (tum mapping values +
aria change-detection values, all published; see its header). Before (2026-09-30): configs/aria/room0.yaml. Our sensor is a Kinect (like
GaME's tum configs), but the tum configs are for static scenes: depth_change_threshold 4.0 m, i.e.
GaME's own change detection is off; GaME's README tests change handling on Flat and Aria only, so
aria is its only published real-world setting for changing scenes (keyframe every 1 cm, 50 + 200
iterations per keyframe).
=====================================================================================================

Configs as published: synthetic = configs/flat/flat.yaml (depth and translation x10), real =
configs/aria/room0.yaml (metres). This adapter
  * feeds our frames in GaME's conventions: world-to-camera pose, depth and translation times the
    config's scale, invalid depth 0; each frame is cropped (and subsampled) so that the principal
    point is the image centre, since GaME's cameras are FoV-based;
  * gives GaME the segment masks its SAM automatic masks would give (a cover of the image): every
    physical instance, plus every connected region of one semantic class among the pixels without
    an instance;
  * never integrates dynamic pixels (people): depth 0 (never seeded) and excluded from the
    keyframe's losses (GaME's per-keyframe occlusion mask);
  * keeps per Gaussian a stable id, its identity (the instance of the keyframe pixel it was seeded
    from, by projection) and last_update (the last keyframe that measured it on its surface,
    within the 5 cm sensor tolerance); its extent is 3 sigma of its largest axis (the
    rasterizer's cut-off);
  * own update on = GaME's change handling (detect_additions / detect_removals) as published; off =
    none, and retire() removes Gaussians the way GaME removes content: prune them and mask where
    they were seen in the stored keyframes, so that old keyframes do not grow them back;
  * snapshot(t) renders the map from its own keyframe views and back-projects depth D / alpha where
    alpha > 0.5, [E1, since 2026-10-01 17:30] not at a depth edge (>5% jump to a neighbour: there D / alpha
    blends foreground and background into a surface in free space) and inside the view's measured-depth
    area (real A test, 150 frames, judged against measured depth: free-space samples 25.3% -> 7.6%,
    the points backend's own 7.5%; it never compares rendered with measured depth values). E1 is applied
    to real data only: the synthetic GaME runs (rows 1-4, 2026-09-30) were made without it and are kept
    as they are (synthetic depth has no invalid pixels); identity per pixel from rendering (id, id^2, 1): a pixel with ~0 variance belongs
    to one identity, mixed pixels are dropped; one sample per 2 cm (background) / 1 cm (object) voxel.
The GaME model (Gaussians + keyframes) is carried across sessions as GaME's own multi-session
setting does.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2 as cv
import numpy as np
import torch
import yaml
from pytorch_msssim import ssim
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from ...eval.scene import EvaluationScene, SceneObject
from ...eval.scenelist import UINT64_MAX, SceneListTimeline
from ...frames import FlatSession, Frame, Intrinsics, SessionSpec, dynamic_mask
from ...interface import Backend, DatasetInfo, Elements

GAME = Path("/home/jixian/Desktop/FT/baselines/GaME")
if str(GAME) not in sys.path:
    sys.path.insert(0, str(GAME))
from src.entities.game import GaME  # noqa: E402
from src.entities.losses import isotropic_loss, l1_loss  # noqa: E402
from src.flashsplat.gaussian_renderer import flashsplat_render  # noqa: E402
from src.utils import utils as gu  # noqa: E402

CONFIGS = {"synthetic": GAME / "configs/flat/flat.yaml",
           "real": Path(__file__).resolve().parent / "configs/kinect_real.yaml"}   # tum mapping + aria change detection
STEP = {"synthetic": 2, "real": 4}      # the image resolution the update layer reads


class Crop:
    """Crop + subsample so that the principal point is the exact image centre.

    Pixel-centre convention: a sample at integer (u, v) has continuous coordinate u + offset;
    a centred principal point satisfies (c - offset - first) / step == (n - 1) / 2.
    """

    def __init__(self, K: Intrinsics, step: int):
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
        self.step = step
        self.top, self.rows = axis(K.cy, K.offset, K.height)
        self.left, self.cols = axis(K.cx, K.offset, K.width)
        self.K = np.array([[K.fx / step, 0, (self.cols - 1) / 2.0],
                           [0, K.fy / step, (self.rows - 1) / 2.0], [0, 0, 1]])

    def __call__(self, img: np.ndarray) -> np.ndarray:
        s = self.step
        return img[self.top:self.top + s * (self.rows - 1) + 1:s, self.left:self.left + s * (self.cols - 1) + 1:s]


class _NoFrames(Exception):
    pass


class TrackedGaME(GaME):
    """GaME with per-Gaussian id / identity / birth / last_update kept through add and prune."""

    def __init__(self, config: dict):
        super().__init__(config, wandb_online=False)
        self.next_uid = 0
        self.frame_counter = 0
        self.uid = torch.zeros(0, dtype=torch.int64, device="cuda")
        self.identity = torch.zeros(0, dtype=torch.int64, device="cuda")
        self.last_update = torch.zeros(0, dtype=torch.int64, device="cuda")
        self.now = 0
        gm = self.gaussian_model
        post, prune = gm.densification_postfix, gm.prune_points

        def densification_postfix(new_xyz, *rest):
            n = new_xyz.shape[0]
            post(new_xyz, *rest)
            self.uid = torch.cat([self.uid, torch.arange(self.next_uid, self.next_uid + n, device="cuda")])
            self.identity = torch.cat([self.identity, torch.full((n,), -1, dtype=torch.int64, device="cuda")])
            self.last_update = torch.cat([self.last_update, torch.full((n,), self.now, dtype=torch.int64,
                                                                       device="cuda")])
            self.next_uid += n

        def prune_points(mask):
            keep = ~mask.to("cuda").bool()
            prune(mask)
            self.uid, self.identity, self.last_update = self.uid[keep], self.identity[keep], self.last_update[keep]

        gm.densification_postfix = densification_postfix
        gm.prune_points = prune_points

    def is_keyframe(self, pose: np.ndarray) -> bool:
        """[CHANGED vs published: C1, C2 in the module notes] GaME.is_keyframe (game.py:487)."""
        if not self.keyframes:
            return True
        last_pose = self.keyframes[self._last_keyframe_id]["pose"].detach().numpy()
        delta_pose = last_pose @ np.linalg.inv(pose)                          # C1 (published: inv(last) @ pose)
        translation_diff = np.linalg.norm(delta_pose[:3, 3])
        rot_euler_diff = np.abs(Rotation.from_matrix(delta_pose[:3, :3]).as_euler("xyz", degrees=True))  # C2
        return translation_diff > self.config["keyframe_translation_diff"] or np.any(rot_euler_diff > 50)

    def _sample_valid_keyframe(self, selected_frames, only_frame_id=None):
        """[CHANGED vs published: C3] GaME's sampler without its endless loop: frames found covered
        are dropped from the draw."""
        if only_frame_id is not None:
            return only_frame_id
        thresh = self.config["occlusion_ignore_threshold"]
        active = [f for f in selected_frames if f not in self.ignored_frames]
        while active:
            keyframe_id = random.choice(active)
            if keyframe_id not in self.occlusion_masks:
                return keyframe_id
            all_masks = self.keyframes[keyframe_id]["masks"].cuda()
            ignore_mask = self.occlusion_masks[keyframe_id]
            covered = (ignore_mask * all_masks).sum(dim=(1, 2)) / all_masks.sum(dim=(1, 2))
            if (covered > thresh).any():
                self.ignored_frames.add(keyframe_id)
                active.remove(keyframe_id)
            else:
                return keyframe_id
        raise _NoFrames

    def optimize_model(self, iterations=100, only_frame_id=None, refinement=False):
        """GaME.optimize_model (game.py:125-196) with two changes:
        [CHANGED vs published: C3] a step without any usable keyframe is skipped;
        [CHANGED vs published: C4] the loss mask also requires measured depth (gt_depth > 0)."""
        selected_frames = list(self.keyframes.keys())
        if len(selected_frames) == 0 or len(self.ignored_frames) == len(self.keyframes):
            print("no frames available")
            return
        background = torch.zeros(3).cuda()
        pipe = gu.flashsplat_pipe()
        for iteration in tqdm(range(iterations), "Refinement", disable=not refinement):
            try:
                keyframe_id = self._sample_valid_keyframe(selected_frames, only_frame_id)
            except _NoFrames:                                                       # C3
                self.gaussian_model.optimizer.zero_grad(set_to_none=True)
                print("no frames available")
                break
            keyframe = self.keyframes[keyframe_id]
            gt_color, gt_depth = keyframe["color"].cuda().clone(), keyframe["depth"].cuda().clone()
            pose, intrinsics = keyframe["pose"].cuda().clone(), keyframe["intrinsics"]
            flashsplat_view = gu.flashsplat_cam(gt_color, gt_depth, None, intrinsics, pose.cpu(), keyframe_id)
            del keyframe
            render_pkg = flashsplat_render(flashsplat_view, self.gaussian_model, pipe, background,
                                           obj_num=self.num_label_channels)
            image, depth, viewspace_point_tensor, visibility_filter, radii = (
                render_pkg["render"].clone(), render_pkg["depth"].clone(),
                render_pkg["viewspace_points"], render_pkg["visibility_filter"].clone(),
                render_pkg["radii"].clone())
            viewspace_point_tensor.retain_grad()
            del render_pkg
            mask = (~torch.isnan(depth)).squeeze(0).to(image.device)
            mask = mask & (gt_depth.reshape(mask.shape) > 0)                         # C4
            if self.occlusion_masks.get(keyframe_id) is not None:
                mask = mask * ~self.occlusion_masks[keyframe_id].squeeze(0).to(image.device)
            color_loss = (((1.0 - self.opt_params.lambda_dssim)
                           * l1_loss(image, gt_color, agg="none") * mask).mean()
                          + (self.opt_params.lambda_dssim
                             * (1.0 - ssim(image.unsqueeze(0), gt_color.unsqueeze(0), data_range=1.,
                                           mask=mask.unsqueeze(0).tile((3, 1, 1)).unsqueeze(0)))))
            depth_loss = (l1_loss(depth, gt_depth, agg="none") * mask).mean()
            reg_loss = self.config["isotropic_reg_weight"] * isotropic_loss(self.gaussian_model.get_scaling.clone())
            total_loss = color_loss + depth_loss + reg_loss
            total_loss.backward()
            with torch.no_grad():
                if not refinement:
                    if iteration == (iterations // 2) or iteration == iterations:
                        prune_mask = (self.gaussian_model.get_opacity.detach() < 0.1).squeeze()
                        self.gaussian_model.prune_points(prune_mask)
                else:
                    self._densification_step(iteration, total_loss, visibility_filter, radii,
                                             viewspace_point_tensor)
                self.gaussian_model.optimizer.step()
                self.gaussian_model.optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _find_added_geometry_masks(self, keyframe) -> list:
        """GaME._find_added_geometry_masks (game.py:198-232) with one change:
        [CHANGED vs published: C5] only pixels with measured depth (gt_depth > 0) are judged."""
        EPSILON = self.config["depth_change_threshold"]
        gt_depth = keyframe["depth"]
        pose, intrinsics = keyframe["pose"], keyframe["intrinsics"]
        flashsplat_view = gu.flashsplat_cam(keyframe["color"], gt_depth, None, intrinsics, pose.cpu(), None)
        render_label_pkg = flashsplat_render(flashsplat_view, self.gaussian_model, gu.flashsplat_pipe(),
                                             torch.zeros(3).cuda(), gt_mask=None, obj_num=1)
        render_depth, render_alpha = render_label_pkg["depth"], render_label_pkg["alpha"]
        valid_according_2alpha = render_alpha.squeeze() > self.config["min_opacity"]
        valid_according_2alpha = valid_according_2alpha & (gt_depth.squeeze() > 0)  # C5
        depth_diff = (gt_depth - render_depth).squeeze()
        new_geometry_indices = []
        for i, mask in enumerate(keyframe["masks"]):
            mask_alpha = mask * valid_according_2alpha
            mask_add = mask_alpha * (depth_diff < -EPSILON)
            add_area = mask_add.sum() / (mask_alpha * (depth_diff < EPSILON)).sum()
            if add_area > self.config["addition_coverage_threshold"]:
                new_geometry_indices.append(i)
        return new_geometry_indices


class GameBackend(Backend):
    name = "game"
    CHANGES = ("C1 keyframe test uses the true relative pose (published: origin-dependent)",
               "C2 keyframe rotation threshold in degrees (published: radians vs 50, never true)",
               "C3 keyframe sampler without endless loop",
               "C4 training loss only on pixels with measured depth (published: also depth 0)",
               "C5 addition test only on pixels with measured depth (published: depth 0 = addition)",
               "E1 surface export, real data only (ours, not GaME): no depth-edge pixels, only the view's measured-depth area",
               "K1 real config configs/kinect_real.yaml: tum mapping + aria change detection (published values)")

    def __init__(self, info: DatasetInfo, own_update: bool, tolerance: float = 0.05,
                 min_alpha: float = 0.5, bg_voxel: float = 0.02, obj_voxel: float = 0.01,
                 min_mask_px_full: int = 50, work_dir=None):
        super().__init__(info, own_update, work_dir)
        cfg = yaml.safe_load(CONFIGS[info.name].read_text())["game"]
        cfg.setdefault("scale", 1.0)
        self.config = cfg
        self.scale = float(cfg["scale"])
        self.step = STEP[info.name]
        self.tolerance = tolerance
        self.min_alpha = min_alpha
        self.export_e1 = info.name == "real"     # E1 on real data only: the synthetic runs keep their export
        self.bg_voxel, self.obj_voxel = bg_voxel, obj_voxel
        self.min_mask_px = max(1, min_mask_px_full // (self.step * self.step))
        self.game: Optional[TrackedGaME] = None
        self.semantic_of: Dict[int, int] = {}

    # -- session ------------------------------------------------------------------------------
    def start_session(self, spec: SessionSpec, prior: Optional[TrackedGaME]) -> None:
        if prior is None:
            if self.game is not None:
                del self.game                   # the model and its patched methods form a cycle
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            prior = TrackedGaME(dict(self.config))
        self.game = prior
        self.session = FlatSession(spec, pixel_step=1)
        self.crop = Crop(self.session.K, self.step)
        self.stamps, self.scenes = [], []

    def end_session(self) -> TrackedGaME:
        return self.game

    def prior_state(self, g: TrackedGaME) -> dict:
        """The carried model as GaME's own checkpoint does it (GaussianModel.capture: parameters and
        optimizer state), plus the keyframes themselves and our per-Gaussian bookkeeping."""
        return dict(model=g.gaussian_model.capture(), keyframes=g.keyframes, estimated_poses=g.estimated_poses,
                    occlusion_masks={k: v.cpu() for k, v in g.occlusion_masks.items()},
                    ignored_frames=g.ignored_frames, last_keyframe_id=g._last_keyframe_id,
                    next_uid=g.next_uid, frame_counter=g.frame_counter, uid=g.uid.cpu(),
                    identity=g.identity.cpu(), last_update=g.last_update.cpu(), now=g.now,
                    semantic_of=dict(self.semantic_of))

    def prior_from_state(self, s: dict) -> TrackedGaME:
        g = TrackedGaME(dict(self.config))
        g.gaussian_model.restore(s["model"], g.opt_params)      # GaME.load does the same
        g.keyframes, g.estimated_poses, g.ignored_frames = s["keyframes"], s["estimated_poses"], s["ignored_frames"]
        g.occlusion_masks = {k: v.cuda() for k, v in s["occlusion_masks"].items()}
        g._last_keyframe_id, g.next_uid, g.frame_counter, g.now = (s["last_keyframe_id"], s["next_uid"],
                                                                  s["frame_counter"], s["now"])
        g.uid, g.identity, g.last_update = s["uid"].cuda(), s["identity"].cuda(), s["last_update"].cuda()
        self.semantic_of = dict(s["semantic_of"])
        return g

    # -- frames -------------------------------------------------------------------------------
    def _sample(self, index: int) -> Tuple[dict, np.ndarray, np.ndarray]:
        f = self.session.load(index, color=True)
        c = self.crop
        depth = c(f.depth)
        instance = c(f.instance).astype(np.int64)
        semantic = c(f.semantic) if f.semantic is not None else None
        small = Frame(f.index, f.stamp_ns, depth, instance, semantic, f.T_world_cam, f.K)
        dyn = dynamic_mask(small, self.info.dynamic_semantics)
        depth = np.nan_to_num(depth, nan=0.0).astype(np.float32)
        depth[dyn] = 0.0
        for i in np.unique(instance[instance > 0]).tolist():
            if i not in self.semantic_of and semantic is not None:
                vals, cnt = np.unique(semantic[instance == i], return_counts=True)
                self.semantic_of[i] = int(vals[np.argmax(cnt)])
        pose = np.linalg.inv(f.T_world_cam)                     # world-to-camera
        pose[:3, 3] *= self.scale
        sample = dict(color=np.ascontiguousarray(c(f.color)), depth=depth * self.scale,
                      masks=self._masks(instance, semantic, dyn), pose=pose.astype(np.float32),
                      intrinsics=c.K)
        return sample, instance, dyn

    def _masks(self, instance: np.ndarray, semantic: Optional[np.ndarray], dyn: np.ndarray) -> torch.Tensor:
        masks = []
        ids, counts = np.unique(instance[instance > 0], return_counts=True)
        masks += [instance == i for i, n in zip(ids, counts) if n >= self.min_mask_px]
        if semantic is not None:
            rest = (instance <= 0) & ~dyn
            for lab in np.unique(semantic[rest]).tolist():
                n, comp = cv.connectedComponents((rest & (semantic == lab)).astype(np.uint8), connectivity=4)
                for k in range(1, n):
                    m = comp == k
                    if m.sum() >= self.min_mask_px:
                        masks.append(m)
        h, w = instance.shape
        return torch.from_numpy(np.stack(masks)) if masks else torch.zeros((0, h, w), dtype=torch.bool)

    def integrate(self, frame: Frame) -> None:
        """One frame of GaME.train (game.py:599-645) at the layer's rate; keyframes as GaME decides."""
        g = self.game
        g.now = frame.stamp_ns
        frame_id = g.frame_counter
        g.frame_counter += 1
        pose = np.linalg.inv(frame.T_world_cam)                 # world-to-camera, scaled
        pose[:3, 3] *= self.scale
        pose = pose.astype(np.float32)
        g.estimated_poses[frame_id] = pose
        if not g.is_keyframe(pose):
            return
        sample, instance, dyn = self._sample(frame.index)
        kf = {"color": gu.np2torch(sample["color"], device="cuda").permute(2, 0, 1) / 255.0,
              "depth": gu.np2torch(sample["depth"], device="cuda"),
              "masks": sample["masks"].cuda(),
              "pose": gu.np2torch(pose, device="cuda"),
              "intrinsics": sample["intrinsics"]}
        first = not g.keyframes
        if self.own_update and len(g.keyframes) > 2:
            g.detect_additions(kf)
        g.keyframes[frame_id] = gu.dict2device(kf, "cpu")
        g._last_keyframe_id = frame_id
        if dyn.any():
            g.occlusion_masks[frame_id] = torch.from_numpy(dyn).cuda()
        g._add_gaussians(kf["color"], kf["depth"], None, kf["pose"], kf["intrinsics"])
        self._seed_identity(instance, pose, sample["intrinsics"])
        g.optimize_model(50, only_frame_id=frame_id)
        if self.own_update:
            g.detect_removals(frame_id, kf)
        g.optimize_model(g.config["first_keyframe_iters"] if first else g.config["keyframe_iters"])
        self._mark_support(kf, frame.stamp_ns)

    @torch.no_grad()
    def _seed_identity(self, instance: np.ndarray, pose: np.ndarray, K: np.ndarray) -> None:
        g = self.game
        new = torch.nonzero(g.identity < 0).squeeze(1)
        if not len(new):
            return
        T = torch.as_tensor(pose, dtype=torch.float32, device="cuda")
        cam = g.gaussian_model.get_xyz[new] @ T[:3, :3].T + T[:3, 3]
        u = torch.round(K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]).long()
        v = torch.round(K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]).long()
        h, w = instance.shape
        ok = (cam[:, 2] > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        inst = torch.as_tensor(instance, device="cuda")
        ident = torch.zeros(len(new), dtype=torch.int64, device="cuda")
        ident[ok] = inst[v[ok], u[ok]].clamp(min=0)
        g.identity[new] = ident

    @torch.no_grad()
    def _mark_support(self, kf: dict, stamp: int) -> None:
        g = self.game
        _, h, w = kf["color"].shape
        view = gu.flashsplat_cam(kf["color"], kf["depth"], None, kf["intrinsics"], kf["pose"].cpu(), None)
        pkg = flashsplat_render(view, g.gaussian_model, gu.flashsplat_pipe(), torch.zeros(3).cuda(),
                                gt_mask=None, obj_num=1)
        xy, gsd = pkg["proj_xy"], pkg["gs_depth"]
        idx = torch.nonzero(pkg["visibility_filter"] & (xy[0] > 0) & (xy[0] < w) & (xy[1] > 0) &
                            (xy[1] < h)).squeeze(1)
        if not len(idx):
            return
        px = xy[:, idx].to(torch.int64)
        d = kf["depth"][px[1], px[0]]
        on = (d > 0) & ((d - gsd[idx]).abs() <= self.tolerance * self.scale)
        g.last_update[idx[on]] = stamp

    # -- the layer's view ---------------------------------------------------------------------
    @torch.no_grad()
    def elements(self) -> Elements:
        g = self.game
        xyz = g.gaussian_model.get_xyz.detach() / self.scale
        n = len(xyz)
        sigma = g.gaussian_model.get_scaling.detach().max(dim=1).values / self.scale
        return Elements(g.uid.clone(), xyz.float(), torch.full((n, 3), float("nan"), device="cuda"),
                        g.identity.clamp(min=0), g.last_update.clone(), (3.0 * sigma).float())

    @torch.no_grad()
    def retire(self, ids: torch.Tensor, stamp: int) -> None:
        g = self.game
        mask = torch.isin(g.uid, ids.to("cuda"))
        if not mask.any():
            return
        pipe, bg = gu.flashsplat_pipe(), torch.zeros(3).cuda()
        for kid, kf in g.keyframes.items():
            view = gu.flashsplat_cam(kf["color"].cuda(), kf["depth"].cuda(), None, kf["intrinsics"],
                                     kf["pose"], kid)
            pkg = flashsplat_render(view, g.gaussian_model, pipe, bg, obj_num=1, used_mask=mask)
            seen = (pkg["alpha"] > g.config["min_opacity"]).squeeze()
            if seen.any():
                prev = g.occlusion_masks.get(kid)
                g.occlusion_masks[kid] = seen if prev is None else (prev.to(seen.device) | seen)
        g.gaussian_model.prune_points(mask)

    # -- evaluation ---------------------------------------------------------------------------
    def snapshot(self, stamp: int) -> None:
        self.stamps.append(int(stamp))
        self.scenes.append(self._render_scene(stamp))

    def timeline(self) -> SceneListTimeline:
        return SceneListTimeline(self.stamps, self.scenes)

    @torch.no_grad()
    def _render_scene(self, t_ns: int) -> EvaluationScene:
        g = self.game
        gm = g.gaussian_model
        empty = EvaluationScene(t_ns, np.zeros((0, 3), np.float32), np.zeros(0, np.uint32), [],
                                np.zeros((0, 3), np.float32))
        if gm.get_xyz.shape[0] == 0 or not g.keyframes:
            return empty
        idf = g.identity.clamp(min=0).to(torch.float32)
        codes = torch.stack([idf, idf * idf, torch.ones_like(idf)], dim=1)
        pipe, bg = gu.flashsplat_pipe(), torch.zeros(3).cuda()
        # One sample per voxel (and per identity for objects), the first in render order. Merged after
        # every keyframe view, so memory is bounded by the map, not by (keyframes x pixels): the
        # one-shot union over all views (every keyframe carried from A and B too) filled the GPU on
        # real C (2026-10-01) and returned a garbage index. Same result: the kept sample stays first.
        empty3 = torch.zeros((0, 3), device="cuda")
        acc_bg = [empty3, torch.zeros(0, dtype=torch.int64, device="cuda"), empty3]
        acc_ob = [empty3, torch.zeros(0, dtype=torch.int64, device="cuda"), empty3]

        def union(run, q, ids, nn, voxel, with_id):
            q, ids, nn = torch.cat([run[0], q]), torch.cat([run[1], ids]), torch.cat([run[2], nn])
            if len(q) == 0:
                return [q, ids, nn]
            cols = torch.floor(q / voxel).long()
            if with_id:
                cols = torch.cat([cols, ids[:, None]], dim=1)
            _, inv = torch.unique(cols, dim=0, return_inverse=True)
            first = torch.full((int(inv.max()) + 1,), len(q), dtype=torch.int64, device="cuda")
            first.scatter_reduce_(0, inv, torch.arange(len(q), device="cuda"), reduce="amin")
            return [q[first], ids[first], nn[first]]

        for kid, kf in g.keyframes.items():
            K = kf["intrinsics"]
            _, h, w = kf["color"].shape
            pose = kf["pose"].cpu()
            view = gu.flashsplat_cam(torch.zeros((3, h, w), device="cuda"), torch.zeros((h, w), device="cuda"),
                                     None, K, pose, None)
            pkg = flashsplat_render(view, gm, pipe, bg, override_color=codes, obj_num=1)
            alpha = pkg["alpha"].squeeze()
            full = pkg["depth"].squeeze() / alpha.clamp(min=1e-6)
            # [E1] no sample where the alpha-blended depth mixes layers (a >5% jump to a neighbour) and
            # none outside the part of this view that had measured depth (the part GaME is trained on, C4)
            jump = torch.zeros_like(full)
            jump[:, 1:-1] = (full[:, 2:] - full[:, :-2]).abs()
            jump[1:-1, :] = torch.maximum(jump[1:-1, :], (full[2:, :] - full[:-2, :]).abs())
            ok = alpha > self.min_alpha
            if self.export_e1:
                ok &= (jump <= 0.05 * full) & (kf["depth"].to(full.device) > 0)
            depth = torch.where(ok, full, torch.zeros_like(alpha))
            depth = depth / self.scale
            ok &= depth > 0
            img = pkg["render"]
            weight = img[2].clamp(min=1e-6)
            mean = img[0] / weight
            var = img[1] / weight - mean * mean
            v, u = torch.nonzero(ok, as_tuple=True)
            z = depth[v, u]
            x = (u.float() - K[0, 2]) / K[0, 0] * z
            y = (v.float() - K[1, 2]) / K[1, 1] * z
            T_world_cam = torch.linalg.inv(pose.to(torch.float64))
            T_world_cam[:3, 3] /= self.scale
            Tw = T_world_cam.to("cuda", torch.float32)
            p = torch.stack([x, y, z], 1) @ Tw[:3, :3].T + Tw[:3, 3]
            pure = var[v, u] < 0.25
            ident = torch.where(pure, torch.round(mean[v, u]).long(), torch.full_like(v, -1))
            # normals from rendered-depth differences (as frames.torch_pixel_normals)
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
            n_w = nc[v, u] @ Tw[:3, :3].T
            b, o = ident == 0, ident > 0                       # mixed-identity pixels (-1) are dropped
            acc_bg = union(acc_bg, p[b], ident[b], n_w[b], self.bg_voxel, False)
            acc_ob = union(acc_ob, p[o], ident[o], n_w[o], self.obj_voxel, True)
        if not len(acc_bg[0]) and not len(acc_ob[0]):
            return empty
        bp, bi, bn = acc_bg
        op, oi, on = acc_ob
        objects = []
        oi_np, op_np, on_np = oi.cpu().numpy(), op.cpu().numpy(), on.cpu().numpy().astype(np.float32)
        for i in np.unique(oi_np):
            m = oi_np == i
            objects.append(SceneObject(id=str(int(i)), instance_id=int(i),
                                       semantic=int(self.semantic_of.get(int(i), -1)),
                                       points=op_np[m].astype(np.float32), present=True,
                                       first_observed_ns=[0], last_observed_ns=[UINT64_MAX],
                                       normals=on_np[m]))
        return EvaluationScene(t_ns, bp.cpu().numpy().astype(np.float32), np.zeros(len(bp), np.uint32),
                               objects, bn.cpu().numpy().astype(np.float32))
