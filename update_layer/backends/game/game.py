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
  [C1, C2] sources (checked 2026-10-07; comments only, no change of behaviour):
       paper      GaME sec. 4.3 (as read by the supervisor, 10-07): "Following [11, 40], keyframes are selected
                  every time a frame exceeds a translation theta_translation or a rotation theta_rotation
                  threshold"; [11] SplaTAM, [40] Gaussian-SLAM (the same first author, Yugay).
       Gaussian-SLAM  src/utils/mapper_utils.py exceeds_motion_thresholds (github.com/VladimirYugay/Gaussian-SLAM,
                  file at commit ce8c887, 2024-05-17; fetched and read 10-07): delta = inv(last_submap_c2w) @
                  current_c2w with both poses camera-to-world (the relative motion, origin-independent), the
                  rotation through rotation_to_euler, which returns degrees (x 180 / pi), thresholds 50 deg and 0.5.
       GaME       datasets.py:384 pose = np.linalg.inv(self.poses[idx]) inverts the files' camera-to-world poses
                  into world-to-camera (its inline comment "Camera to world" is wrong; the docstring at :341 says
                  world-to-camera); utils.py:49-50 (flashsplat_cam: R = inverse(pose)[:3, :3], T = pose[:3, 3])
                  uses the world-to-camera convention; game.py:503 keeps Gaussian-SLAM's inv(last) @ now on these
                  world-to-camera poses, so its translation depends on the origin (C1); game.py:505-507 as_euler
                  without degrees=True compares radians with 50, never true (C2). C1/C2 restore what sec. 4.3 and
                  Gaussian-SLAM do. GaME's GitHub had one issue on 2026-10-07 (an Aria dataset question, as
                  reported by the supervisor); neither point was reported there.
       effect     full-trajectory simulation (10-07, 30 Hz input, flat config): keyframes published / C1+C2 / in
                  both: real A 478 / 209 / 36, B 440 / 189 / 17, C 784 / 420 / 61; synthetic A 2548 / 544 / 364,
                  B 1933 / 650 / 394 (records/experiments_20261005.csv): rows with C1/C2 select keyframes as the
                  paper describes, not as the released code does.
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
Coupling with the update layer (ours, not a GaME change):
  [I1] identity of each Gaussian (ours, since 2026-10-06): the FlashSplat optimal assignment (Shen et al.,
       ECCV 2024) of the instance masks, accumulated over keyframes (GameBackend._assign_identity). Before:
       the instance at the Gaussian centre's pixel in the keyframe after it was created, never updated; on real
       A (rows 3/4, 2026-10-01) 34-47% of the measured surface of objects 5/7/10/19 carried background identity.
  [T1] time-indexed maps (ours, since 2026-10-06; replaces R1). Khronos eq. (5): the measurement of time t is
       generated by the scene of time t. GaME renders every keyframe with the current Gaussians, i.e. it assumes
       a static scene, so a changed scene leaves stale keyframes that grow old content back (GaME sec. 4.3,
       table 4). Here every Gaussian has a birth and an end; keyframe k renders and trains only the
       Gaussians alive at its stamp t_k (opacity x alive). A Gaussian of identity l belongs to the object state
       of l that was open when it was created (layer state intervals) and lives from that state's birth until it
       ends; a background Gaussian lives from -infinity (a static wall exists at every keyframe, also those before
       GaME seeded it; since 2026-10-06 14:23, before: birth = creation stamp, t1.py); the layer's element
       retirements end a Gaussian at that round. Consequences, all from the same loss: a new state is not trained by keyframes
       from before it existed (README R03), keyframes from before a new object arrived do not erode it but
       still train what lies behind it (R04), ended states keep the geometry of their time (history, 4D:
       the map of time t = the Gaussians alive at t), and nothing is masked by hand.
  [A1] GaME's addition handling (detect_additions) as published in every row (since 2026-10-06; before: only
       with own update on). Only the removal decision is the layer's in rows 4/5.
  [R1] (until T1) retire() (row 4) renders the retired Gaussians into every stored keyframe and excludes those pixels
       from that keyframe's training loss, so old keyframes do not grow the retired content back. Since
       2026-10-01 17:45 these pixels are kept in TrackedGaME.retired_masks, used by the loss only. Before,
       they were written into GaME's occlusion_masks, which also drive GaME's keyframe-ignore rule (a
       keyframe whose segment mask is > occlusion_ignore_threshold covered is never trained again): real
       row 4, session A dropped 370 of its 391 keyframes, the map stayed untrained (46% of its background
       samples in free space against the measured depth, points backend 11%). GaME's own removals (row 3)
       still use occlusion_masks as published.
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
  * own update on = GaME's removal handling (detect_removals) as published (additions: A1, every row); off =
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
    [R2, since 2026-10-06, every row and both datasets] the default snapshot readout is the first echo instead
    (readout.py): depth and identity of the Gaussian at which T first drops to 0.5 (the median depth of the
    median-depth export), measured-depth pixels only (C4); no variance or jump threshold.
The GaME model (Gaussians + keyframes) is carried across sessions as GaME's own multi-session
setting does.
"""
from __future__ import annotations

import math
import os
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
from ...frames import FlatSession, Frame, Intrinsics, SessionSpec, dynamic_mask, motion_mask
from ...interface import Backend, DatasetInfo, Elements
from . import t1

GAME = Path("/home/jixian/Desktop/FT/baselines/GaME")
if str(GAME) not in sys.path:
    sys.path.insert(0, str(GAME))
from src.entities.game import GaME  # noqa: E402
from src.entities.losses import isotropic_loss, l1_loss  # noqa: E402
from src.flashsplat.gaussian_renderer import GaussianModel, flashsplat_render  # noqa: E402
from src.flashsplat.utils.general_utils import build_rotation as _build_rotation  # noqa: E402
from src.utils import utils as gu  # noqa: E402

CONFIGS = {"synthetic": GAME / "configs/flat/flat.yaml",
           "real": Path(__file__).resolve().parent / "configs/kinect_real.yaml"}   # tum mapping + aria change detection
STEP = {"synthetic": 1, "real": 2}      # run.INPUT_STEP: the input every backend integrates (since 2026-10-01 20:10;
                                        # before: 2 / 4, the update layer's resolution)


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


INT64_MAX = torch.iinfo(torch.int64).max


def probe_render(view, pc, gt_mask: Optional[torch.Tensor] = None, obj_num: int = 1,
                 meas_depth: Optional[torch.Tensor] = None, vote_tol: float = 0.0):
    """Forward-only render with the update_layer build of the FlashSplat rasterizer (ul_rasterizer/): the same
    image, depth, alpha and per-label weights as flashsplat_render, plus the median depth (the depth where the
    transmittance first drops to 0.5; 2DGS's median depth) and, with vote_tol > 0 and the measured z-depth of
    the view, label weights only from Gaussians within vote_tol of the measured depth of the pixel.
    Returns dict(render, depth, alpha, median, used_count, radii). Opacity follows pc.get_opacity (T1 alive)."""
    from ul_flashsplat_rasterization import _C as ulc
    H, W = int(view.image_height), int(view.image_width)
    gt = gt_mask if gt_mask is not None else torch.ones(1, W, H, device="cuda")
    meas = meas_depth.float().contiguous() if meas_depth is not None else torch.empty(0, device="cuda")
    color, depth, alpha, median, used, radii = ulc.rasterize_gaussians_probe(
        gt.float().contiguous(), torch.tensor([], dtype=torch.int), torch.zeros(3, device="cuda"),
        pc.get_xyz, torch.Tensor([]), pc.get_opacity, pc.get_scaling, pc.get_rotation, 1.0, torch.Tensor([]),
        view.world_view_transform, view.full_proj_transform, math.tan(view.FoVx * 0.5), math.tan(view.FoVy * 0.5),
        H, W, pc.get_features, pc.active_sh_degree, view.camera_center, False, obj_num, False, meas, float(vote_tol))
    return dict(render=color, depth=depth, alpha=alpha, median=median, used_count=used, radii=radii)


def probe_render_fe(view, pc):
    """probe_render with the v2 build (ul_rasterizer_v2/): also the index of the Gaussian at which the transmittance
    first drops to 0.5 (the first echo; -1 where it never does). Returns dict(render, depth, alpha, median,
    median_index, radii)."""
    from ul_flashsplat_rasterization_v2 import _C as ulc2
    H, W = int(view.image_height), int(view.image_width)
    gt = torch.ones(1, W, H, device="cuda")
    color, depth, alpha, median, used, radii, median_index = ulc2.rasterize_gaussians_probe(
        gt, torch.tensor([], dtype=torch.int), torch.zeros(3, device="cuda"),
        pc.get_xyz, torch.Tensor([]), pc.get_opacity, pc.get_scaling, pc.get_rotation, 1.0, torch.Tensor([]),
        view.world_view_transform, view.full_proj_transform, math.tan(view.FoVx * 0.5), math.tan(view.FoVy * 0.5),
        H, W, pc.get_features, pc.active_sh_degree, view.camera_center, False, 1, False,
        torch.empty(0, device="cuda"), 0.0)
    return dict(render=color, depth=depth, alpha=alpha, median=median, median_index=median_index, radii=radii)


class _TimedGaussianModel(GaussianModel):
    """[T1] GaME's Gaussian model whose rendering can be restricted to the Gaussians alive at one time:
    `alive` (bool per Gaussian, None = all) multiplies the opacity, so a Gaussian that is not alive adds
    nothing to the render and receives no gradient. Parameters and optimizer state are unchanged."""
    alive = None

    @property
    def get_opacity(self):
        o = self.opacity_activation(self._opacity)
        return o if self.alive is None else o * self.alive[:, None].to(o.dtype)

    @property
    def raw_opacity(self):
        return self.opacity_activation(self._opacity)

    # [T1] GaussianModel.densify_and_split / densify_and_clone (flashsplat/scene/gaussian_model.py:506-577)
    # unchanged except that the rows of the parents are recorded (self.parents) before the new Gaussians are
    # appended, so that a clone or split child inherits its parent's creation, end and identity.
    parents = None

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, limit_num=-1):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)
        if limit_num > 0:
            idx = padded_grad.argsort(dim=0, descending=True)
            sorted_pts_mask = torch.zeros(padded_grad.size(0), dtype=torch.bool, device=grads.device)
            inc_num = limit_num - self.get_num_pts
            if inc_num <= 0:
                return
            inc_num = min(inc_num, selected_pts_mask.sum().item())
            sorted_pts_mask[idx[:inc_num]] = 1
            selected_pts_mask = torch.logical_and(selected_pts_mask, sorted_pts_mask)
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = _build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        self.parents = torch.nonzero(selected_pts_mask).squeeze(1).repeat(N)                  # [T1]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, limit_num):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)
        if limit_num > 0:
            idx = grads.argsort(dim=0, descending=True)
            sorted_pts_mask = torch.zeros(grads.size(0), dtype=torch.bool, device=grads.device)
            inc_num = limit_num - self.get_num_pts
            if inc_num <= 0:
                return
            inc_num = min(inc_num, selected_pts_mask.sum().item())
            sorted_pts_mask[idx[:inc_num]] = 1
            selected_pts_mask = torch.logical_and(selected_pts_mask, sorted_pts_mask)
        self.parents = torch.nonzero(selected_pts_mask).squeeze(1)                             # [T1]
        self.densification_postfix(self._xyz[selected_pts_mask], self._features_dc[selected_pts_mask],
                                   self._features_rest[selected_pts_mask], self._opacity[selected_pts_mask],
                                   self._scaling[selected_pts_mask], self._rotation[selected_pts_mask])


class TrackedGaME(GaME):
    """GaME with per-Gaussian id / identity / birth / last_update kept through add and prune."""

    def __init__(self, config: dict):
        super().__init__(config, wandb_online=False)
        # [M1] num_label_channels (flat.yaml 256) is only the obj_num of renders without a gt_mask (GaME game.py:159
        # training, :444 removal, :570 seeding; our optimize_model): obj_num only sizes FlashSplat's label accumulation
        # buffer used_count = (obj_num + 1) x N floats, which without a gt_mask is written in row 0 alone, is read by no
        # GaME code and does not enter the gradients (mask_grad False). 1 instead of 256: identical renders and
        # gradients (checked bitwise), ~1 KB less GPU memory per Gaussian in every such render.
        self.num_label_channels = 1
        self.next_uid = 0
        self.frame_counter = 0
        self.retired_masks: Dict[int, torch.Tensor] = {}    # R1: per keyframe, where the layer's retired Gaussians were
        self.uid = torch.zeros(0, dtype=torch.int64, device="cuda")
        self.identity = torch.zeros(0, dtype=torch.int64, device="cuda")
        self.last_update = torch.zeros(0, dtype=torch.int64, device="cuda")
        # I1: accumulated FlashSplat label weights, column j = physical identity label_ids[j] (0 = background)
        self.label_ids = [0]
        self.label_weight = torch.zeros((0, 1), dtype=torch.float32, device="cuda")
        # T1: per Gaussian its creation stamp and two ends: the end of the object state it belongs to
        # (set_state_intervals) and the end decided by the layer's evidence for this element (retire)
        self.created = torch.zeros(0, dtype=torch.int64, device="cuda")
        self.state_birth = torch.zeros(0, dtype=torch.int64, device="cuda")   # birth of its object state (t1.py)
        self.death_state = torch.zeros(0, dtype=torch.int64, device="cuda")
        self.death_evidence = torch.zeros(0, dtype=torch.int64, device="cuda")
        self.kf_stamp: Dict[int, int] = {}                  # keyframe id -> its sensor stamp (ns)
        self.now = 0
        gm = self.gaussian_model
        gm.__class__ = _TimedGaussianModel
        post, prune = gm.densification_postfix, gm.prune_points

        def densification_postfix(new_xyz, *rest):
            n = new_xyz.shape[0]
            par, gm.parents = gm.parents, None                    # T1: rows of the parents (densify) or None (seeding)
            post(new_xyz, *rest)
            if par is not None and len(par) == n:
                self.uid = torch.cat([self.uid, torch.arange(self.next_uid, self.next_uid + n, device="cuda")])
                self.identity = torch.cat([self.identity, self.identity[par]])
                self.last_update = torch.cat([self.last_update, self.last_update[par]])
                self.label_weight = torch.cat([self.label_weight, self.label_weight[par]])
                self.created = torch.cat([self.created, self.created[par]])
                self.state_birth = torch.cat([self.state_birth, self.state_birth[par]])
                self.death_state = torch.cat([self.death_state, self.death_state[par]])
                self.death_evidence = torch.cat([self.death_evidence, self.death_evidence[par]])
                self.next_uid += n
                return
            self.uid = torch.cat([self.uid, torch.arange(self.next_uid, self.next_uid + n, device="cuda")])
            self.identity = torch.cat([self.identity, torch.full((n,), -1, dtype=torch.int64, device="cuda")])
            self.last_update = torch.cat([self.last_update, torch.full((n,), self.now, dtype=torch.int64,
                                                                       device="cuda")])
            self.label_weight = torch.cat([self.label_weight, self.label_weight.new_zeros((n, len(self.label_ids)))])
            self.created = torch.cat([self.created, torch.full((n,), self.now, dtype=torch.int64, device="cuda")])
            self.state_birth = torch.cat([self.state_birth, torch.full((n,), INT64_MAX, dtype=torch.int64,
                                                                       device="cuda")])
            self.death_state = torch.cat([self.death_state, torch.full((n,), INT64_MAX, dtype=torch.int64, device="cuda")])
            self.death_evidence = torch.cat([self.death_evidence,
                                             torch.full((n,), INT64_MAX, dtype=torch.int64, device="cuda")])
            self.next_uid += n

        def prune_points(mask):
            keep = ~mask.to("cuda").bool()
            prune(mask)
            self.uid, self.identity, self.last_update = self.uid[keep], self.identity[keep], self.last_update[keep]
            self.label_weight = self.label_weight[keep]
            self.created, self.death_state, self.death_evidence = (self.created[keep], self.death_state[keep],
                                                                   self.death_evidence[keep])
            self.state_birth = self.state_birth[keep]

        gm.densification_postfix = densification_postfix
        gm.prune_points = prune_points

    timed = False         # T1 is on once the layer drives this map (state intervals / retirements); rows 1-3: off

    def alive_at(self, t: Optional[int]) -> Optional[torch.Tensor]:
        """[T1] Gaussians that exist in the map of time t: born at or before t and not yet ended (birth: background
        -infinity, an object Gaussian its state's birth; t1.py). None (= all, GaME as published) when no layer
        drives this map."""
        if t is None or not self.timed:
            return None
        return t1.alive_at(t, self.identity, self.created, self.state_birth, self.death_state, self.death_evidence)

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
            # T1: a keyframe observed the scene of its own time, so it renders (and trains) only the
            # Gaussians alive at its stamp
            self.gaussian_model.alive = self.alive_at(self.kf_stamp.get(keyframe_id))
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
            if keyframe_id in self.retired_masks:                                    # R1
                mask = mask * ~self.retired_masks[keyframe_id].squeeze(0).to(image.device)
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
                        prune_mask = (self.gaussian_model.raw_opacity.detach() < 0.1).squeeze()   # T1: own opacity
                        self.gaussian_model.prune_points(prune_mask)
                else:
                    self.gaussian_model.alive = None              # T1: densify/prune on the Gaussians' own opacity
                    self._densification_step(iteration, total_loss, visibility_filter, radii,
                                             viewspace_point_tensor)
                self.gaussian_model.optimizer.step()
                self.gaussian_model.optimizer.zero_grad(set_to_none=True)
            self.gaussian_model.alive = None
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
    consumes_state_intervals = True                                                 # T1
    CHANGES = ("C1 keyframe test uses the true relative pose (published: origin-dependent)",
               "C2 keyframe rotation threshold in degrees (published: radians vs 50, never true)",
               "C3 keyframe sampler without endless loop",
               "C4 training loss only on pixels with measured depth (published: also depth 0)",
               "C5 addition test only on pixels with measured depth (published: depth 0 = addition)",
               "E1 surface export, real data only (ours, not GaME): no depth-edge pixels, only the view's measured-depth area",
               "K1 real config configs/kinect_real.yaml: tum mapping + aria change detection (published values)",
               "T1 time-indexed maps: each keyframe renders/trains only the Gaussians alive at its stamp; "
               "Gaussians follow their object state's interval; layer retirements end Gaussians instead of pruning",
               "I1 Gaussian identity = FlashSplat optimal assignment of the instance masks, accumulated per keyframe",
               "F2 GaME's published final refinement (refinement_iters) once after the chain's last session, as GaME "
               "after all runs (all rows); sessions before it are not refined; outputs before (pre_ref) and after (post_ref)",
               "R2 snapshot readout = first echo (T first <= 0.5: median depth and its Gaussian's identity), all rows",
               "A1 GaME's addition handling as published in every row (removals: own update / the layer)",
               "M1 renders without a gt_mask use obj_num 1 instead of 256 (the unread label buffer; identical results)")

    def __init__(self, info: DatasetInfo, own_update: bool, tolerance: float = 0.05,
                 min_alpha: float = 0.5, bg_voxel: float = 0.02, obj_voxel: float = 0.01,
                 min_mask_px_full: int = 50, work_dir=None):
        super().__init__(info, own_update, work_dir)
        # GAME_CONFIG=<yaml> replaces the dataset's config (debugging: configs/flat/flat.yaml trains 50 iterations
        # per keyframe instead of kinect_real's 600, ~10x faster); recorded in run.json backend_changes.
        path = Path(os.environ["GAME_CONFIG"]) if os.environ.get("GAME_CONFIG") else CONFIGS[info.name]
        if path != CONFIGS[info.name]:
            self.CHANGES = tuple(c for c in self.CHANGES if not c.startswith("K1")) + (
                f"DEBUG config {path} (not the dataset's {CONFIGS[info.name].name})",)
        cfg = yaml.safe_load(path.read_text())["game"]
        # GAME_IDENTITY=seed turns I1 off (ablation only): identity from the seeding projection, as before I1.
        self.identity_mode = os.environ.get("GAME_IDENTITY", "flashsplat")
        if self.identity_mode == "seed":
            self.CHANGES = tuple(c for c in self.CHANGES if not c.startswith("I1")) + (
                "ABLATION I1 off: identity from the seeding projection (GAME_IDENTITY=seed)",)
        cfg.setdefault("scale", 1.0)
        self.config = cfg
        self.scale = float(cfg["scale"])
        self.step = STEP[info.name]
        self.tolerance = tolerance
        self.min_alpha = min_alpha
        self.export_e1 = info.name == "real"     # E1 on real data only: the synthetic runs keep their export
        # snapshot readout: "e1" (alpha-weighted class-id mean, dropped when its variance >= 0.25, depth D/alpha with
        # the 5% jump test) or "first_echo" (readout.py: depth and identity of the Gaussian at which T first drops to
        # 0.5 -- the median depth of the median-depth export; measured-depth pixels only, C4)
        self.readout = "first_echo"                                                 # R2
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
                    retired_masks={k: v.cpu() for k, v in g.retired_masks.items()},
                    ignored_frames=g.ignored_frames, last_keyframe_id=g._last_keyframe_id,
                    next_uid=g.next_uid, frame_counter=g.frame_counter, uid=g.uid.cpu(),
                    identity=g.identity.cpu(), last_update=g.last_update.cpu(), now=g.now,
                    label_ids=list(g.label_ids), label_weight=g.label_weight.cpu(),
                    created=g.created.cpu(), death_state=g.death_state.cpu(), death_evidence=g.death_evidence.cpu(),
                    state_birth=g.state_birth.cpu(),
                    kf_stamp=dict(g.kf_stamp), timed=g.timed,
                    semantic_of=dict(self.semantic_of))

    def prior_from_state(self, s: dict) -> TrackedGaME:
        g = TrackedGaME(dict(self.config))
        g.gaussian_model.restore(s["model"], g.opt_params)      # GaME.load does the same
        g.keyframes, g.estimated_poses, g.ignored_frames = s["keyframes"], s["estimated_poses"], s["ignored_frames"]
        g.occlusion_masks = {k: v.cuda() for k, v in s["occlusion_masks"].items()}
        g.retired_masks = {k: v.cuda() for k, v in s.get("retired_masks", {}).items()}
        g._last_keyframe_id, g.next_uid, g.frame_counter, g.now = (s["last_keyframe_id"], s["next_uid"],
                                                                  s["frame_counter"], s["now"])
        g.uid, g.identity, g.last_update = s["uid"].cuda(), s["identity"].cuda(), s["last_update"].cuda()
        n = len(g.uid)
        if "created" in s:                                                          # T1
            g.created, g.death_state, g.death_evidence = (s["created"].cuda(), s["death_state"].cuda(),
                                                          s["death_evidence"].cuda())
            g.state_birth = (s["state_birth"].cuda() if "state_birth" in s
                             else torch.full_like(g.created, INT64_MAX))       # older checkpoints: created
            g.kf_stamp = dict(s["kf_stamp"])
            # checkpoints written before the flag existed: the layer drove the map iff some Gaussian has an end
            g.timed = bool(s["timed"]) if "timed" in s else bool(
                (g.death_evidence < INT64_MAX).any() or (g.death_state < INT64_MAX).any())
        else:
            g.created = torch.zeros(n, dtype=torch.int64, device="cuda")
            g.state_birth = torch.full((n,), INT64_MAX, dtype=torch.int64, device="cuda")
            g.death_state = torch.full((n,), INT64_MAX, dtype=torch.int64, device="cuda")
            g.death_evidence = torch.full((n,), INT64_MAX, dtype=torch.int64, device="cuda")
        if "label_weight" in s:                                                     # I1
            g.label_ids, g.label_weight = list(s["label_ids"]), s["label_weight"].cuda()
        else:
            g.label_weight = torch.zeros((len(g.uid), 1), dtype=torch.float32, device="cuda")
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
        motion = motion_mask(f)                  # the layer's D1 motion pixels (rows 4/5), full resolution
        if motion is not None:
            dyn = dyn | c(motion)
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
        if len(g.keyframes) > 2:                          # A1: GaME's addition handling, as published, every row
            g.gaussian_model.alive = g.alive_at(g.now)    # T1: GaME's current model = the map of now
            g.detect_additions(kf)
            g.gaussian_model.alive = None
        g.keyframes[frame_id] = gu.dict2device(kf, "cpu")
        g.kf_stamp[frame_id] = frame.stamp_ns                                       # T1
        g._last_keyframe_id = frame_id
        if dyn.any():
            g.occlusion_masks[frame_id] = torch.from_numpy(dyn).cuda()
        g.gaussian_model.alive = g.alive_at(g.now)       # T1: seed where the map of now explains nothing
        g._add_gaussians(kf["color"], kf["depth"], None, kf["pose"], kf["intrinsics"])
        g.gaussian_model.alive = None
        self._seed_identity(instance, pose, sample["intrinsics"])
        g.optimize_model(50, only_frame_id=frame_id)
        if self.own_update:
            g.detect_removals(frame_id, kf)
        g.optimize_model(g.config["first_keyframe_iters"] if first else g.config["keyframe_iters"])
        g.gaussian_model.alive = g.alive_at(g.now)
        self._mark_support(kf, frame.stamp_ns)
        self._assign_identity(kf, instance, dyn)                                    # I1
        g.gaussian_model.alive = None

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
    def _assign_identity(self, kf: dict, instance: np.ndarray, dyn: np.ndarray) -> None:
        """[I1] Identity of every Gaussian = the FlashSplat optimal assignment (Shen et al., ECCV 2024).

        Rendering a label image is linear in the per-Gaussian labels (alpha blending), so the labelling
        that best reproduces the 2D instance masks over all views is solved in closed form: per Gaussian,
        sum its blending weights alpha*T over every pixel of each label and take the largest sum. The
        FlashSplat rasterizer in GaME's repository accumulates exactly these sums ('used_count' for a
        gt_mask of label indices); GaME itself calls it with gt_mask=None and assigns by Gaussian centre.
        Sums are accumulated keyframe by keyframe, each with the model as optimised on that keyframe.
        Pixels of people and pixels without measured depth do not vote (label index 0 of the kernel).
        A Gaussian that has no weight yet keeps its seeded identity."""
        g = self.game
        if self.identity_mode == "seed" or not len(g.label_weight):
            return
        col = {lab: j for j, lab in enumerate(g.label_ids)}
        new = [int(i) for i in np.unique(instance[instance > 0]).tolist() if int(i) not in col]
        if new:
            g.label_ids += new
            col.update({lab: len(col) + k for k, lab in enumerate(new)})
            g.label_weight = torch.cat([g.label_weight, g.label_weight.new_zeros((len(g.label_weight), len(new)))], 1)
        lut = np.zeros(int(max(g.label_ids)) + 1, dtype=np.float32)
        for lab, j in col.items():
            lut[lab] = j + 1                                                        # kernel index 0 = no vote
        label = lut[np.clip(instance, 0, None)]
        depth = kf["depth"].detach().cpu().numpy().reshape(label.shape)
        label[dyn | ~(depth > 0)] = 0
        view = gu.flashsplat_cam(kf["color"], kf["depth"], None, kf["intrinsics"], kf["pose"].cpu(), None)
        pkg = flashsplat_render(view, g.gaussian_model, gu.flashsplat_pipe(), torch.zeros(3).cuda(),
                                gt_mask=torch.from_numpy(label).cuda(), obj_num=len(g.label_ids))
        g.label_weight += pkg["used_count"][1:len(g.label_ids) + 1].T
        best, j = g.label_weight.max(dim=1)
        voted = best > 0
        g.identity[voted] = torch.as_tensor(g.label_ids, device="cuda")[j[voted]]

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
        """The Gaussians of the map of now (T1: ended ones are kept for the keyframes of their time)."""
        g = self.game
        n_all = g.gaussian_model.get_xyz.shape[0]
        for name in ("uid", "identity", "last_update", "created", "state_birth", "death_state", "death_evidence",
                     "label_weight"):
            assert len(getattr(g, name)) == n_all, f"per-Gaussian {name}: {len(getattr(g, name))} != {n_all}"
        live = g.alive_at(g.now)
        if live is None:                                   # T1 off (no layer yet / rows 1-3): every Gaussian
            live = torch.ones(n_all, dtype=torch.bool, device="cuda")
        xyz = g.gaussian_model.get_xyz.detach()[live] / self.scale
        n = len(xyz)
        sigma = g.gaussian_model.get_scaling.detach()[live].max(dim=1).values / self.scale
        return Elements(g.uid[live].clone(), xyz.float(), torch.full((n, 3), float("nan"), device="cuda"),
                        g.identity[live].clamp(min=0), g.last_update[live].clone(), (3.0 * sigma).float(),
                        g.created[live].clone())

    def finish_session(self, stamp: int) -> None:
        """[F2] GaME's published end-of-run refinement (run.py:36-37 of GaME: optimize_model(refinement_iters,
        refinement=True) over all keyframes, with densification), once after the last session of the chain, as
        GaME does after all its runs (run.py calls this then; since 2026-10-06, before: F1 after every session).
        With the layer (T1) every keyframe trains only the Gaussians alive at its stamp under the final state
        intervals."""
        g = self.game
        g.now = stamp
        iters = int(self.config.get("refinement_iters", 0))
        if iters and g.keyframes:
            g.optimize_model(iterations=iters, refinement=True)

    @torch.no_grad()
    def set_state_intervals(self, intervals: dict, stamp: int) -> None:
        """[T1] A Gaussian of identity l belongs to the state of l that was open when the Gaussian was
        created (the latest state born at or before its creation): it lives from that state's birth until the
        state ends (t1.py). Background Gaussians live from -infinity and end only by the layer's evidence
        (retire). Ended Gaussians that no stored keyframe of their lifetime can render are pruned: they can no
        longer contribute to any map."""
        g = self.game
        g.timed = True
        g.state_birth, g.death_state = t1.state_membership(g.identity, g.created, intervals)
        self._prune_unrenderable(stamp)

    def _prune_unrenderable(self, stamp: int) -> None:
        g = self.game
        end = torch.minimum(g.death_state, g.death_evidence)
        dead = end <= stamp
        if not dead.any():
            return
        kfs = torch.tensor(sorted(g.kf_stamp.values()), dtype=torch.int64, device="cuda")
        if len(kfs):
            born = t1.birth(g.identity, g.created, g.state_birth)[dead]
            pos = torch.searchsorted(kfs, born.clamp(min=int(kfs[0])))
            has = pos < len(kfs)
            has[has.clone()] = kfs[pos[has]] < end[dead][has]
        else:
            has = torch.zeros(int(dead.sum()), dtype=torch.bool, device="cuda")
        drop = torch.zeros_like(dead)
        drop[torch.nonzero(dead).squeeze(1)[~has]] = True
        if drop.any():
            g.gaussian_model.prune_points(drop)

    @torch.no_grad()
    @torch.no_grad()
    def render_map(self, T_world_cam: np.ndarray, K: Intrinsics):
        """Median depth (z, metres) and alpha of the map of now (every live Gaussian) from a camera of intrinsics K,
        at the centred window of K; returns (median, alpha, top, left)."""
        return self.render_identity(None, T_world_cam, K)

    @torch.no_grad()
    def render_identity(self, identity, T_world_cam: np.ndarray, K: Intrinsics):
        """Median depth (z, metres; where T first drops to 0.5) and alpha of one identity's live Gaussians alone,
        seen from a camera of intrinsics K: rendered at the centred window of K (GaME's camera has its principal
        point at the image centre, Crop); returns (median, alpha, top, left) of that window."""
        crop = Crop(K, 1)
        H, W = crop.rows, crop.cols
        pose = np.linalg.inv(np.asarray(T_world_cam, dtype=np.float64)).astype(np.float32)
        pose[:3, 3] *= self.scale
        view = gu.flashsplat_cam(torch.zeros(3, H, W, device="cuda"), torch.zeros(H, W, device="cuda"), None,
                                 crop.K, torch.from_numpy(pose), None)
        g = self.game
        gm = g.gaussian_model
        alive = g.alive_at(g.now)
        mask = torch.ones(len(g.identity), dtype=torch.bool, device="cuda") if identity is None else g.identity == identity
        if alive is not None:
            mask &= alive
        gm.alive = mask
        try:
            pkg = probe_render(view, gm)
        finally:
            gm.alive = None
        return pkg["median"].reshape(H, W) / self.scale, pkg["alpha"].reshape(H, W), crop.top, crop.left

    def retire(self, ids: torch.Tensor, stamp: int) -> None:
        """[T1] The element ends at `stamp`: it leaves the map of now and of every later time, and stays
        in the maps (and the training) of the keyframes before `stamp` (replaces R1's masks)."""
        g = self.game
        g.timed = True
        mask = torch.isin(g.uid, ids.to("cuda"))
        if not mask.any():
            return
        g.death_evidence[mask] = torch.minimum(g.death_evidence[mask], torch.full_like(g.death_evidence[mask], stamp))
        self._prune_unrenderable(stamp)

    # -- evaluation ---------------------------------------------------------------------------
    def snapshot(self, stamp: int) -> None:
        self.stamps.append(int(stamp))
        self.scenes.append(self._render_scene(stamp))

    def timeline(self) -> SceneListTimeline:
        return SceneListTimeline(self.stamps, self.scenes)

    @torch.no_grad()
    def _render_scene(self, t_ns: int) -> EvaluationScene:
        """[T1] The map of time t_ns: only the Gaussians alive at t_ns are rendered."""
        self.game.gaussian_model.alive = self.game.alive_at(t_ns)
        try:
            return self._render_scene_alive(t_ns)
        finally:
            self.game.gaussian_model.alive = None

    def _render_scene_alive(self, t_ns: int) -> EvaluationScene:
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
            if self.readout == "first_echo":
                fe = probe_render_fe(view, gm)
                mi = fe["median_index"].squeeze().long()
                ok = (mi >= 0) & (kf["depth"].to(mi.device).reshape(mi.shape) > 0)
                depth = torch.where(ok, fe["median"].squeeze(), torch.zeros_like(fe["median"].squeeze())) / self.scale
                ok &= depth > 0
                v, u = torch.nonzero(ok, as_tuple=True)
                ident = g.identity[mi[v, u]].clamp(min=0)
                z = depth[v, u]
                Tw = torch.linalg.inv(pose.to(torch.float64))
                Tw[:3, 3] /= self.scale
                Tw = Tw.to("cuda", torch.float32)
                p = torch.stack([(u.float() - K[0, 2]) / K[0, 0] * z, (v.float() - K[1, 2]) / K[1, 1] * z, z], 1) \
                    @ Tw[:3, :3].T + Tw[:3, 3]
                uu = (torch.arange(w, device="cuda", dtype=torch.float32) - K[0, 2]) / K[0, 0]
                vv = (torch.arange(h, device="cuda", dtype=torch.float32) - K[1, 2]) / K[1, 1]
                dd = torch.where(ok, depth, torch.full_like(depth, float("nan")))
                Pc = torch.stack([uu[None, :] * dd, vv[:, None] * dd, dd], dim=-1)
                dx, dy = torch.full_like(Pc, float("nan")), torch.full_like(Pc, float("nan"))
                dx[:, 1:-1] = Pc[:, 2:] - Pc[:, :-2]
                dy[1:-1, :] = Pc[2:, :] - Pc[:-2, :]
                nc = torch.cross(dx, dy, dim=-1)
                nc = nc / torch.linalg.norm(nc, dim=-1, keepdim=True)
                nc = torch.where(((nc * Pc).sum(-1) > 0)[..., None], -nc, nc)
                n_w = nc[v, u] @ Tw[:3, :3].T
                b, o = ident == 0, ident > 0
                acc_bg = union(acc_bg, p[b], ident[b], n_w[b], self.bg_voxel, False)
                acc_ob = union(acc_ob, p[o], ident[o], n_w[o], self.obj_voxel, True)
                continue
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
