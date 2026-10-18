"""GaME (CVPR 2026, Gaussian mapping of evolving scenes) as a backend of the update layer, with EV-T for the second visit.

GaME as published (FT/baselines/GaME @1c971d6; its files are not edited, GaME_1c971d6.patch only adds a missing #include),
with five fixes of the released code:
  C1  the keyframe test uses the true relative pose last_w2c @ inv(now_w2c) (published: inv(last) @ now on world-to-camera
      poses, which depends on where a dataset puts its origin; GaME sec. 4.3 and Gaussian-SLAM use the relative motion)
  C2  the keyframe rotation threshold in degrees (published: Euler angles in radians against 50, never true)
  C3  the keyframe sampler without the endless loop when every keyframe is covered by occlusion masks
  C4  the training loss only on pixels with measured depth (published: a pixel without depth trains the depth towards 0)
  C5  the addition test only on pixels with measured depth (published: depth 0 counts as new geometry)

Coupling with the update layer (core/): I1 the identity of a Gaussian is FlashSplat's optimal assignment of the instance
masks (Shen et al., ECCV 2024), accumulated over keyframes; T1 every Gaussian has a lifetime [birth, end) and a keyframe
of the session renders the Gaussians alive at its stamp (an earlier session's keyframe: the map of now on its M1 pixels) (t1.py: the layer's object states and retirements set the ends; an earlier
session's Gaussian that leaves the map is ended, not deleted, so the keyframes of its time still render it); R2 the
snapshot readout is the first echo (readout.py); A1 GaME's addition handling as published.

Which update runs is the framework's row, no switch of this file: without the layer (rows 1-3) the map is GaME's own
(own_update: GaME's removal handling as published); when the layer drives the map (row 4) the second visit is EV-T.

EV-T. An earlier session's Gaussian changes only on this session's evidence: the update layer's memory test
(core/session_end.py decide()) with a Gaussian as the element. At each keyframe of this session, before its seeding,
every live earlier Gaussian is compared with the keyframe's measured depth at its own pixel, r = reading - its depth, with
tau = max(3 sigma, support_tol) (session_end.py's tau = max(half, sigma(q)), the layer's surface tolerance standing in for
sigma(q) as in that module):
  hit       |r| <= tau
  through   r > tau
  covered   -band <= r < -tau, the keyframe seeing it from the side the earlier session saw it from (band: the export
            fusion's truncation, sdf_trunc = 10 cm, eval/tsdf_export.py)
  occluded  any other r < -tau: behind what the keyframe measured, so not part of that keyframe's renders
and, over the session,
  ended     through > hit, or neither hit nor seen through and covered more often than occluded: it leaves the map now
  present   hit or covered: this session's keyframes train it
  shared    hit and never covered: the earlier keyframes train it too
  held      anything else: this session's keyframes leave it as it is (parameters and optimiser moments, no prune); the
            earlier keyframes keep maintaining it, as the memory of their own session
Training is GaME's loop. Each step draws one keyframe: this session's keyframes count 1 each, an earlier keyframe the
share of shared rows among the rows it hit that this session has decided on (present or ended; a held row says nothing
either way); the step moves only the rows its keyframe's session may train, every other row stays exactly as it is
(parameters and Adam moments). GaME's own opacity prune deletes this session's Gaussians only: an earlier Gaussian
leaves the map only as ended above (its fade under the loss is no evidence and can be undone; the earlier keyframes
maintain it). No other constant: 3 sigma (the Gaussian's support), support_tol (the layer's surface_match_tolerance,
5 cm), 10 cm (sdf_trunc), 0.1 m (the sensor's minimum depth).

M1. Which pixels of a drawn keyframe train (mapping and refinement, every keyframe; from the refinement's R15U rule,
code_f3e5_r22 game.py:927-1052): a pixel trains colour and depth together or not at all. Each measured pixel's reading
is compared with the later reading of the same place: the witness is the keyframe that last measured the surface the
map of now shows there (last_update of its first echo). The witness's reading beyond it: the place is gone, the pixel is
out. The same surface: the reading is confirmed. An earlier session's keyframe supervises the map of now, as GaME's
keyframes train the map of now outside their masks; where the first echo now is a Gaussian not of the scene the keyframe
saw (not alive at its stamp -- a later state -- or of a later session), it speaks for the pixel only if its reading is
confirmed there (that Gaussian then stands in front of a surface measured later: a filler GaME's unnormalised depth
grows behind surfaces); unconfirmed -- a new surface in front, or readings too noisy to tell -- the pixel is out: the
absence of a contradiction is no confirmation (without this: real B 473d7ac 92.04, analysis/g1dec_m1_20261018; the map
of the keyframe's own time instead: the bathroom fillers stay, quick c57040d dark rows 12). A keyframe of this session
renders the map of its own time (T1). Readings against readings: neither the layer's states nor a Gaussian's centre
decide it, so a change the layer missed and a large Gaussian whose centre hides behind a surface are both caught. (R21's
'colour of the latest session' -- an earlier pixel's colour out on a surface a later session re-observed, its depth
kept -- is not used: the earlier views then train such surfaces through depth alone; real B ac3c328 92.28 lost a wall
where 61 % of their pixels were depth-only, and its gain in B's colours was never measurable -- B's test pixels are off
by +0.7..+2.3 grey levels in every class.)
"""
from __future__ import annotations

import gc
import math
import random
import sys
import time
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
from ...refinement import RefinementProgress
from . import t1
from .growth import clear_densification_rows, restored_session_settings
from .label_table import LabelTable

GAME = Path("/home/jixian/Desktop/FT/baselines/GaME")
if str(GAME) not in sys.path:
    sys.path.insert(0, str(GAME))
from src.entities.game import GaME  # noqa: E402
from src.entities.losses import isotropic_loss, l1_loss  # noqa: E402
from src.flashsplat.gaussian_renderer import GaussianModel, flashsplat_render  # noqa: E402
from src.flashsplat.utils.general_utils import build_rotation as _build_rotation  # noqa: E402
from src.utils import utils as gu  # noqa: E402

CONFIGS = {"synthetic": GAME / "configs/flat/flat.yaml", "real": GAME / "configs/flat/flat.yaml"}
STEP = {"synthetic": 1, "real": 2}      # run.INPUT_STEP: the input every backend integrates
BAND_M = 0.10                           # the export fusion's sdf_trunc (5 voxels of 2 cm, eval/tsdf_export.py, after 2DGS)
NEAR_M = 0.1                            # the sensor's minimum depth
INT64_MAX = torch.iinfo(torch.int64).max
ROWS = ("identity", "last_update", "created", "state_birth", "death_state", "death_evidence", "death_prune", "frozen")
EV = ("ev_hit", "ev_thr", "ev_ovw", "ev_far", "ev_dir")


# ------------------------------------------------------------------------------------------------------- evidence
@torch.no_grad()
def measured_surface(depth: torch.Tensor, K: torch.Tensor):
    """The surface a depth image measured: per pixel its point (camera frame), the unit normal facing the camera (tangents
    from the neighbour on the smoother side along each image axis, so a depth edge does not bend the plane), and where both
    tangents exist."""
    H, W = depth.shape
    vv, uu = torch.meshgrid(torch.arange(H, device=depth.device, dtype=torch.float32),
                            torch.arange(W, device=depth.device, dtype=torch.float32), indexing="ij")
    P = torch.stack([(uu - K[0, 2]) / K[0, 0] * depth, (vv - K[1, 2]) / K[1, 1] * depth, depth], -1)
    valid = depth > 0

    def tangent(axis):
        f, b = torch.roll(P, -1, dims=axis) - P, P - torch.roll(P, 1, dims=axis)
        fv, bv = valid & torch.roll(valid, -1, dims=axis), valid & torch.roll(valid, 1, dims=axis)
        if axis == 1:
            fv[:, -1] = bv[:, 0] = False
        else:
            fv[-1, :] = bv[0, :] = False
        inf = torch.full_like(depth, float("inf"))
        smoother = torch.where(fv, f[..., 2].abs(), inf) <= torch.where(bv, b[..., 2].abs(), inf)
        return torch.where(smoother[..., None], f, b), fv | bv

    tx, ox = tangent(1)
    ty, oy = tangent(0)
    n = torch.cross(tx, ty, dim=-1)
    n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    n = torch.where(((n * P).sum(-1) > 0)[..., None], -n, n)
    return P, n, valid & ox & oy


@torch.no_grad()
def evidence_counts(xyz, tau, pose, K, depth, near, band=None, side=None):
    """hit / through / blocked of rows against one keyframe at each row's own pixel (the layer's rounding, session_end.py).
    The residual r is the signed distance of the row to the surface the keyframe measured at that pixel (point to plane,
    > 0 in front of it), the signed distance a TSDF stands for; along the viewing ray it would grow as 1 / cos of the
    incidence angle.
    xyz (N, 3) and tau (N,) in map units, pose world -> camera (4, 4), K (3, 3), depth (H, W) measured (0 = none), near: the
    smallest depth considered. With `band` and `side` (N, bool: seen from the earlier session's side): (hit, through,
    covered, occluded) -- covered = blocked within the band on that side, occluded = every other block."""
    cam = xyz @ pose[:3, :3].T + pose[:3, 3]
    z = cam[:, 2]
    ok = z > near
    zs = torch.where(ok, z, torch.ones_like(z))
    u = torch.floor(cam[:, 0] / zs * K[0, 0] + K[0, 2] + 0.5).to(torch.int64)
    v = torch.floor(cam[:, 1] / zs * K[1, 1] + K[1, 2] + 0.5).to(torch.int64)
    H, W = depth.shape
    ok &= (u >= 0) & (v >= 0) & (u < W) & (v < H)
    vc, uc = v.clamp(0, H - 1), u.clamp(0, W - 1)
    P, nrm, measured = measured_surface(depth, K)
    ok &= measured[vc, uc]
    r = ((cam - P[vc, uc]) * nrm[vc, uc]).sum(-1)                   # as d - z for a frontal surface: > 0 the ray went on past
    hit, thr, blk = ok & (r.abs() <= tau), ok & (r > tau), ok & (r < -tau)
    if band is None:
        return hit, thr, blk
    ovw = blk & (-r <= band) & side
    return hit, thr, ovw, blk & ~ovw


def evidence_decide(hit, thr, ovw, far):
    """(ended, present, shared) from this session's counts, core/session_end.py decide()."""
    ended = (thr > hit) | ((hit == 0) & (thr == 0) & (ovw > far))
    present = ~ended & ((hit > 0) | (ovw > 0))
    shared = ~ended & (hit > 0) & (ovw == 0)
    return ended, present, shared


def uid_bits(uids: torch.Tensor, end: int) -> torch.Tensor:
    """A bit table over the uids [0, end): bit u set for every u in `uids` (distinct, below end)."""
    bits = torch.zeros((end + 7) // 8, dtype=torch.int32, device=uids.device)
    bits.index_add_(0, uids >> 3, (1 << (uids & 7)).to(torch.int32))      # distinct bits of one byte add up to their union
    return bits.to(torch.uint8)


def uid_bits_count(bits: torch.Tensor, uids32: torch.Tensor) -> torch.Tensor:
    """How many of `uids32` (int32, all below the table's end) are members of a bit table."""
    byte = torch.index_select(bits, 0, uids32 >> 3)
    return ((byte.to(torch.int32) >> (uids32 & 7)) & 1).sum()


def uid_bits_lookup(bits: torch.Tensor, end: int, uids: torch.Tensor) -> torch.Tensor:
    """Membership of `uids` in a bit table over [0, end); uids at or beyond end are not members."""
    inb = uids < end
    u = torch.where(inb, uids, torch.zeros_like(uids))
    return inb & (((bits[u >> 3].to(torch.int64) >> (u & 7)) & 1) > 0)


# ------------------------------------------------------------------------------------------------------- helpers
class Crop:
    """Crop + subsample so that the principal point is the exact image centre (GaME's cameras are FoV-based).

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


def probe_render(view, pc, gt_mask: Optional[torch.Tensor] = None, obj_num: int = 1,
                 meas_depth: Optional[torch.Tensor] = None, vote_tol: float = 0.0):
    """Forward-only render with the update_layer build of the FlashSplat rasterizer (ul_rasterizer/): image, depth, alpha,
    per-label weights, and the median depth (where the transmittance first drops to 0.5; 2DGS's median depth)."""
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
    """probe_render with the v2 build (ul_rasterizer_v2/): also the index of the Gaussian at which the transmittance first
    drops to 0.5 (the first echo; -1 where it never does)."""
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


def _to_gpu(x):
    """Every tensor of a nested tuple/list/dict on the GPU (parameters stay parameters)."""
    if isinstance(x, torch.nn.Parameter):
        return torch.nn.Parameter(x.detach().cuda(), requires_grad=x.requires_grad)
    if torch.is_tensor(x):
        return x.cuda()
    if isinstance(x, (tuple, list)):
        return type(x)(_to_gpu(v) for v in x)
    if isinstance(x, dict):
        return {k: _to_gpu(v) for k, v in x.items()}
    return x


def _label_weight_store(lw) -> torch.Tensor:
    """The per-Gaussian label sums as stored in a checkpoint: CSR when that is smaller, else dense (values exact)."""
    if isinstance(lw, LabelTable):
        return lw.to_sparse_csr()
    lw = lw.cpu()
    if lw.dim() != 2 or not lw.numel():
        return lw
    nz = int((lw != 0).sum())
    return lw.to_sparse_csr() if nz * 16 + 8 * (lw.shape[0] + 1) < lw.numel() * lw.element_size() else lw


_STRIP_SOURCES = None


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _strip_sources(kf: dict):
    """The dataset frames a stripped keyframe is rebuilt from (eval/kf_strip.py, the stripped checkpoint's one implementation)."""
    global _STRIP_SOURCES
    from ...eval import kf_strip
    if _STRIP_SOURCES is None:
        _STRIP_SOURCES = kf_strip._sources("real" if "real" in kf["session"] else "synthetic")[0]
    return kf_strip, _STRIP_SOURCES


def _keyframe_depth(kf: dict) -> torch.Tensor:
    """A stored keyframe's measured depth (map units) on the GPU, its own or rebuilt from the dataset when stripped."""
    if kf.get("stripped"):
        kf_strip, src = _strip_sources(kf)
        d = kf_strip.restore_depth(src, kf)
    else:
        d = kf["depth"]
    return d.reshape(d.shape[-2:]).cuda().float()


class _TimedGaussianModel(GaussianModel):
    """GaME's Gaussian model whose rendering is restricted to the Gaussians alive at one time: `alive` (bool per Gaussian,
    None = all) multiplies the opacity, so a Gaussian that is not alive adds nothing to the render and gets no gradient.
    Densification records the parents of the new rows (a child inherits its parent's bookkeeping) and goes through the
    map's policies (which rows may densify; how a prune is applied)."""
    alive = None
    parents = None
    prune_policy = None
    densify_policy = None

    @property
    def get_opacity(self):
        o = self.opacity_activation(self._opacity)
        return o if self.alive is None else o * self.alive[:, None].to(o.dtype)

    @property
    def raw_opacity(self):
        return self.opacity_activation(self._opacity)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, limit_num=-1):
        """GaussianModel.densify_and_prune (flashsplat/scene/gaussian_model.py:579-599); the prune through prune_policy."""
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        if ((limit_num > 0) and (self.get_num_pts < limit_num)) or (limit_num < 0):
            self.densify_and_clone(grads, max_grad, extent, limit_num=limit_num)
            self.densify_and_split(grads, max_grad, extent, limit_num=limit_num)
            prune_mask = (self.get_opacity < min_opacity).squeeze()
            if max_screen_size:
                big_points_vs = self.max_radii2D > max_screen_size
                big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
                prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            (self.prune_policy or self.prune_points)(prune_mask)
        torch.cuda.empty_cache()

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, limit_num=-1):
        """GaussianModel.densify_and_split (gaussian_model.py:506-552) with densify_policy and the parents recorded."""
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)
        if self.densify_policy is not None:
            selected_pts_mask = self.densify_policy(selected_pts_mask)
        if limit_num > 0:
            idx = padded_grad.masked_fill(~selected_pts_mask, -torch.inf).argsort(dim=0, descending=True)
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
        self.parents = torch.nonzero(selected_pts_mask).squeeze(1).repeat(N)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, limit_num):
        """GaussianModel.densify_and_clone (gaussian_model.py:554-577) with densify_policy and the parents recorded."""
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)
        if self.densify_policy is not None:
            selected_pts_mask = self.densify_policy(selected_pts_mask)
        if limit_num > 0:
            idx = torch.norm(grads, dim=-1).masked_fill(~selected_pts_mask, -torch.inf).argsort(descending=True)
            sorted_pts_mask = torch.zeros(grads.size(0), dtype=torch.bool, device=grads.device)
            inc_num = limit_num - self.get_num_pts
            if inc_num <= 0:
                return
            inc_num = min(inc_num, selected_pts_mask.sum().item())
            sorted_pts_mask[idx[:inc_num]] = 1
            selected_pts_mask = torch.logical_and(selected_pts_mask, sorted_pts_mask)
        self.parents = torch.nonzero(selected_pts_mask).squeeze(1)
        self.densification_postfix(self._xyz[selected_pts_mask], self._features_dc[selected_pts_mask],
                                   self._features_rest[selected_pts_mask], self._opacity[selected_pts_mask],
                                   self._scaling[selected_pts_mask], self._rotation[selected_pts_mask])


# ------------------------------------------------------------------------------------------------------- the map
class TrackedGaME(GaME):
    """GaME with per-Gaussian bookkeeping (uid, identity, lifetime) and EV-T's evidence for the second visit."""

    timed = False      # T1 is on once the layer drives this map (state intervals / retirements)

    def __init__(self, config: dict):
        super().__init__(config, wandb_online=False)
        self.num_label_channels = 1          # renders without a gt_mask: one label channel (identical renders and gradients)
        self.next_uid = 0
        self.frame_counter = 0
        self.uid = torch.zeros(0, dtype=torch.int64, device="cuda")
        for k in ROWS:
            setattr(self, k, torch.zeros(0, dtype=torch.bool if k == "frozen" else torch.int64, device="cuda"))
        self.label_ids = [0]                 # I1: column j of the label sums = physical identity label_ids[j] (0 = background)
        self.label_weight = LabelTable(0, 1)
        self.tracked_labels: Optional[torch.Tensor] = None
        self.session_start: Optional[int] = None
        self.support_tol = 0.0
        self.kf_stamp: Dict[int, int] = {}   # keyframe id -> its sensor stamp (ns)
        self.kf_session: Dict[int, int] = {} # keyframe id -> the start stamp of its session (M1: a later session's rows)
        self.now = 0
        self._replay_cache = {}              # earlier keyframes' images restored once from the dataset
        self._m1_cache, self._m1_sig, self._m1_slots, self._m1_buf, self._m1_occ_of = {}, None, {}, None, {}
        self._new_evidence()
        gm = self.gaussian_model
        gm.__class__ = _TimedGaussianModel
        post, prune, reset = gm.densification_postfix, gm.prune_points, gm.reset_opacity

        def densification_postfix(new_xyz, *rest):
            n = new_xyz.shape[0]
            par, gm.parents = gm.parents, None
            ev = self._ev_live()
            post(new_xyz, *rest)
            if par is not None and len(par) == n:                         # densified: a child carries its parent's rows
                new = {k: getattr(self, k)[par] for k in ROWS + (EV if ev else ())}
                self.label_weight = self.label_weight.append_rows(self.label_weight[par])
            else:                                                          # seeded: rows of now, no evidence
                full = lambda v: torch.full((n,), v, dtype=torch.int64, device="cuda")
                new = dict(identity=full(-1), last_update=full(self.now), created=full(self.now), state_birth=full(INT64_MAX),
                           death_state=full(INT64_MAX), death_evidence=full(INT64_MAX), death_prune=full(INT64_MAX),
                           frozen=torch.zeros(n, dtype=torch.bool, device="cuda"))
                if ev:
                    new.update({k: torch.zeros(n, dtype=torch.int32, device="cuda") for k in EV[:4]})
                    new["ev_dir"] = torch.zeros((n, 3), dtype=torch.float32, device="cuda")
                self.label_weight = self.label_weight.append_empty_rows(n)
            self.uid = torch.cat([self.uid, torch.arange(self.next_uid, self.next_uid + n, device="cuda")])
            self.next_uid += n
            for k, v in new.items():
                setattr(self, k, torch.cat([getattr(self, k), v]))

        def prune_points(mask, force=False):
            """GaME's prune; an ended (frozen) row leaves only when no keyframe of its lifetime can render it (force)."""
            mask = mask.to("cuda").bool()
            if not force:
                mask = mask & ~self.frozen
            if not mask.any():
                return
            keep = ~mask
            ev = self._ev_live()
            prune(mask)
            self.uid = self.uid[keep]
            for k in ROWS + (EV if ev else ()):
                setattr(self, k, getattr(self, k)[keep])
            self.label_weight = self.label_weight[keep]

        def reset_opacity():
            """GaME's opacity reset (refinement) restarts the opacities of a map optimised from scratch. A map the layer drives
            enters the refinement already trained over its sessions, its memory included: it is not reset. GaME's own rows: as
            published."""
            if not self.timed:
                reset()

        gm.densification_postfix = densification_postfix
        gm.prune_points = prune_points
        gm.reset_opacity = reset_opacity
        gm.prune_policy = self._prune_policy
        gm.densify_policy = lambda candidates: candidates & ~self.frozen

    # -- time -----------------------------------------------------------------------------------------------------
    def alive_at(self, t: Optional[int]) -> Optional[torch.Tensor]:
        """The Gaussians of the map of time t (t1.py); None (= all, GaME as published) when no layer drives this map."""
        if t is None or not self.timed:
            return None
        return t1.alive_at(t, self.identity, self.created, self.state_birth, self.death_state, self.death_evidence,
                           self.death_prune)

    def ended(self) -> torch.Tensor:
        return t1.end(self.death_state, self.death_evidence, self.death_prune)

    def earlier_session(self, keyframe_id: int) -> bool:
        return bool(self.timed and self.session_start is not None
                    and self.kf_stamp.get(keyframe_id, self.session_start) < self.session_start)

    def freeze_rows(self, mask: torch.Tensor) -> None:
        """Ended rows: rendered by the keyframes of their time, never optimised, densified or pruned again."""
        new = mask.to("cuda").bool() & ~self.frozen
        if not new.any():
            return
        self.frozen = self.frozen | new
        clear_densification_rows(self.gaussian_model, new)

    def hold_rows(self, rows: torch.Tensor):
        """Rows a step may not train stay as they are through it, in place: their gradient and first moment are set to zero
        before the step, so Adam's update of them is exactly zero, and the one decay the step gives their second moment is
        undone after it (returned). A parameter without a gradient (every one right after a prune) is not stepped and is left
        alone. The second moment is what damps a row's steps: set to zero (EV-T's reset), the row's first steps once it may
        train again would be several times the learning rate (Adam's bias correction counts the optimiser's steps, not the
        row's; GaME's eps is 1e-15)."""
        if not rows.any():
            return lambda: None
        opt = self.gaussian_model.optimizer
        decayed = []
        for group in opt.param_groups:
            prm = group["params"][0]
            if prm.grad is None:
                continue
            per_row = rows.view((-1,) + (1,) * (prm.dim() - 1))
            prm.grad.masked_fill_(per_row, 0)
            st = opt.state.get(prm)
            if st and "exp_avg" in st:
                st["exp_avg"].masked_fill_(per_row, 0)
                decayed.append((st["exp_avg_sq"], torch.where(per_row, 1.0 / group["betas"][1], 1.0)))

        def undo():
            for v, factor in decayed:
                v.mul_(factor)
        return undo

    def end_rows(self, rows: torch.Tensor, stamp: int) -> None:
        """An earlier session's row leaves the map at `stamp`: ended, not deleted (the keyframes of its time render it)."""
        self.death_prune[rows] = torch.minimum(self.death_prune[rows], torch.full_like(self.death_prune[rows], int(stamp)))
        m = torch.zeros(len(self.uid), dtype=torch.bool, device="cuda")
        m[rows] = True
        self.freeze_rows(m)

    def _prune_policy(self, mask: torch.Tensor) -> None:
        """GaME's opacity prune deletes this session's rows. An earlier session's row leaves the map only on this session's
        evidence (evidence_step): a fade under the loss is no evidence, and the earlier keyframes maintain it."""
        if self.timed and self.session_start is not None:
            mask = mask.to("cuda").bool() & (self.created >= self.session_start)
        self.gaussian_model.prune_points(mask)

    def begin_session(self, session_start: int, support_tol: float) -> None:
        self.session_start = session_start
        self.support_tol = support_tol
        self._new_evidence()
        if self.timed and len(self.uid):
            self.freeze_rows(self.ended() <= session_start)

    # -- EV-T -----------------------------------------------------------------------------------------------------
    def _new_evidence(self) -> None:
        """This session's evidence: per row the counts and the side the earlier session saw it from; per earlier keyframe
        the rows it hit (flat int32 uids in host memory with offsets); per keyframe of this session the occluded rows
        (a bit table over the earlier uids, host memory)."""
        for k in EV:
            setattr(self, k, None)
        self._a_keys, self._a_uids, self._a_off = [], None, None
        self._occluded: Dict[int, torch.Tensor] = {}
        self._uid_end = 0
        self._draws = [0, 0]
        self._m1_px = [0, 0]
        self._calls = self._ended_total = 0

    def _ev_live(self) -> bool:
        return self.ev_hit is not None and len(self.ev_hit) == len(self.uid)

    def _ev_rows(self, idx: torch.Tensor):
        """Positions and tolerances tau = max(3 sigma, support_tol) of rows."""
        gm = self.gaussian_model
        return gm.get_xyz.detach()[idx].float(), torch.clamp(3.0 * gm.get_scaling.detach()[idx].float().max(dim=1).values,
                                                             min=float(self.support_tol))

    @torch.no_grad()
    def _ev_init(self, old: torch.Tensor, near: float) -> None:
        t0 = time.time()
        n = len(self.uid)
        self.ev_hit, self.ev_thr, self.ev_ovw, self.ev_far = (torch.zeros(n, dtype=torch.int32, device="cuda") for _ in range(4))
        self.ev_dir = torch.zeros((n, 3), dtype=torch.float32, device="cuda")
        self._uid_end = int(self.uid[old].max()) + 1 if old.any() else 0
        oi = torch.nonzero(old).squeeze(1)
        xa, ta = self._ev_rows(oi)
        keys, uids = [], []
        for k, kfa in self.keyframes.items():
            if not self.earlier_session(k):
                continue
            pa = kfa["pose"].cuda().float()
            Ka = torch.as_tensor(np.asarray(kfa["intrinsics"]), dtype=torch.float32, device="cuda")
            hit, _, _ = evidence_counts(xa, ta, pa, Ka, _keyframe_depth(kfa), near)
            if not hit.any():
                continue
            h = oi[hit]
            v = -(pa[:3, :3].T @ pa[:3, 3])[None, :] - xa[hit]
            self.ev_dir.index_add_(0, h, v / v.norm(dim=1, keepdim=True).clamp(min=1e-12))
            uids.append(self.uid[h].to(torch.int32).cpu())
            keys.append(k)
        self.ev_dir = self.ev_dir / self.ev_dir.norm(dim=1, keepdim=True).clamp(min=1e-12)
        lens = [len(x) for x in uids]
        self._a_keys = keys
        self._a_uids = torch.cat(uids) if uids else torch.zeros(0, dtype=torch.int32)
        self._a_off = [0]
        for m in lens:
            self._a_off.append(self._a_off[-1] + m)
        print(f"EV_INIT earlier rows {len(oi)} uid end {self._uid_end} earlier keyframes with hits {len(keys)} "
              f"hit pairs {len(self._a_uids)} {time.time() - t0:.1f} s", flush=True)

    @torch.no_grad()
    def evidence_step(self, kf: dict, kid: int, near: float, band: float) -> None:
        """This keyframe's evidence on every live earlier row, before its seeding; the rows it ends leave the map now."""
        if not (self.timed and self.session_start is not None and len(self.uid)):
            return
        old = self.created < self.session_start
        if not self._ev_live():
            self._ev_init(old, near)
        live = self.alive_at(self.now)
        idx = torch.nonzero(old & ~self.frozen & (live if live is not None else torch.ones_like(old))).squeeze(1)
        if len(idx):
            xb, tb = self._ev_rows(idx)
            pose = kf["pose"].cuda().float()
            K = torch.as_tensor(np.asarray(kf["intrinsics"]), dtype=torch.float32, device="cuda")
            d = kf["depth"].reshape(kf["depth"].shape[-2:]).cuda().float()
            side = ((-(pose[:3, :3].T @ pose[:3, 3]))[None, :] - xb).mul(self.ev_dir[idx]).sum(1) > 0
            hit, thr, ovw, far = evidence_counts(xb, tb, pose, K, d, near, band=band, side=side)
            self.ev_hit[idx] += hit.to(torch.int32)
            self.ev_thr[idx] += thr.to(torch.int32)
            self.ev_ovw[idx] += ovw.to(torch.int32)
            self.ev_far[idx] += far.to(torch.int32)
            if self._uid_end:
                self._occluded[kid] = uid_bits(self.uid[idx[far]], self._uid_end).cpu()
            ended, _, _ = evidence_decide(self.ev_hit[idx], self.ev_thr[idx], self.ev_ovw[idx], self.ev_far[idx])
            if ended.any():
                self.end_rows(idx[ended], self.now)
                self._ended_total += int(ended.sum())
        self._calls += 1
        if self._calls % 20 == 1:
            _, present, shared = evidence_decide(self.ev_hit, self.ev_thr, self.ev_ovw, self.ev_far)
            keep = old & ~self.frozen
            print(f"EV call {self._calls} live_earlier={len(idx)} ended_total={self._ended_total} "
                  f"present={int((present & keep).sum())} shared={int((shared & keep).sum())} held={int((keep & ~present).sum())} "
                  f"draws earlier/this={self._draws[0]}/{self._draws[1]} M1 earlier pixels {self._m1_px[0]} out "
                  f"{self._m1_px[1]}", flush=True)

    def occluded(self, kid) -> Optional[torch.Tensor]:
        """The earlier rows keyframe `kid` of this session found behind its measured surface: not part of its renders."""
        bits = self._occluded.get(kid)
        if bits is None or not self._uid_end:
            return None
        return uid_bits_lookup(bits.to("cuda", non_blocking=True), self._uid_end, self.uid)

    def held(self) -> torch.Tensor:
        """Earlier rows this session has no evidence for (with the ended ones): nothing moves them."""
        if not (self.timed and self.session_start is not None and len(self.uid)):
            return torch.zeros(len(self.uid), dtype=torch.bool, device="cuda")
        old = self.created < self.session_start
        if not self._ev_live():
            return old
        _, present, _ = evidence_decide(self.ev_hit, self.ev_thr, self.ev_ovw, self.ev_far)
        return old & ~present

    def fixed(self, earlier_keyframe: bool, held: torch.Tensor) -> torch.Tensor:
        """The rows a step may not move: ended rows; for a keyframe of this session the held rows; for an earlier keyframe
        the earlier rows this session took over (hit or covered, not shared) -- it maintains the rest of its own memory."""
        if not (earlier_keyframe and self._ev_live()):
            return self.frozen | held
        _, present, shared = evidence_decide(self.ev_hit, self.ev_thr, self.ev_ovw, self.ev_far)
        return self.frozen | ((self.created < self.session_start) & present & ~shared)

    @torch.no_grad()
    def _replay_weights(self) -> Optional[list]:
        """Per earlier keyframe, of the rows it hit that this session has decided on (present or ended), the share that is
        shared: how much of what it showed this session checked and found as it was. A held row is not counted."""
        if self._a_uids is None or not len(self._a_keys) or not self._ev_live():
            return None
        _, present, shared = evidence_decide(self.ev_hit, self.ev_thr, self.ev_ovw, self.ev_far)
        old = self.created < self.session_start
        tables = [uid_bits(self.uid[m & old], self._uid_end) for m in (shared & ~self.frozen, present | self.frozen)]
        num, den, i, K = [], [], 0, len(self._a_keys)
        while i < K:                                                    # the host relation in pieces of about 16 M uids
            j = i + 1
            while j < K and self._a_off[j + 1] - self._a_off[i] <= (1 << 24):
                j += 1
            piece = self._a_uids[self._a_off[i]:self._a_off[j]].to("cuda", non_blocking=True)
            for t in range(i, j):
                seg = piece[self._a_off[t] - self._a_off[i]:self._a_off[t + 1] - self._a_off[i]]
                num.append(uid_bits_count(tables[0], seg))
                den.append(uid_bits_count(tables[1], seg))
            i = j
        return (torch.stack(num).float() / torch.stack(den).float().clamp(min=1.0)).cpu().tolist()

    def _replay_frame(self, kid) -> Optional[dict]:
        """An earlier keyframe's images: its own, or restored once from the dataset (stripped checkpoints), colour as uint8."""
        kf = self.keyframes[kid]
        if "color" in kf and "depth" in kf:
            return kf
        if not kf.get("stripped"):
            return None
        c = self._replay_cache.get(kid)
        if c is None:
            kf_strip, src = _strip_sources(kf)
            r = kf_strip.restore_one(src, kf)
            c = {"color8": (r["color"] * 255.0).round().clamp(0, 255).to(torch.uint8), "depth": r["depth"].float()}
            self._replay_cache[kid] = c
        # the division on the GPU, as GameBackend.integrate and kf_strip do (on the CPU it differs in the last bit)
        return {"color": c["color8"].cuda().float() / 255.0, "depth": c["depth"], "pose": kf["pose"], "intrinsics": kf["intrinsics"]}

    # -- M1 -------------------------------------------------------------------------------------------------------
    def _m1_slot(self, key) -> Optional[int]:
        """A witness keyframe's row in the GPU buffers of M1: its raw depth (flat), pose and intrinsics, kept (a
        keyframe's reading and pose never change), and its occlusion mask, refreshed when GaME replaces it. None when its
        images are not at hand (this session's keyframes hold them, an earlier one's come from the replay cache)."""
        j = self._m1_slots.get(key)
        kf = self.keyframes[key]
        if j is None:
            if "depth" in kf:
                d = kf["depth"]
            else:
                if key not in self._replay_cache and self._replay_frame(key) is None:
                    return None
                d = self._replay_cache[key]["depth"]
            d = d.reshape(-1).to("cuda", torch.float32)
            j = len(self._m1_slots)
            b = self._m1_buf
            if b is None or j == len(b["depth"]):
                n = max(64, 2 * j)
                grown = dict(depth=torch.zeros((n, d.numel()), dtype=torch.float32, device="cuda"),
                             occ=torch.zeros((n, d.numel()), dtype=torch.bool, device="cuda"),
                             pose=torch.zeros((n, 4, 4), dtype=torch.float64, device="cuda"),
                             K=torch.zeros((n, 3, 3), dtype=torch.float64, device="cuda"))
                if b is not None:
                    for name, t in b.items():
                        grown[name][:j] = t
                self._m1_buf = b = grown
            b["depth"][j] = d
            b["pose"][j] = torch.as_tensor(np.asarray(_np(kf["pose"])), dtype=torch.float64)
            b["K"][j] = torch.as_tensor(np.asarray(_np(kf["intrinsics"])), dtype=torch.float64)
            self._m1_slots[key] = j
        o = self.occlusion_masks.get(key)
        if self._m1_occ_of.get(key) is not o:
            self._m1_buf["occ"][j] = o.reshape(-1).bool() if o is not None else False
            self._m1_occ_of[key] = o
        return j

    @torch.no_grad()
    def measured_pixels(self, keyframe_id, view, depth, pose, K) -> torch.Tensor:
        """[M1] The pixels of a drawn keyframe that leave its loss: per measured pixel its reading against the later reading
        of the same place, the witness being the keyframe that last measured the surface the map of now shows there
        (last_update of the first echo). The witness's reading beyond it: gone, out; the same surface: confirmed. An earlier
        session's keyframe speaks for a pixel whose first echo now is not of the scene it saw (not alive at its stamp, or of
        a later session) only where it is confirmed. Without the layer (GaME's rows): nothing out."""
        shape = tuple(depth.shape[-2:])
        H, W = shape
        if not self.timed:
            return torch.zeros(shape, dtype=torch.bool, device="cuda")
        sig = (len(self.keyframes), int(self.now))
        if self._m1_sig != sig:
            self._m1_cache.clear()
            self._m1_sig = sig
        if keyframe_id not in self._m1_cache:
            dev, f64 = "cuda", torch.float64
            out = torch.zeros(H * W, dtype=torch.bool, device=dev)
            t_src = self.kf_stamp.get(keyframe_id, -1)
            later = sorted((t, k) for k, t in self.kf_stamp.items() if t > t_src)
            if later:
                gm = self.gaussian_model
                before = gm.alive
                gm.alive = self.alive_at(self.now)
                idx = probe_render_fe(view, gm)["median_index"].reshape(-1).long()
                gm.alive = before
                stamp = torch.where(idx >= 0, self.last_update[idx.clamp(min=0)], torch.full_like(idx, -1))
                # not of the scene this keyframe saw: not alive at its stamp (a later state), or of a later session
                newer = torch.zeros_like(idx, dtype=torch.bool)
                if self.earlier_session(keyframe_id):
                    s_src = self.kf_session.get(keyframe_id, -1)
                    nxt = min(t for t in set(self.kf_session.values()) | {self.session_start} if t > s_src)
                    seen = self.alive_at(t_src) & (self.created < nxt)
                    newer = (idx >= 0) & ~seen[idx.clamp(min=0)]
                confirmed = torch.zeros(H * W, dtype=torch.bool, device=dev)
                del idx
                d0 = depth.reshape(-1).to(dev, f64)
                own = d0 > 0
                if self.occlusion_masks.get(keyframe_id) is not None:
                    own &= ~self.occlusion_masks[keyframe_id].reshape(-1).to(dev).bool()
                ts = torch.tensor([t for t, _ in later], dtype=torch.int64, device=dev)
                pos = torch.searchsorted(ts, stamp).clamp(max=len(later) - 1)
                px = torch.nonzero(own & (ts[pos] == stamp)).squeeze(1)       # pixels with a later witness
                uniq, inv = torch.unique(pos[px], return_inverse=True)
                rows = [self._m1_slot(later[u][1]) for u in uniq.tolist()]
                keep = [j for j, r in enumerate(rows) if r is not None]
                if keep:
                    slot = torch.full((len(uniq),), -1, dtype=torch.int64, device=dev)
                    slot[torch.tensor(keep, device=dev)] = torch.arange(len(keep), device=dev)
                    w = slot[inv]
                    px, w = px[w >= 0], w[w >= 0]
                    buf = self._m1_buf
                    rows = torch.tensor([rows[j] for j in keep], device=dev)
                    Kn = torch.as_tensor(np.asarray(_np(K)), dtype=f64, device=dev)
                    inv_P = torch.linalg.inv(torch.as_tensor(np.asarray(_np(pose)), dtype=f64, device=dev))
                    T = (buf["pose"][rows] @ inv_P)[w]                                 # source camera -> witness camera
                    Kw = buf["K"][rows][w]
                    yy, xx = (px // W).to(f64), (px % W).to(f64)
                    pts = torch.stack(((xx - Kn[0, 2]) / Kn[0, 0], (yy - Kn[1, 2]) / Kn[1, 1], torch.ones_like(xx)), dim=1)
                    pts = torch.einsum("nij,nj->ni", T[:, :3, :3], pts * d0[px, None]) + T[:, :3, 3]
                    front = pts[:, 2] > 0
                    z = torch.where(front, pts[:, 2], torch.ones_like(pts[:, 2]))
                    u = torch.round(Kw[:, 0, 0] * pts[:, 0] / z + Kw[:, 0, 2]).long()
                    v = torch.round(Kw[:, 1, 1] * pts[:, 1] / z + Kw[:, 1, 2]).long()
                    look = v.clamp(0, H - 1) * W + u.clamp(0, W - 1)
                    zw = buf["depth"][rows[w], look]
                    comparable = front & (u >= 0) & (u < W) & (v >= 0) & (v < H) & (zw > 0) & ~buf["occ"][rows[w], look]
                    r = zw.to(f64) - pts[:, 2]
                    tol = float(self.support_tol)
                    out[px[comparable & (r > tol)]] = True                              # gone
                    confirmed[px[comparable & (r.abs() <= tol)]] = True                 # the same surface
                out |= newer & ~confirmed                                              # not its scene, unconfirmed
            self._m1_cache[keyframe_id] = out.reshape(shape)
        return self._m1_cache[keyframe_id]

    # -- GaME ----------------------------------------------------------------------------------------------------
    def is_keyframe(self, pose: np.ndarray) -> bool:
        """GaME.is_keyframe (game.py:487) with C1 and C2."""
        if not self.keyframes:
            return True
        last_pose = self.keyframes[self._last_keyframe_id]["pose"].detach().numpy()
        delta_pose = last_pose @ np.linalg.inv(pose)                                                      # C1
        translation_diff = np.linalg.norm(delta_pose[:3, 3])
        rot_euler_diff = np.abs(Rotation.from_matrix(delta_pose[:3, :3]).as_euler("xyz", degrees=True))  # C2
        return translation_diff > self.config["keyframe_translation_diff"] or np.any(rot_euler_diff > 50)

    def _sample_valid_keyframe(self, selected_frames, only_frame_id=None):
        """GaME's sampler (C3): frames found covered leave the draw; none left raises _NoFrames."""
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
        """GaME.optimize_model (game.py:125-196) with C3, C4, T1, EV-T and M1: during this session's mapping the draws are
        this session's keyframes and, by the shared share of what this session decided of them, the earlier ones; a keyframe
        of this session renders the map of its time without the rows it found occluded, an earlier one the map of now; each
        trains on its M1 pixels only, and a step moves only the rows its keyframe's session may train."""
        progress = RefinementProgress(int(iterations))
        if refinement:
            self.refinement_progress = progress       # callers can report completed steps after a CUDA failure
        if iterations == 0:
            return progress.as_dict()
        layered = self.timed and self.session_start is not None
        frames = list(self.keyframes.keys())
        if layered and not refinement:
            frames = [k for k in frames if self.kf_stamp.get(k, -1) >= self.session_start]
        if len(frames) == 0 or len(self.ignored_frames) == len(self.keyframes):
            print("no frames available")
            progress.stop("no_usable_frames")
            return progress.as_dict()
        replay = layered and not refinement and only_frame_id is None
        weights = self._replay_weights() if replay else None
        wa = float(sum(weights)) if weights else 0.0
        background, pipe = torch.zeros(3).cuda(), gu.flashsplat_pipe()
        gm = self.gaussian_model
        for iteration in tqdm(range(iterations), "Refinement", disable=not refinement):
            kid_a = kf = None
            if wa > 0 and random.random() * (wa + len(frames)) < wa:
                kid_a = random.choices(self._a_keys, weights=weights)[0]
                kf = self._replay_frame(kid_a)
                if kf is None:
                    kid_a = None
            try:
                keyframe_id = kid_a if kid_a is not None else self._sample_valid_keyframe(frames, only_frame_id)
            except _NoFrames:                                                                     # C3
                gm.optimizer.zero_grad(set_to_none=True)
                print("no frames available")
                progress.stop("no_usable_frames")
                break
            if replay:
                self._draws[0 if kid_a is not None else 1] += 1
            keyframe = kf if kid_a is not None else self.keyframes[keyframe_id]
            gt_color, gt_depth = keyframe["color"].cuda().clone(), keyframe["depth"].cuda().clone()
            pose, intrinsics = keyframe["pose"].cuda().clone(), keyframe["intrinsics"]
            view = gu.flashsplat_cam(gt_color, gt_depth, None, intrinsics, pose.cpu(), keyframe_id)
            del keyframe, kf
            earlier = self.earlier_session(keyframe_id)
            # T1: a keyframe of this session renders the map of its stamp; an earlier one the map of now, on the M1 pixels
            gm.alive = self.alive_at(self.now if earlier else self.kf_stamp.get(keyframe_id))
            occ = self.occluded(keyframe_id) if not refinement and not earlier else None
            if occ is not None:
                gm.alive = ~occ if gm.alive is None else gm.alive & ~occ
            render_pkg = flashsplat_render(view, gm, pipe, background, obj_num=self.num_label_channels)
            image, depth, viewspace_point_tensor, visibility_filter, radii = (
                render_pkg["render"].clone(), render_pkg["depth"].clone(), render_pkg["viewspace_points"],
                render_pkg["visibility_filter"].clone(), render_pkg["radii"].clone())
            viewspace_point_tensor.retain_grad()
            del render_pkg
            mask = (~torch.isnan(depth)).squeeze(0).to(image.device)
            mask = mask & (gt_depth.reshape(mask.shape) > 0)                                        # C4
            if self.occlusion_masks.get(keyframe_id) is not None:
                mask = mask * ~self.occlusion_masks[keyframe_id].squeeze(0).to(image.device)
            out = self.measured_pixels(keyframe_id, view, gt_depth, pose, intrinsics)                    # M1
            if earlier:
                self._m1_px[0] += int(mask.sum()); self._m1_px[1] += int((mask & out).sum())
            mask = mask & ~out
            color_loss = (((1.0 - self.opt_params.lambda_dssim) * l1_loss(image, gt_color, agg="none") * mask).mean()
                          + (self.opt_params.lambda_dssim
                             * (1.0 - ssim(image.unsqueeze(0), gt_color.unsqueeze(0), data_range=1.,
                                           mask=mask.unsqueeze(0).tile((3, 1, 1)).unsqueeze(0)))))
            depth_loss = (l1_loss(depth, gt_depth, agg="none") * mask).mean()
            reg_loss = self.config["isotropic_reg_weight"] * isotropic_loss(gm.get_scaling.clone())
            total_loss = color_loss + depth_loss + reg_loss
            total_loss.backward()
            with torch.no_grad():
                # the rows this step may not move: in the mapping as the sessions' evidence decides (fixed); in the refinement
                # every observation is in, so each keyframe trains every row of its time (T1) except the ended ones
                still = self.frozen if refinement else self.fixed(earlier, self.held())
                if not refinement:
                    if iteration == (iterations // 2) or iteration == iterations:
                        self._prune_policy((gm.raw_opacity.detach() < 0.1).squeeze())
                else:
                    gm.alive = None                         # densify / prune on the Gaussians' own opacity
                    if viewspace_point_tensor.grad is not None:     # the rows it may not train gather no statistics
                        viewspace_point_tensor.grad[still] = 0
                    self._densification_step(iteration, total_loss, visibility_filter, radii, viewspace_point_tensor)
                undo = self.hold_rows(still)
                gm.optimizer.step()
                undo()
                progress.advance()
                gm.optimizer.zero_grad(set_to_none=True)
            gm.alive = None
        torch.cuda.empty_cache()
        return progress.as_dict()

    @torch.no_grad()
    def _find_added_geometry_masks(self, keyframe) -> list:
        """GaME._find_added_geometry_masks (game.py:198-232) with C5."""
        EPSILON = self.config["depth_change_threshold"]
        gt_depth = keyframe["depth"]
        pose, intrinsics = keyframe["pose"], keyframe["intrinsics"]
        flashsplat_view = gu.flashsplat_cam(keyframe["color"], gt_depth, None, intrinsics, pose.cpu(), None)
        render_label_pkg = flashsplat_render(flashsplat_view, self.gaussian_model, gu.flashsplat_pipe(),
                                             torch.zeros(3).cuda(), gt_mask=None, obj_num=1)
        render_depth, render_alpha = render_label_pkg["depth"], render_label_pkg["alpha"]
        valid_according_2alpha = render_alpha.squeeze() > self.config["min_opacity"]
        valid_according_2alpha = valid_according_2alpha & (gt_depth.squeeze() > 0)                  # C5
        depth_diff = (gt_depth - render_depth).squeeze()
        new_geometry_indices = []
        for i, mask in enumerate(keyframe["masks"]):
            mask_alpha = mask * valid_according_2alpha
            mask_add = mask_alpha * (depth_diff < -EPSILON)
            add_area = mask_add.sum() / (mask_alpha * (depth_diff < EPSILON)).sum()
            if add_area > self.config["addition_coverage_threshold"]:
                new_geometry_indices.append(i)
        return new_geometry_indices


# ------------------------------------------------------------------------------------------------------- backend
class GameBackend(Backend):
    name = "game"
    consumes_state_intervals = True
    CHANGES = ("C1 keyframe test uses the true relative pose (published: origin-dependent)",
               "C2 keyframe rotation threshold in degrees (published: radians vs 50, never true)",
               "C3 keyframe sampler without endless loop",
               "C4 training loss only on pixels with measured depth (published: also depth 0)",
               "C5 addition test only on pixels with measured depth (published: depth 0 = addition)",
               "I1 Gaussian identity = FlashSplat optimal assignment of the instance masks, accumulated per keyframe",
               "T1 time-indexed maps: a keyframe of the session renders/trains the Gaussians alive at its stamp (an earlier "
               "session's keyframe: the map of now on its M1 pixels); an earlier session's Gaussian that leaves the map is "
               "ended, not deleted",
               "R2 snapshot readout = first echo (T first <= 0.5: median depth and its Gaussian's identity)",
               "A1 GaME's addition handling as published in every row (removals: own update / the layer)",
               "F2 GaME's published final refinement (refinement_iters) once after the chain's last session; when the layer drives "
               "the map: no opacity reset (the map enters it trained), each keyframe trains every row of its time except the "
               "ended ones",
               "EV-T an earlier session's Gaussian changes only on this session's evidence (core/session_end.py decide(), "
               "tau = max(3 sigma, support_tol), band 10 cm): ended / present / shared / held; the earlier keyframes are "
               "drawn by the shared share of the rows this session decided on (held rows not counted) and maintain their own "
               "memory except the rows this session took over; GaME's opacity prune deletes this session's rows only",
               "M1 a drawn keyframe trains only its pixels whose reading the later reading of the same place does not "
               "contradict (gone: out; same surface: confirmed), colour and depth together; an earlier "
               "session's keyframe supervises the map of now, and where a Gaussian not of the scene it saw stands first, "
               "only on confirmed pixels (mapping and refinement)")

    def __init__(self, info: DatasetInfo, own_update: bool, tolerance: float = 0.05, bg_voxel: float = 0.02,
                 obj_voxel: float = 0.01, min_mask_px_full: int = 50, work_dir=None):
        super().__init__(info, own_update, work_dir)
        cfg = yaml.safe_load(CONFIGS[info.name].read_text())["game"]
        cfg.setdefault("scale", 1.0)
        self.config = cfg
        self.scale = float(cfg["scale"])
        self.step = STEP[info.name]
        self.tolerance = tolerance
        self.readout = "first_echo"      # interface: R2, the only snapshot readout; the test harness (quick_map.py) reads it
        self.bg_voxel, self.obj_voxel = bg_voxel, obj_voxel
        self.min_mask_px = max(1, min_mask_px_full // (self.step * self.step))
        self.game: Optional[TrackedGaME] = None
        self.semantic_of: Dict[int, int] = {}

    # -- session --------------------------------------------------------------------------------------------------
    def start_session(self, spec: SessionSpec, prior: Optional[TrackedGaME]) -> None:
        if prior is None and self.game is not None:
            del self.game                        # the model and its patched methods form a cycle
            self.game = None
            gc.collect()
            torch.cuda.empty_cache()
        self.game = prior if prior is not None else TrackedGaME(dict(self.config))
        self.session = FlatSession(spec, pixel_step=1)
        self.crop = Crop(self.session.K, self.step)
        self.stamps, self.scenes = [], []
        self.session_start = self.session.stamp_ns(0)
        self.spec_name = spec.name
        self.game.begin_session(self.session_start, self.tolerance * self.scale)
        self.kf_index: Dict[int, int] = {}

    def end_session(self) -> TrackedGaME:
        return self.game

    def finish_session(self, stamp: int) -> dict:
        """GaME's published end-of-run refinement (refinement_iters), once after the chain's last session."""
        g = self.game
        g.now = stamp
        iters = int(self.config.get("refinement_iters", 0))
        if iters and g.keyframes:
            if g.timed and g.session_start is None:
                raise RuntimeError("Cannot refine a timed checkpoint without its recorded session boundary")
            g.label_weight = g.label_weight.new_zeros((len(g.label_weight), 0))   # identities are voted only at keyframes
        return g.optimize_model(iterations=iters, refinement=True)

    def prior_state(self, g: TrackedGaME) -> dict:
        """The carried model as GaME's checkpoint does it (GaussianModel.capture), the keyframes and the bookkeeping."""
        return dict(model=g.gaussian_model.capture(), keyframes=g.keyframes, estimated_poses=g.estimated_poses,
                    occlusion_masks={k: v.cpu() for k, v in g.occlusion_masks.items()},
                    ignored_frames=g.ignored_frames, last_keyframe_id=g._last_keyframe_id,
                    next_uid=g.next_uid, frame_counter=g.frame_counter, now=g.now, uid=g.uid.cpu(),
                    **{k: getattr(g, k).cpu() for k in ROWS},
                    label_ids=list(g.label_ids), label_weight=_label_weight_store(g.label_weight),
                    tracked_labels=g.tracked_labels.cpu() if g.tracked_labels is not None else None,
                    session_start=g.session_start, support_tol=g.support_tol, kf_stamp=dict(g.kf_stamp),
                    kf_session=dict(g.kf_session), timed=g.timed,
                    semantic_of=dict(self.semantic_of))

    def prior_from_state(self, s: dict) -> TrackedGaME:
        if s.get("split") or s.get("bg_birth") is not None or s.get("retired_masks"):
            raise ValueError("checkpoint of a removed variant (split memory / background birth / retired masks)")
        g = TrackedGaME(dict(self.config))
        g.gaussian_model.restore(_to_gpu(s["model"]), g.opt_params)
        g.keyframes, g.estimated_poses, g.ignored_frames = s["keyframes"], s["estimated_poses"], s["ignored_frames"]
        g.occlusion_masks = {k: v.cuda() for k, v in s["occlusion_masks"].items()}
        g._last_keyframe_id, g.next_uid, g.frame_counter, g.now = (s["last_keyframe_id"], s["next_uid"],
                                                                  s["frame_counter"], s["now"])
        g.uid = s["uid"].cuda()
        for k in ROWS:
            setattr(g, k, s[k].cuda())
        g.kf_stamp, g.timed = dict(s["kf_stamp"]), bool(s["timed"])
        if g.frozen.any():
            clear_densification_rows(g.gaussian_model, g.frozen)
        tl = s.get("tracked_labels")
        g.tracked_labels = tl.cuda() if tl is not None else None
        g.session_start, g.support_tol = restored_session_settings(s, self.tolerance * self.scale)
        # M1: a checkpoint written before kf_session: its keyframes from its session's start on are that session's, the
        # ones before it of one earlier session
        g.kf_session = {k: s.get("kf_session", {}).get(
            k, g.session_start if g.session_start is not None and t >= g.session_start else -1) for k, t in g.kf_stamp.items()}
        lw = s["label_weight"]
        g.label_ids = list(s["label_ids"])
        g.label_weight = lw if isinstance(lw, LabelTable) else LabelTable.from_tensor(lw)
        self.semantic_of = dict(s["semantic_of"])
        return g

    # -- frames ---------------------------------------------------------------------------------------------------
    def _sample(self, index: int) -> Tuple[dict, np.ndarray, np.ndarray]:
        f = self.session.load(index, color=True)
        c = self.crop
        depth = c(f.depth)
        instance = c(f.instance).astype(np.int64)
        semantic = c(f.semantic) if f.semantic is not None else None
        small = Frame(f.index, f.stamp_ns, depth, instance, semantic, f.T_world_cam, f.K)
        dyn = dynamic_mask(small, self.info.dynamic_semantics)
        motion = motion_mask(f)                  # the layer's D1 motion pixels, full resolution
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
                      masks=self._masks(instance, semantic, dyn), pose=pose.astype(np.float32), intrinsics=c.K)
        return sample, instance, dyn

    def _masks(self, instance: np.ndarray, semantic: Optional[np.ndarray], dyn: np.ndarray) -> torch.Tensor:
        """The segment masks GaME's SAM automatic masks would give: every instance, plus every connected region of one
        semantic class among the pixels without an instance."""
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
        """One frame of GaME.train (game.py:599-645); keyframes as GaME decides. A keyframe of this session gives its
        evidence on the earlier rows before its seeding; seeding and training see the map of now without its occluded rows."""
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
        if len(g.keyframes) > 2:                                  # A1
            g.gaussian_model.alive = g.alive_at(g.now)
            g.detect_additions(kf)
            g.gaussian_model.alive = None
        g.keyframes[frame_id] = gu.dict2device(kf, "cpu")
        g.kf_stamp[frame_id] = frame.stamp_ns
        g.kf_session[frame_id] = g.session_start
        self.kf_index[frame_id] = frame.index
        g._last_keyframe_id = frame_id
        if dyn.any():
            g.occlusion_masks[frame_id] = torch.from_numpy(dyn).cuda()
        g.evidence_step(kf, frame_id, NEAR_M * self.scale, BAND_M * self.scale)
        g.gaussian_model.alive = g.alive_at(g.now)
        occ = g.occluded(frame_id)
        if occ is not None:
            g.gaussian_model.alive = ~occ if g.gaussian_model.alive is None else g.gaussian_model.alive & ~occ
        g._add_gaussians(kf["color"], kf["depth"], None, kf["pose"], kf["intrinsics"])
        g.gaussian_model.alive = None
        self._seed_identity(instance, pose, sample["intrinsics"])
        g.optimize_model(50, only_frame_id=frame_id)
        if self.own_update:
            g.detect_removals(frame_id, kf)
        g.optimize_model(g.config["first_keyframe_iters"] if first else g.config["keyframe_iters"])
        g.gaussian_model.alive = g.alive_at(g.now)
        self._support(kf, frame.stamp_ns)
        self._assign_identity(kf, instance, dyn)
        g.gaussian_model.alive = None

    @torch.no_grad()
    def _seed_identity(self, instance: np.ndarray, pose: np.ndarray, K: np.ndarray) -> None:
        """A newly seeded Gaussian's identity: the instance at its centre's pixel in the keyframe it was seeded from."""
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
        """I1: per Gaussian, the label with the largest accumulated blending weight alpha*T over the keyframes' instance
        masks (FlashSplat's closed-form assignment). Pixels of people and without measured depth do not vote. An earlier
        session's Gaussian keeps the identity it entered with (its state membership is that session's history)."""
        g = self.game
        if not len(g.label_weight):
            return
        col = {lab: j for j, lab in enumerate(g.label_ids)}
        new = [int(i) for i in np.unique(instance[instance > 0]).tolist() if int(i) not in col]
        if new:
            g.label_ids += new
            col.update({lab: len(col) + k for k, lab in enumerate(new)})
            g.label_weight = g.label_weight.add_columns(len(new))
        lut = np.zeros(int(max(g.label_ids)) + 1, dtype=np.float32)
        for lab, j in col.items():
            lut[lab] = j + 1                                                        # kernel index 0 = no vote
        label = lut[np.clip(instance, 0, None)]
        depth = kf["depth"].detach().cpu().numpy().reshape(label.shape)
        label[dyn | ~(depth > 0)] = 0
        view = gu.flashsplat_cam(kf["color"], kf["depth"], None, kf["intrinsics"], kf["pose"].cpu(), None)
        pkg = flashsplat_render(view, g.gaussian_model, gu.flashsplat_pipe(), torch.zeros(3).cuda(),
                                gt_mask=torch.from_numpy(label).cuda(), obj_num=len(g.label_ids))
        g.label_weight = g.label_weight.add_dense(pkg["used_count"][1:len(g.label_ids) + 1].T)
        best, j = g.label_weight.max(dim=1)
        voted = best > 0
        if g.timed and g.session_start is not None:
            voted &= g.created >= g.session_start
        g.identity[voted] = torch.as_tensor(g.label_ids, device="cuda")[j[voted]]

    @torch.no_grad()
    def _support(self, kf: dict, stamp: int) -> None:
        """last_update of the Gaussians this keyframe measured on their surface (within the sensor tolerance)."""
        g = self.game
        _, h, w = kf["color"].shape
        view = gu.flashsplat_cam(kf["color"], kf["depth"], None, kf["intrinsics"], kf["pose"].cpu(), None)
        pkg = flashsplat_render(view, g.gaussian_model, gu.flashsplat_pipe(), torch.zeros(3).cuda(), gt_mask=None, obj_num=1)
        xy, gsd = pkg["proj_xy"], pkg["gs_depth"]
        idx = torch.nonzero(pkg["visibility_filter"] & (xy[0] > 0) & (xy[0] < w) & (xy[1] > 0) & (xy[1] < h)).squeeze(1)
        if not len(idx):
            return
        px = xy[:, idx].to(torch.int64)
        d = kf["depth"][px[1], px[0]]
        on = (d > 0) & ((d - gsd[idx]).abs() <= self.tolerance * self.scale)
        g.last_update[idx[on]] = stamp

    # -- the layer's view -----------------------------------------------------------------------------------------
    @torch.no_grad()
    def elements(self) -> Elements:
        """The Gaussians of the map of now; an earlier session's label the layer never tracked is background to it."""
        g = self.game
        n_all = g.gaussian_model.get_xyz.shape[0]
        for name in ("uid", "label_weight") + ROWS:
            assert len(getattr(g, name)) == n_all, f"per-Gaussian {name}: {len(getattr(g, name))} != {n_all}"
        live = g.alive_at(g.now)
        if live is None:
            live = torch.ones(n_all, dtype=torch.bool, device="cuda")
        xyz = g.gaussian_model.get_xyz.detach()[live] / self.scale
        sigma = g.gaussian_model.get_scaling.detach()[live].max(dim=1).values / self.scale
        nrm = torch.full((len(xyz), 3), float("nan"), device="cuda")     # Gaussians carry no normal: no facing test
        ident = g.identity[live].clamp(min=0)
        if g.session_start is not None and g.tracked_labels is not None:
            untracked = (ident > 0) & (g.created[live] < g.session_start) & ~torch.isin(ident, g.tracked_labels)
            ident = torch.where(untracked, torch.zeros_like(ident), ident)
        return Elements(g.uid[live].clone(), xyz.float(), nrm, ident, g.last_update[live].clone(),
                        (3.0 * sigma).float(), g.created[live].clone())

    @torch.no_grad()
    def set_state_intervals(self, intervals: dict, stamp: int) -> None:
        """A Gaussian of identity l lives from the birth of the state of l open when it was created until that state ends."""
        g = self.game
        g.timed = True
        g.state_birth, g.death_state = t1.state_membership(g.identity, g.created, intervals, g.session_start)
        g.tracked_labels = torch.tensor(sorted(int(i) for i in intervals), dtype=torch.int64, device="cuda")
        self._ended_by(stamp)

    def retire(self, ids: torch.Tensor, stamp: int) -> None:
        """The layer's evidence ends these elements at `stamp`."""
        g = self.game
        g.timed = True
        mask = torch.isin(g.uid, ids.to("cuda"))
        if mask.any():
            g.death_evidence[mask] = torch.minimum(g.death_evidence[mask], torch.full_like(g.death_evidence[mask], stamp))
            self._ended_by(stamp)

    def _ended_by(self, stamp: int) -> None:
        """Rows ended by `stamp`: deleted when no stored keyframe of their lifetime can render them, else frozen."""
        g = self.game
        end = g.ended()
        dead = end <= stamp
        if dead.any():
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
                g.gaussian_model.prune_points(drop, force=True)
        g.freeze_rows(g.ended() <= stamp)

    @torch.no_grad()
    def render_map(self, T_world_cam: np.ndarray, K: Intrinsics):
        """Median depth (metres) and alpha of the map of now from a camera of intrinsics K; (median, alpha, top, left)."""
        return self.render_identity(None, T_world_cam, K)

    @torch.no_grad()
    def render_identity(self, identity, T_world_cam: np.ndarray, K: Intrinsics):
        """Median depth (metres) and alpha of one identity's live Gaussians (all with None), at the centred window of K."""
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

    # -- evaluation -----------------------------------------------------------------------------------------------
    def snapshot(self, stamp: int) -> None:
        self.stamps.append(int(stamp))
        self.scenes.append(self._render_scene(stamp))

    def timeline(self) -> SceneListTimeline:
        return SceneListTimeline(self.stamps, self.scenes)

    @torch.no_grad()
    def _render_scene(self, t_ns: int) -> EvaluationScene:
        """The map of time t_ns, read out by the first echo from every stored keyframe view: depth and identity of the
        Gaussian at which T first drops to 0.5, on measured-depth pixels; one sample per voxel (2 cm background, 1 cm per
        object identity), the first in render order."""
        g = self.game
        gm = g.gaussian_model
        empty = EvaluationScene(t_ns, np.zeros((0, 3), np.float32), np.zeros(0, np.uint32), [], np.zeros((0, 3), np.float32))
        if gm.get_xyz.shape[0] == 0 or not g.keyframes:
            return empty
        gm.alive = g.alive_at(t_ns)
        try:
            acc_bg = acc_ob = None
            for kf in g.keyframes.values():
                p, ident, n_w = self._first_echo_points(gm, g.identity, kf)
                b, o = ident == 0, ident > 0
                acc_bg = _voxel_union(acc_bg, p[b], ident[b], n_w[b], self.bg_voxel, False)
                acc_ob = _voxel_union(acc_ob, p[o], ident[o], n_w[o], self.obj_voxel, True)
        finally:
            gm.alive = None
        if acc_bg is None or (not len(acc_bg[0]) and not len(acc_ob[0])):
            return empty
        bp, _, bn = acc_bg
        op, oi, on = acc_ob
        objects = []
        oi_np, op_np, on_np = oi.cpu().numpy(), op.cpu().numpy(), on.cpu().numpy().astype(np.float32)
        for i in np.unique(oi_np):
            m = oi_np == i
            objects.append(SceneObject(id=str(int(i)), instance_id=int(i), semantic=int(self.semantic_of.get(int(i), -1)),
                                       points=op_np[m].astype(np.float32), present=True, first_observed_ns=[0],
                                       last_observed_ns=[UINT64_MAX], normals=on_np[m]))
        return EvaluationScene(t_ns, bp.cpu().numpy().astype(np.float32), np.zeros(len(bp), np.uint32),
                               objects, bn.cpu().numpy().astype(np.float32))

    def _first_echo_points(self, gm, identity: torch.Tensor, kf: dict):
        """World points (metres), identities and normals (measured_surface of the rendered first-echo depth) of one view."""
        K = kf["intrinsics"]
        if kf.get("stripped"):
            d = _keyframe_depth(kf)
            h, w = d.shape
            valid = d > 0
        else:
            _, h, w = kf["color"].shape
            valid = kf["depth"].cuda().reshape(h, w) > 0
        pose = kf["pose"].cpu()
        view = gu.flashsplat_cam(torch.zeros((3, h, w), device="cuda"), torch.zeros((h, w), device="cuda"), None, K, pose, None)
        fe = probe_render_fe(view, gm)
        mi = fe["median_index"].squeeze().long()
        ok = (mi >= 0) & valid.to(mi.device).reshape(mi.shape)
        depth = torch.where(ok, fe["median"].squeeze(), torch.zeros_like(fe["median"].squeeze())) / self.scale
        ok &= depth > 0
        v, u = torch.nonzero(ok, as_tuple=True)
        ident = identity[mi[v, u]].clamp(min=0)
        P, nrm, measured = measured_surface(depth, K)
        nrm = torch.where(measured[..., None], nrm, torch.full_like(nrm, float("nan")))
        Tw = torch.linalg.inv(pose.to(torch.float64))
        Tw[:3, 3] /= self.scale
        Tw = Tw.to("cuda", torch.float32)
        return P[v, u] @ Tw[:3, :3].T + Tw[:3, 3], ident, nrm[v, u] @ Tw[:3, :3].T


def _voxel_union(run, q, ids, nn, voxel, with_id):
    """The samples so far plus new ones, one per voxel (and per identity), the first in order; memory bounded by the map."""
    if run is not None:
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
