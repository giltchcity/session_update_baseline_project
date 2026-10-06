"""Session-end memory test of the TSDF version (session_refusion.cpp step 5 @192c1cf, lines 1045-1059 and
1129-1208, decisions 1254-1294), on any backend's elements.

A memory element -- one that existed before this session's first frame and that the object reasoning does
not own (background, or an object whose physical identity is current and whose state did not begin in this
session) -- is tested against every stored frame of the session (the layer's evidence store: the ranges the
present measured, with the rejected dynamic/motion pixels removed):
  hit       some pixel of its footprint (radius f tau sqrt(xn^2 + yn^2 + 1) / q) reads within tau and the
            reading's point lies inside the element's ball of radius tau;
  through   every pixel of the footprint is in the image, valid, and reads beyond it by more than tau;
  blocked   its own pixel reads nearer than -tau; blocked_band: of those, within its truncation.
It gives way to the present when seen_through = through > hit, or hidden = no hit, no through and
2 blocked_band > blocked. displaced (needs the sessions' depth scales, P41) is not ported yet: off.

Representation terms (CSV 2026-10-06 08:55): tau = max(half, sigma(q)); half = the element's extent
(interface.Elements.extent: a Gaussian's 3 sigma, a surfel's radius, half a cell's diagonal, 0 for points);
the truncation = the distance within which the representation cannot hold a second surface behind the
observed one = the element's extent (for a Gaussian its own support; assumption: two surfaces closer than
it are not resolved by it), so for tau >= extent the band is empty and `hidden` cannot fire. sigma(q): the
P37 range table is not ported; the layer's surface tolerance stands in (temporary). Frames: the evidence
store's (5 Hz) instead of every frame (30 Hz); hit and through are counted over the same frames. The
`stale` pixel removal of the archive (step 1b) and the removal of object pixels before t_L (step 1) are not
ported.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .evidence import DEV

# SessionRefusion::Config defaults (session_refusion.h @192c1cf:64-66)
RANGE_BIN = 0.5
NUM_BINS = 16
HISTOGRAM_RESOLUTION = 0.0005


def sigma_from_histogram(hist: np.ndarray, resolution: float) -> np.ndarray:
    """sigmaFromHistogram (session_refusion.cpp:224-293): per range bin 1.4826 x the median residual, as
    log sigma observations with the Gaussian-MAD sampling variance 1/(16 n phi(z)^2 z^2), z = Phi^-1(3/4);
    a random walk over the bins with step variance delta^2 = the moment excess of squared adjacent
    differences; Rauch-Tung-Striebel smoothing."""
    nb = len(hist)
    z = 0.6744897501960817
    phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    y = [float("nan")] * nb
    r = [float("inf")] * nb
    for b in range(nb):
        cnt = int(hist[b].sum())
        if cnt == 0:
            continue
        half = 0.5 * cnt
        cum = 0
        for k in range(len(hist[b])):
            prev = cum
            cum += int(hist[b][k])
            if cum >= half:
                sigma = 1.4826 * (k + (half - prev) / max(int(hist[b][k]), 1)) * resolution
                if sigma > 0.0:
                    y[b] = math.log(sigma)
                    r[b] = 1.0 / (16.0 * cnt * phi * phi * z * z)
                break
    excess, pairs = 0.0, 0
    for b in range(nb - 1):
        if not (math.isfinite(y[b]) and math.isfinite(y[b + 1])):
            continue
        excess += (y[b + 1] - y[b]) ** 2 - r[b] - r[b + 1]
        pairs += 1
    delta2 = max(0.0, excess / pairs) if pairs else 0.0
    inf = float("inf")
    m, p, mp, pp = [0.0] * nb, [inf] * nb, [0.0] * nb, [inf] * nb
    mean, var = 0.0, inf
    for b in range(nb):
        if b > 0:
            var += delta2
        mp[b], pp[b] = mean, var
        if math.isfinite(y[b]):
            if math.isinf(var):
                mean, var = y[b], r[b]
            else:
                gain = var / (var + r[b])
                mean += gain * (y[b] - mean)
                var *= 1.0 - gain
        m[b], p[b] = mean, var
    out = np.zeros(nb, np.float32)
    if math.isinf(var):
        return out                                   # no bin measured
    smooth = list(m)
    for b in range(nb - 2, -1, -1):
        gain = 1.0 if (math.isinf(p[b]) or not pp[b + 1] > 0.0) else p[b] / pp[b + 1]
        smooth[b] = smooth[b + 1] if math.isinf(p[b]) else m[b] + gain * (smooth[b + 1] - mp[b + 1])
    return np.exp(np.array(smooth)).astype(np.float32)


def _depth_from_range(store, f: int, rejected) -> torch.Tensor:
    K = store.K
    H, W = store.rng.shape[1:]
    u = (torch.arange(W, device=DEV, dtype=torch.float32) + K.offset - K.cx) / K.fx
    v = (torch.arange(H, device=DEV, dtype=torch.float32) + K.offset - K.cy) / K.fy
    scale = torch.sqrt(u[None, :] ** 2 + v[:, None] ** 2 + 1.0)
    rng = store.rng[f].to(torch.float32) * 1e-3
    if rejected is not None and rejected[f] is not None:
        rng = torch.where(rejected[f], torch.zeros_like(rng), rng)
    return rng / scale, rng


def present_surface(store, voxel: float, rejected=None):
    """Step 2: the present -- a TSDF of this session's frames (object voxel, truncation 2 voxels) and its
    marching-cubes surface (vertices, faces, vertex normals = normalised sums of face normals)."""
    import open3d as o3d
    K = store.K
    H, W = store.rng.shape[1:]
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=2.0 * voxel, color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
    intr = o3d.camera.PinholeCameraIntrinsic(W, H, K.fx, K.fy, K.cx - K.offset, K.cy - K.offset)
    dummy = o3d.geometry.Image(np.zeros((H, W, 3), np.uint8))
    for f in range(store.n):
        depth, _ = _depth_from_range(store, f, rejected)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            dummy, o3d.geometry.Image(np.ascontiguousarray(depth.cpu().numpy())), depth_scale=1.0,
            depth_trunc=1e9, convert_rgb_to_intensity=False)
        vol.integrate(rgbd, intr, store.T[f].cpu().numpy().astype(np.float64))
    mesh = vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    V = torch.as_tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=DEV)
    F = torch.as_tensor(np.asarray(mesh.triangles), dtype=torch.int64, device=DEV)
    N = torch.as_tensor(np.asarray(mesh.vertex_normals), dtype=torch.float32, device=DEV)
    return V, F, N


def rendered_surface(store, render, rejected=None, cell: float = 0.02):
    """The present as the representation's own current surface: per stored frame, the map of now rendered from
    that camera (median depth where T first drops to 0.5, where alpha >= 0.5). Returns per frame the rendered range
    image at the store's pixels (0 where none) and the back-projected surface points (one per `cell`)."""
    K = store.K
    H, W = store.rng.shape[1:]
    ranges, pts = [], []
    vv, uu = torch.meshgrid(torch.arange(H, device=DEV), torch.arange(W, device=DEV), indexing="ij")
    xn = (uu.to(torch.float32) + K.offset - K.cx) / K.fx
    yn = (vv.to(torch.float32) + K.offset - K.cy) / K.fy
    scale = torch.sqrt(xn * xn + yn * yn + 1.0)
    for f in range(store.n):
        T = store.T[f]
        median, alpha, top, left = render(torch.linalg.inv(T).cpu().numpy(), K)
        z = torch.zeros((H, W), device=DEV)
        hh, ww = median.shape
        z[top:top + hh, left:left + ww] = torch.where(alpha >= 0.5, median, torch.zeros_like(median))
        r = z * scale
        ranges.append(r)
        ok = z > 0
        cam = torch.stack([xn[ok] * z[ok], yn[ok] * z[ok], z[ok]], 1)
        R_wc, t_wc = T[:3, :3].T, -(T[:3, :3].T @ T[:3, 3])
        pts.append(cam @ R_wc.T + t_wc)
    P = torch.cat(pts) if pts else torch.zeros((0, 3), device=DEV)
    if len(P):
        keys = torch.floor(P / cell).to(torch.int64)
        _, first = np.unique(keys.cpu().numpy(), axis=0, return_index=True)
        P = P[torch.as_tensor(np.sort(first), device=DEV)]
    return ranges, P


def noise_table_rendered(store, ranges, trunc: float, rejected=None) -> np.ndarray:
    """Step 3 with the rendered present: per stored frame and pixel, |reading - rendered range| within one
    truncation (the rendered surface is the front-facing one along that ray), sigmaFromHistogram unchanged."""
    nh = int(math.floor(trunc / HISTOGRAM_RESOLUTION + 1e-9)) + 1
    hist = torch.zeros(NUM_BINS * nh, dtype=torch.int64, device=DEV)
    for f in range(store.n):
        _, rng = _depth_from_range(store, f, rejected)
        q = ranges[f].to(torch.float64)
        d = rng.to(torch.float64)
        ok = (q > 0) & (d > 0)
        r = (d - q).abs()
        ok &= r <= trunc
        b = torch.clamp((q[ok] / RANGE_BIN).floor().to(torch.int64), max=NUM_BINS - 1)
        c = torch.clamp((r[ok] / HISTOGRAM_RESOLUTION).floor().to(torch.int64), max=nh - 1)
        hist += torch.bincount(b * nh + c, minlength=NUM_BINS * nh)
    return sigma_from_histogram(hist.view(NUM_BINS, nh).cpu().numpy(), HISTOGRAM_RESOLUTION)


def noise_table(store, V: torch.Tensor, N: torch.Tensor, trunc: float, rejected=None) -> np.ndarray:
    """Step 3 (session_refusion.cpp:940-1007): sigma(q) per range bin = 1.4826 x the median |reading - range|
    of front-facing present vertices within one truncation, smoothed over the bins."""
    K = store.K
    H, W = store.rng.shape[1:]
    nh = int(math.floor(trunc / HISTOGRAM_RESOLUTION + 1e-9)) + 1
    hist = torch.zeros(NUM_BINS * nh, dtype=torch.int64, device=DEV)
    for f in range(store.n):
        _, rng = _depth_from_range(store, f, rejected)
        T = store.T[f]
        cam = V @ T[:3, :3].T + T[:3, 3]
        z = cam[:, 2]
        ok = z > 0.1
        zs = torch.where(ok, z, torch.ones_like(z))
        u = torch.floor(cam[:, 0] / zs * K.fx + K.cx - K.offset + 0.5).to(torch.int64)
        v = torch.floor(cam[:, 1] / zs * K.fy + K.cy - K.offset + 0.5).to(torch.int64)
        ok &= (u >= 0) & (v >= 0) & (u < W) & (v < H)
        centre = -(T[:3, :3].T @ T[:3, 3])                                       # camera centre, world
        ok &= ((centre[None, :] - V) * N).sum(1) > 0                            # front-facing
        d = rng[v.clamp(0, H - 1), u.clamp(0, W - 1)].to(torch.float64)
        q = torch.linalg.norm(cam, dim=1).to(torch.float64)
        r = (d - q).abs()
        ok &= (d > 0) & (r <= trunc)
        b = torch.clamp((q[ok] / RANGE_BIN).floor().to(torch.int64), max=NUM_BINS - 1)
        c = torch.clamp((r[ok] / HISTOGRAM_RESOLUTION).floor().to(torch.int64), max=nh - 1)
        hist += torch.bincount(b * nh + c, minlength=NUM_BINS * nh)
    return sigma_from_histogram(hist.view(NUM_BINS, nh).cpu().numpy(), HISTOGRAM_RESOLUTION)


def depth_scale(store, association: float, stride: int, rejected=None) -> Tuple[float, dict]:
    """depthScale (session_refusion.cpp:377-477, P41): the scale s that makes the frames agree best -- every
    reading scaled by (1 + s) minimises the median disagreement of re-measured points (pixels of one frame
    projected into another, associated within `association`) in measured units, m_s / (1 + s); coarse
    +-10 % in 0.2 % steps, fine 0.02 % steps; used only when n log(m_0 / m_s) > (1/2) log n (Laplace,
    one parameter). Up to 64 frames, one pixel per `stride`."""
    K = store.K
    H, W = store.rng.shape[1:]
    step = max(1, store.n // 64)
    views = list(range(0, store.n, step))[:64]
    if len(views) < 2:
        return 0.0, dict(samples=0)
    vv, uu = torch.meshgrid(torch.arange(0, H, stride, device=DEV), torch.arange(0, W, stride, device=DEV),
                            indexing="ij")
    vv, uu = vv.reshape(-1), uu.reshape(-1)
    ray = torch.stack([(uu.to(torch.float32) + K.offset - K.cx) / K.fx,
                       (vv.to(torch.float32) + K.offset - K.cy) / K.fy, torch.ones(len(uu), device=DEV)], 1)
    ray = ray / torch.linalg.norm(ray, dim=1, keepdim=True)
    rngs = {f: _depth_from_range(store, f, rejected)[1] for f in views}
    origin, direction, other, rng_a, rng_b = [], [], [], [], []
    for g in views:
        R_wc, t_wc = store.R[g], -(store.T[g][:3, :3].T @ store.T[g][:3, 3])
        d = rngs[g][vv, uu]
        ok = d > 0
        dirw = ray[ok] @ R_wc.T
        pt = t_wc + dirw * d[ok, None]
        for f in views:
            if f == g:
                continue
            T = store.T[f]
            cam = pt @ T[:3, :3].T + T[:3, 3]
            z = cam[:, 2]
            m = z > 0.1
            zs = torch.where(m, z, torch.ones_like(z))
            pu = torch.floor(cam[:, 0] / zs * K.fx + K.cx - K.offset + 0.5).to(torch.int64)
            pv = torch.floor(cam[:, 1] / zs * K.fy + K.cy - K.offset + 0.5).to(torch.int64)
            m &= (pu >= 0) & (pv >= 0) & (pu < W) & (pv < H)
            q = torch.linalg.norm(cam, dim=1)
            e = rngs[f][pv.clamp(0, H - 1), pu.clamp(0, W - 1)]
            m &= (e > 0) & ((e - q).abs() <= association)
            if not m.any():
                continue
            n = int(m.sum())
            origin.append(t_wc.expand(n, 3))
            direction.append(dirw[m])
            other.append((-(T[:3, :3].T @ T[:3, 3])).expand(n, 3))
            rng_a.append(d[ok][m])
            rng_b.append(e[m])
    if not origin:
        return 0.0, dict(samples=0)
    O, Dr, Ot, Ra, Rb = (torch.cat(x) for x in (origin, direction, other, rng_a, rng_b))

    def measured(sc: float) -> float:
        point = O + Dr * (Ra * (1.0 + sc))[:, None]
        res = (Rb * (1.0 + sc) - torch.linalg.norm(point - Ot, dim=1)).abs()
        k = len(res) // 2
        mid = torch.kthvalue(res.cpu(), k + 1).values.item()      # nth_element at size/2
        return mid / (1.0 + sc)
    best_s, best = 0.0, measured(0.0)
    for k in range(-50, 51):
        sc = 0.002 * k
        m = measured(sc)
        if m < best:
            best, best_s = m, sc
    coarse = best_s
    for k in range(-10, 11):
        sc = coarse + 0.0002 * k
        m = measured(sc)
        if m < best:
            best, best_s = m, sc
    n = float(len(Ra))
    exact = measured(0.0)
    gain = n * math.log(exact / best) if best > 0 and exact > 0 else 0.0
    fitted = gain > 0.5 * math.log(n)
    return (best_s if fitted else 0.0), dict(samples=int(n), median_exact=exact, median_fitted=best, scale=best_s,
                                             log_gain=gain, penalty=0.5 * math.log(n), fitted=fitted)


def memory_test(store, centroid: torch.Tensor, half: torch.Tensor, trunc: torch.Tensor, sigma,
                rejected=None, chunk: int = 65536) -> Dict[str, torch.Tensor]:
    """Evidence of every tested element over the store's frames: hit, through, blocked, blocked_band.
    sigma: the noise table (per range bin of RANGE_BIN) or one value."""
    sig = torch.as_tensor(np.atleast_1d(np.asarray(sigma, dtype=np.float32)), device=DEV)
    n = len(centroid)
    hit = torch.zeros(n, dtype=torch.int32, device=DEV)
    through = torch.zeros_like(hit)
    blocked = torch.zeros_like(hit)
    band = torch.zeros_like(hit)
    q_reach = torch.full((n,), float("inf"), device=DEV)
    cam_reach = torch.zeros((n, 3), device=DEV)
    if not n or not store.n:
        return dict(hit=hit, through=through, blocked=blocked, blocked_band=band, q_reach=q_reach, cam_reach=cam_reach)
    K = store.K
    H, W = store.rng.shape[1:]
    for f in range(store.n):
        rng = store.rng[f].to(torch.float32) * 1e-3                    # metres, 0 = invalid
        if rejected is not None and rejected[f] is not None:
            rng = torch.where(rejected[f], torch.zeros_like(rng), rng)
        T = store.T[f]
        for a in range(0, n, chunk):
            p = centroid[a:a + chunk]
            cam = p @ T[:3, :3].T + T[:3, 3]
            x, y, z = cam[:, 0], cam[:, 1], cam[:, 2]
            ok = z > 0.1
            zs = torch.where(ok, z, torch.ones_like(z))
            u = torch.floor(x / zs * K.fx + K.cx - K.offset + 0.5).to(torch.int64)
            v = torch.floor(y / zs * K.fy + K.cy - K.offset + 0.5).to(torch.int64)
            ok &= (u >= 0) & (v >= 0) & (u < W) & (v < H)
            idx = torch.nonzero(ok).squeeze(1)
            if not len(idx):
                continue
            u, v, cam = u[idx], v[idx], cam[idx]
            q = torch.linalg.norm(cam, dim=1)
            sb = sig[torch.clamp((q / RANGE_BIN).floor().to(torch.int64), max=len(sig) - 1)]
            tau = torch.maximum(half[a:a + chunk][idx], sb)                     # tauOf(half, q)
            tr = trunc[a:a + chunk][idx]
            gi = idx + a
            # blocked / blocked_band at the element's own pixel
            d0 = rng[v, u]
            r0 = d0 - q
            bl = (d0 > 0) & (r0 < -tau)
            blocked[gi] += bl.to(torch.int32)
            band[gi] += (bl & (r0 >= -tr)).to(torch.int32)
            # reached: at most the truncation in front (the nearest such frame)
            reach = (d0 > 0) & (r0 >= -tr) & (q < q_reach[gi])
            q_reach[gi[reach]] = q[reach]
            cam_reach[gi[reach]] = -(T[:3, :3].T @ T[:3, 3])
            # footprint
            xn = (u.to(torch.float32) + K.offset - K.cx) / K.fx
            yn = (v.to(torch.float32) + K.offset - K.cy) / K.fy
            rp = K.fx * tau * torch.sqrt(xn * xn + yn * yn + 1.0) / q
            R = torch.floor(rp).to(torch.int64)
            h = torch.zeros(len(idx), dtype=torch.bool, device=DEV)
            all_ok = torch.ones(len(idx), dtype=torch.bool, device=DEV)
            for r_val in torch.unique(R).tolist():
                m = torch.nonzero(R == r_val).squeeze(1)
                dv, du = torch.meshgrid(torch.arange(-r_val, r_val + 1, device=DEV),
                                        torch.arange(-r_val, r_val + 1, device=DEV), indexing="ij")
                du, dv = du.reshape(-1), dv.reshape(-1)
                d2 = (du * du + dv * dv).to(torch.float32)
                for s in range(0, len(m), max(1, (1 << 22) // len(du))):
                    mm = m[s:s + max(1, (1 << 22) // len(du))]
                    inside_disk = (d2[None, :] <= rp[mm, None] ** 2) | ((du == 0) & (dv == 0))[None, :]
                    xx, yy = u[mm, None] + du[None, :], v[mm, None] + dv[None, :]
                    in_img = (xx >= 0) & (yy >= 0) & (xx < W) & (yy < H)
                    d = rng[yy.clamp(0, H - 1), xx.clamp(0, W - 1)]
                    valid = in_img & (d > 0)
                    r = d - q[mm, None]
                    t = tau[mm, None]
                    ray = torch.stack([(xx.to(torch.float32) + K.offset - K.cx) / K.fx,
                                       (yy.to(torch.float32) + K.offset - K.cy) / K.fy,
                                       torch.ones_like(d)], dim=-1)
                    point = ray / torch.linalg.norm(ray, dim=-1, keepdim=True) * d[..., None]
                    ball = ((point - cam[mm, None, :]) ** 2).sum(-1) <= t * t
                    hh = (inside_disk & valid & (r.abs() <= t) & ball).any(dim=1)
                    # every footprint pixel valid and beyond by more than tau
                    beyond = valid & (r > t)
                    ao = (~inside_disk | beyond).all(dim=1)
                    h[mm] = hh
                    all_ok[mm] = ao
            hit[gi] += h.to(torch.int32)
            through[gi] += (~h & all_ok).to(torch.int32)
    return dict(hit=hit, through=through, blocked=blocked, blocked_band=band, q_reach=q_reach, cam_reach=cam_reach)


def inside_test(store, xyz: torch.Tensor, identity: torch.Tensor, render, margin: torch.Tensor) -> torch.Tensor:
    """INSIDE (session_refusion.cpp:318-375, 1307-1316) for a representation that renders: an object's memory
    element gives way when it lies inside the same object's present surface. GOF's opacity field (Gaussian Opacity
    Fields, arXiv 2404.10772 sec. 4.1: a point's opacity is its minimum over the views; >= 0.5 is inside), with
    the object rendered alone (its own live elements: median depth where T first drops to 0.5, and alpha): a view
    counts where the element projects inside the object's silhouette (alpha >= 0.5); projecting outside it in any
    view means outside the object; INSIDE = in at least one view and behind the object's median-depth surface in
    every view (opacity >= 0.5 there) by more than `margin`. The margin is the TSDF's 'farther than one voxel from
    any present surface' (session_refusion.cpp:362): an element whose support reaches the surface is part of it --
    a Gaussian centred just behind the 0.5 level set carries the rest of that surface's opacity, and removing it
    thins the surface (offline smoke3 B without the margin: 38157 removed, the median-depth export changed,
    P@5 87.65 -> 87.43). The TSDF's voxel is that representation's own resolution; the 3DGS counterpart, as for
    blocked_band, is the element's own support: margin = its extent (3 sigma, the rasterizer's cut-off). The depth
    behind along each ray bounds the 3D distance from above, so this is necessary, not sufficient. The TSDF decides
    'inside' by 64 fixed directions (more first hits from behind); here GOF's view criterion (opacity = min over the
    views >= 0.5) replaces it, the representation's own, literature-backed form.
    render(identity, T_world_cam, K) -> (median, alpha, top, left)."""
    out = torch.zeros(len(xyz), dtype=torch.bool, device=DEV)
    K = store.K
    H, W = store.rng.shape[1:]
    for ident in torch.unique(identity).tolist():
        idx = torch.nonzero(identity == ident).squeeze(1)
        p = xyz[idx]
        mg = margin[idx]
        seen = torch.zeros(len(idx), dtype=torch.bool, device=DEV)
        outside = torch.zeros_like(seen)
        behind_all = torch.ones_like(seen)
        for f in range(store.n):
            T = store.T[f]
            cam = p @ T[:3, :3].T + T[:3, 3]
            z = cam[:, 2]
            ok = z > 0.1
            zs = torch.where(ok, z, torch.ones_like(z))
            u = torch.floor(cam[:, 0] / zs * K.fx + K.cx - K.offset + 0.5).to(torch.int64)
            v = torch.floor(cam[:, 1] / zs * K.fy + K.cy - K.offset + 0.5).to(torch.int64)
            ok &= (u >= 0) & (v >= 0) & (u < W) & (v < H)
            if not ok.any():
                continue
            T_world_cam = torch.linalg.inv(T).cpu().numpy()
            median, alpha, top, left = render(int(ident), T_world_cam, K)
            hh, ww = median.shape
            uc, vc = u - left, v - top
            ok &= (uc >= 0) & (vc >= 0) & (uc < ww) & (vc < hh)
            a = alpha[vc.clamp(0, hh - 1), uc.clamp(0, ww - 1)]
            m = median[vc.clamp(0, hh - 1), uc.clamp(0, ww - 1)]
            in_sil = ok & (a >= 0.5)
            outside |= ok & ~in_sil
            seen |= in_sil
            behind_all &= ~in_sil | (z - m > mg)
        out[idx] = seen & ~outside & behind_all
    return out


def decide(ev: Dict[str, torch.Tensor], centroid=None, half=None, sigma=None, error_per_metre: float = 0.0,
           present=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(seen_through, hidden, displaced) of step 5 (session_refusion.cpp:1254-1294). displaced: neither, a
    frame reached it, and the nearest present surface within tau(half, q_reach) + error_per_metre q_reach
    lies farther than 2 half from it on the camera side (the position error two sessions' depth scales
    explain: (|s_prev| + |s_now|) q)."""
    seen_through = ev["through"] > ev["hit"]
    hidden = (ev["hit"] == 0) & (ev["through"] == 0) & (2 * ev["blocked_band"] > ev["blocked"]) & ~seen_through
    displaced = torch.zeros_like(seen_through)
    if present is not None and error_per_metre > 0.0:
        cand = ~seen_through & ~hidden & torch.isfinite(ev["q_reach"])
        idx = torch.nonzero(cand).squeeze(1)
        if len(idx):
            c = centroid[idx]
            V, F = present
            if F is None:                                   # present as surface points (rendered)
                from scipy.spatial import cKDTree
                _, j = cKDTree(V.cpu().numpy()).query(c.cpu().numpy(), k=1)
                closest = V[torch.as_tensor(j, device=DEV)]
            else:
                import open3d as o3d
                mesh = o3d.t.geometry.TriangleMesh()
                mesh.vertex.positions = o3d.core.Tensor(V.cpu().numpy())
                mesh.triangle.indices = o3d.core.Tensor(F.cpu().numpy().astype(np.uint32))
                scene = o3d.t.geometry.RaycastingScene()
                scene.add_triangles(mesh)
                closest = torch.as_tensor(scene.compute_closest_points(
                    o3d.core.Tensor(c.cpu().numpy()))["points"].numpy(), device=DEV)
            d = torch.linalg.norm(closest - c, dim=1)
            qr = ev["q_reach"][idx]
            sig = torch.as_tensor(np.atleast_1d(np.asarray(sigma, dtype=np.float32)), device=DEV)
            tau = torch.maximum(half[idx], sig[torch.clamp((qr / RANGE_BIN).floor().to(torch.int64), max=len(sig) - 1)])
            window = tau + error_per_metre * qr
            near = d <= window
            camera_side = ((closest - c) * (ev["cam_reach"][idx] - c)).sum(1) > 0
            displaced[idx] = near & (d > 2.0 * half[idx]) & camera_side
    return seen_through, hidden, displaced
