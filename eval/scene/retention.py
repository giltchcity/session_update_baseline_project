"""Unobserved retention and observed-absence residue of inherited map content.

For session k with inherited map M_{k-1} (the previous session's final scene) and final map
M_k, every inherited surface sample is classified by session k's own raw depth frames:
  observed-present   some frame measured it within the sensor tolerance
  observed-absent    some frame measured beyond it (seen through) and none measured it present
  unobserved         no frame measured it at all (outside every view, occluded, or invalid)
and it is "shown" when M_k has a sample within `radius` of it.

  unobserved retention = shown(unobserved) / unobserved             (memory kept, higher better)
  absent residue       = shown(observed-absent) / observed-absent    (stale surface, lower better)

The classification uses only the session's depth and poses (method-independent), the same
projection and 5 cm tolerance as the layer's evidence; "seen through" additionally requires the
sample to face the camera within 60 deg when a normal is known. Samples are reduced to one per
2 cm voxel so that dense and sparse representations are weighted by surface, not by count.

  python -m eval.scene.retention PRIOR_TIMELINE CURRENT_TIMELINE {real|synthetic} STAGE OUT.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/home/jixian/Desktop/FT")
sys.path.insert(0, "/home/jixian/Desktop/FT/wt_layer_43c")
from eval.scene.emit_harness import first_per_voxel  # noqa: E402
from update_layer.frames import FlatSession, real_session, synthetic_session  # noqa: E402
from update_layer.timeline import LayerTimeline  # noqa: E402


def scene_samples(scene, with_owner: bool = False):
    pts = [scene.background] + [o.points for o in scene.objects]
    nrm = [scene.background_normal if scene.background_normal is not None
           else np.full((len(scene.background), 3), np.nan)]
    nrm += [o.normals if o.normals is not None else np.full((len(o.points), 3), np.nan)
            for o in scene.objects]
    owner = [np.zeros(len(scene.background), np.int64)] + \
        [np.full(len(o.points), o.instance_id, np.int64) for o in scene.objects]
    pts, nrm = np.concatenate(pts).astype(np.float64), np.concatenate(nrm).astype(np.float64)
    owner = np.concatenate(owner)
    keep = first_per_voxel(pts, 0.02)
    if with_owner:
        return pts[keep], nrm[keep], owner[keep]
    return pts[keep], nrm[keep]


def changed_identities(dataset: str, stage: str) -> set:
    """GT identities that changed by the end of `stage` since the previous session's end.

    Their inherited surfaces are judged by D3 and the residue, not by retention: dropping the
    unseen part of a state that GT says moved is correct, not lost memory.
    """
    import csv
    ft = Path("/home/jixian/Desktop/FT")
    if dataset == "synthetic":
        out = set()
        for r in csv.DictReader(open(ft / "datasets/synthetic_ab/GT_AB_BOUNDARY.csv")):
            if float(r["center_displacement_m"]) > 0.01 or r["a_end_render_visible"] != r["b_start_render_visible"]:
                out.add(int(r["instance_id"]))
        for r in csv.DictReader(open(ft / "datasets/synthetic_ab/GT_ANIMATION_INTERVALS.csv")):
            if r["session"] == stage:
                out.add(int(r["instance_id"]))
        return out
    events = json.loads((ft / "results/abc_eval_final_baseline/events.json").read_text())
    return {int(e["object_id"]) for e in events if e["session"] == stage.upper()}


def classify(points, normals, session: FlatSession, hz: float = 2.0, tol: float = 0.05,
             max_incidence_deg: float = 60.0):
    """Per inherited sample: observed present / observed absent / unobserved (GPU, torch)."""
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P = torch.as_tensor(points, dtype=torch.float32, device=dev)
    N = torch.as_tensor(np.nan_to_num(normals), dtype=torch.float32, device=dev)
    has_n = torch.as_tensor(np.isfinite(normals).all(axis=1), device=dev)
    present = torch.zeros(len(P), dtype=torch.bool, device=dev)
    through = torch.zeros(len(P), dtype=torch.bool, device=dev)
    min_cos = math.cos(math.radians(max_incidence_deg))
    step = max(1, int(round(30.0 / hz)))
    from concurrent.futures import ThreadPoolExecutor
    indices = list(range(0, len(session.ids), step))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in pool.map(session.load, indices):
            K = f.K
            T = torch.as_tensor(np.linalg.inv(f.T_world_cam), dtype=torch.float32, device=dev)
            R = torch.as_tensor(f.T_world_cam[:3, :3], dtype=torch.float32, device=dev)
            depth = torch.as_tensor(f.depth, dtype=torch.float32, device=dev)
            cam = P @ T[:3, :3].T + T[:3, 3]
            z = cam[:, 2]
            u = torch.floor(K.fx * cam[:, 0] / z + K.cx - K.offset + 0.5)
            v = torch.floor(K.fy * cam[:, 1] / z + K.cy - K.offset + 0.5)
            ok = (z > 0) & torch.isfinite(u) & torch.isfinite(v) & (u >= 0) & (u < K.width) & \
                (v >= 0) & (v < K.height)
            ui, vi = u.clamp(0, K.width - 1).long(), v.clamp(0, K.height - 1).long()
            d = torch.where(ok, depth[vi, ui], torch.full_like(z, float("nan")))
            meas = ok & torch.isfinite(d)
            # optical-Z comparison along the pixel ray (same pixel, same scale on both sides)
            delta = d - z
            view_w = (cam / torch.linalg.norm(cam, dim=1, keepdim=True)) @ R.T
            facing = ~has_n | (torch.abs((N * view_w).sum(1)) >= min_cos)
            present |= meas & (torch.abs(delta) <= tol)
            through |= meas & (delta > tol) & facing
    present, through = present.cpu().numpy(), through.cpu().numpy()
    absent = through & ~present
    unobserved = ~present & ~through
    return present, absent, unobserved


def main() -> None:
    prior_path, current_path, dataset, stage, out = sys.argv[1:6]
    prior, current = LayerTimeline.load(prior_path), LayerTimeline.load(current_path)
    spec = synthetic_session(stage) if dataset == "synthetic" else real_session(stage)
    session = FlatSession(spec, pixel_step=2 if dataset == "synthetic" else 4)
    p_pts, p_nrm, owner = scene_samples(prior.scene(prior.stamps()[-1]), with_owner=True)
    c_pts, _ = scene_samples(current.scene(current.stamps()[-1]))
    present, absent, unobserved = classify(p_pts, p_nrm, session)
    shown = np.zeros(len(p_pts), bool)
    if len(c_pts):
        d, _ = cKDTree(c_pts).query(p_pts, distance_upper_bound=0.05)
        shown = np.isfinite(d)
    changed = changed_identities(dataset, stage)
    unchanged = ~np.isin(owner, sorted(changed)) if changed else np.ones(len(owner), bool)
    rate = lambda m: float(shown[m].mean()) if m.any() else None
    row = dict(prior=prior_path, current=current_path, dataset=dataset, stage=stage,
               inherited_samples=int(len(p_pts)), observed_present=int(present.sum()),
               observed_absent=int(absent.sum()), unobserved=int(unobserved.sum()),
               changed_identities=sorted(changed),
               unobserved_retention=rate(unobserved & unchanged),
               unobserved_retention_all=rate(unobserved),
               unobserved_unchanged=int((unobserved & unchanged).sum()),
               absent_residue=rate(absent),
               present_kept=rate(present))
    Path(out).write_text(json.dumps(row, indent=2))
    print(json.dumps(row))


if __name__ == "__main__":
    main()
