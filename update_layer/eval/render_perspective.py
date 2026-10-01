"""Perspective figure of the final maps of a session, seen from a real camera of that session.

  python -m update_layer.eval.render_perspective VIEW_FRAME [OUT.png]                (real C, e.g. 7071)
  python -m update_layer.eval.render_perspective DATASET STAGE VIEW [OUT.png]
      VIEW = a frame index, or a change site (real: events.json id, synthetic: GT object name):
      then the frame of STAGE that sees through most of the object's old surface (render_closeup)

Columns: the camera's photo | one column per backend (render_topdown.RUNS[DATASET]); rows 2 / 3 / 4.
Points, surfels and Gaussians' samples are coloured by the latest frame (A, B, C at 1 fps) that
measured them on their surface; never measured -> grey. wavemap is drawn as its occupied 5 cm cells
coloured by height (its own look). Default output: D:/3Study/ETH/FT/101.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the project root
from update_layer.eval.render_topdown import (AXES, FIGURES, ROW_NAME, RUNS, WAVEMAP_CELL, colourize,  # noqa: E402
                                              frames, timeline_path, final_scene, voxel_first)
from update_layer.frames import FlatSession, real_session, synthetic_session  # noqa: E402

SESSION = {"real": real_session, "synthetic": synthetic_session}
PIXEL_STEP = {"real": 2, "synthetic": 1}                            # both about 960 / 680 px wide

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def samples(rel, stage):
    sc = final_scene(rel, stage)
    return np.concatenate([sc.background] + [o.points for o in sc.objects]).astype(np.float32)


def render(P, C, view, spacing, max_r):
    """z-buffered splats of radius fx * spacing / z (1..max_r px) into the view camera."""
    K = view.K
    T = np.linalg.inv(view.T_world_cam)
    cam = P @ T[:3, :3].T + T[:3, 3]
    z = cam[:, 2]
    keep = z > 0.1
    cam, z, C = cam[keep], z[keep], C[keep]
    u = np.floor(K.fx * cam[:, 0] / z + K.cx - K.offset + 0.5).astype(np.int64)
    v = np.floor(K.fy * cam[:, 1] / z + K.cy - K.offset + 0.5).astype(np.int64)
    r = np.clip(np.round(K.fx * spacing / z), 1, max_r).astype(np.int64)
    h, w = K.height, K.width
    zb = torch.full((h * w,), float("inf"), device=DEV)
    img = torch.zeros((h * w, 3), device=DEV)
    U, V, Z, Cc, Rr = (torch.as_tensor(a, device=DEV) for a in (u, v, z, C, r))
    offsets = [(dx, dy) for dy in range(-max_r, max_r + 1) for dx in range(-max_r, max_r + 1)]
    for depth_pass in (True, False):
        for dx, dy in offsets:
            m = dx * dx + dy * dy <= Rr * Rr
            uu, vv = U + dx, V + dy
            m &= (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            pix = (vv * w + uu)[m]
            if depth_pass:
                zb.scatter_reduce_(0, pix, Z[m].float(), reduce="amin")
            else:
                win = Z[m].float() <= zb[pix] + 1e-6
                img[pix[win]] = Cc[m][win].float()
    return img.reshape(h, w, 3).cpu().numpy()


def main():
    args = sys.argv[1:]
    if args[0].isdigit():
        args = ["real", "c"] + args
    dataset, stage, key = args[:3]
    if key.isdigit():
        view_i, tag = int(key), f"view{key}"
    else:
        from update_layer.eval.render_closeup import pick_frame, site_points
        view_i, tag = pick_frame(dataset, stage, site_points(dataset, stage, key), "after")[0], key
    out = Path(args[3]) if len(args) > 3 else FIGURES / f"{dataset}_{stage.upper()}_final_{tag}.png"
    view = FlatSession(SESSION[dataset](stage), pixel_step=PIXEL_STEP[dataset]).load(view_i, color=True)
    runs = {b: rows for b, rows in RUNS[dataset].items()
            if any(timeline_path(rel, stage).exists() for rel in rows.values())}
    fr = frames(dataset, stage)
    h, w = view.color.shape[:2]
    fig, axes = plt.subplots(3, len(runs) + 1, figsize=(5 * (len(runs) + 1), 15 * h / w + 1.2))
    for r, row in enumerate((2, 3, 4)):
        axes[r][0].imshow(view.color)
        axes[r][0].set_title(f"photo, session {stage.upper()} frame {view_i}", fontsize=9)
        for c, backend in enumerate(runs, start=1):
            ax = axes[r][c]
            rel = runs[backend].get(row)
            if rel is None or not timeline_path(rel, stage).exists():
                ax.set_title(f"{backend} - {ROW_NAME[row]}\n(not run)", fontsize=9)
                continue
            P = samples(rel, stage)
            if backend == "wavemap":
                _, va, up = AXES[dataset]
                h = up * P[:, va]
                t = (h - np.percentile(h, 1)) / max(1e-6, np.percentile(h, 99) - np.percentile(h, 1))
                img = render(P, plt.get_cmap("viridis")(np.clip(t, 0, 1))[:, :3], view, WAVEMAP_CELL / 2, 12)
            else:
                P = P[voxel_first(P, 0.01)]
                img = render(P, colourize(P, fr, 0.05), view, 0.006, 4)
            ax.imshow(np.clip(img, 0, 1))
            ax.set_title(f"{backend} - {ROW_NAME[row]}", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"{dataset} session {stage.upper()}, final maps rendered from the camera of the photo", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main()
