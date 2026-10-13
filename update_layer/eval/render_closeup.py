"""Close-up of one change site: rows 2 / 3 / 4 x backends, final maps of STAGE, cropped, true map colours.

  python -m update_layer.eval.render_closeup real c D3_BC_I2 [OUT.png]          (real: an events.json id)
  python -m update_layer.eval.render_closeup synthetic a chair_0001 [OUT.png]   (synthetic: a GT object name)

Site = the old surface of the object that changed:
  real       the GT surface of the event's old state (results/abc_eval_final_baseline/events.json)
  synthetic  that object's samples in the final row 2 map (carried, no update) of the points backend
             that row 4 removed (farther than 5 cm from its row 4 samples)
View = the frame of STAGE that sees through most of the site (the object is gone there), latest on ties.
First column: photos: before (the frame, of the previous session or of STAGE, that measures most of
the site on its surface) and after (the view). Map samples are coloured as in render_topdown rgb mode
(latest frame that measured them); nothing is highlighted. Default output: D:/3Study/ETH/FT/101.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the project root
from update_layer.eval.render_topdown import FIGURES, ROW_NAME, RUNS, WAVEMAP_CELL, colourize, final_scene, frames, \
    timeline_path, voxel_first  # noqa: E402
from update_layer.eval.render_perspective import render  # noqa: E402
from update_layer.frames import FlatSession, real_session, synthetic_session  # noqa: E402

FT = Path("/home/jixian/Desktop/FT")
SESSION = {"real": real_session, "synthetic": synthetic_session}
PREVIOUS = {"a": None, "b": "a", "c": "b"}


def site_points(dataset: str, stage: str, key: str) -> np.ndarray:
    if dataset == "real":
        ev = {e["event_id"]: e for e in json.loads((FT / "results/abc_eval_final_baseline/events.json").read_text())}[key]
        return np.load(ev["old"]["surface"]).astype(np.float32)
    ident = {r["name"]: int(r["instance_id"]) for r in csv.DictReader(open(FT / "datasets/synthetic_ab/GT_AB_BOUNDARY.csv"))}[key]
    rows = RUNS["synthetic"]["points"]
    pick = lambda sc: np.concatenate([o.points for o in sc.objects if o.instance_id == ident] or [np.zeros((0, 3))])
    old, new = pick(final_scene(rows[2], stage)), pick(final_scene(rows[4], stage))
    if len(new):
        old = old[cKDTree(new).query(old, distance_upper_bound=0.05)[0] > 0.05]
    return old.astype(np.float32)


def project(P, f):
    T = np.linalg.inv(f.T_world_cam)
    cam = P @ T[:3, :3].T + T[:3, 3]
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.floor(f.K.fx * cam[:, 0] / z + f.K.cx - f.K.offset + 0.5)
        v = np.floor(f.K.fy * cam[:, 1] / z + f.K.cy - f.K.offset + 0.5)
    ok = (z > 0.5) & (z < 4.0) & (u >= 0) & (u < f.K.width) & (v >= 0) & (v < f.K.height)
    return u, v, z, ok


def pick_frame(dataset, stage, P, mode, step=10):
    """'after': most site samples seen through; 'before': most measured on their surface."""
    fs = FlatSession(SESSION[dataset](stage), pixel_step=2)
    best, best_n = 0, -1
    for i in range(0, len(fs.ids), step):
        f = fs.load(i)
        u, v, z, ok = project(P, f)
        d = f.depth[v[ok].astype(int), u[ok].astype(int)]
        zz = z[ok]
        hit = (d > zz + 0.1) if mode == "after" else (np.abs(d - zz) < 0.05)
        n = int((np.isfinite(d) & hit).sum())
        if n >= best_n:
            best, best_n = i, n
    return best, best_n


def crop_box(P, f, margin=0.6, min_px=160):
    u, v, z, ok = project(P, f)
    if not ok.any():
        return 0, f.K.width, 0, f.K.height
    u0, u1, v0, v1 = np.percentile(u[ok], 2), np.percentile(u[ok], 98), np.percentile(v[ok], 2), np.percentile(v[ok], 98)
    cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
    half = max(min_px / 2, (1 + margin) * max(u1 - u0, (v1 - v0) * 4 / 3) / 2)
    hw, hh = half, half * 3 / 4
    a, b = int(max(0, cu - hw)), int(min(f.K.width, cu + hw))
    c, d = int(max(0, cv - hh)), int(min(f.K.height, cv + hh))
    return a, b, c, d


def main():
    dataset, stage, key = sys.argv[1], sys.argv[2], sys.argv[3]
    out = Path(sys.argv[4]) if len(sys.argv) > 4 else FIGURES / f"closeup_{dataset}_{stage.upper()}_{key}.png"
    site = site_points(dataset, stage, key)
    name = key
    if dataset == "real":
        ev = {e["event_id"]: e for e in json.loads((FT / "results/abc_eval_final_baseline/events.json").read_text())}[key]
        name = f'{ev["old"]["name"]} {ev["change_type"]}'
    if not len(site):
        sys.exit(f"no site samples for {key}")
    after_i, n_after = pick_frame(dataset, stage, site, "after")
    prev = PREVIOUS[stage] if dataset == "real" and key.startswith("D3") else stage
    if dataset == "synthetic" and prev == stage and PREVIOUS[stage] is not None:
        prev = PREVIOUS[stage]                    # synthetic B changes happened between the sessions
    before_i, _ = pick_frame(dataset, prev, site, "before")
    view = FlatSession(SESSION[dataset](stage), pixel_step=1).load(after_i, color=True)
    before = FlatSession(SESSION[dataset](prev), pixel_step=1).load(before_i, color=True)
    a, b, c, d = crop_box(site, view)
    ba, bb, bc, bd = crop_box(site, before)
    fr = frames(dataset, stage)
    runs = {k: v for k, v in RUNS[dataset].items() if any(timeline_path(r, stage).exists() for r in v.values())}
    fig, axes = plt.subplots(3, len(runs) + 1, figsize=(4 * (len(runs) + 1), 3 * 2.9))
    axes[0][0].imshow(before.color[bc:bd, ba:bb])
    axes[0][0].set_title(f"photo, before ({prev.upper()})", fontsize=9)
    axes[1][0].imshow(view.color[c:d, a:b])
    axes[1][0].set_title(f"photo, now ({stage.upper()})", fontsize=9)
    for c_, backend in enumerate(runs, start=1):
        for r_, row in enumerate((2, 3, 4)):
            ax = axes[r_][c_]
            rel = runs[backend].get(row)
            if rel is None or not timeline_path(rel, stage).exists():
                ax.set_title(f"{backend} - row {row}\n(not run)", fontsize=9)
                continue
            sc = final_scene(rel, stage)
            P = np.concatenate([sc.background] + [o.points for o in sc.objects]).astype(np.float32)
            u, v, z, ok = project(P, view)
            inside = ok & (u >= a - 20) & (u < b + 20) & (v >= c - 20) & (v < d + 20)
            P = P[inside]
            wm = backend == "wavemap"
            if not wm:
                P = P[voxel_first(P, 0.005)]
            C = colourize(P, fr, 0.05 + (0.5 * np.sqrt(3.0) * WAVEMAP_CELL if wm else 0.0))
            img = render(P, C, view, WAVEMAP_CELL / 2 if wm else 0.004, 14 if wm else 6)
            ax.imshow(np.clip(img[c:d, a:b], 0, 1))
            ax.set_title(f"{backend} - {ROW_NAME[row]}", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"{dataset} {stage.upper()}: {name}, final maps", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print("wrote", out, "after", after_i, n_after, "before", prev, before_i, "site", len(site))


if __name__ == "__main__":
    main()
