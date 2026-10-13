"""Top-down figure of the final maps: rows 2 / 3 / 4 (carried, own update, + update layer) x backends.

  python -m update_layer.eval.render_topdown synthetic b [identity|rgb] [OUT.png]
  python -m update_layer.eval.render_topdown real c [identity|rgb] [OUT.png]

identity  background grey by height, objects one colour per physical identity (same in every panel)
rgb       every map sample in the colour of the latest frame (1 fps, current session first, then
          earlier sessions) that measured it on its surface -- the colour a coloured map keeps;
          never measured -> grey. Objects that changed in the session (GT) are tinted with their
          identity colour. GaME and wavemap store no colour of their own; they are coloured the same way.
Only points below 2 m above the floor are drawn (no ceiling). Default output: D:/3Study/ETH/FT/101.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the project root
from update_layer.eval.scenelist import load_timeline  # noqa: E402
from update_layer.eval.retention import changed_identities  # noqa: E402
from update_layer.frames import FlatSession, real_session, synthetic_session  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
R = Path("/home/jixian/Desktop/FT/runs")
FIGURES = Path("/mnt/d/3Study/ETH/FT/101")
# the final run of every backend and row (update when a run is replaced)
RUNS = {
    "synthetic": {
        "points":        {2: "update_layer_points_syn_20260930/row2", 3: "update_layer_points_syn_20260930/row3",
                          4: "update_layer_points_syn_nomoved_20260930/row4"},
        "SurfelMeshing": {2: "update_layer_surfelmeshing_syn_fix_20260930/row2",
                          3: "update_layer_surfelmeshing_syn_fix_20260930/row3",
                          4: "update_layer_surfelmeshing_syn_fix_20260930/row4"},
        "GaME (3DGS)":   {2: "update_layer_game_syn_kf_20260930/row2", 3: "update_layer_game_syn_kf_20260930/row3",
                          4: "update_layer_game_syn_kf_20260930/row4"},
        "wavemap":       {2: "update_layer_wavemap_syn_20260930/row2", 3: "update_layer_wavemap_syn_20260930/row3",
                          4: "update_layer_wavemap_syn_20260930/row4"},
    },
    "real": {
        "points":        {2: "update_layer_points_real_20260930/row2", 3: "update_layer_points_real_20260930/row3",
                          4: "update_layer_points_real_final_20260930/row4"},
        "SurfelMeshing": {2: "update_layer_surfelmeshing_real_fix_20260930/row2",
                          3: "update_layer_surfelmeshing_real_fix_20260930/row3",
                          4: "update_layer_surfelmeshing_real_fix_20260930/row4"},
        "GaME (3DGS)":   {3: "update_layer_game_real_20260930/row3", 4: "update_layer_game_real_20260930/row4"},
        "wavemap":       {2: "update_layer_wavemap_real_20260930/row2", 3: "update_layer_wavemap_real_20260930/row3",
                          4: "update_layer_wavemap_real_20260930/row4"},
    },
}
ROW_NAME = {2: "row 2: carried, no update", 3: "row 3: backend's own update", 4: "row 4: + our update layer",
            5: "row 5: own update + our layer"}
# (horizontal axes, vertical axis, sign of up): synthetic world z up; real world = session A camera (y down)
AXES = {"synthetic": ((0, 1), 2, 1.0), "real": ((0, 2), 1, -1.0)}
WAVEMAP_CELL = 0.02                      # wavemap runs since 2026-10-01 20:10 (5 cm before)

rng = np.random.default_rng(0)
palette = np.concatenate([plt.get_cmap(n)(np.linspace(0, 1, 20))[:, :3] for n in ("tab20", "tab20b", "tab20c")])


def colour(identity: int) -> np.ndarray:
    return palette[(identity * 7) % len(palette)]


def timeline_path(rel: str, stage: str) -> Path:
    return R / rel / f"session_{stage}" / "timeline.pkl"


def final_scene(rel: str, stage: str):
    tl = load_timeline(timeline_path(rel, stage))
    return tl.scene(tl.stamps()[-1])


_FRAMES = None


def frames(dataset: str, stage: str):
    """Depth + colour + pose at 1 fps, current session first, latest first (cached)."""
    global _FRAMES
    if _FRAMES is None:
        make = synthetic_session if dataset == "synthetic" else real_session
        order = {"a": "a", "b": "ba", "c": "cba"}[stage]
        _FRAMES = []
        for st in order:
            fs = FlatSession(make(st), pixel_step=2)
            for i in list(range(0, len(fs.ids), 30))[::-1]:
                f = fs.load(i, color=True)
                _FRAMES.append((f.K, torch.as_tensor(np.linalg.inv(f.T_world_cam), dtype=torch.float32, device=DEV),
                                torch.as_tensor(f.depth, device=DEV),
                                torch.as_tensor(f.color, device=DEV).float() / 255.0))
    return _FRAMES


def colourize(P: np.ndarray, fr, tol: float) -> np.ndarray:
    """Colour of each point in the latest frame that measured it within tol (grey if none)."""
    Pt = torch.as_tensor(P, dtype=torch.float32, device=DEV)
    col = torch.full((len(P), 3), -1.0, device=DEV)
    for K, T, depth, rgb in fr:
        todo = torch.nonzero(col[:, 0] < 0).squeeze(1)
        if not len(todo):
            break
        cam = Pt[todo] @ T[:3, :3].T + T[:3, 3]
        z = cam[:, 2]
        u = torch.floor(K.fx * cam[:, 0] / z + K.cx - K.offset + 0.5).long()
        v = torch.floor(K.fy * cam[:, 1] / z + K.cy - K.offset + 0.5).long()
        ok = (z > 0.1) & (u >= 0) & (u < K.width) & (v >= 0) & (v < K.height)
        d = torch.full_like(z, float("nan"))
        d[ok] = depth[v[ok], u[ok]]
        hit = ok & torch.isfinite(d) & (torch.abs(d - z) < tol)
        col[todo[hit]] = rgb[v[hit], u[hit]]
    col[col[:, 0] < 0] = 0.6
    return col.cpu().numpy()


def voxel_first(P: np.ndarray, cell: float) -> np.ndarray:
    k = np.floor(P / cell).astype(np.int64)
    _, idx = np.unique(k, axis=0, return_index=True)
    return np.sort(idx)


def main():
    dataset, stage = sys.argv[1], sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "rgb"
    out = Path(sys.argv[4]) if len(sys.argv) > 4 else FIGURES / f"{dataset}_{stage.upper()}_final_topdown_{mode}.png"
    changed = changed_identities(dataset, stage) if mode == "rgb" else set()
    (ha, hb), va, up = AXES[dataset]
    runs = {b: rows for b, rows in RUNS[dataset].items()
            if any(timeline_path(rel, stage).exists() for rel in rows.values())}
    ref = final_scene(runs["points"][4], stage)
    allp = np.concatenate([ref.background] + [o.points for o in ref.objects])
    h = up * allp[:, va]
    floor = np.percentile(h, 1)
    lo = np.percentile(allp[:, [ha, hb]], 0.5, axis=0) - 0.3
    hi = np.percentile(allp[:, [ha, hb]], 99.5, axis=0) + 0.3
    cut = floor + 2.0
    rows, cols = [2, 3, 4], list(runs)
    w, hgt = hi[0] - lo[0], hi[1] - lo[1]
    fig, axes = plt.subplots(len(rows), len(cols), squeeze=False,
                             figsize=(4.2 * len(cols) * w / max(w, hgt) + 1, 4.2 * len(rows) * hgt / max(w, hgt) + 1.2))
    for c, backend in enumerate(cols):
        for r, row in enumerate(rows):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            rel = runs[backend].get(row)
            if rel is None or not timeline_path(rel, stage).exists():
                ax.set_title(f"{backend} - {ROW_NAME[row]}\n(not run)", fontsize=8)
                continue
            sc = final_scene(rel, stage)
            bg = sc.background
            bh = up * bg[:, va]
            keep = bh < cut
            bg, bh = bg[keep], bh[keep]
            if len(bg) > 400_000:
                s = rng.choice(len(bg), 400_000, replace=False)
                bg, bh = bg[s], bh[s]
            pts = [bg]
            cols_ = [np.repeat(np.clip(0.88 - 0.22 * (bh - floor)[:, None], 0.45, 0.9), 3, axis=1)]
            owner = [np.zeros(len(bg), np.int64)]
            n_obj = 0
            for o in sc.objects:
                p = o.points[up * o.points[:, va] < cut]
                if not len(p):
                    continue
                n_obj += 1
                if len(p) > 20_000:
                    p = p[rng.choice(len(p), 20_000, replace=False)]
                pts.append(p)
                cols_.append(np.tile(colour(o.instance_id), (len(p), 1)))
                owner.append(np.full(len(p), o.instance_id, np.int64))
            P, C, own = np.concatenate(pts), np.concatenate(cols_), np.concatenate(owner)
            if mode == "rgb":
                keep = voxel_first(P, 0.01)
                P, own = P[keep], own[keep]
                tol = 0.05 + (0.5 * np.sqrt(3.0) * WAVEMAP_CELL if backend == "wavemap" else 0.0)
                C = colourize(P, frames(dataset, stage), tol)
                hl = np.isin(own, list(changed))
                if hl.any():
                    C[hl] = 0.45 * C[hl] + 0.55 * np.stack([colour(i) for i in own[hl]])
            order = np.argsort(up * P[:, va])                        # higher points drawn on top
            size = 1.6 if backend == "wavemap" else 0.15             # wavemap: 5 cm cells, drawn cell-sized
            ax.scatter(P[order, ha], P[order, hb], c=C[order], s=size, marker="s" if backend == "wavemap" else "o",
                       linewidths=0, rasterized=True)
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
            ax.set_title(f"{backend} - {ROW_NAME[row]}\n{n_obj} objects shown", fontsize=8)
    what = ("map colours (latest frame that measured each sample); tinted: objects that changed in this session (GT)"
            if mode == "rgb" else "objects coloured by identity")
    fig.suptitle(f"{dataset} session {stage.upper()}: final map, top-down (below {cut - floor:.1f} m); {what}", fontsize=10)
    fig.tight_layout(h_pad=2.2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
