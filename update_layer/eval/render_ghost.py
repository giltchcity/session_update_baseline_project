"""Ghost figure: where a removed object was, final map of row 3 (backend's own update) vs row 4 (+ layer).

  python -m update_layer.eval.render_ghost BACKEND STAGE EVENT_ID [OUT.png]
  e.g.  python -m update_layer.eval.render_ghost SurfelMeshing c D3_BC_I2

BACKEND is a key of render_topdown.RUNS["real"]. The old object's site = the probes of EVENT_ID in the
ghost evaluation (eval_real/ghost/STAGE/evidence_final_STAGE.npz of the row 3 run) that the cameras
saw through (known empty: strictfree >= 3 and support < 3, as cleanup_evaluate_h.py). The view is the
frame of the session that sees through most of them. Red = map samples within 5 cm of those probes
(an illustration); the titles give the official numbers (final_full_confirmed_wrong_surface /
final_full_initial_supported_known_empty of event_summary_STAGE.csv). Default output: D:/3Study/ETH/FT/101.
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
from update_layer.eval.render_topdown import FIGURES, R, RUNS, colourize, final_scene, frames, voxel_first  # noqa: E402
from update_layer.eval.render_perspective import render  # noqa: E402
from update_layer.frames import FlatSession, real_session  # noqa: E402

NEAR = 0.05
RED = np.array([1.0, 0.1, 0.1])


def ghost_dir(rel: str, stage: str) -> Path:
    return R / rel / "eval_real" / "ghost" / stage


def known_empty_probes(rel: str, stage: str, event: str) -> np.ndarray:
    d = ghost_dir(rel, stage)
    ev = np.load(d / f"evidence_final_{stage}.npz")
    lay = json.loads((d / f"evidence_layout_{stage}.json").read_text())[event]
    sl = slice(lay["probe_offset"], lay["probe_offset"] + lay["probe_count"])
    known = (ev["strictfree"][sl] >= 3) & (ev["support"][sl] < 3)
    return ev["points"][sl][known].astype(np.float32)


def official(rel: str, stage: str, event: str) -> str:
    for r in csv.DictReader(open(ghost_dir(rel, stage) / f"event_summary_{stage}.csv")):
        if r["event_id"] == event:
            w, n = int(r["final_full_confirmed_wrong_surface"]), int(r["final_full_initial_supported_known_empty"])
            return f"official: {w}/{n} old-site probes still on map surface ({100 * w / max(1, n):.0f}%), {r['status']}"
    return "official: event not scored"


def best_view(probes: np.ndarray, stage: str, step: int = 10) -> int:
    """Frame of the session that sees through the most probes (measured depth well behind them)."""
    fs = FlatSession(real_session(stage), pixel_step=2)
    best, best_n = 0, -1
    for i in range(0, len(fs.ids), step):
        f = fs.load(i)
        T = np.linalg.inv(f.T_world_cam)
        cam = probes @ T[:3, :3].T + T[:3, 3]
        z = cam[:, 2]
        ok = z > 0.3
        u = np.floor(f.K.fx * cam[ok, 0] / z[ok] + f.K.cx - f.K.offset + 0.5).astype(int)
        v = np.floor(f.K.fy * cam[ok, 1] / z[ok] + f.K.cy - f.K.offset + 0.5).astype(int)
        inside = (u >= 0) & (u < f.K.width) & (v >= 0) & (v < f.K.height)
        d = f.depth[v[inside], u[inside]]
        n = int((np.isfinite(d) & (d > z[ok][inside] + 0.1)).sum())
        if n > best_n:
            best, best_n = i, n
    return best


def main():
    backend, stage, event = sys.argv[1], sys.argv[2], sys.argv[3]
    out = Path(sys.argv[4]) if len(sys.argv) > 4 else FIGURES / f"ghost_{backend.split()[0]}_{stage.upper()}_{event}.png"
    rows = RUNS["real"][backend]
    probes = known_empty_probes(rows[3], stage, event)
    view_i = best_view(probes, stage)
    view = FlatSession(real_session(stage), pixel_step=2).load(view_i, color=True)
    fr = frames("real", stage)
    tree = cKDTree(probes)
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.9))
    axes[0].imshow(view.color)
    axes[0].set_title(f"photo, session {stage.upper()} frame {view_i}: the object is gone", fontsize=9)
    for ax, row in zip(axes[1:], (3, 4)):
        sc = final_scene(rows[row], stage)
        P = np.concatenate([sc.background] + [o.points for o in sc.objects]).astype(np.float32)
        P = P[voxel_first(P, 0.01)]
        C = colourize(P, fr, 0.05)
        ghost = tree.query(P, distance_upper_bound=NEAR)[0] < NEAR
        C[ghost] = RED
        ax.imshow(np.clip(render(P, C, view, 0.006, 4), 0, 1))
        name = "row 3: backend's own update" if row == 3 else "row 4: + our update layer"
        ax.set_title(f"{backend} {name}, end of {stage.upper()}\nred: map surface at the old site ({int(ghost.sum())} samples)\n"
                     f"{official(rows[row], stage, event)}", fontsize=8)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"ghost of a removed object ({event}): real session {stage.upper()}, final maps from the same camera",
                 fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print("wrote", out, "view", view_i, "known-empty probes", len(probes))


if __name__ == "__main__":
    main()
