"""Look at the final session-A maps from real cameras of A: photo | points | GaME before fix | GaME C4C5E1K1 (no R1)."""
import sys, numpy as np
sys.path.insert(0, "/home/jixian/Desktop/FT/session_update_baseline_project")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from update_layer.eval.render_topdown import colourize, frames, voxel_first
from update_layer.eval.render_perspective import render
from update_layer.eval.scenelist import load_timeline
from update_layer.frames import FlatSession, real_session
R = "/home/jixian/Desktop/FT/runs/"
cols = {"points": "update_layer_points_real_final_20260930/row4",
        "GaME before fix": "update_layer_game_real_20260930/row4_noC4C5_20261001",
        "GaME C4 C5 E1 K1": "update_layer_game_real_20260930/row4_noR1_20261001"}
P = {}
for k, rel in cols.items():
    tl = load_timeline(R + rel + "/session_a/timeline.pkl"); sc = tl.scene(tl.stamps()[-1])
    p = np.concatenate([sc.background] + [o.points for o in sc.objects]).astype(np.float32); P[k] = p[voxel_first(p, 0.01)]
fr = frames("real", "a"); C = {k: colourize(p, fr, 0.05) for k, p in P.items()}
views = [int(v) for v in sys.argv[1:]]
fig, axes = plt.subplots(len(views), 4, figsize=(22, 3.3 * len(views)))
fs = FlatSession(real_session("a"), pixel_step=2)
for r, vi in enumerate(views):
    view = fs.load(vi, color=True)
    axes[r][0].imshow(view.color); axes[r][0].set_title(f"photo, A frame {vi}", fontsize=10)
    for c, k in enumerate(cols, start=1):
        axes[r][c].imshow(np.clip(render(P[k], C[k], view, 0.006, 4), 0, 1)); axes[r][c].set_title(k, fontsize=10)
for ax in axes.ravel(): ax.axis("off")
fig.tight_layout(); out = R + "update_layer_game_real_20260930/inspect/look_a.png"; fig.savefig(out, dpi=90); print("wrote", out)
