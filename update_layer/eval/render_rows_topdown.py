"""Top-down figure of one session's final maps, rows side by side (e.g. GaME rows 1 / 3 / 4), from exported PLYs.

  python -m update_layer.eval.render_rows_topdown real STAGE OUT.png [--probes GHOST_DIR] [--reference REF.ply] \
      [--box H0,H1,V0,V1] [--offref] [--counts OUT.json] "LABEL=MAP.ply[:G1.json[:GHOST.json]]" ...

--box: a close-up of that rectangle of the top-down plane (real: x and z, metres). --offref: red = map samples farther
than 5 cm from the reference surface, without old-site probes (e.g. where people walked: surface the present scene does
not explain). --counts: per panel the red and drawn sample counts of the full map (before the drawing subsample).

MAP.ply: a mesh (the median-depth TSDF export, tsdf_final.ply) or a point/surfel PLY (eval_real/final_<s>.ply). Every
panel is coloured the same way (render_topdown.colourize): each sample in the colour of the latest frame (1 fps, current
session first, then earlier ones) that measured it on its surface; never measured -> grey.
Old sites (--probes: an eval_real/ghost/<stage> directory; the probes and their evidence come from the session's
frames, the same for every row): the old-state surface probes of every D2/D3 event of the session that the cameras saw
empty (strictfree >= 3 and support < 3, as cleanup_evaluate_h.py) are drawn as small black dots. Map samples within
5 cm of them and farther than 5 cm from the session's reference surface (--reference: the G1 reference of the session,
the surface that is there now) are drawn red: map surface on a seen-empty old site that the present scene does not
explain (ghost; an illustration -- the official number is GHOST.json's, given in the title when passed). Titles also
give G1 P/R/F1 @5 cm when its json is passed. Only samples below 2 m above the floor are drawn (no ceiling).
Real world frame = session A camera (y down).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the project root

NEAR = 0.05
AXES = {"real": ((0, 2), 1, -1.0), "synthetic": ((0, 1), 2, 1.0)}


def load_ply(path: str):
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(path)
    if len(m.vertices):
        return np.asarray(m.vertices)
    return np.asarray(o3d.io.read_point_cloud(path).points)


def old_site_probes(ghost_dir: Path, stage: str) -> np.ndarray:
    ev = np.load(ghost_dir / f"evidence_final_{stage}.npz")
    known = (ev["strictfree"] >= 3) & (ev["support"] < 3)
    return ev["points"][known].astype(np.float64)


def main():
    from scipy.spatial import cKDTree
    from update_layer.eval.render_topdown import colourize, frames

    args = sys.argv[1:]
    probes_dir = ref_path = box = counts_path = None
    offref = "--offref" in args
    if offref:
        args.remove("--offref")
    for flag in ("--probes", "--reference", "--box", "--counts"):
        if flag in args:
            k = args.index(flag)
            v = args[k + 1]
            if flag == "--probes":
                probes_dir = Path(v)
            elif flag == "--reference":
                ref_path = v
            elif flag == "--box":
                box = [float(x) for x in v.split(",")]
            else:
                counts_path = Path(v)
            args = args[:k] + args[k + 2:]
    dataset, stage, out = args[0], args[1], Path(args[2])
    panels = []
    for spec in args[3:]:
        label, rest = spec.split("=", 1)
        parts = rest.split(":")
        panels.append((label, parts[0], parts[1] if len(parts) > 1 and parts[1] else None,
                       parts[2] if len(parts) > 2 and parts[2] else None))
    (ha, hb), va, up = AXES[dataset]
    probes = old_site_probes(probes_dir, stage) if probes_dir else np.zeros((0, 3))
    ptree = cKDTree(probes) if len(probes) else None
    rtree = cKDTree(load_ply(ref_path)) if ref_path else None
    maps = [load_ply(p) for _, p, _, _ in panels]
    allp = np.concatenate(maps)
    floor = np.percentile(up * allp[:, va], 1)
    cut = floor + 2.0
    lo = np.percentile(allp[:, [ha, hb]], 0.5, axis=0) - 0.3
    hi = np.percentile(allp[:, [ha, hb]], 99.5, axis=0) + 0.3
    if box:
        lo, hi = np.array([box[0], box[2]]), np.array([box[1], box[3]])
        inb = lambda Q: ((Q[:, ha] >= lo[0]) & (Q[:, ha] <= hi[0]) & (Q[:, hb] >= lo[1]) & (Q[:, hb] <= hi[1]))
        maps = [M[inb(M)] for M in maps]
        if len(probes):
            probes = probes[inb(probes)]
            ptree = cKDTree(probes) if len(probes) else None
    counts = {}
    w, h = hi[0] - lo[0], hi[1] - lo[1]
    fig, axes = plt.subplots(1, len(panels), squeeze=False,
                             figsize=(5.0 * len(panels) * w / max(w, h) + 0.6, 5.0 * h / max(w, h) + 1.3))
    rng = np.random.default_rng(0)
    for i, ((label, path, g1, ghost), P) in enumerate(zip(panels, maps)):
        ax = axes[0][i]
        ax.set_xticks([]); ax.set_yticks([])
        hgt = up * P[:, va]
        keep = hgt < cut
        P, hgt = P[keep], hgt[keep]

        def red_of(Q):
            r = np.zeros(len(Q), bool)
            if offref and rtree is not None:
                dr, _ = rtree.query(Q, k=1, distance_upper_bound=NEAR)
                return ~np.isfinite(dr)
            if ptree is not None:
                d, _ = ptree.query(Q, k=1, distance_upper_bound=NEAR)
                r = np.isfinite(d)
                if rtree is not None and r.any():
                    dr, _ = rtree.query(Q[r], k=1, distance_upper_bound=NEAR)
                    r[np.nonzero(r)[0][np.isfinite(dr)]] = False
            return r
        counts[label] = {"red": int(red_of(P).sum()), "samples": int(len(P))}
        if len(P) > 1_500_000:
            s = rng.choice(len(P), 1_500_000, replace=False)
            P, hgt = P[s], hgt[s]
        C = colourize(P.astype(np.float32), frames(dataset, stage), NEAR)
        red = red_of(P)
        C = C.copy()
        C[red] = [1.0, 0.1, 0.1]
        order = np.argsort(hgt + 10.0 * red)                      # higher drawn on top, ghost samples last
        ax.scatter(P[order, ha], P[order, hb], c=C[order], s=0.6, linewidths=0, rasterized=True)
        if len(probes):
            pv = probes[up * probes[:, va] < cut]
            ax.scatter(pv[:, ha], pv[:, hb], c="k", s=0.02, linewidths=0, rasterized=True, alpha=0.25)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
        title = [label]
        if g1 and Path(g1).exists():
            d = json.loads(Path(g1).read_text())
            title.append(f"G1@5 P/R/F1 {100 * d['P_05cm']:.2f}/{100 * d['R_05cm']:.2f}/{100 * d['F1_05cm']:.2f}")
        if ghost and Path(ghost).exists():
            g = json.loads(Path(ghost).read_text())
            pct = g.get("old_site_full_map_ghost_pct")
            title.append(f"ghost {pct}% (official)" if pct is not None else "ghost n/a (no inherited old sites)")
        if offref:
            title.append(f"{counts[label]['red']} samples off the present surface (red) of {counts[label]['samples']}")
        elif len(probes):
            title.append(f"{counts[label]['red']} samples on seen-empty old sites, off the present surface (red)")
        ax.set_title("\n".join(title), fontsize=8)
    import textwrap
    head = (f"{dataset} session {stage.upper()}: final maps, top-down (below 2 m), coloured by the latest frame that "
            f"measured each sample; red = map surface on old sites the session saw empty that the present scene "
            f"does not explain (ghost); black dots = those old sites")
    fig.suptitle("\n".join(textwrap.wrap(head, 60 * len(panels))), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    if counts_path:
        counts_path.parent.mkdir(parents=True, exist_ok=True)
        counts_path.write_text(json.dumps({"stage": stage, "box": box, "offref": offref, "panels": counts}, indent=1))
    print("wrote", out, json.dumps(counts))


if __name__ == "__main__":
    main()
