"""Per-object 4D surface F1 (synthetic; user 10-22: new experiment 1): for every round of a session's live map (its
timeline: the map as it stood at the round; with the 4D snapshots also the map of every time up to the round) and every
evaluation unit, the unit's map points (its identity) against the unit's true surface at that moment, at 5 cm, with the
geometry of Mesh F1 (evaluate_cross_session_geometry.py): exact USD meshes of the visible members at the round's frame,
recall over the causal reference cells seen by the round (first stamp <= round), projected onto the true surface.
  precision: the unit's points within 5 cm of its true surface now (points left at an old place, a smeared or
             stretched object count against it)
  recall:    the seen true surface within 5 cm of the unit's points
A unit is scored at a round once its current pose has been seen (a reference cell of the pose with first stamp <= the
round); before that the map cannot know the change (the user's rule: a delay until the change is seen is no error).
A unit with no visible member (taken away) is not scored. Round k stands for (round k-1, round k] (the 4D map of a time
is the first round at or after it); a session's carried-in round covers no time of the session. Reported: the duration-
weighted mean over rounds of each unit's F1, P and R, averaged over units, and every (unit, round) row.
  usage: object_surface_f1.py RUN SESSION(a|b) OUT_DIR"""
from __future__ import annotations

import csv
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/mnt/d/3Study/ETH/FT/Synthetic_ABC/scripts")
from cvpr_eval_common import DATA, GT, IX, ORIGIN, REG, worldmesh  # noqa: E402
import open3d as o3d  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from update_layer.eval.scenelist import load_timeline  # noqa: E402

TH = 0.05
BANK = DATA / "evaluation_complete_20260917/causal_reference"
UNITS = DATA / "results/cvpr_surface_final_20260916/paper_aligned/EVAL_SCOPE_UNITS.csv"
refmeta = {r["key"]: r for r in json.loads((BANK / "manifest.json").read_text())}


def key(s, i, f):
    return str(i) + ":" + ",".join(map(str, np.round(GT[s]["T_world_object"][f, IX[s][i]].ravel(), 5)))


@lru_cache(maxsize=512)
def refload(file):
    a = np.load(BANK / file)
    return a["points"], a["first_stamp_ns"]


def main():
    run, s, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    units = [dict(name=u["unit"], ident=int(u["core_members"].split(",")[0]),
                  members=[int(m) for m in u["all_members"].split(",")]) for u in csv.DictReader(UNITS.open())]
    tl = load_timeline(run / f"session_{s}" / "timeline.pkl")
    stamps = [int(t) for t in tl._stamps]
    start, nf = ORIGIN[s], len(GT[s]["time_s"])
    rows = []
    for k, stamp in enumerate(stamps):
        lo = max(stamps[k - 1], start) if k else start
        dur = max(0, stamp - lo) / 1e9
        if dur == 0:
            continue                                       # a carried-in round: no time of this session
        frame = int(np.clip(round((stamp - start) / 1e9 * 30), 0, nf - 1))
        scene = tl.scenes[k]
        pts = {int(o.instance_id): np.asarray(o.points, np.float64) for o in scene.objects if len(o.points)}
        for u in units:
            vis = [i for i in u["members"] if i in IX[s] and GT[s]["render_visible"][frame, IX[s][i]] and i in REG]
            if not vis:
                continue                                   # taken away: not scored
            refs = []
            for i in vis:
                r = refmeta.get(key(s, i, frame))
                if r is not None:
                    p, t = refload(r["file"])
                    refs.append(p[t <= stamp])
            refs = np.concatenate(refs) if refs else np.zeros((0, 3))
            if not len(refs):
                continue                                   # its current pose not seen yet
            gt = o3d.t.geometry.RaycastingScene(nthreads=4)
            for i in vis:
                v, f = worldmesh(s, i, frame)
                if len(f):
                    gt.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(v), o3d.core.Tensor(f)))
            q = gt.compute_closest_points(o3d.core.Tensor(refs.astype(np.float32)), nthreads=4)["points"].numpy()
            q = q[np.linalg.norm(q - refs, axis=1) <= TH]  # the seen true surface (Mesh F1's reference rule)
            if not len(q):
                continue
            mine = pts.get(u["ident"], np.zeros((0, 3)))
            if len(mine):
                d = gt.compute_distance(o3d.core.Tensor(mine.astype(np.float32)), nthreads=4).numpy()
                P = float((d <= TH).mean())
                R = float((cKDTree(mine).query(q, distance_upper_bound=TH)[0] <= TH).mean())
            else:
                P = R = 0.0
            F = 2 * P * R / (P + R) if P + R else 0.0
            rows.append(dict(session=s, round=k, stamp_ns=stamp, time_s=round((stamp - start) / 1e9, 2), duration_s=dur,
                             frame=frame, unit=u["name"], identity=u["ident"], map_points=len(mine),
                             reference_cells=len(q), precision=P, recall=R, f1=F))
        print(f"round {k} t={(stamp - start) / 1e9:6.1f} s: {sum(r['round'] == k for r in rows)} units scored", flush=True)
    with open(out / f"OBJECT_SURFACE_{s}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    per = {}
    for r in rows:
        a = per.setdefault(r["unit"], np.zeros(4))
        a += [r["duration_s"], r["duration_s"] * r["f1"], r["duration_s"] * r["precision"], r["duration_s"] * r["recall"]]
    unit_rows = [dict(unit=n, seconds=a[0], f1=a[1] / a[0], precision=a[2] / a[0], recall=a[3] / a[0]) for n, a in sorted(per.items())]
    with open(out / f"OBJECT_SURFACE_UNITS_{s}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(unit_rows[0])); w.writeheader(); w.writerows(unit_rows)
    m = {k2: float(np.mean([u[k2] for u in unit_rows])) for k2 in ("f1", "precision", "recall")}
    summary = dict(run=str(run), session=s, threshold_m=TH, units_scored=len(unit_rows), unit_rounds=len(rows), **m)
    (out / f"OBJECT_SURFACE_SUMMARY_{s}.json").write_text(json.dumps(summary, indent=1))
    print(f"{run.name} {s}: object surface F1 {100 * m['f1']:.2f} (P {100 * m['precision']:.2f}, R {100 * m['recall']:.2f}) "
          f"over {len(unit_rows)} units, {len(rows)} unit-rounds")


if __name__ == "__main__":
    main()
