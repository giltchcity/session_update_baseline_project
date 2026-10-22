"""Per-object 4D surface F1 (synthetic; user 10-22: new experiment 1): for every round of a session's live map (its
timeline: the map as it stood at the round; with the 4D snapshots also the map of every time up to the round) and every
evaluation unit, the unit's map points (its identity) against the unit's true surface at that moment, at 5 cm, with the
geometry of Mesh F1 (evaluate_cross_session_geometry.py): exact USD meshes of the visible members at the round's frame,
recall over the causal reference cells seen by the round (first stamp <= round), projected onto the true surface.
  precision: the unit's points within 5 cm of its true surface now (points left at an old place, a smeared or
             stretched object count against it)
  recall:    the seen true surface within 5 cm of the unit's points
Every unit at every round, against the truth of that moment (no grace for a change not yet seen: both maps pay it):
  present and seen:            F1 of P and R
  present, its surface not seen yet: F1 = P (what the map shows of it must be where it is now; nothing shown: 1)
  taken away:                  map points of it left: 0; none: 1
Reported over all units and separately over the units that change (has_GT_change) and those that do not. Round k stands for (round k-1, round k] (the 4D map of a time
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
    units = [dict(name=u["unit"], ident=int(u["core_members"].split(",")[0]), changed=u["has_GT_change"] == "True",
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
            mine = pts.get(u["ident"], np.zeros((0, 3)))
            P = R = None
            q = np.zeros((0, 3))
            if vis:
                gt = o3d.t.geometry.RaycastingScene(nthreads=4)
                for i in vis:
                    v, f = worldmesh(s, i, frame)
                    if len(f):
                        gt.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(v), o3d.core.Tensor(f)))
                refs = []
                for i in vis:
                    r = refmeta.get(key(s, i, frame))
                    if r is not None:
                        pp, t = refload(r["file"])
                        refs.append(pp[t <= stamp])
                refs = np.concatenate(refs) if refs else np.zeros((0, 3))
                if len(refs):
                    q = gt.compute_closest_points(o3d.core.Tensor(refs.astype(np.float32)), nthreads=4)["points"].numpy()
                    q = q[np.linalg.norm(q - refs, axis=1) <= TH]          # the seen true surface (Mesh F1's rule)
                if len(mine):
                    P = float((gt.compute_distance(o3d.core.Tensor(mine.astype(np.float32)), nthreads=4).numpy() <= TH).mean())
                if len(q):
                    R = float((cKDTree(mine).query(q, distance_upper_bound=TH)[0] <= TH).mean()) if len(mine) else 0.0
            elif len(mine):
                P = 0.0                                    # taken away: every point of it left in the map is wrong
            if P is None and R is None:
                F = 1.0                                    # nothing there and nothing seen to cover: right
            elif R is None:
                F = P                                      # its present surface not seen yet: only what the map shows counts
            elif P is None:
                F = 0.0                                    # seen, not in the map
            else:
                F = 2 * P * R / (P + R) if P + R else 0.0
            rows.append(dict(session=s, round=k, stamp_ns=stamp, time_s=round((stamp - start) / 1e9, 2), duration_s=dur,
                             frame=frame, unit=u["name"], identity=u["ident"], changed=u["changed"], present=bool(vis),
                             map_points=len(mine), reference_cells=len(q), precision=P, recall=R, f1=F))
        print(f"round {k} t={(stamp - start) / 1e9:6.1f} s", flush=True)
    with open(out / f"OBJECT_SURFACE_{s}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    per = {}
    for r in rows:
        a = per.setdefault((r["unit"], r["changed"]), {"seconds": 0.0, "f1": 0.0, "precision": [0.0, 0.0], "recall": [0.0, 0.0]})
        a["seconds"] += r["duration_s"]; a["f1"] += r["duration_s"] * r["f1"]
        for k2 in ("precision", "recall"):
            if r[k2] is not None:
                a[k2][0] += r["duration_s"] * r[k2]; a[k2][1] += r["duration_s"]
    unit_rows = [dict(unit=n, changed=c, seconds=a["seconds"], f1=a["f1"] / a["seconds"],
                      precision=a["precision"][0] / a["precision"][1] if a["precision"][1] else None,
                      recall=a["recall"][0] / a["recall"][1] if a["recall"][1] else None) for (n, c), a in sorted(per.items())]
    with open(out / f"OBJECT_SURFACE_UNITS_{s}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(unit_rows[0])); w.writeheader(); w.writerows(unit_rows)
    mean = lambda rr, k2: float(np.mean([u[k2] for u in rr if u[k2] is not None])) if rr else None
    summary = dict(run=str(run), session=s, threshold_m=TH, units=len(unit_rows), unit_rounds=len(rows))
    for tag, rr in (("all", unit_rows), ("changed", [u for u in unit_rows if u["changed"]]),
                    ("unchanged", [u for u in unit_rows if not u["changed"]])):
        summary[tag] = {k2: mean(rr, k2) for k2 in ("f1", "precision", "recall")}
    (out / f"OBJECT_SURFACE_SUMMARY_{s}.json").write_text(json.dumps(summary, indent=1))
    f3 = lambda d: f"{100 * d['f1']:.2f} (P {100 * d['precision']:.2f}, R {100 * d['recall']:.2f})"
    print(f"{run.name} {s}: object surface F1 all {f3(summary['all'])}; changed units {f3(summary['changed'])}; "
          f"unchanged {f3(summary['unchanged'])}")


if __name__ == "__main__":
    main()
