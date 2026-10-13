"""Which end took the surface a session's frames measured but the map lacks (CPU).

  python -m update_layer.eval.death_attribution STAGE CHECKPOINT.pt MAP.ply SESSION_START_NS OUT.json [--scale 10] [--radius 0.05]

The 'observed but missing' reference samples are those of eval/g1_decompose.py (G1 reference, crop, 2 cm area samples,
> 5 cm from MAP, measured on their surface by a 1 fps frame of the session). For each, the nearest Gaussian of the
checkpoint that has ENDED by the map's time (T1: death_state or death_evidence set) within --radius is looked up and its
end classified: 'state' (death_state: the object state it belongs to ended), 'g5' (death_evidence == the session's first
stamp: the session-end memory test), 'in_session' (death_evidence inside the session: the layer's element retirements,
element_rule or closed_background -- the checkpoint does not tell them apart), 'before' (ended before the session).
Samples with no ended Gaussian within the radius are 'no_ended_gaussian' (never reconstructed there, or pruned: pruned
Gaussians are not in the checkpoint). A living Gaussian closer than the ended one is reported too.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    a = sys.argv[1:]
    stage, ckpt, map_ply, start, out = a[0].lower(), a[1], a[2], int(a[3]), Path(a[4])
    scale = float(a[a.index("--scale") + 1]) if "--scale" in a else 10.0
    radius = float(a[a.index("--radius") + 1]) if "--radius" in a else 0.05
    import open3d as o3d
    from scipy.spatial import cKDTree
    from update_layer.eval import g1_decompose as D
    G, rotate = D.G, D.rotate
    S = stage.upper()
    FT = D.FT
    crop = json.loads((FT / f"eval/crops/real_{S}.json").read_text())
    lo, hi, yaw = np.array(crop["lo"]), np.array(crop["hi"]), crop["yaw_deg"]
    ref = o3d.io.read_triangle_mesh(str(FT / f"results/abc_eval_v2/geometry/{S}_reference_1cm.ply"))
    pred = o3d.io.read_triangle_mesh(map_ply)
    rp, _ = G.surface_sample(ref)
    rp = rp[~((rotate(rp, yaw) < lo) | (rotate(rp, yaw) > hi)).any(1)].astype(np.float64)
    miss = G.distances(rp, pred) > 0.05
    rm = rp[miss]
    chg = D.change_points(S)
    in_change = np.zeros(len(rm), bool)
    if len(chg):
        dc, _ = cKDTree(chg).query(rm, distance_upper_bound=0.10)
        in_change = np.isfinite(dc)
    seen = D.observed(rm, D.session_frames(S))
    target = rm[~in_change & seen]
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    b = ck["backend"]
    xyz = b["model"][1].detach().numpy() / scale
    big = np.iinfo(np.int64).max
    ds, de = b["death_state"].numpy(), b["death_evidence"].numpy()
    t_end = int(ck["prev_final"])
    ended = np.minimum(ds, de) <= t_end
    cause = np.full(len(xyz), "alive", dtype=object)
    cause[ended & (ds <= t_end)] = "state"
    ev = ended & (ds > t_end)
    cause[ev & (de == start)] = "g5"
    cause[ev & (de > start)] = "in_session"
    cause[ev & (de < start)] = "before"
    idx_e = np.flatnonzero(ended)
    d_e, j_e = cKDTree(xyz[idx_e]).query(target, distance_upper_bound=radius)
    d_a, _ = cKDTree(xyz[~ended]).query(target, distance_upper_bound=radius)
    c = Counter()
    for de_i, je_i, da_i in zip(d_e, j_e, d_a):
        if not np.isfinite(de_i):
            c["no_ended_gaussian" + ("" if np.isfinite(da_i) else " (none alive either)")] += 1
        else:
            k = cause[idx_e[je_i]]
            c[k + (" (a living one closer)" if np.isfinite(da_i) and da_i < de_i else "")] += 1
    n = len(rp)
    res = dict(stage=S, checkpoint=ckpt, map=map_ply, reference_samples=n, missing=int(miss.sum()),
               observed_missing=int(len(target)), observed_missing_share=round(len(target) / n, 4), radius_m=radius,
               ended_gaussians=int(ended.sum()), ended_by_cause=dict(Counter(cause[ended].tolist())),
               nearest_ended_cause={k: [v, round(v / max(len(target), 1), 4)] for k, v in c.most_common()})
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
