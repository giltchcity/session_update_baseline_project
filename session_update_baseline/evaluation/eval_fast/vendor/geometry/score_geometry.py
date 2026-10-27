"""Observed-domain geometry score for one exported mesh against the 1 cm pseudo-GT.

Uses the canonical evaluator's sampling, observed domain and distances
(tools/abc_eval_v2/geometry_score.py) and substitutes the 1 cm reference, which
is what every real-data comparison since 2026-09-19 is built on. The published
Table 1 used a 2 cm reference that is no longer on disk; numbers here are not
comparable to it, only to each other.

usage: score_geometry.py SESSION NAME MESH.ply [--json OUT.json]
"""
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geometry_score as G

CACHE = Path(__file__).resolve().parent / 'ref_cache'
CACHE.mkdir(exist_ok=True)


def reference(session):
    mesh = o3d.io.read_triangle_mesh(str(G.OUT / (session + '_reference_1cm.ply')))
    cache = CACHE / (session + '_ref.npz')
    if cache.exists():
        d = np.load(cache)
        return mesh, d['points'], d['eligible']
    pts, _ = G.surface_sample(mesh)
    elig, _, _ = G.observed_domain(pts, np.load(G.OUT / (session + '_domain.npz')))
    np.savez_compressed(cache, points=pts, eligible=elig)
    return mesh, pts, elig


def main():
    session, name, mesh_path = sys.argv[1], sys.argv[2], sys.argv[3]
    out = sys.argv[sys.argv.index('--json') + 1] if '--json' in sys.argv else None
    ref, rp, re = reference(session)
    pred = o3d.io.read_triangle_mesh(mesh_path)
    pp, meta = G.surface_sample(pred)
    pe, _, _ = G.observed_domain(pp, np.load(G.OUT / (session + '_domain.npz')))
    dp, dr = G.distances(pp, ref), G.distances(rp, pred)
    s = G.summaries(dp[pe >= 3], dr[re >= 3])
    row = dict(session=session, name=name, area_m2=round(meta['area_m2'], 1))
    for t in ('05', '10', '20'):
        row.update({'P_%scm' % t: round(s['P_%scm' % t], 4), 'R_%scm' % t: round(s['R_%scm' % t], 4),
                    'F1_%scm' % t: round(s['F1_%scm' % t], 4)})
    print(json.dumps(row), flush=True)
    if out:
        Path(out).write_text(json.dumps(row, indent=2))


if __name__ == '__main__':
    main()
