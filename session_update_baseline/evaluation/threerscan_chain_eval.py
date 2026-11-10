#!/usr/bin/env python3
"""3RScan multi-revisit chains (P3): a room's visits in order (datasets/3rscan/mapping_configs/ROOM/chain.json), scored
with threerscan_eval.py's events and criterion (>= 50 % of a GT state within 10 cm of the predicted surface) once per
rescan, against the visit before it, then summed over the chain; and each visit's geometry, as threerscan_geom.py.
Step k (visit k-1 -> visit k): GT states = the two visits' annotated meshes in the reference frame. Changes from
3RScan.json, which lists each rescan against the reference: moved = listed rigid in either visit (a listed object whose
visit k-1 state is still represented by its visit k state is static, as threerscan_eval does for near-identity
entries; one removed now is removed); removed = removed in visit k, not yet in visit k-1; non-rigid = listed in exactly one of the two (listed in
both: not evaluated). Ids in the reference's ambiguity list are not evaluated.
Maps, per method: ours = METHOD_DIR/visit_K/final.4dmap.zpk (all its snapshots; a state our output check refused:
visit_K.rejected/state, marked rejected); khronos = METHOD_DIR/state/
final.4dmap.zpk, the one continuous map, its snapshots in visit K's window (visit K starts at 1 + 200 K s) and the
next visit's first snapshot (its backend takes a visit's objects in only then, about 2 s into the next visit, less than
the 3 s an object needs to leave the active window, so that snapshot holds no object of the next visit), matched by
semantic class like the main table's Khronos rows; panoptic = METHOD_DIR/visit_K/session_vK.panmap (its saved end
state, panoptic_export.py), ids from the submap names.
Geometry: the visit's end map (ours, Khronos: geometry_export viewer of the visit's last snapshot; Panoptic:
current_confirmed.ply, the submaps its paper evaluates: New, Matched, Persistent) against the visit's own mesh.refined.v2.obj in the reference frame; P/R/F1 at 5/10/20 cm.
A chain broken by a visit that was never run (ours: no state, accepted or refused, after a refused one): strict = every
expected event of the steps not run is FN (statics count nothing); completed = only the steps both of whose visits ran.
usage: threerscan_chain_eval.py ROOM ours|khronos|panoptic METHOD_DIR   (writes METHOD_DIR/CHAIN_EVAL.json)
"""
import json, os, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

FT = Path('/home/jixian/Desktop/FT'); D3 = FT / 'datasets/3rscan'
sys.path.insert(0, str(FT / 'eval')); sys.path.insert(0, str(FT / 'eval_fast/vendor/pack')); sys.path.insert(0, str(FT / 'tools/abc_eval_v2'))
import threerscan_eval as te  # noqa: E402
import zpk_snapshot as zs  # noqa: E402
VENDOR = FT / 'eval_fast/vendor'
CANONICAL = ['bash', '-c', 'source "$1" >/dev/null 2>&1 || exit 90; shift; exec "$@"', 'chain-eval', str(FT / 'eval_fast/runtime_khronos_env.sh')]
VISIT_NS = lambda k: 1000000000 + 200000000000 * k


def mesh_in_reference(cfg, visits, k, name):
    """A visit's annotated (labels...ply) or reconstructed mesh vertices, in the reference frame."""
    scan = visits[k]
    T = np.eye(4) if k == 0 else np.loadtxt(cfg / f'world_transform_{scan}_to_{visits[0]}.txt')
    if name == 'labels':
        v, f, o = te.read_ply(D3 / 'scans' / scan / 'labels.instances.annotated.v2.ply')
        return (T[:3, :3] @ v.T).T + T[:3, 3], o
    m = o3d.io.read_triangle_mesh(str(D3 / 'scans' / scan / 'mesh.refined.v2.obj')); m.transform(T)
    return m


def zpk_snapshots(path, lo, hi, work, next_one=False):
    """export_timeline records of the map's snapshots with lo <= stamp < hi (next_one: and the first one after); the
    last one cut alone for geometry."""
    stamps = zs.stamps(str(path)); idx = [i for i, s in enumerate(stamps) if lo <= s < hi]
    if next_one: idx += [i for i, s in enumerate(stamps) if s >= hi][:1]
    if not idx: raise SystemExit(f'no snapshots of {path} in [{lo}, {hi})')
    sliced = Path(work) / 'visit.4dmap'; zs.cut_many(str(path), idx, out=str(sliced), work=work)
    subprocess.run([*CANONICAL, str(VENDOR / 'timeline/export_timeline'), str(sliced), f'{work}/timeline'], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    snaps = [json.loads(l) for l in open(f'{work}/timeline/snapshots.jsonl')]
    last = Path(work) / 'last.4dmap'; zs.cut_many(str(path), [idx[-1]], out=str(last), work=work)
    subprocess.run([*CANONICAL, str(VENDOR / 'geometry/geometry_export'), str(last), f'{work}/end.ply', 'viewer'], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return snaps, f'{work}/end.ply'


def panoptic_snapshots(panmap, work):
    from panoptic_export import export
    export(Path(panmap), Path(work) / 'export', False)
    subprocess.run([sys.executable, str(FT / 'eval/panoptic_to_snapshots.py'), f'{work}/export', f'{work}/snap'], check=True,
                   stdout=subprocess.DEVNULL)
    return [json.loads(l) for l in open(f'{work}/snap/snapshots.jsonl')], f'{work}/export/current_confirmed.ply'


def fscore(gt_mesh, pred_ply):
    pm = o3d.io.read_triangle_mesh(pred_ply)
    g = np.asarray(gt_mesh.sample_points_uniformly(200000).points)
    p = np.asarray(pm.sample_points_uniformly(200000).points) if len(pm.triangles) else np.asarray(pm.vertices)
    lo, hi = g.min(0) - 0.2, g.max(0) + 0.2; p = p[np.all((p >= lo) & (p <= hi), 1)]   # as threerscan_geom.py
    dp = cKDTree(g).query(p)[0]; dg = cKDTree(p).query(g)[0]; out = {}
    for t in (0.05, 0.10, 0.20):
        P = (dp <= t).mean(); R = (dg <= t).mean(); out['%dcm' % round(100 * t)] = [round(100 * P, 1), round(100 * R, 1), round(100 * 2 * P * R / max(1e-9, P + R), 1)]
    return out


def main(room, method, mdir):
    cfg = D3 / 'mapping_configs' / room; chain = json.load((cfg / 'chain.json').open()); visits = chain['visits']
    excluded = set(chain.get('ambiguous_ids', []))
    cat = json.load((cfg / 'physical_catalog.json').open())
    things = {e['physical_instance_id']: e['name'].split(' (')[0] for e in cat['entities'] if e['semantic_id'] != 1}
    te.SEM_OF = {e['physical_instance_id']: e['semantic_id'] for e in cat['entities'] if e['semantic_id'] != 1}
    te.MATCH = 'semantic' if method == 'khronos' else 'id'
    changes = {s: json.load((cfg / f'changes_gt_{s}.json').open()) for s in visits[1:]}
    rigid = lambda k: {x['instance_reference'] for x in changes[visits[k]]['rigid']} if k else set()
    removed = lambda k: set(changes[visits[k]]['removed']) if k else set()
    nonrigid = lambda k: set(changes[visits[k]]['nonrigid']) if k else set()
    gt, vis, geometry, steps = {}, {}, [], []
    for k in range(len(visits)):
        v, o = mesh_in_reference(cfg, visits, k, 'labels')
        gt[k] = {i: v[o == i] for i in things if (o == i).sum() >= 20}; vis[k] = te.visible(visits[k])
    ran = lambda k: method != 'ours' or (mdir / f'visit_{k}/final.4dmap.zpk').exists() or (mdir / f'visit_{k}.rejected/state/final.4dmap.zpk').exists()
    for k in range(len(visits)):
        if not ran(k):
            geometry.append(dict(visit=k, scan=visits[k], run=False))
            if k == 0: continue
            both = nonrigid(k) & nonrigid(k - 1); gone = removed(k) - removed(k - 1)
            rows, _, _, _ = te.score(things, gt[k - 1], gt[k], (rigid(k) | rigid(k - 1)) - both - gone, nonrigid(k) ^ nonrigid(k - 1),
                                     gone, vis[k - 1], vis[k], [{}], [0.0], excluded | both)
            fn = sum(r[7].count(':') for r in rows if r[7] and ('disappeared:' in r[7] or 'appeared:' in r[7]))
            steps.append(dict(step=k, before=visits[k - 1], after=visits[k], run=False, TP=0, FP=0, FN=fn,
                              rows=[r[:7] + ('not run: every expected event FN' if r[7] and ('disappeared:' in r[7] or 'appeared:' in r[7]) else r[7],) for r in rows]))
            print(f'{room} {method} visit {k - 1} -> {k}: NOT RUN (strict: {fn} expected events FN)', flush=True)
            continue
        with tempfile.TemporaryDirectory(prefix='chain-eval-', dir='/dev/shm') as work:
            rejected = method == 'ours' and not (mdir / f'visit_{k}/final.4dmap.zpk').exists()
            if method == 'ours':
                snaps, end = zpk_snapshots(mdir / (f'visit_{k}.rejected/state' if rejected else f'visit_{k}') / 'final.4dmap.zpk', 0, 2**64, work)
            elif method == 'khronos':
                snaps, end = zpk_snapshots(mdir / 'state/final.4dmap.zpk', VISIT_NS(k), VISIT_NS(k + 1), work, next_one=True)
            else:
                snaps, end = panoptic_snapshots(mdir / f'visit_{k}/session_v{k}.panmap', work)
            geometry.append(dict(visit=k, scan=visits[k], rejected=rejected, **fscore(mesh_in_reference(cfg, visits, k, 'mesh'), end)))
            if k == 0: continue
            base = snaps[0]['stamp_ns']; times = [(s['stamp_ns'] - base) / 1e9 for s in snaps]; S = [te.surfaces(s) for s in snaps]
            both = nonrigid(k) & nonrigid(k - 1); gone = removed(k) - removed(k - 1)   # moved before, removed now: removed
            rows, tp, fp, fn = te.score(things, gt[k - 1], gt[k], (rigid(k) | rigid(k - 1)) - both - gone, nonrigid(k) ^ nonrigid(k - 1),
                                        gone, vis[k - 1], vis[k], S, times, excluded | both)
            steps.append(dict(step=k, before=visits[k - 1], after=visits[k], run=True, rejected=rejected, snapshots=len(S), TP=tp, FP=fp, FN=fn, rows=rows))
            print(f'{room} {method} visit {k - 1} -> {k}: TP {tp} FP {fp} FN {fn}' + (' (state rejected by our output check)' if rejected else ''), flush=True)
            for r in rows:
                if r[2] != 'static' or r[7] != 'kept': print('   %3d %-17s %-11s %s' % (r[0], r[1][:17], r[2], r[7]))
    def total(chosen):
        TP, FP, FN = (sum(s[x] for s in chosen) for x in ('TP', 'FP', 'FN'))
        P = TP / max(1, TP + FP); R = TP / max(1, TP + FN); F = 2 * P * R / max(1e-9, P + R)
        return dict(TP=TP, FP=FP, FN=FN, P=round(100 * P, 1), R=round(100 * R, 1), F1=round(100 * F, 1), n_steps=len(chosen))
    strict, completed = total(steps), total([s for s in steps if s['run'] and ran(s['step'] - 1)])
    summary = dict(room=room, method=method, visits=visits, excluded_ids=sorted(excluded), match=te.MATCH, **strict,
                   completed=completed, broken=any(not s['run'] for s in steps), steps=steps, geometry=geometry)
    (mdir / 'CHAIN_EVAL.json').write_text(json.dumps(summary, indent=1, default=str) + '\n')
    print(f'{room} {method} chain: TP {strict["TP"]} FP {strict["FP"]} FN {strict["FN"]}  P/R/F1 {strict["P"]}/{strict["R"]}/{strict["F1"]}'
          + (f'  (completed steps only: TP {completed["TP"]} FP {completed["FP"]} FN {completed["FN"]}, F1 {completed["F1"]})' if summary['broken'] else ''))
    for g in geometry: print('   visit %d geometry @5cm P/R/F1 %s' % (g['visit'], g.get('5cm', 'not run')))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], Path(sys.argv[3]).resolve())
