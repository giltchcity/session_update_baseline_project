#!/usr/bin/env python3
"""3RScan cross-session (D3) evaluation of a reference->rescan run. GT = annotated meshes (objectId per vertex) of the
reference (session A states) and the rescan (session B states, transformed rescan->reference), change list from
3RScan.json. Prediction = per-object surfaces of the B map over time (export_timeline snapshots, physical ids).
State representedness follows protocol_v1: >= 50 % of the GT state's vertices within 0.10 m of the predicted surface of
the SAME physical id (3RScan ids are shared by reference and rescan, so identity is exact, no semantic matching).
Events: moved -> old state must end (not represented at the end) and the new state must be represented;
removed -> old state ends; new -> new state represented; static -> the (single) state stays represented.
usage: threerscan_eval.py SCENE RUN_DIR [id|semantic]   (RUN_DIR = runs/3rscan_<scene>_<tag>; 'semantic' = match by ADE class like
 protocol_v1, for maps without physical ids such as official Khronos; also B_MAP env to point at a non-standard map path)"""
import json, os, subprocess, sys
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
sys.path.insert(0, '/home/jixian/Desktop/FT/tools/dataset_adapters'); from threerscan_to_flat import read_ply
ROOT = Path('/home/jixian/Desktop/FT'); D3 = ROOT / 'datasets/3rscan'
EXPORT = ROOT / 'tools/abc_eval_final_baseline/export_timeline'; SUB = ROOT / 'session_update_baseline_project/session_update_baseline'
GATE, FRAC = 0.10, 0.5

def export(map_path, out):
    if not (out / 'snapshots.jsonl').exists():
        out.mkdir(parents=True, exist_ok=True)
        cmd = f'unset SESSION_UPDATE_CANONICAL_ROOT SESSION_UPDATE_CANONICAL_PREFIX SESSION_UPDATE_CANONICAL_BUILD BASE1_CANONICAL_PREFIX BASE1_KHRONOS_SETUP BASE1_BUILD_DIR; source {SUB}/scripts/khronos_env.sh >/dev/null 2>&1; {EXPORT} {map_path} {out}'
        subprocess.run(['bash', '-c', cmd], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    snaps = [json.loads(l) for l in (out / 'snapshots.jsonl').open()]
    return snaps

MATCH = 'id'
SEM_OF = {}
def surfaces(r):
    d = {}
    for p in r['objects']:
        if p.get('surface') and p.get('present'):
            if MATCH == 'semantic':
                key = ('sem', int(p.get('semantic', -1)))
            else:
                iid = p['details'].get('instance_id'); iid = (iid[0] if iid else None) if isinstance(iid, list) else iid; key = int(iid) if iid else None
            if key is not None: d.setdefault(key, []).append(np.asarray(p['surface'], float))
    out = {k: np.vstack(v) for k, v in d.items()}
    if MATCH == 'semantic':  # expose the union surface of each class under every object id of that class
        return {i: out.get(('sem', SEM_OF[i])) for i in SEM_OF if ('sem', SEM_OF[i]) in out}
    return out

def represented(gt_pts, pred):
    if pred is None or len(pred) == 0: return 0.0
    return float((cKDTree(pred).query(gt_pts)[0] <= GATE).mean())

# an object is evaluable in a session only if that session's frames actually show it (mask >= MIN_PX in >= MIN_FRAMES frames)
MIN_PX, MIN_FRAMES = 1778, 20   # >= 2 s of frames at 10 Hz: below that the object extractor never gets a stable segment (scene3: 11-19 frames -> none extracted, >= 21 -> all extracted)
def visible(sid):
    import glob
    from PIL import Image
    cnt = {}
    for f in sorted(glob.glob(f'{D3}/sessions/{sid}/instance_labels/*_segmentation.png')):
        m = np.array(Image.open(f)); ids, n = np.unique(m, return_counts=True)
        for i, c in zip(ids, n):
            if i and c >= MIN_PX: cnt[int(i)] = cnt.get(int(i), 0) + 1
    return {i for i, c in cnt.items() if c >= MIN_FRAMES}

def score(things, gtA, gtB, moved, nonrigid, removed, visA, visB, S, times, excluded=()):
    """Per-object verdicts of a map's snapshots S (times in s) between state A (gtA, visA) and state B (gtB, visB);
    excluded ids are listed, not evaluated."""
    rows = []; tp = fp = fn = 0
    for i in sorted(set(gtA) | set(gtB)):
        if i in excluded: rows.append((i, things[i], 'ambiguous', None, None, None, None, 'ambiguous annotation (3RScan ambiguity list): not evaluated')); continue
        kind = 'moved' if i in moved else 'nonrigid' if i in nonrigid else 'removed' if i in removed else 'new' if i not in gtA else 'unannotated' if i not in gtB else 'static'
        if kind == 'moved' and i in gtA and i in gtB and represented(gtA[i], gtB[i]) >= FRAC:
            kind = 'static'  # listed rigid, but the reference pose is still represented by the rescan pose (3RScan's rigid
                             # list also holds near-identity alignments, e.g. scene1 armchair 13: 2.5 cm) -> same state
        if kind == 'unannotated': rows.append((i, things[i], kind, None, None, None, None, 'not in rescan annotation, not listed as removed: not evaluated')); continue
        need = {'static': [visA | visB], 'moved': [visA, visB], 'nonrigid': [visA, visB], 'removed': [visA], 'new': [visB]}[kind]  # a removed object has no B mask by definition
        if not all(i in v for v in need) or (kind == 'static' and i not in (visA | visB)):
            rows.append((i, things[i], kind, None, None, None, None, 'not visible enough in the frames (A %s, B %s): not evaluated' % (i in visA, i in visB))); continue
        if kind == 'static' and i not in visA: kind = 'new'   # never seen in the reference frames: for the map it is new in B
        ga, gb = gtA.get(i), gtB.get(i)
        if kind == 'moved':
            # a move smaller than the object leaves most of the surface within the 10 cm gate of both poses; judge each
            # state on its discriminative part (old points > gate from the new pose, new points > gate from the old pose)
            da = cKDTree(gb).query(ga)[0]; db = cKDTree(ga).query(gb)[0]
            if (da > GATE).sum() >= 20: ga = ga[da > GATE]
            if (db > GATE).sum() >= 20: gb = gb[db > GATE]
        old = [represented(ga, s.get(i)) if ga is not None else None for s in S]; new = [represented(gb, s.get(i)) if gb is not None else None for s in S]
        old_end = old[-1] if old[-1] is not None else None; new_end = new[-1] if new[-1] is not None else None
        t_old_gone = next((t for t, o in zip(times, old) if o is not None and o < FRAC), None); t_new = next((t for t, n in zip(times, new) if n is not None and n >= FRAC), None)
        exp_dis = kind in ('moved', 'removed'); exp_app = kind in ('moved', 'new', 'nonrigid')  # non-rigid: the new shape must be represented
        got_dis = old_end is not None and old_end < FRAC and (kind != 'static'); got_app = new_end is not None and new_end >= FRAC and (kind != 'static')
        if kind == 'static':
            ok = new_end is not None and new_end >= FRAC
            if not ok: fp += 1  # a static state reported gone = false disappearance
            verdict = 'kept' if ok else 'FALSE-DISAPPEARED'
        else:
            v = []
            if exp_dis: tp += got_dis; fn += (not got_dis); v.append('disappeared:' + ('TP' if got_dis else 'FN'))
            if exp_app: tp += got_app; fn += (not got_app); v.append('appeared:' + ('TP' if got_app else 'FN'))
            verdict = ' '.join(v)
        rows.append((i, things[i], kind, old_end, new_end, t_old_gone, t_new, verdict))
    return rows, tp, fp, fn

def main(scene, run):
    cfg = D3 / 'mapping_configs' / scene; ch = json.load((cfg / 'changes_gt.json').open()); ref, res = ch['reference'], ch['rescan']
    T = np.loadtxt(cfg / f'world_transform_{res}_to_{ref}.txt'); cat = json.load((cfg / 'physical_catalog.json').open())
    things = {e['physical_instance_id']: e['name'].split(' (')[0] for e in cat['entities'] if e['semantic_id'] != 1}  # STATIC_OTHER classes cannot be objects
    global SEM_OF; SEM_OF = {e['physical_instance_id']: e['semantic_id'] for e in cat['entities'] if e['semantic_id'] != 1}
    vA, _, oA = read_ply(D3 / 'scans' / ref / 'labels.instances.annotated.v2.ply'); vB, _, oB = read_ply(D3 / 'scans' / res / 'labels.instances.annotated.v2.ply')
    vB = (T[:3, :3] @ vB.T).T + T[:3, 3]
    gtA = {i: vA[oA == i] for i in things if (oA == i).sum() >= 20}; gtB = {i: vB[oB == i] for i in things if (oB == i).sum() >= 20}
    moved = {x['instance_reference'] for x in ch['rigid']}; nonrigid = set(ch['nonrigid']); removed = set(ch['removed'])
    visA, visB = visible(ref), visible(res)
    run = Path(run); bmap = Path(os.environ.get('B_MAP', run / 'session_b/final.4dmap')); snaps = export(bmap, run / ('export_b_' + MATCH)); base = snaps[0]['stamp_ns']
    times = [(s['stamp_ns'] - base) / 1e9 for s in snaps]; S = [surfaces(s) for s in snaps]
    # a map built from scratch in the rescan's own frame (official Khronos, no --world-transform) is brought to the
    # reference frame with the same rescan->reference transform the GT uses
    if os.environ.get('PRED_IN_RESCAN_FRAME') == '1':
        S = [{k: (T[:3, :3] @ v.T).T + T[:3, 3] for k, v in s.items() if v is not None} for s in S]; print('(predictions transformed rescan -> reference)')
    rows, tp, fp, fn = score(things, gtA, gtB, moved, nonrigid, removed, visA, visB, S, times)
    print('%s  B run %s  (%d snapshots, %.1f s)' % (scene, run.name, len(S), times[-1]))
    print(' id  name              kind     old@end new@end  t_old_gone  t_new   verdict'); 
    for r in rows: print(' %3d %-17s %-8s %7s %7s %10s %7s   %s' % (r[0], r[1][:17], r[2], '-' if r[3] is None else '%.2f' % r[3], '-' if r[4] is None else '%.2f' % r[4], '-' if r[5] is None else '%.1f' % r[5], '-' if r[6] is None else '%.1f' % r[6], r[7]))
    P = tp / max(1, tp + fp); R = tp / max(1, tp + fn); F = 2 * P * R / max(1e-9, P + R)
    print('D3 events: TP %d FP %d FN %d  P/R/F1 = %.1f / %.1f / %.1f' % (tp, fp, fn, 100 * P, 100 * R, 100 * F))
    json.dump({'rows': rows, 'TP': tp, 'FP': fp, 'FN': fn, 'P': P, 'R': R, 'F1': F, 'match': MATCH}, (run / ('D3_EVAL_%s.json' % MATCH)).open('w'), indent=1)

if __name__ == '__main__':
    if len(sys.argv) > 3: MATCH = sys.argv[3]
    main(sys.argv[1], sys.argv[2])
