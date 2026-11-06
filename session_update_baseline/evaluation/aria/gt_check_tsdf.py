"""gt_check_adt.py (the GaME session's ADT check, used only to check, not to choose or tune) on our TSDF maps: for every ADT
object in view of a room's two clips, whether it stayed, moved, left or came, and how much of visit A's map at its visit-A
place (inside its visit-A box, above its base) is still there at the end of visit B. The map points: the vertices of our
map's final snapshot of each visit, the background and the objects present then (analysis/history_export); B's objects
first seen in B are not counted (the front numbers them above A's). Then the objects our map closed in each visit (present
in an earlier snapshot of the visit, absent or without surface at its end; by physical id), each with the ADT object whose visit-A box holds
most of its last surface, and that object's status.
usage: gt_check_tsdf.py ROOM RUN_DIR   (RUN_DIR/session_{a,b}/final.4dmap.zpk, aria/run_tsdf.sh)"""
import json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/jixian/Desktop/FT/eval_fast/vendor/pack'); import zpk_snapshot as zs
GT_CHECK = Path('/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/analysis/aria_20261101/gt_check_adt.py')
EXPORT = '/home/jixian/Desktop/FT/wt_takeover_runs/analysis/history_export'
LIBS = ':'.join(f'/home/jixian/ros2_ws/install/{d}/lib' for d in sorted(os.listdir('/home/jixian/ros2_ws/install')))
ENV = dict(os.environ, LD_LIBRARY_PATH=f'{LIBS}:/opt/ros/jazzy/lib:/opt/ros/jazzy/lib/x86_64-linux-gnu')
room, run = sys.argv[1], Path(sys.argv[2])

# The ground truth part of gt_check_adt.py, unchanged (everything before its map loading): poses, boxes, names, in_view.
src = GT_CHECK.read_text(); head = src[:src.index('if kind == "points":')]
sys.argv = [str(GT_CHECK), room, str(run), 'points']
exec(compile(head, str(GT_CHECK), 'exec'))
inside_src = src[src.index('def inside('):src.index('rows = []')]
exec(compile(inside_src, str(GT_CHECK), 'exec'))


def snapshot(path, i):
    """Slice i of a 4D map: {owner: (vertices, present)} of the objects, and the background vertices."""
    with tempfile.TemporaryDirectory(dir='/dev/shm') as tmp:
        zs.cut_many(str(path), [i], out=f'{tmp}/s.4dmap', work=tmp)
        subprocess.run([EXPORT, f'{tmp}/s.4dmap', f'{tmp}/s.ply'], env=ENV, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        m = json.load(open(f'{tmp}/s.ply.json')); raw = open(f'{tmp}/s.ply', 'rb').read()
    V = np.frombuffer(raw, np.float32, m['vertices'] * 3, raw.index(b'end_header\n') + 11).reshape(-1, 3).astype(np.float64)
    objs, bg = {}, []
    for l in m['layers']:
        part = V[l['vertex_start']:l['vertex_start'] + l['vertices']]
        if l['layer'] == 'background': bg.append(part)
        elif l['layer'] == 'object' and not l.get('trajectory_points'):
            # keyed by the physical id (the front's): node ids are renumbered between snapshots
            key = l.get('instance_id') if l.get('instance_id') is not None else l['owner']
            P, present, _ = objs.get(key, (np.zeros((0, 3)), False, None))
            objs[key] = (np.concatenate([P, part]), present or bool(l['native_present']) and len(part) > 0, l.get('instance_id'))
    return objs, (np.concatenate(bg) if bg else np.zeros((0, 3)))


def visit(s):
    path = run / f'session_{s}/final.4dmap.zpk'; n = len(zs.stamps(str(path)))
    last, present_end, ids = {}, set(), {}
    for i in range(n):
        objs, bg = snapshot(path, i)
        for o, (P, present, inst) in objs.items():
            if present and len(P): last[o] = P; ids[o] = inst
        if i == n - 1:
            present_end = {o for o, (P, present, _) in objs.items() if present and len(P)}
            final = np.concatenate([bg] + [objs[o][0] for o in present_end]) if present_end else bg
            final_objects = {o: objs[o][0] for o in present_end}
    closed = {o: last[o] for o in last if o not in present_end}
    return final, final_objects, closed, ids


XA, objA, closedA, idsA = visit('a')
XB_all, objB, closedB, idsB = visit('b')
a_ids = {i for i in idsA.values() if i is not None}
a_max = max(a_ids) if a_ids else 0
born_b = [P for o, P in objB.items() if idsB.get(o) is not None and idsB[o] > a_max]   # objects first seen in B: not counted
XB = XB_all
if born_b:
    from scipy.spatial import cKDTree
    t = cKDTree(np.concatenate(born_b)); XB = XB_all[t.query(XB_all)[0] > 1e-6]

rows = []
for uid in sorted(set(PA) | set(PB)):
    if uid not in BX: continue
    a, b = PA.get(uid), PB.get(uid)
    status = 'came' if a is None else 'gone' if b is None else ('moved' if float(np.linalg.norm(a[0] - b[0])) > 0.01 else 'stayed')
    ref = a if a is not None else b
    if in_view(ref[0]) < 1: continue
    lo, hi = BX[uid]; vol = float(np.prod(hi - lo))
    sel_a, sel_b = inside(XA, uid, ref), inside(XB, uid, ref)
    if status == 'moved':
        sel_a &= ~inside(XA, uid, b); sel_b &= ~inside(XB, uid, b)
    na, nb = int(sel_a.sum()), int(sel_b.sum())
    if na < 20 and status != 'came': continue
    rows.append((status, names.get(str(uid), {}).get('instance_name', str(uid)), vol, na, nb))
for st in ('moved', 'gone', 'came', 'stayed'):
    rs = [r for r in rows if r[0] == st]
    if not rs: continue
    keep = [r[4] / max(r[3], 1) for r in rs]
    print(f"{st}: {len(rs)} objects, visit A's map {'where it was and no longer is' if st == 'moved' else 'at its visit-A place'} "
          f"kept at the end of B: median {np.median(keep):.2f}")
    for r in sorted(rs, key=lambda r: -r[2]):
        if st == 'stayed' and r[4] / max(r[3], 1) > 0.5: continue
        print(f"  {r[1][:30]:30s} vol {r[2]:.4f} m3{' (<0.005)' if r[2] < 0.005 else '         '}  end of A {r[3]:6d}  end of B {r[4]:6d}  kept {r[4] / max(r[3], 1):.2f}")


def which(P):
    """The ADT object whose visit-A box holds most of P, with its status."""
    best = (0, None)
    for uid in BX:
        if 'ApartmentEnv' in names.get(str(uid), {}).get('instance_name', ''): continue   # the apartment itself
        ref = PA.get(uid) or PB.get(uid)
        if ref is None: continue
        k = int(inside(P, uid, ref).sum())
        if k > best[0]: best = (k, uid)
    if best[1] is None: return 'no ADT object', '-', 0.0
    uid = best[1]; a, b = PA.get(uid), PB.get(uid)
    status = 'came' if a is None else 'gone' if b is None else ('moved' if float(np.linalg.norm(a[0] - b[0])) > 0.01 else 'stayed')
    return names.get(str(uid), {}).get('instance_name', str(uid)), status, best[0] / max(len(P), 1)


for s, closed, ids in (('A', closedA, idsA), ('B', closedB, idsB)):
    print(f'\nclosed in visit {s}: {len(closed)}')
    for o, P in sorted(closed.items(), key=lambda x: -len(x[1])):
        name, status, share = which(P)
        print(f'  id {ids.get(o)}  {len(P):6d} vertices, centre {np.round(P.mean(0), 2)}  -> {name[:30]} ({status}, {100 * share:.0f}% of its surface in that box)')
