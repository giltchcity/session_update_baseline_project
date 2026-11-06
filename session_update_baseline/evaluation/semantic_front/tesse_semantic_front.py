"""The semantic object front on TESSE (user 11-03: try it where Khronos itself makes objects from semantics and tracking;
as robust as the annotation, which is semantics plus a number): each frame's semantic image (the scene's ground-truth
semantics) cut into regions per static object class (the label space's object labels without its dynamic ones: the
annotation gives people no id, the core takes them from the semantics), the regions named with physical ids by the GaME
session's front (update_layer/front.py at e4a02cf), full resolution, range 5 m (the TESSE runs' sensor range). Writes
OUT/instance_labels/<frame>_segmentation.png (uint16: physical id, 0 none), OUT/physical_catalog.json (each id with its
class) and OUT/FRONT_CHECK.txt (the ground-truth instance ids only read to check: splits and merges).
Without flags: regions are 8-connected image pieces, tracked by the front's steps 1-3 (InstanceForwarding, MaxIoUTracker
over 0.2 m voxels in a 3 s window, re-identification by the voxels an id was ever seen in).
--3d: regions as Khronos' ConnectedSemantics makes them from semantic input (khronos_uHumans2_cd5.yaml: use_3d,
grid_size 0.1 m, full connectivity): each class's depth pixels in 0.1 m cells of the world, 26-connected; a pixel
without depth takes the cell region most of its image region's depth pixels fall in.
--place: named by surface place, the front's step 4 (place_voxel 0.05 m, the map's resolution; surface tolerance
0.05 m), read with semantics: a pixel is the record on whose surface it lies only inside a region of the record's class
(step 4 is class-agnostic, as SAM's masks are: with it a wall pixel by an object's surface was that object and grew it);
new surface is organised by its region as in step 4; a region's pixels without depth get the id its depth pixels got
most.
In every mode a track (record) is named (its id written) once its surface, the points it got in earlier frames, seen
again has been measured again (within the surface tolerance, 0.05 m) more often than seen through (the ray going on
past it), the core's own evidence (Supported against Free; the surface as the mean point per voxel of the map): an object stays at its place, a thing in motion (the
office's kicked ball, of a static class) leaves surface behind that is seen through and keeps no id, as in the
annotation, so the core tracks its motion.
Runs (11-03): v1 no flags, people named, no naming rule; v3 --place --3d, people named by a tracker of their own, no
naming rule; v4 --place --3d; v5 --3d.
usage: tesse_semantic_front.py office|apartment OUT [--place] [--3d]"""
import json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import yaml
from PIL import Image
from scipy import ndimage
sys.path.insert(0, '/home/jixian/Desktop/FT/wt_takeover_runs/aria/front_e4a02cf')
from update_layer.frames import FlatSession, backproject, tesse_session
from update_layer.front import FrontConfig, KhronosObjectFront, _place_keys
scene, OUT, PLACE, CONNECT_3D = sys.argv[1], Path(sys.argv[2]), '--place' in sys.argv[3:], '--3d' in sys.argv[3:]; (OUT / 'instance_labels').mkdir(parents=True, exist_ok=True)
T = Path('/home/jixian/Desktop/FT/wt_takeover_runs/tesse')
cfg = T / 'configs' / ('apartment' if scene == 'apartment' else '')
space = yaml.safe_load((cfg / 'label_space.yaml').read_text())
STATIC = [c for c in space['object_labels'] if c not in space['dynamic_labels']]
VOXEL = yaml.safe_load((cfg / 'mapper_tesse_5cm.yaml').read_text())['active_window']['volumetric_map']['voxel_size']   # the map's


class SemanticFront(KhronosObjectFront):
    def __init__(self, cfg):
        super().__init__(cfg); self.named, self.pending, self.evidence = set(), {}, {}

    def _judge(self, frame):
        R, t = frame.T_world_cam[:3, :3], frame.T_world_cam[:3, 3]; tol = self.cfg.surface_tolerance
        for r, parts in list(self.pending.items()):                  # its surface from earlier frames: mean point per place
            P = np.concatenate(parts); uk, inv = np.unique(_place_keys(P, VOXEL), return_inverse=True)
            S = np.stack([np.bincount(inv, P[:, j]) for j in range(3)], 1) / np.bincount(inv)[:, None]; self.pending[r] = [S]
            u, v, z = frame.K.project((S - t) @ R); ok = (u >= 0) & (z <= self.cfg.max_range)
            d, z = frame.depth[v[ok], u[ok]], z[ok]; m = np.isfinite(d)
            e = self.evidence.setdefault(r, [0, 0])
            e[0] += int(np.count_nonzero(m & (np.abs(d - z) <= tol))); e[1] += int(np.count_nonzero(m & (d > z + tol)))
            if e[0] > e[1]: self.named.add(r); del self.pending[r]

    def _clusters(self, frame):
        out = super()._clusters(frame)
        cls = np.zeros(frame.depth.shape, np.int64)
        for c in out: cls[c['mask']] = c['semantic']
        self._point_class = cls[np.isfinite(frame.depth) & (frame.depth <= self.cfg.max_range)]   # _process_by_place's points
        return out

    def _measure(self, pts):
        out = super()._measure(pts); rec = out >= 0
        if rec.any():
            semantic_of = np.array([t.semantic for t in self.tracks], np.int64)
            out[rec & (semantic_of[np.maximum(out, 0)] != self._point_class)] = -1
        return out

    def process(self, frame):
        self._judge(frame); out = super().process(frame)
        P, V, U = backproject(frame, (out > 0) & (frame.depth <= self.cfg.max_range)); r = out[V, U] - 1
        for t in np.unique(r).tolist():
            if t not in self.named: self.pending.setdefault(t, []).append(P[r == t])
        out[(out > 0) & ~np.isin(out - 1, list(self.named))] = 0
        return out


front = SemanticFront(FrontConfig(STATIC, max_range=5.0, place_voxel=VOXEL if PLACE else 0.0, surface_tolerance=VOXEL))
session = FlatSession(tesse_session('apt' if scene == 'apartment' else 'a'), pixel_step=1)


def segments(sem):
    out = np.zeros(sem.shape, np.int32); n = 0
    for c in STATIC:
        lab, k = ndimage.label(sem == c, structure=np.ones((3, 3), bool))
        if k: out[lab > 0] = lab[lab > 0] + n; n += k
    return out


GRID = 0.1                                            # ConnectedSemantics grid_size, Khronos' TESSE configuration


def segments_3d(frame, sem):
    out = np.zeros(sem.shape, np.int32); n = 0
    valid = np.isfinite(frame.depth) & (frame.depth <= 5.0)
    for c in STATIC:
        m = (sem == c) & valid
        if not m.any(): continue
        pts, V, U = backproject(frame, m)
        key = np.floor(pts / GRID).astype(np.int64); key -= key.min(0)
        grid = np.zeros(tuple(key.max(0) + 1), bool); grid[tuple(key.T)] = True
        lab, k = ndimage.label(grid, structure=np.ones((3, 3, 3), bool))
        out[V, U] = lab[tuple(key.T)] + n
        flat, j = ndimage.label(sem == c, structure=np.ones((3, 3), bool))   # pixels without depth: their image region's
        for r in np.unique(flat[(flat > 0) & ~valid]):
            rm = flat == r; got = out[rm & valid]
            if len(got): out[rm & ~valid] = np.bincount(got).argmax()
        n += k
    return out


pairs = defaultdict(int); t0 = time.time()
front.begin_session()
for i in range(len(session.ids)):
    frame = session.load(i)
    truth = frame.instance.copy()
    frame.instance = segments_3d(frame, frame.semantic) if CONNECT_3D else segments(frame.semantic)
    ids = front.process(frame)
    if PLACE:                                       # a region's pixels without depth: the id its depth pixels got most
        valid = np.isfinite(frame.depth) & (frame.depth <= 5.0)
        for k in np.unique(frame.instance[(frame.instance > 0) & ~valid]):
            m = frame.instance == k; got = ids[m & valid]; got = got[got > 0]
            if len(got): ids[m & ~valid & (ids == 0)] = np.bincount(got).argmax()
    Image.fromarray(ids.astype(np.uint16)).save(OUT / 'instance_labels' / f'{session.ids[i]}_segmentation.png')
    sel = (truth > 0) | (ids > 0)
    key, cnt = np.unique(np.stack([truth[sel], ids[sel]], 1), axis=0, return_counts=True)
    for (a, b), c in zip(key.tolist(), cnt.tolist()): pairs[(a, b)] += c
    if i % 500 == 0: print(f'{scene} {i}/{len(session.ids)}: {len(front.tracks)} front ids, {time.time() - t0:.0f} s', flush=True)
catalog = dict(schema='session_update_physical_catalog/v1', policy='A nonzero physical-instance pixel receives the catalog semantic '
               'class; ids: the semantic front (front id + 1), each with its track class',
               semantic_id_range=[0, int(space['total_semantic_labels']) - 1],
               entities=[dict(physical_instance_id=t.id + 1, semantic_id=int(t.semantic), name=f'front_{t.id + 1}') for t in front.tracks
                         if t.id in front.named])
(OUT / 'physical_catalog.json').write_text(json.dumps(catalog, indent=1))
lines = [f'{scene}: front {front.log}; {len(front.tracks)} front ids for {len(session.ids)} frames', '',
         'per ground-truth instance: pixels, main front id share, front ids with >= 5 % (splits)']
splits = merges = 0
for a in sorted({a for a, _ in pairs if a > 0}):
    tot = {b: c for (x, b), c in pairs.items() if x == a}; n = sum(tot.values()); main = max(tot, key=tot.get)
    big = [b for b, c in sorted(tot.items(), key=lambda x: -x[1]) if c >= 0.05 * n and b > 0]
    splits += len(big) > 1
    lines.append(f'  GT {a:<4d} {n:8d} px  main front id {main} {100 * tot[main] / n:5.1f} %  ids >= 5 %: {big}')
lines += ['', 'per front id holding >= 5 % of 2+ ground-truth instances (merges)']
for b in sorted({b for _, b in pairs if b > 0}):
    tot = {a: c for (a, x), c in pairs.items() if x == b}; n = sum(tot.values())
    held = sorted(a for a, c in tot.items() if a > 0 and c >= 0.05 * n)
    if len(held) >= 2: merges += 1; lines.append(f'  front {b}: {n} px, GT instances {held}')
lines.insert(1, f'GT instances split over 2+ front ids: {splits}; front ids merging 2+ GT instances: {merges}')
(OUT / 'FRONT_CHECK.txt').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines[:3]))
