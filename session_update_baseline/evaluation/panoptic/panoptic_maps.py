"""The maps one Panoptic Mapping visit saved (analysis/panoptic_rerun.sh: its maps during the visit and its final map)
as the per-snapshot records every method's map is scored from (eval/scene: snapshots.jsonl, and for real B and C
cleanup_probes.bin and cleanup_mesh.jsonl, by eval/scene/emit_harness.py's writers; synthetic:
the native export the published synthetic harness reads).
Each map is read by its own submap states only (Panoptic Multi-TSDFs, Schmid et al., ICRA 2022, Sec. III-D/E), in
two ways, one per question:
  the scene it shows (G1, object F1: objects.jsonl, final_present.ply, the native export): as the paper evaluates a
    map ("only present, i.e. active or persistent submaps, are counted") and map_evaluator.cpp computeMeshError
    selects, a submap is shown when its change state is New, Matched or Persistent; Absent (conflicts with newer
    observations) and Unobserved (loaded from the previous visit, not observed again: unknown) are not. Its other
    instance submaps stay in the map with their surfaces, not present at that time, as Khronos' and our maps hand the
    evaluator their objects no longer present (the official evaluator associates every object of a map, then counts
    by presence);
  the changes it reports (the change protocols: snapshots.jsonl, the cleanup probes, native_changes/): a submap is
    gone only when Panoptic judged it Absent; an Unobserved one is no judgment and stays in the map (the paper's map
    queries still answer from it as the expected state), so it is kept with its surface.
Free space is neither. Nothing is carried from one map to another: an object is each map's instance submap, with its
surface (area samples of its mesh, tools/abc_eval_v2/panoptic_export.py, the validated decoder of the archived maps)
when it is kept by the reading; the background is the kept non-instance submaps.
  semantic: real = the label contract's ADE class (panoptic_export's 'semantic'); flat, synthetic, 3rscan = the class
    id of the input's groundtruth_labels.csv;
  stamps: a map's frame stamp (its file name; the final map: the visit's last frame) plus --stamp-base;
  snapshots.jsonl (the change protocols, the changes reading): with --seed, first the previous visit's final map at
    --seed-ns (the start of this visit; Panoptic's loadMap then marks every loaded submap Unobserved until it is
    observed again): what the visit inherited, the "before" of the between-visit events; then the visit's maps;
  objects.jsonl (the object F1, the scene reading) and the native export: the visit's own maps, after, with --start-ns, the map at
    the visit's start, in which nothing is shown (a first visit starts empty; a later one shows nothing of what it
    loaded until it is observed again).
Also writes final_present.ply (the final map's shown mesh, panoptic_export's current_confirmed.ply) and maps.json.
  usage: panoptic_maps.py KIND OUT_DIR SNAPSHOT_DIR FINAL_MAP END_STAMP_NS [--stamp-base NS] [--seed MAP --seed-ns NS]
                          [--start-ns NS] [--probes RESIDUAL_PROBES.json MESH_PROBES.json] [--native]
                          (KIND: real | flat | synthetic; --native writes OUT_DIR/native/ (shown meshes) and
                          OUT_DIR/native_changes/ (kept meshes, the same maps))
"""
import argparse, contextlib, io, json, sys, tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

FT = Path('/home/jixian/Desktop/FT')
sys.path.insert(0, str(FT)); sys.path.insert(0, str(FT / 'tools/abc_eval_v2'))
from eval.scene.scene import INVALID_LABEL, EvaluationScene, SceneObject  # noqa: E402

SHOWN = ('New', 'Matched', 'Persistent')       # the scene a map shows
GONE = ('Absent',)                             # the changes it reports: only a judged absence removes a submap
NEVER = 2**64 - 1          # last observed of an object still present, as every method's timeline (the evaluator needs one)


def ply_vertices(path):
    """panoptic_export.write_ply's triangle soup: every 3 vertices one face."""
    data = Path(path).read_bytes()
    head = data.index(b'end_header\n') + len(b'end_header\n')
    n = int(data[:head].split(b'element vertex ')[1].split(b'\n')[0])
    return np.frombuffer(data, '<f4', 3 * n, head).reshape(-1, 3)


def write_native_mesh(out, prefix, vertices):
    """complete_native_export.cpp layout (eval/scene/emit_synthetic._write): vertices, faces, colours."""
    np.asarray(vertices, np.float32).tofile(out / f'{prefix}.f32')
    np.arange(len(vertices), dtype=np.uint32).reshape(-1, 3).tofile(out / f'{prefix}.u32')
    np.full((len(vertices), 3), 180, np.uint8).tofile(out / f'{prefix}.rgb')
    return len(vertices), len(vertices) // 3


def read_map(job):
    """One saved map -> its objects and background under both readings and its state counts; the final map also keeps
    its shown mesh; with native export indices, the map's rows of native/ and native_changes/ (mesh files written)."""
    kind, path, stamp, keep_mesh, native = job
    import panoptic_export as pe
    with tempfile.TemporaryDirectory(prefix='panoptic-map-', dir='/dev/shm') as tmp:
        with contextlib.redirect_stdout(io.StringIO()):
            manifest = pe.export(Path(path), Path(tmp))
        objects, counts = [], {}
        bg = {'scene': [], 'changes': []}; bg_label = {'scene': [], 'changes': []}; bg_mesh = {'scene': [], 'changes': []}
        rows = {'scene': [], 'changes': []}
        for r in manifest['submaps']:
            counts[r['change_state']] = counts.get(r['change_state'], 0) + 1
            if r['panoptic_label'] == 'FreeSpace':
                continue
            semantic = r['semantic'] if kind == 'real' else r['class_id']
            semantic = -1 if semantic is None else int(semantic)
            oid = str(r['load_ordinal'])      # the submap's place in the map: unique (its frame name is not once a visit
                                              # has loaded a map: new submaps reuse the loaded ones' numbers)
            keep = {'scene': r['change_state'] in SHOWN and r['vertex_count'] > 0,
                    'changes': r['change_state'] not in GONE and r['vertex_count'] > 0}
            geometry = r['vertex_count'] > 0
            points = np.load(r['surface_points_path'])['points'].astype(np.float32) if geometry else np.zeros((0, 3), np.float32)
            mesh = ply_vertices(r['mesh_path']) if native and geometry else None
            if r['panoptic_label'] == 'Instance':
                objects.append(dict(id=oid, instance_id=int(r['instance_id']), semantic=semantic, points=points, **keep))
            for reading, kept in keep.items():
                if r['panoptic_label'] == 'Instance' and native and reading in native:
                    prefix = f"{native[reading][1]}_{oid}"
                    listed = geometry if reading == 'scene' else kept      # the scene lists the map's every object
                    nv, nf = write_native_mesh(native[reading][0], prefix, mesh) if listed else (0, 0)
                    rows[reading].append(dict(id=oid, semantic=semantic, present=kept, current=kept and nv > 0, vertices=nv,
                                              faces=nf, prefix=prefix, first_observed_ns=[stamp],
                                              last_observed_ns=[NEVER] if kept else [stamp],
                                              trajectory_timestamps=[], trajectory_positions=[]))
                elif r['panoptic_label'] != 'Instance' and kept:
                    bg[reading].append(points)
                    bg_label[reading].append(np.full(len(points), INVALID_LABEL if semantic < 0 else semantic, np.uint32))
                    if native and reading in native:
                        bg_mesh[reading].append(mesh)
        final_mesh = (Path(tmp) / 'current_confirmed.ply').read_bytes() if keep_mesh else None
    native_rows = None
    if native:
        native_rows = {}
        for reading, (out, index) in native.items():
            write_native_mesh(out, f'{index}_background', np.concatenate(bg_mesh[reading]) if bg_mesh[reading] else np.zeros((0, 3), np.float32))
            native_rows[reading] = dict(index=index, stamp_ns=stamp, objects=rows[reading], background_prefix=f'{index}_background')
    cat = lambda xs, empty: np.concatenate(xs) if xs else empty
    return dict(stamp=stamp, source=str(path), counts=counts, objects=objects,
                background={k: cat(v, np.zeros((0, 3), np.float32)) for k, v in bg.items()},
                background_label={k: cat(v, np.zeros(0, np.uint32)) for k, v in bg_label.items()},
                mesh=final_mesh, native=native_rows)


def scene(m, reading, t=None):
    """The map under one reading ('scene' or 'changes'), at its stamp or at t."""
    t = m['stamp'] if t is None else t
    objs = [SceneObject(id=o['id'], instance_id=o['instance_id'], semantic=o['semantic'],
                        points=o['points'] if o[reading] or reading == 'scene' else np.zeros((0, 3), np.float32), present=o[reading],
                        first_observed_ns=[t], last_observed_ns=[NEVER] if o[reading] else [t])
            for o in m['objects']]
    return EvaluationScene(t, m['background'][reading], m['background_label'][reading], objs)


def empty_scene(t):
    return EvaluationScene(t, np.zeros((0, 3), np.float32), np.zeros(0, np.uint32), [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=('real', 'flat', 'synthetic', '3rscan'))
    ap.add_argument('out', type=Path); ap.add_argument('snapshots', type=Path); ap.add_argument('final', type=Path)
    ap.add_argument('end_ns', type=int)
    ap.add_argument('--stamp-base', type=int, default=0)
    ap.add_argument('--seed', type=Path); ap.add_argument('--seed-ns', type=int)
    ap.add_argument('--start-ns', type=int)
    ap.add_argument('--probes', nargs=2, type=Path)
    ap.add_argument('--native', action='store_true')
    ap.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    nat = {'scene': a.out / 'native', 'changes': a.out / 'native_changes'}
    if a.native:
        for d in nat.values():
            d.mkdir(exist_ok=True)
    start = 0 if a.start_ns is None else 1               # objects.jsonl / native index of the visit's first map
    visit = [(p, a.stamp_base + int(p.stem.split('_')[1])) for p in sorted(a.snapshots.glob('*.panmap'))]
    visit = [(p, t) for p, t in visit if t < a.end_ns] + [(a.final, a.end_ns)]   # one saved at the last frame is the final map's moment
    for seq in (([] if a.seed is None else [a.seed_ns]) + [t for _, t in visit],
                ([] if a.start_ns is None else [a.start_ns]) + [t for _, t in visit]):
        assert seq == sorted(seq) and len(set(seq)) == len(seq), 'map stamps out of order'
    seed_native = {'changes': (nat['changes'], 0)} if a.native and start else None    # native_changes starts with it
    jobs = ([] if a.seed is None else [(a.kind, a.seed, a.seed_ns, False, seed_native)]) + [
        (a.kind, p, t, p == a.final, {k: (d, start + i) for k, d in nat.items()} if a.native else None)
        for i, (p, t) in enumerate(visit)]
    from eval.scene.emit_harness import snapshot_record
    with ProcessPoolExecutor(a.jobs) as pool:
        maps = list(pool.map(read_map, jobs))
    seed, maps = (maps[0], maps[1:]) if a.seed is not None else (None, maps)
    # the changes reading: the seed (what the visit inherited) first; the scene reading: the visit's maps, after the
    # visit's start, at which nothing is shown
    changes = ([] if seed is None else [scene(seed, 'changes')]) + [scene(m, 'changes') for m in maps]
    shown = ([] if a.start_ns is None else [empty_scene(a.start_ns)]) + [scene(m, 'scene') for m in maps]
    with (a.out / 'snapshots.jsonl').open('w') as f:
        f.writelines(json.dumps(snapshot_record(i, sc)) + '\n' for i, sc in enumerate(changes))
    with (a.out / 'objects.jsonl').open('w') as f:
        f.writelines(json.dumps(snapshot_record(i, sc)) + '\n' for i, sc in enumerate(shown))
    if a.probes:      # the CPU writers (emit_gpu.py's GPU versions write the same files, but need a newer torch)
        from eval.scene.emit_harness import write_cleanup_binary, write_cleanup_mesh
        at = {sc.t_ns: sc for sc in changes}
        write_cleanup_binary(list(at), at.__getitem__, a.probes[0], a.out / 'cleanup_probes.bin')
        write_cleanup_mesh(list(at), at.__getitem__, a.probes[1], a.out / 'cleanup_mesh.jsonl')
    if a.native:
        for reading, d in nat.items():
            rows = [m['native'][reading] for m in maps]
            if start and reading == 'changes' and seed is not None:   # the visit's start: what it inherited
                rows.insert(0, seed['native']['changes'])
            elif start:                                               # ... or nothing (the scene shows nothing then)
                write_native_mesh(d, '0_background', np.zeros((0, 3), np.float32))
                rows.insert(0, dict(index=0, stamp_ns=a.start_ns, objects=[], background_prefix='0_background'))
            (d / 'meshes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (a.out / 'final_present.ply').write_bytes(maps[-1]['mesh'])
    (a.out / 'maps.json').write_text(json.dumps(dict(
        kind=a.kind, scene='shown = change state New, Matched or Persistent, not free space (paper Sec. III-E, computeMeshError)',
        changes='kept = change state not Absent (Unobserved is no judgment), not free space',
        seed=None if a.seed is None else str(a.seed), start_ns=a.start_ns,
        maps=[dict(index=i, stamp_ns=m['stamp'], source=m['source'], states=m['counts'],
                   objects_shown=sum(o['scene'] for o in m['objects']), objects_kept=sum(o['changes'] for o in m['objects']),
                   objects_listed=len(m['objects'])) for i, m in enumerate(([] if seed is None else [seed]) + maps)]), indent=1))
    (a.out / 'COMPLETE.json').write_text(json.dumps(dict(map=str(a.final), snapshots=len(changes), provenance='panoptic_maps.py'), indent=2))
    print(f'{len(maps)} maps ({int(seed is not None)} seed, {start} start), last: {maps[-1]["counts"]}', flush=True)


if __name__ == '__main__':
    main()
