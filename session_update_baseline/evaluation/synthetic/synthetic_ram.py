#!/usr/bin/env python3
"""Validation input adapter for unchanged synthetic native exporters/evaluators.

Only one .4dmap snapshot lives in /dev/shm. Native mesh bytes and object scoring
are untouched; their local snapshot/query indices are restored to global indices.
Usage: synthetic_ram.py export PACK OUTPUT
       synthetic_ram.py objects PACK MANIFEST OUTPUT
SYNTHETIC_TOOLS=khronos reads official Khronos maps with the exporter and evaluator built against official Khronos
(the ones the published Khronos rows came from) in its own runtime; default: ours, with the exporter and evaluator
built against our build (eval/synthetic_base_v38/harness/compile_ours_tools.py, into eval_tools) reading the map through
the library that wrote it (wt_takeover_build).
"""
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path('/home/jixian/Desktop/FT')
sys.path.insert(0, str(ROOT / 'eval_fast/vendor/pack'))
import map4d_pack as mp
import zpk_snapshot as zs
if os.environ.get('SYNTHETIC_TOOLS', 'ours') == 'khronos':
    BIN, RUNTIME, TOOL = (ROOT / 'datasets/synthetic_ab/evaluation_complete_20260917',
                          ROOT / 'build_artifacts/official_khronos_63faadde_abc/environment.sh', 'khronos')
else:
    BIN, RUNTIME, TOOL = ROOT / 'wt_takeover_runs/eval_tools', ROOT / 'eval_fast/runtime_khronos_env.sh', 'ours'
    os.environ['SESSION_UPDATE_CANONICAL_PREFIX'] = str(ROOT / 'wt_takeover_build/canonical/install')
BIN = Path(os.environ.get('SYNTHETIC_TOOLS_BIN', BIN))   # e.g. the 0.5 m object-matching tools


def native(name, *args):
    subprocess.run(['bash', '-c', 'source "$1" >/dev/null 2>&1 || exit 90; shift; exec "$@"',
                    'synthetic-ram', str(RUNTIME),
                    str(BIN / name), *map(str, args)], check=True,
                   stdout=subprocess.DEVNULL)


def export(source, output):
    stamps = mp.read_manifest(source)['source']['stamps']
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'meshes.jsonl').open('w') as metadata:
        for index, stamp in enumerate(stamps):
            with tempfile.TemporaryDirectory(prefix='synthetic-export-', dir='/dev/shm') as tmp:
                tmp = Path(tmp)
                sliced, exported = tmp / 'one.4dmap', tmp / 'native'
                zs.cut(str(source), index, str(sliced))
                native(TOOL + '_export', sliced, exported)
                rows = (exported / 'meshes.jsonl').read_text().splitlines()
                assert len(rows) == 1
                row = json.loads(rows[0])
                assert row['index'] == 0 and row['stamp_ns'] == stamp
                row['index'] = index
                row['background_prefix'] = f'{index}_background'
                for obj in row['objects']:
                    assert obj['prefix'] == '0_' + obj['id']
                    obj['prefix'] = f"{index}_{obj['id']}"
                for path in exported.iterdir():
                    if path.name == 'meshes.jsonl':
                        continue
                    assert path.name.startswith('0_')
                    shutil.copyfile(path, output / (str(index) + path.name[1:]))
                metadata.write(json.dumps(row, separators=(',', ':'))+'\n')
                metadata.flush()
            print('EXPORT', index+1, '/', len(stamps), flush=True)


def objects(source, manifest_path, output):
    stamps = mp.read_manifest(source)['source']['stamps']
    indices = {stamp: i for i, stamp in enumerate(stamps)}
    manifest = json.loads(manifest_path.read_text())
    groups = {}
    for index, query in enumerate(manifest['queries']):
        stamp = query['snapshot_stamp_ns']
        groups.setdefault(stamp, []).append((index, query))
    rows = [None] * len(manifest['queries'])
    fieldnames = None
    output.mkdir(parents=True, exist_ok=False)
    for stamp, queries in groups.items():
        with tempfile.TemporaryDirectory(prefix='synthetic-objects-', dir='/dev/shm') as tmp:
            tmp = Path(tmp)
            sliced, selected, scored = tmp / 'one.4dmap', tmp / 'manifest.json', tmp / 'score'
            zs.cut(str(source), indices[stamp] if stamp is not None else 0, str(sliced))
            selected.write_text(json.dumps(dict(manifest, queries=[q for _, q in queries])))
            native(TOOL + '_historical_object_eval', sliced, selected, scored)
            reader = csv.DictReader(io.StringIO((scored / 'object.csv').read_text()))
            actual = list(reader)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            assert fieldnames == reader.fieldnames and len(actual) == len(queries)
            for local, ((index, query), row) in enumerate(zip(queries, actual)):
                assert int(row['Name']) == local and int(row['Query']) == query['query_stamp_ns']
                row['Name'] = str(index)
                rows[index] = row
        print('OBJECT', stamp, len(queries), flush=True)
    assert all(row is not None for row in rows)
    with (output / 'object.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    action, path, *arguments = sys.argv[1:]
    source = Path(path).resolve(strict=True)
    os.environ['TMPDIR'] = '/dev/shm'
    if action == 'export':
        export(source, Path(arguments[0]))
    elif action == 'objects':
        objects(source, Path(arguments[0]), Path(arguments[1]))
    else:
        raise ValueError(action)
