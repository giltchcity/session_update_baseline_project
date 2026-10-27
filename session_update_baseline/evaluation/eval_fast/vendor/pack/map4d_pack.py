#!/usr/bin/env python3
"""Lossless delta packing for Khronos v1 .4dmap files.

A v1 .4dmap stores one full scene-graph snapshot per update (~every 2 s) and
writes every byte as two (msgpack UINT8 tag 0xcc + value). Consecutive snapshots
differ by ~1 MB out of 100-250 MB, so each snapshot is stored as a zstd patch
against the previous one, with a standalone keyframe every K snapshots.

`unpack` reproduces the source file byte for byte, so every existing reader
(evaluators, viewer, --input-state) keeps working on the unpacked file. `pack`
publishes its output only after a full reconstruction hashes to the source's
sha256; the source is never touched unless --remove-source is given, and then
only after that check passed.

  map4d_pack.py pack   MAP.4dmap [--out MAP.4dmap.zpk] [--keyframe 10] [--remove-source]
  map4d_pack.py verify MAP.4dmap.zpk
  map4d_pack.py unpack MAP.4dmap.zpk [--out MAP.4dmap]
  map4d_pack.py cat    MAP.4dmap.zpk > FIFO     # stream the original bytes, no disk
  map4d_pack.py info   MAP.4dmap.zpk

`cat` emits each snapshot only after it matched its recorded sha256, so a
damaged container truncates the stream instead of passing wrong bytes on; the
exit status is non-zero unless the whole stream matched the source sha256.
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

ZSTD = shutil.which('zstd') or '/home/jixian/Desktop/miniconda3/bin/zstd'
MAGIC = b'4DMAPZPK\x00\x01'
TRAILER_MAGIC = b'ZPKEND01'
TRAILER = struct.Struct('<QQ8s')  # manifest offset, manifest length, magic
UINT8_TAG = 0xcc
ARR32_TAG = 0xdd
LEVEL = 3


# ---------------------------------------------------------------- v1 layout

def _read(f, n):
    b = f.read(n)
    if len(b) != n:
        raise EOFError('truncated .4dmap')
    return b


def _read_int(f):
    t = _read(f, 1)[0]
    fmt = {0xcc: '<B', 0xcd: '<H', 0xce: '<I', 0xcf: '<Q',
           0xd0: '<b', 0xd1: '<h', 0xd2: '<i', 0xd3: '<q'}.get(t)
    if fmt is None:
        raise ValueError(f'expected an integer, got pack type {t:#x}')
    return struct.unpack(fmt, _read(f, struct.calcsize(fmt)))[0]


def _read_bool(f):
    t = _read(f, 1)[0]
    if t not in (0xc2, 0xc3):
        raise ValueError(f'expected a bool, got pack type {t:#x}')
    return t == 0xc3


def _read_arr_len(f):
    t = _read(f, 1)[0]
    if t != ARR32_TAG:
        raise ValueError(f'expected ARR32, got pack type {t:#x}')
    return struct.unpack('<I', _read(f, 4))[0]


def index_v1(path):
    """Metadata and (header_offset, decoded_length) of every snapshot. Seeks only."""
    with open(path, 'rb') as f:
        version = _read_int(f)
        if version != 1:
            raise ValueError(f'{path}: unsupported .4dmap serialization version {version}')
        _read_bool(f)                      # finalize_incrementally
        n = _read_int(f)
        stamps = [_read_int(f) for _ in range(_read_arr_len(f))]
        _read_int(f)                       # earliest
        _read_int(f)                       # latest
        _read_bool(f)                      # finalized
        f.seek(2 * _read_arr_len(f), 1)    # spark-dsg file header, typed bytes
        snaps = []
        for _ in range(n):
            header_offset = f.tell()
            length = _read_arr_len(f)
            snaps.append((header_offset, length))
            f.seek(2 * length, 1)
        end = f.tell()
        size = os.fstat(f.fileno()).st_size
    if end != size:
        raise ValueError(f'{path}: {size - end} trailing bytes after the last snapshot')
    if len(stamps) != n:
        raise ValueError(f'{path}: {len(stamps)} stamps for {n} snapshots')
    return dict(n=n, stamps=stamps, snaps=snaps, size=size)


# ---------------------------------------------------------------- zstd

def _zstd(args, data=None):
    run = subprocess.run([ZSTD, '-q', *args], input=data, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, check=False)
    if run.returncode != 0:
        raise RuntimeError(f'zstd {" ".join(args)} failed: {run.stderr.decode().strip()}')
    return run.stdout


def _compress(cur_path, prev_path):
    if prev_path is None:
        return _zstd([f'-{LEVEL}', '-T8', '-c', cur_path])
    return _zstd([f'-{LEVEL}', '--long=31', f'--patch-from={prev_path}', '-c', cur_path])


def _decompress(blob, prev_path):
    if prev_path is None:
        return _zstd(['-d', '-c'], blob)
    return _zstd(['-d', '--long=31', f'--patch-from={prev_path}', '-c'], blob)


def _interleave(payload):
    out = np.empty(2 * len(payload), dtype=np.uint8)
    out[0::2] = UINT8_TAG
    out[1::2] = np.frombuffer(payload, dtype=np.uint8)
    return out.data


# ---------------------------------------------------------------- container

def read_manifest(path):
    with open(path, 'rb') as f:
        if f.read(len(MAGIC)) != MAGIC:
            raise ValueError(f'{path}: not a 4dmap-zpk file')
        f.seek(-TRAILER.size, 2)
        offset, length, magic = TRAILER.unpack(f.read(TRAILER.size))
        if magic != TRAILER_MAGIC:
            raise ValueError(f'{path}: missing trailer (incomplete write?)')
        f.seek(offset)
        return json.loads(_read(f, length))


def reconstruct(path, sink, work):
    """Stream the original .4dmap bytes into sink(); returns their sha256."""
    manifest = read_manifest(path)
    digest = hashlib.sha256()
    prev_path = os.path.join(work, 'prev.bin')

    def emit(chunk):
        digest.update(chunk)
        sink(chunk)

    with open(path, 'rb') as f:
        f.seek(manifest['prefix']['offset'])
        emit(_read(f, manifest['prefix']['length']))
        for i, snap in enumerate(manifest['snapshots']):
            f.seek(snap['offset'])
            blob = _read(f, snap['length'])
            payload = _decompress(blob, None if snap['kind'] == 'key' else prev_path)
            if len(payload) != snap['decoded_len'] or \
                    hashlib.sha256(payload).hexdigest() != snap['sha256']:
                raise ValueError(f'{path}: snapshot {i} does not decode to its recorded content')
            emit(bytes.fromhex(snap['header']))
            emit(_interleave(payload))
            with open(prev_path, 'wb') as p:
                p.write(payload)
    return digest.hexdigest(), manifest


def pack(src, out, keyframe, work, remove_source):
    t0 = time.time()
    idx = index_v1(src)
    tmp = f'{out}.tmp.{os.getpid()}'
    digest = hashlib.sha256()
    cur_path, prev_path = os.path.join(work, 'cur.bin'), os.path.join(work, 'prev.bin')
    snapshots = []
    try:
        with open(src, 'rb') as f, open(tmp, 'wb') as o:
            o.write(MAGIC)
            prefix_len = idx['snaps'][0][0] if idx['n'] else idx['size']
            prefix = _read(f, prefix_len)
            digest.update(prefix)
            prefix_entry = dict(offset=o.tell(), length=prefix_len)
            o.write(prefix)
            for i, (header_offset, length) in enumerate(idx['snaps']):
                assert f.tell() == header_offset
                header = _read(f, 5)
                raw = _read(f, 2 * length)
                digest.update(header)
                digest.update(raw)
                typed = np.frombuffer(raw, dtype=np.uint8)
                if not (typed[0::2] == UINT8_TAG).all():
                    raise ValueError(f'{src}: snapshot {i} is not a plain typed-uint8 array')
                payload = typed[1::2].tobytes()
                del raw, typed
                with open(cur_path, 'wb') as c:
                    c.write(payload)
                kind = 'key' if i % keyframe == 0 else 'delta'
                blob = _compress(cur_path, None if kind == 'key' else prev_path)
                snapshots.append(dict(header=header.hex(), decoded_len=length, kind=kind,
                                      offset=o.tell(), length=len(blob),
                                      sha256=hashlib.sha256(payload).hexdigest()))
                o.write(blob)
                os.replace(cur_path, prev_path)
                print(f'  [{i + 1}/{idx["n"]}] {kind:5s} {length / 1e6:7.1f} MB -> '
                      f'{len(blob) / 1e6:6.2f} MB', flush=True)
            manifest = dict(
                format='4dmap-zpk', format_version=1,
                source=dict(name=os.path.basename(src), size=idx['size'],
                            sha256=digest.hexdigest(), n=idx['n'], stamps=idx['stamps']),
                zstd=_zstd(['--version']).decode().strip(), level=LEVEL, keyframe=keyframe,
                prefix=prefix_entry, snapshots=snapshots,
                created=datetime.datetime.now().isoformat(timespec='seconds'))
            data = json.dumps(manifest).encode()
            manifest_offset = o.tell()
            o.write(data)
            o.write(TRAILER.pack(manifest_offset, len(data), TRAILER_MAGIC))

        # Publish only a container that rebuilds the exact source.
        rebuilt, _ = reconstruct(tmp, lambda chunk: None, work)
        if rebuilt != manifest['source']['sha256']:
            raise ValueError(f'{src}: reconstruction sha256 {rebuilt} != source '
                             f'{manifest["source"]["sha256"]}')
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    packed = os.path.getsize(out)
    print(f'PACKED {src} ({idx["n"]} snapshots) {idx["size"] / 1e9:.2f} GB -> '
          f'{packed / 1e9:.3f} GB ({idx["size"] / packed:.0f}x), verified sha256 '
          f'{manifest["source"]["sha256"][:16]}..., {time.time() - t0:.0f}s')
    if remove_source:
        os.remove(src)
        print(f'REMOVED {src}')


def unpack(path, out, work):
    tmp = f'{out}.tmp.{os.getpid()}'
    try:
        with open(tmp, 'wb') as o:
            rebuilt, manifest = reconstruct(path, o.write, work)
        if rebuilt != manifest['source']['sha256'] or \
                os.path.getsize(tmp) != manifest['source']['size']:
            raise ValueError(f'{path}: unpacked bytes do not match the recorded source')
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    print(f'UNPACKED {path} -> {out} ({manifest["source"]["size"] / 1e9:.2f} GB, sha256 ok)')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('pack')
    p.add_argument('map')
    p.add_argument('--out')
    p.add_argument('--keyframe', type=int, default=10)
    p.add_argument('--remove-source', action='store_true',
                   help='delete the .4dmap after the packed file verified bit-exact')
    for name in ('verify', 'info', 'cat'):
        sub.add_parser(name).add_argument('zpk')
    u = sub.add_parser('unpack')
    u.add_argument('zpk')
    u.add_argument('--out')
    args = ap.parse_args()

    if args.cmd == 'info':
        m = read_manifest(args.zpk)
        s = m['source']
        keys = sum(x['kind'] == 'key' for x in m['snapshots'])
        print(json.dumps(dict(source=s['name'], size_gb=round(s['size'] / 1e9, 3),
                              packed_gb=round(os.path.getsize(args.zpk) / 1e9, 3),
                              snapshots=s['n'], keyframes=keys, sha256=s['sha256'],
                              created=m['created']), indent=1))
        return

    with tempfile.TemporaryDirectory(prefix='map4d_pack_') as work:
        if args.cmd == 'pack':
            if args.keyframe < 1:
                ap.error('--keyframe must be >= 1')
            pack(args.map, args.out or args.map + '.zpk', args.keyframe, work,
                 args.remove_source)
        elif args.cmd == 'verify':
            rebuilt, m = reconstruct(args.zpk, lambda chunk: None, work)
            ok = rebuilt == m['source']['sha256']
            print(f'{"VERIFIED" if ok else "MISMATCH"} {args.zpk} sha256 {rebuilt}')
            sys.exit(0 if ok else 1)
        elif args.cmd == 'cat':
            out = sys.stdout.buffer
            rebuilt, m = reconstruct(args.zpk, out.write, work)
            out.flush()
            if rebuilt != m['source']['sha256']:
                print(f'MISMATCH {args.zpk} sha256 {rebuilt}', file=sys.stderr)
                sys.exit(1)
        else:
            out = args.out or (args.zpk[:-4] if args.zpk.endswith('.zpk') else args.zpk + '.4dmap')
            if os.path.exists(out):
                ap.error(f'{out} already exists')
            unpack(args.zpk, out, work)


if __name__ == '__main__':
    main()
