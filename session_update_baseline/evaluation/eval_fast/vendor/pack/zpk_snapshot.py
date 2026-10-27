#!/usr/bin/env python3
"""Cut ONE moment out of a packed map (.4dmap.zpk) without unpacking the rest: a valid single-snapshot .4dmap in RAM.

Why: a real map is 16-26 GB unpacked (63-112 full scene-graph snapshots, 2 bytes per byte); a figure or a check needs
1-5 of them. The pack stores a zstd keyframe every K (=10) snapshots and a patch against the previous snapshot otherwise,
plus a manifest with every entry's offset, so snapshot i needs only its keyframe k = i - i % K and the <= K-1 patches after
it: O(K * snapshot) decompression in memory instead of O(whole map) written to and read back from disk.

The output keeps the original file header (version, flags, spark-dsg header) but lists only this snapshot
(n = 1, stamps = [stamp_i], earliest = latest = stamp_i), so every existing reader (4dmap_mesh_server, export_timeline,
geometry_export, the viewer) opens it unchanged. Each decoded payload is checked against the manifest's sha256.

  zpk_snapshot.py MAP.4dmap[.zpk] INDEX|final|t=SECONDS[,...] [--origin-ns N] [--out /dev/shm/x.4dmap]
      one spec -> single-snapshot map; several comma-separated specs -> ONE sparse map holding exactly those moments
  python API: cut(src, index, out), cut_many(src, [indices], out), stamps(src), pick(src, spec, origin_ns)
Works on packed (.zpk) and on plain .4dmap files (a plain file is cut by seeking, nothing else is read).
"""
import hashlib, io, json, os, struct, subprocess, sys, tempfile, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import map4d_pack as mp  # noqa: E402

SHM = '/dev/shm'


def _int_raw(f):
    """Read one msgpack integer, return (value, raw bytes) so it can be re-emitted with the same width."""
    t = mp._read(f, 1)
    fmt = {0xcc: '<B', 0xcd: '<H', 0xce: '<I', 0xcf: '<Q', 0xd0: '<b', 0xd1: '<h', 0xd2: '<i', 0xd3: '<q'}.get(t[0])
    if fmt is None: raise ValueError('expected an integer, got %#x' % t[0])
    b = mp._read(f, struct.calcsize(fmt))
    return struct.unpack(fmt, b)[0], t + b


def _same_width(raw, value):
    t = raw[0]; fmt = {0xcc: '<B', 0xcd: '<H', 0xce: '<I', 0xcf: '<Q', 0xd0: '<b', 0xd1: '<h', 0xd2: '<i', 0xd3: '<q'}[t]
    return bytes([t]) + struct.pack(fmt, value)


def _header_parts(prefix):
    f = io.BytesIO(prefix)
    version, v_raw = _int_raw(f)
    if version != 1: raise ValueError('unsupported .4dmap version %d' % version)
    fin_inc = mp._read(f, 1)
    n, n_raw = _int_raw(f)
    t = mp._read(f, 1)
    if t[0] != mp.ARR32_TAG: raise ValueError('stamps are not ARR32')
    m = struct.unpack('<I', mp._read(f, 4))[0]
    stamps_raw = [_int_raw(f) for _ in range(m)]
    earliest, e_raw = _int_raw(f); latest, l_raw = _int_raw(f)
    finalized = mp._read(f, 1)
    rest = prefix[f.tell():]                       # spark-dsg file header (typed bytes), unchanged
    return dict(v_raw=v_raw, fin_inc=fin_inc, n_raw=n_raw, stamps=[s for s, _ in stamps_raw],
                stamp_raw=[r for _, r in stamps_raw], e_raw=e_raw, l_raw=l_raw, finalized=finalized, rest=rest)


def _decode(f, snap, prev_file):
    f.seek(snap['offset']); blob = mp._read(f, snap['length'])
    payload = mp._decompress(blob, None if snap['kind'] == 'key' else prev_file)
    if len(payload) != snap['decoded_len'] or hashlib.sha256(payload).hexdigest() != snap['sha256']:
        raise ValueError('snapshot does not decode to its recorded content')
    return payload


def cut(zpk, index, out=None, work=SHM):
    t0 = time.time()
    man = mp.read_manifest(zpk); snaps = man['snapshots']; n = len(snaps)
    if index < 0: index += n
    K = man.get('keyframe', 10); k = index - index % K
    while snaps[k]['kind'] != 'key': k -= 1
    with open(zpk, 'rb') as f:
        f.seek(man['prefix']['offset']); prefix = mp._read(f, man['prefix']['length'])
        h = _header_parts(prefix)
        with tempfile.TemporaryDirectory(dir=work) as tmp:
            prev = os.path.join(tmp, 'prev.bin'); payload = None
            for j in range(k, index + 1):
                payload = _decode(f, snaps[j], prev)
                if j < index:
                    with open(prev, 'wb') as p: p.write(payload)
    stamp = h['stamps'][index]
    out = out or os.path.join(work, '%s_%d.4dmap' % (hashlib.sha1(os.path.abspath(zpk).encode()).hexdigest()[:12], index))
    tmp_out = out + '.part'
    with open(tmp_out, 'wb') as o:
        o.write(h['v_raw']); o.write(h['fin_inc']); o.write(_same_width(h['n_raw'], 1))
        o.write(bytes([mp.ARR32_TAG]) + struct.pack('<I', 1)); o.write(h['stamp_raw'][index])
        o.write(_same_width(h['e_raw'], stamp)); o.write(_same_width(h['l_raw'], stamp)); o.write(h['finalized'])
        o.write(h['rest'])
        o.write(bytes.fromhex(snaps[index]['header'])); o.write(mp._interleave(payload))
    os.replace(tmp_out, out)
    return dict(path=out, index=index, stamp_ns=stamp, chain=index - k + 1, seconds=round(time.time() - t0, 2),
                size_mb=round(os.path.getsize(out) / 1e6, 1))


def _is_zpk(src):
    with open(src, 'rb') as f: return f.read(len(mp.MAGIC)) == mp.MAGIC


def stamps(src):
    return mp.read_manifest(src)['source']['stamps'] if _is_zpk(src) else mp.index_v1(src)['stamps']


def cut_many(src, indices, out=None, work=SHM):
    """One sparse .4dmap with the given moments (sorted, unique). Packed input: each moment decodes from its keyframe;
    moments in the same keyframe segment share the walk. Plain input: the snapshot bytes are copied by seeking."""
    t0 = time.time(); indices = sorted(set(indices))
    if _is_zpk(src):
        man = mp.read_manifest(src); snaps = man['snapshots']; K = man.get('keyframe', 10)
        with open(src, 'rb') as f:
            f.seek(man['prefix']['offset']); prefix = mp._read(f, man['prefix']['length'])
        heads = [bytes.fromhex(snaps[i]['header']) for i in indices]
    else:
        idx = mp.index_v1(src)
        with open(src, 'rb') as f: prefix = mp._read(f, idx['snaps'][0][0])
    h = _header_parts(prefix); st = [h['stamps'][i] for i in indices]
    out = out or os.path.join(work, '%s_%s.4dmap' % (hashlib.sha1(os.path.abspath(src).encode()).hexdigest()[:12],
                                                     hashlib.sha1(str(indices).encode()).hexdigest()[:8]))
    tmp_out = out + '.part'
    with open(tmp_out, 'wb') as o:
        o.write(h['v_raw']); o.write(h['fin_inc']); o.write(_same_width(h['n_raw'], len(indices)))
        o.write(bytes([mp.ARR32_TAG]) + struct.pack('<I', len(indices)))
        for i in indices: o.write(h['stamp_raw'][i])
        o.write(_same_width(h['e_raw'], st[0])); o.write(_same_width(h['l_raw'], st[-1])); o.write(h['finalized'])
        o.write(h['rest'])
        if _is_zpk(src):
            with open(src, 'rb') as f, tempfile.TemporaryDirectory(dir=work) as tmp:
                prev = os.path.join(tmp, 'prev.bin'); have = None; payload = None
                for n_, i in enumerate(indices):
                    k = i - i % K
                    while snaps[k]['kind'] != 'key': k -= 1
                    j = have + 1 if (have is not None and k <= have < i) else k   # continue the walk inside a segment
                    if have == i: j = i + 1
                    for jj in range(j, i + 1):
                        payload = _decode(f, snaps[jj], prev)
                        if jj < i or n_ + 1 < len(indices):
                            with open(prev, 'wb') as p: p.write(payload)
                        have = jj
                    o.write(heads[n_]); o.write(mp._interleave(payload))
        else:
            with open(src, 'rb') as f:
                for i in indices:
                    off, length = idx['snaps'][i]; f.seek(off); left = 5 + 2 * length
                    while left: b = f.read(min(left, 1 << 26)); o.write(b); left -= len(b)
    os.replace(tmp_out, out)
    return dict(path=out, indices=indices, stamps_ns=st, seconds=round(time.time() - t0, 2), size_mb=round(os.path.getsize(out) / 1e6, 1))


def pick(zpk, spec, origin_ns=0):
    stamps_ = stamps(zpk)
    if spec == 'final': return len(stamps_) - 1
    if spec.startswith('t='):
        q = (origin_ns or stamps_[0]) + int(float(spec[2:]) * 1e9)
        return max([i for i, s in enumerate(stamps_) if s <= q] or [0])
    return int(spec)


def _pick_old(zpk, spec, origin_ns=0):
    man = mp.read_manifest(zpk); stamps = man['source']['stamps']
    if spec == 'final': return len(stamps) - 1
    if spec.startswith('t='):
        q = (origin_ns or stamps[0]) + int(float(spec[2:]) * 1e9)
        return max([i for i, s in enumerate(stamps) if s <= q] or [0])
    return int(spec)


if __name__ == '__main__':
    a = sys.argv[1:]
    if len(a) < 2: sys.exit(__doc__)
    src = a[0] if (a[0].endswith('.zpk') or os.path.exists(a[0])) else a[0] + '.zpk'
    origin = int(a[a.index('--origin-ns') + 1]) if '--origin-ns' in a else 0
    out = a[a.index('--out') + 1] if '--out' in a else None
    ix = [pick(src, x, origin) for x in a[1].split(',')]
    print(json.dumps(cut_many(src, ix, out)))
