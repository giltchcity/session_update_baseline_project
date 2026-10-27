"""Reader for export_cleanup_binary output."""
from pathlib import Path
import struct
import numpy as np

MAGIC = b'ABCCLN1\0'

def read_binary(path):
    path = Path(path)
    with path.open('rb') as f:
        magic = f.read(8)
        if magic != MAGIC:
            raise ValueError(f'bad magic in {path}: {magic!r}')
        n_steps = struct.unpack('<I', f.read(4))[0]
        snapshots = []
        for _ in range(n_steps):
            index = struct.unpack('<I', f.read(4))[0]
            stamp = struct.unpack('<Q', f.read(8))[0]
            n_events = struct.unpack('<I', f.read(4))[0]
            events = {}
            for _ in range(n_events):
                elen = struct.unpack('<I', f.read(4))[0]
                eid = f.read(elen).decode('utf-8')
                n = struct.unpack('<I', f.read(4))[0]
                dt = np.dtype([('gd', '<f4'), ('gl', '<u4'), ('od', '<f4'), ('oi', '<u4')])
                raw = np.frombuffer(f.read(dt.itemsize * n), dtype=dt)
                events[eid] = dict(global_d=raw['gd'].copy(), global_label=raw['gl'].copy(),
                                   obj_d=raw['od'].copy(), obj_owner=raw['oi'].copy())
            n_owners = struct.unpack('<I', f.read(4))[0]
            owners = []
            for _ in range(n_owners):
                olen = struct.unpack('<I', f.read(4))[0]
                oid = f.read(olen).decode('utf-8')
                sem = struct.unpack('<I', f.read(4))[0]
                owners.append(dict(id=oid, semantic=sem))
            snapshots.append(dict(index=index, stamp_ns=stamp, events=events, owners=owners))
    return snapshots

if __name__ == '__main__':
    import sys
    snap = read_binary(sys.argv[1])
    print('snapshots', len(snap))
    for s in snap[:2] + snap[-1:]:
        print('index', s['index'], 'stamp', s['stamp_ns'], 'events', len(s['events']), 'owners', len(s['owners']))
        for eid, e in list(s['events'].items())[:2]:
            print(' ', eid, 'n', len(e['global_d']), 'global<=.1', float((e['global_d'] <= .10).mean()),
                  'obj<=.1', float((e['obj_d'] <= .10).mean()), 'owners', len(set(e['obj_owner'].tolist())))
