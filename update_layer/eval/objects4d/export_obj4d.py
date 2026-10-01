"""Timeline (any representation) -> OBJ4D object snapshots for objects4d build (official Object F1).

  python -m update_layer.eval.objects4d.export_obj4d TIMELINE.pkl OUT.obj4d [--voxel 0.01]

One snapshot per timeline snapshot; one object node per physical identity shown at that snapshot
(its surface samples, reduced to one per voxel, in their axis-aligned box), semantic = the scene object's class, node id =
Khronos' object symbol 'O' with the identity as index. first_observed_ns = the first snapshot of
this timeline that shows the identity (computed here for every backend, so maps whose scenes carry
no appearance time are treated alike); last_observed_ns = [UINT64_MAX] as Khronos writes for an
object still present (the official evaluator skips objects with an empty list): an object that is
gone is not in the snapshot at all (the backends delete it).
The OBJ4D layout is documented in objects4d.cpp.
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))       # the project root
from update_layer.eval.scenelist import load_timeline  # noqa: E402

OBJECT_SYMBOL = ord("O") << 56


def voxel_first(P: np.ndarray, cell: float) -> np.ndarray:
    k = np.floor(P / cell).astype(np.int64)
    _, idx = np.unique(k, axis=0, return_index=True)
    return np.sort(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("timeline")
    ap.add_argument("out")
    ap.add_argument("--voxel", type=float, default=0.01)
    a = ap.parse_args()
    tl = load_timeline(a.timeline)
    first_seen = {}
    n_obj = 0
    with open(a.out, "wb") as f:
        f.write(b"OBJ4D002")
        stamps = tl.stamps()
        f.write(struct.pack("<I", len(stamps)))
        for t in stamps:
            sc = tl.scene(t, background=False)
            objs = []
            for o in sc.objects:
                if o.instance_id <= 0 or not len(o.points):
                    continue
                P = np.asarray(o.points, np.float32)
                P = P[np.isfinite(P).all(axis=1)]
                if not len(P):
                    continue
                P = P[voxel_first(P, a.voxel)]
                first_seen.setdefault(o.instance_id, int(t))
                objs.append((o.instance_id, int(o.semantic), P))
            f.write(struct.pack("<QI", int(t), len(objs)))
            for identity, semantic, P in objs:
                lo, hi = P.min(axis=0), P.max(axis=0)
                f.write(struct.pack("<Qi", OBJECT_SYMBOL | identity, semantic))
                f.write(struct.pack("<10f", *(0.5 * (lo + hi)), *(hi - lo), 1.0, 0.0, 0.0, 0.0))
                f.write(struct.pack("<IQ", 1, first_seen[identity]))
                f.write(struct.pack("<IQ", 1, 2**64 - 1))
                f.write(struct.pack("<I", len(P)))
                f.write(np.ascontiguousarray(P, dtype="<f4").tobytes())
            n_obj += len(objs)
    print(f"wrote {len(stamps)} snapshots, {n_obj} object nodes, {len(first_seen)} identities -> {a.out}")


if __name__ == "__main__":
    main()
