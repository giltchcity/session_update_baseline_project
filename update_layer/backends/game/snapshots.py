"""The 4D map as incremental snapshots (user 10-22, as Khronos keeps its map over time; the user's incremental compressed
storage of it): at every round the map as it is then, the Gaussians the readout draws (alive at the round) with all
their parameters and identity, is written as the change since the previous round: Gaussians that left, new Gaussians,
and changed Gaussians as the bitwise XOR of their parameters against their previous values (byte-shuffled, zlib), so
the map of any round is rebuilt exactly. Each session starts with a full record. The map of time t is the map of the
last round at or before t: what the map was then, not a reconstruction from the final map.
  SnapshotLog(dir).record(stamp, uid, params, identity, base)   (during mapping, the rows only read)
  state_at(dir, t) -> (stamp, uid, params, identity)              (any time t)"""
from __future__ import annotations

import io
import json
import zlib
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

FIELDS = (("xyz", 3), ("dc", 3), ("rest", 45), ("scaling", 3), ("rotation", 4), ("opacity", 1))
WIDTH = sum(w for _, w in FIELDS)                    # 59 float32 per Gaussian


def _pack(arrays: Dict[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return zlib.compress(buf.getvalue(), 6)


def _unpack(blob: bytes) -> Dict[str, np.ndarray]:
    with np.load(io.BytesIO(zlib.decompress(blob))) as z:
        return {k: z[k] for k in z.files}


def _shuffle(bits: np.ndarray) -> np.ndarray:
    """(n, WIDTH) uint32 -> its bytes grouped by significance (as Blosc's shuffle): XOR deltas of slightly changed
    floats are zero in their high bytes, which then compress well."""
    return np.ascontiguousarray(bits.view(np.uint8).reshape(-1, 4).T)


def _unshuffle(planes: np.ndarray, n: int) -> np.ndarray:
    return np.ascontiguousarray(planes.T).reshape(-1).view(np.uint32).reshape(n, WIDTH)


class SnapshotLog:
    """Writes one file per round under `root` (00001_<stamp>.snap ...) and an index.json; keeps the previous round's
    map in memory (CPU) to take the change against."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        idx = self.root / "index.json"
        self.index = json.loads(idx.read_text()) if idx.exists() else []
        self.prev: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None      # uid (sorted), bits (n, WIDTH), identity

    def record(self, stamp: int, uid: np.ndarray, params: np.ndarray, identity: np.ndarray, base: bool) -> dict:
        """params: (n, WIDTH) float32 in FIELDS order; rows in any order (sorted here by uid)."""
        o = np.argsort(uid, kind="stable")
        uid, bits, identity = uid[o].astype(np.int64), np.ascontiguousarray(params[o], dtype=np.float32).view(np.uint32), identity[o].astype(np.int64)
        if base or self.prev is None:
            rec = {"kind": "base", "uid": uid, "planes": _shuffle(bits), "identity": identity}
        else:
            pu, pb, pi = self.prev
            keep_new = np.isin(uid, pu, assume_unique=True)
            keep_old = np.isin(pu, uid, assume_unique=True)
            # rows in both rounds, in the same uid order on both sides
            cb, ci = bits[keep_new], identity[keep_new]
            ob, oi = pb[keep_old], pi[keep_old]
            x = cb ^ ob
            changed = (x != 0).any(axis=1) | (ci != oi)
            rec = {"kind": "delta", "removed": pu[~keep_old],
                   "added_uid": uid[~keep_new], "added_planes": _shuffle(bits[~keep_new]), "added_identity": identity[~keep_new],
                   "changed_uid": uid[keep_new][changed], "changed_planes": _shuffle(x[changed]),
                   "changed_identity": ci[changed]}
        blob = _pack(rec)
        name = f"{len(self.index) + 1:05d}_{int(stamp)}.snap"
        tmp = self.root / (name + ".tmp")
        tmp.write_bytes(blob)
        tmp.replace(self.root / name)
        info = {"file": name, "stamp": int(stamp), "kind": rec["kind"], "rows": int(len(uid)), "bytes": len(blob)}
        if rec["kind"] == "delta":
            info.update(removed=int(len(rec["removed"])), added=int(len(rec["added_uid"])), changed=int(len(rec["changed_uid"])))
        self.index.append(info)
        (self.root / "index.json").write_text(json.dumps(self.index, indent=0))
        self.prev = (uid, bits, identity)
        return info


def state_at(root: Path, t: int):
    """(stamp, uid, params (n, WIDTH) float32, identity) of the last recorded round at or before t; None before the
    first one. Records at one stamp (a round and the session's end) apply in order: the later one is the map of t."""
    root = Path(root)
    index = json.loads((root / "index.json").read_text())
    upto = [k for k, r in enumerate(index) if r["stamp"] <= t]
    if not upto:
        return None
    last = upto[-1]
    start = max(k for k in range(last + 1) if index[k]["kind"] == "base")
    uid = bits = ident = None
    for k in range(start, last + 1):
        rec = _unpack((root / index[k]["file"]).read_bytes())
        if rec["kind"] == "base":
            uid, ident = rec["uid"], rec["identity"]
            bits = _unshuffle(rec["planes"], len(uid))
            continue
        keep = ~np.isin(uid, rec["removed"], assume_unique=True)
        uid, bits, ident = uid[keep], bits[keep], ident[keep]
        if len(rec["changed_uid"]):
            pos = np.searchsorted(uid, rec["changed_uid"])
            bits[pos] ^= _unshuffle(rec["changed_planes"], len(rec["changed_uid"]))
            ident[pos] = rec["changed_identity"]
        if len(rec["added_uid"]):
            uid = np.concatenate([uid, rec["added_uid"]])
            bits = np.concatenate([bits, _unshuffle(rec["added_planes"], len(rec["added_uid"]))])
            ident = np.concatenate([ident, rec["added_identity"]])
            o = np.argsort(uid, kind="stable")
            uid, bits, ident = uid[o], bits[o], ident[o]
    return index[last]["stamp"], uid, bits.view(np.float32), ident
