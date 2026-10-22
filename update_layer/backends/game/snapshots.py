"""The 4D map as every round's live map (user 10-22: "4D 不就是加几行不同时间的么", as Khronos): during a session each
round's map, the Gaussians the readout draws at the round (all their parameters, identity, creation stamp, state birth),
is written as its change since the previous round (snapshots/<session>/rounds: the first in full, each later one as its
change; changed rows as the bitwise XOR of their float bits, byte-shuffled, zlib: lossless). The map of time t is the
map of the first round at or after t, the live map as it stood then; a session's last round is its end map (its
checkpoint's live rows).
  SnapshotLog(dir).record(stamp, uid, params, identity, created, state_birth)          (mapping: the rows only read)
  map_at(run_dir, t, cache) -> (stamp, uid, params, identity) | None                   (any time t)"""
from __future__ import annotations

import io
import json
import zlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

WIDTH = 59                                           # xyz 3, dc 3, rest 45, scaling 3, rotation 4, opacity 1 (float32)


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


INTS = ("identity", "created", "state_birth")


class _Chain:
    """A sequence of maps written as a full first record and later records as changes (one file each, index.json)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        p = self.root / "index.json"
        self.index: List[dict] = json.loads(p.read_text()) if p.exists() else []
        self.prev = None                              # uid (sorted), bits, ints of the last record

    def write(self, stamp: int, uid: np.ndarray, bits: np.ndarray, ints: Dict[str, np.ndarray]) -> dict:
        if self.prev is None:
            rec = {"kind": "base", "uid": uid, "planes": _shuffle(bits), **ints}
        else:
            pu, pb, pi = self.prev
            new = np.isin(uid, pu, assume_unique=True)
            old = np.isin(pu, uid, assume_unique=True)
            x = bits[new] ^ pb[old]
            changed = (x != 0).any(axis=1)
            for k in INTS:
                changed |= ints[k][new] != pi[k][old]
            rec = {"kind": "delta", "removed": pu[~old], "added_uid": uid[~new], "added_planes": _shuffle(bits[~new]),
                   **{"added_" + k: ints[k][~new] for k in INTS},
                   "changed_uid": uid[new][changed], "changed_planes": _shuffle(x[changed]),
                   **{"changed_" + k: ints[k][new][changed] for k in INTS}}
        blob = _pack(rec)
        name = f"{len(self.index) + 1:05d}_{int(stamp)}.snap"
        tmp = self.root / (name + ".tmp")
        tmp.write_bytes(blob)
        tmp.replace(self.root / name)
        info = {"file": name, "stamp": int(stamp), "kind": rec["kind"], "rows": int(len(uid)), "bytes": len(blob)}
        self.index.append(info)
        (self.root / "index.json").write_text(json.dumps(self.index, indent=0))
        self.prev = (uid, bits, ints)
        return info


def _replay(root: Path, upto: int, state: Optional[tuple] = None) -> tuple:
    """The map of record `upto` (0-based) of a chain: (upto, stamp, uid, bits, ints). From `state` (an earlier result
    of this chain at a record <= upto) when given, else from the chain's start."""
    index = json.loads((Path(root) / "index.json").read_text())
    if state is not None and state[0] <= upto:
        k0, _, uid, bits, ints = state
        uid, bits, ints = uid.copy(), bits.copy(), {n: v.copy() for n, v in ints.items()}
        start = k0 + 1
    else:
        uid = bits = None
        ints: Dict[str, np.ndarray] = {}
        start = 0
    for k in range(start, upto + 1):
        rec = _unpack((Path(root) / index[k]["file"]).read_bytes())
        if rec["kind"] == "base":
            uid, bits = rec["uid"], _unshuffle(rec["planes"], len(rec["uid"]))
            ints = {n: rec[n] for n in INTS}
            continue
        keep = ~np.isin(uid, rec["removed"], assume_unique=True)
        uid, bits = uid[keep], bits[keep]
        ints = {n: v[keep] for n, v in ints.items()}
        if len(rec["changed_uid"]):
            pos = np.searchsorted(uid, rec["changed_uid"])
            bits[pos] ^= _unshuffle(rec["changed_planes"], len(rec["changed_uid"]))
            for n in INTS:
                ints[n][pos] = rec["changed_" + n]
        if len(rec["added_uid"]):
            uid = np.concatenate([uid, rec["added_uid"]])
            bits = np.concatenate([bits, _unshuffle(rec["added_planes"], len(rec["added_uid"]))])
            ints = {n: np.concatenate([ints[n], rec["added_" + n]]) for n in INTS}
            o = np.argsort(uid, kind="stable")
            uid, bits = uid[o], bits[o]
            ints = {n: v[o] for n, v in ints.items()}
    return upto, index[upto]["stamp"], uid, bits, ints


class SnapshotLog:
    """One session's rounds (session_dir/rounds): every round's map as its change since the previous round."""

    def __init__(self, session_dir: Path):
        self.dir = Path(session_dir)
        self.rounds = _Chain(self.dir / "rounds")

    def record(self, stamp: int, uid: np.ndarray, params: np.ndarray, identity: np.ndarray, created: np.ndarray,
               state_birth: np.ndarray) -> dict:
        o = np.argsort(uid, kind="stable")
        ints = {"identity": identity, "created": created, "state_birth": state_birth}
        return self.rounds.write(stamp, uid[o].astype(np.int64), np.ascontiguousarray(params[o], dtype=np.float32).view(np.uint32),
                                 {k: np.asarray(v)[o].astype(np.int64) for k, v in ints.items()})


def map_at(run_dir: Path, t: int, cache: Optional[dict] = None):
    """(stamp of the round, uid, params (n, WIDTH) float32, identity) of the map of time t: the first round at or after
    t over the run's sessions (a tie: the earlier session); None after the last round. cache (a dict the caller keeps)
    holds each session's last replayed round, so queries in time order replay each change once."""
    cache = {} if cache is None else cache
    best = None
    for sdir in sorted(p for p in (Path(run_dir) / "snapshots").iterdir() if (p / "rounds" / "index.json").exists()):
        index = json.loads((sdir / "rounds" / "index.json").read_text())
        k = next((k for k, r in enumerate(index) if r["stamp"] >= t), None)
        if k is not None and (best is None or index[k]["stamp"] < best[0]):
            best = (index[k]["stamp"], sdir, k)
    if best is None:
        return None
    _, sdir, k = best
    state = _replay(sdir / "rounds", k, cache.get(str(sdir)))
    cache[str(sdir)] = state
    _, stamp, uid, bits, ints = state
    return stamp, uid, bits.view(np.float32), ints["identity"]
