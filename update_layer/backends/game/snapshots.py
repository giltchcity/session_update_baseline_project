"""The 4D map as event snapshots (user 10-22): the map is kept as it was just before each change the final map cannot
show, a state ending (something left: GaME deletes and re-trains what the camera sees through before the layer's round
ends the state). During a session every round's map (the Gaussians the readout draws then: all their parameters,
identity, creation stamp, state birth) is written as its change since the previous round (rounds/, temporary). When the
session ends, the map of the last round before each state end of the session is kept (kept/: the first in full, each
later one as its change since the one before; changed rows as the bitwise XOR of their float bits, byte-shuffled,
zlib: lossless) and the rounds are removed. The map at a session's end is its checkpoint.

The map of time t (map_at): the kept map of the first state end after t within t's session (the map just before that
change: from t to it nothing left, so it shows t's scene as fully as the session saw it), else the session's end map;
objects born after t taken out (a Gaussian's birth: the latest state of its identity born at or before its creation;
background stays).
  SnapshotLog(dir).record(stamp, uid, params, identity, created, state_birth, base)    (mapping: the rows only read)
  keep_session(session_dir, ends, start, end)                                          (session end)
  map_at(run_dir, t, end_maps) -> (stamp, uid, params, identity) | None                (any time t)"""
from __future__ import annotations

import io
import json
import shutil
import zlib
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

WIDTH = 59                                           # xyz 3, dc 3, rest 45, scaling 3, rotation 4, opacity 1 (float32)
INT64_MAX = np.iinfo(np.int64).max


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

    def write(self, stamp: int, uid: np.ndarray, bits: np.ndarray, ints: Dict[str, np.ndarray], meta: dict) -> dict:
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
        info = {"file": name, "stamp": int(stamp), "kind": rec["kind"], "rows": int(len(uid)), "bytes": len(blob), **meta}
        self.index.append(info)
        (self.root / "index.json").write_text(json.dumps(self.index, indent=0))
        self.prev = (uid, bits, ints)
        return info


def _replay(root: Path, upto: int):
    """The map of record `upto` (0-based) of a chain: (stamp, uid, bits, ints)."""
    index = json.loads((Path(root) / "index.json").read_text())
    start = max(k for k in range(upto + 1) if index[k]["kind"] == "base")
    uid = bits = None
    ints: Dict[str, np.ndarray] = {}
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
    return index[upto]["stamp"], uid, bits, ints


def _sorted(uid, params, ints):
    o = np.argsort(uid, kind="stable")
    return (uid[o].astype(np.int64), np.ascontiguousarray(params[o], dtype=np.float32).view(np.uint32),
            {k: np.asarray(v)[o].astype(np.int64) for k, v in ints.items()})


class SnapshotLog:
    """One session's rounds (session_dir/rounds, temporary): every round's map as its change since the previous round."""

    def __init__(self, session_dir: Path):
        self.dir = Path(session_dir)
        self.rounds = _Chain(self.dir / "rounds")

    def record(self, stamp: int, uid: np.ndarray, params: np.ndarray, identity: np.ndarray, created: np.ndarray,
               state_birth: np.ndarray) -> dict:
        u, b, ints = _sorted(uid, params, {"identity": identity, "created": created, "state_birth": state_birth})
        return self.rounds.write(stamp, u, b, ints, {})


def keep_session(session_dir: Path, ends: List[int], start: int, end: int) -> dict:
    """At a session's end: the map of the last round before each state end in (start, end] is kept (kept/, a chain of
    its own), the rounds are removed. Writes kept/index.json even when nothing is kept (the session's range)."""
    session_dir = Path(session_dir)
    rounds = session_dir / "rounds"
    index = json.loads((rounds / "index.json").read_text()) if (rounds / "index.json").exists() else []
    stamps = [r["stamp"] for r in index]
    pick: Dict[int, List[int]] = {}
    for d in sorted({int(d) for d in ends if start < int(d) <= end}):
        before = [k for k, s in enumerate(stamps) if s < d]
        if before:
            pick.setdefault(before[-1], []).append(d)
    kept = session_dir / "kept"
    if kept.exists():
        shutil.rmtree(kept)
    chain = _Chain(kept)
    for k in sorted(pick):
        stamp, uid, bits, ints = _replay(rounds, k)
        chain.write(stamp, uid, bits, ints, {"before_ends": pick[k]})
    (kept / "session.json").write_text(json.dumps({"start": int(start), "end": int(end), "rounds_seen": len(index),
                                                   "kept": len(pick)}, indent=0))
    if rounds.exists():
        shutil.rmtree(rounds)
    return {"rounds": len(index), "kept": len(pick), "bytes": sum(r["bytes"] for r in chain.index)}


def _births(identity: np.ndarray, created: np.ndarray, state_birth: np.ndarray) -> np.ndarray:
    """A Gaussian's birth: background -inf; an object's, the latest state of its identity born at or before its creation
    (t1's rule, applied here so that a row a later session made never shows before its own state); without a state, its
    creation."""
    out = np.full(len(identity), np.iinfo(np.int64).min, dtype=np.int64)
    obj = identity > 0
    out[obj] = created[obj]
    for i in np.unique(identity[obj]):
        m = np.nonzero(identity == i)[0]
        b = np.unique(state_birth[m][state_birth[m] < INT64_MAX])
        if not len(b):
            continue
        pos = np.searchsorted(b, created[m], side="right") - 1
        ok = pos >= 0
        out[m[ok]] = b[pos[ok]]
    return out


def map_at(run_dir: Path, t: int, end_map: Callable[[str], Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]]):
    """(stamp of the map used, uid, params (n, WIDTH) float32, identity) of the map of time t; None outside every session.
    end_map(session_name) -> (uid, params, ints) of that session's end map (its checkpoint's rows alive at its end)."""
    root = Path(run_dir) / "snapshots"
    for sdir in sorted(p for p in root.iterdir() if (p / "kept" / "session.json").exists()):
        ses = json.loads((sdir / "kept" / "session.json").read_text())
        if not ses["start"] <= t <= ses["end"]:
            continue
        index = json.loads((sdir / "kept" / "index.json").read_text()) if (sdir / "kept" / "index.json").exists() else []
        after = [(min(d for d in r["before_ends"] if d > t), k) for k, r in enumerate(index) if any(d > t for d in r["before_ends"])]
        if after:
            stamp, uid, bits, ints = _replay(sdir / "kept", min(after)[1])
        else:
            uid, params, ints = end_map(sdir.name)
            stamp, uid, bits, ints = ses["end"], *_sorted(uid, params, ints)
        born = _births(ints["identity"], ints["created"], ints["state_birth"])
        keep = born <= t
        return stamp, uid[keep], bits[keep].view(np.float32), ints["identity"][keep]
    return None
