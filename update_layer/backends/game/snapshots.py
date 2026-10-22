"""The 4D map as event snapshots (user 10-22): the map is kept as it was just before each change the final map cannot
show, a state ending (something left: GaME deletes and re-trains what the camera sees through before the layer's round
ends the state). During a session every round's map (the Gaussians the readout draws then: all their parameters,
identity, creation stamp, state birth) is written as its change since the previous round (rounds/, temporary). When the
session ends, the map of the last round before each state end of the session is kept (kept/: the first in full, each
later one as its change since the one before; changed rows as the bitwise XOR of their float bits, byte-shuffled,
zlib: lossless) and the rounds are removed, and the session's final state intervals are kept with them (the layer's
intervals as the backend last received them: they say which state ended when). The map at a session's end is its
checkpoint.

The map of time t (map_at): the rows alive at t by T1 under the session's final intervals (t1.state_membership,
t1.alive_at: born at or before t, their state not ended by t), each state taken from one map:
  - an object state standing at the session's end: from the end map (its latest, most complete version);
  - an object state that ended after t: from the map kept just before its end (the last round with it);
  - the background and any other row: from the nearest map at or after t (a kept map, else the end map), which holds
    everything observed up to t.
Rounds are 10.8 s apart and the layer may close a state rounds after its end, so the round maps alone do not separate
the states of a time between rounds; the intervals do. Not represented: a state born and ended between two rounds.
  SnapshotLog(dir).record(stamp, uid, params, identity, created, state_birth)          (mapping: the rows only read)
  keep_session(session_dir, intervals, start, end)                                     (session end)
  map_at(run_dir, t, end_map, cache) -> (stamp, uid, params, identity) | None          (any time t)"""
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


def keep_session(session_dir: Path, intervals: Dict[int, List[Tuple[int, Optional[int]]]], start: int, end: int) -> dict:
    """At a session's end: the map of the last round before each state end in (start, end] is kept (kept/, a chain of
    its own), the rounds are removed. kept/session.json holds the session's range and its final state intervals
    {identity: [(birth, end or None)]}, written even when nothing is kept."""
    session_dir = Path(session_dir)
    rounds = session_dir / "rounds"
    index = json.loads((rounds / "index.json").read_text()) if (rounds / "index.json").exists() else []
    stamps = [r["stamp"] for r in index]
    ends = [d for lst in intervals.values() for _, d in lst if d is not None]
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
                                                   "kept": len(pick),
                                                   "intervals": {str(int(i)): [[int(b), None if d is None else int(d)]
                                                                               for b, d in lst]
                                                                 for i, lst in intervals.items()}}, indent=0))
    if rounds.exists():
        shutil.rmtree(rounds)
    return {"rounds": len(index), "kept": len(pick), "bytes": sum(r["bytes"] for r in chain.index)}


def _lifetimes(ints: Dict[str, np.ndarray], intervals: Dict[int, List[Tuple[int, Optional[int]]]], start: int):
    """(birth, end) of each row of a map of the session by T1 under the session's final intervals, as the backend
    assigns them (t1.state_membership: rows of this session by their creation, rows of earlier sessions keep their
    state's birth and take the end of the state continuing at the session's start; t1.birth)."""
    import torch
    from . import t1
    ident, created, sb0 = (torch.from_numpy(np.ascontiguousarray(ints[k])) for k in INTS)
    none = torch.full_like(created, INT64_MAX)
    sb, ds = t1.state_membership(ident, created, intervals, start, prior=(sb0, none, none))
    return t1.birth(ident, created, sb).numpy(), ds.numpy()


def map_at(run_dir: Path, t: int, end_map: Callable[[str], Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]],
           cache: Optional[dict] = None):
    """(stamp of the nearest map at or after t, uid, params (n, WIDTH) float32, identity) of the map of time t; None
    outside every session. end_map(session_name) -> (uid, params, ints) of that session's end map (its checkpoint's rows
    alive at its end). cache (a dict kept by the caller) holds the maps read, with their rows' lifetimes."""
    cache = {} if cache is None else cache
    root = Path(run_dir) / "snapshots"
    for sdir in sorted(p for p in root.iterdir() if (p / "kept" / "session.json").exists()):
        ses = json.loads((sdir / "kept" / "session.json").read_text())
        if not ses["start"] <= t <= ses["end"]:
            continue
        intervals = {int(i): [(int(b), None if d is None else int(d)) for b, d in lst] for i, lst in ses["intervals"].items()}
        index = json.loads((sdir / "kept" / "index.json").read_text()) if (sdir / "kept" / "index.json").exists() else []

        def load(k):          # map k of the session (kept record k, or the end map: k == len(index)) and its lifetimes
            if (str(sdir), k) not in cache:
                if k < len(index):
                    stamp, uid, bits, ints = _replay(sdir / "kept", k)
                else:
                    stamp, (uid, bits, ints) = ses["end"], _sorted(*end_map(sdir.name))
                born, end = _lifetimes(ints, intervals, ses["start"])
                cache[(str(sdir), k)] = (stamp, uid, bits, ints["identity"], born, end)
            return cache[(str(sdir), k)]

        later = [k for k, r in enumerate(index) if r["stamp"] >= t]
        near = later[0] if later else len(index)                     # the nearest map at or after t
        kept_ends = {d: k for k, r in enumerate(index) for d in r["before_ends"]}
        ends_after = [d for d in kept_ends if d > t]
        parts = []
        for k in sorted({near, len(index)} | {kept_ends[d] for d in ends_after}):
            stamp, uid, bits, ident, born, end = load(k)
            alive, obj = (born <= t) & (t < end), ident != 0
            standing = end > ses["end"]                              # the state stands at the session's end
            if k == len(index):
                take = alive & obj & standing
            else:
                take = alive & obj & np.isin(end, [d for d in index[k]["before_ends"] if d > t])
            if k == near:
                take |= alive & ~(obj & (standing | np.isin(end, list(kept_ends))))
            parts.append((uid[take], bits[take], ident[take]))
        uid = np.concatenate([p[0] for p in parts])
        o = np.argsort(uid, kind="stable")
        return (load(near)[0], uid[o], np.concatenate([p[1] for p in parts])[o].view(np.float32),
                np.concatenate([p[2] for p in parts])[o])
    return None
