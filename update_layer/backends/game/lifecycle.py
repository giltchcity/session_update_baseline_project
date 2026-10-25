"""Bounded, UID-based suppression intervals for reversible element evidence.

Intervals are independent of the permanent T1 state/evidence/prune lifetimes. Each
event uses three int64 values; no per-keyframe or N-by-history dense table is kept.
Only transient UID lookup indices and one current N-element boolean mask are cached.
"""
from __future__ import annotations

from numbers import Integral

import torch


INT64_MAX = torch.iinfo(torch.int64).max
DEFAULT_HISTORY_BYTES = 64 * 1024 * 1024
EVENT_BYTES = 3 * 8


def _finite_stamp(stamp) -> int:
    if isinstance(stamp, bool) or not isinstance(stamp, Integral):
        raise ValueError("suppression timestamp must be an integer")
    stamp = int(stamp)
    if not torch.iinfo(torch.int64).min <= stamp < INT64_MAX:
        raise ValueError("suppression timestamp is outside the finite int64 range")
    return stamp


def recoverable_lifetime(base_alive, identity, death_state, death_evidence, death_prune):
    """A reversible observation cannot override a permanent end or a known closed object state."""
    return (base_alive & (death_evidence == INT64_MAX) & (death_prune == INT64_MAX)
            & ((identity <= 0) | (death_state == INT64_MAX)))


def apply_surface_weights(row_uids, surface_weights, ids, weights) -> int:
    """Write canonical signed confidence by stable UID, with validation before mutation."""
    ids = torch.as_tensor(ids, dtype=torch.int64, device=row_uids.device).reshape(-1)
    weights = torch.as_tensor(weights, dtype=torch.float32, device=row_uids.device).reshape(-1)
    if len(ids) != len(weights):
        raise ValueError("surface weight UID/value counts differ")
    if bool((~torch.isfinite(weights) | (weights < -1) | (weights > 8)).any()):
        raise ValueError("surface weights must be finite and within [-1, 8]")
    if not len(ids):
        return 0
    if len(ids) > 1 and not bool((ids[1:] > ids[:-1]).all()):
        ids, order = torch.sort(ids)
        weights = weights[order]
        if bool((ids[1:] == ids[:-1]).any()):
            raise ValueError("surface weight updates contain duplicate UIDs")
    pos = torch.searchsorted(ids, row_uids)
    matched = pos < len(ids)
    matched &= ids[pos.clamp(max=len(ids) - 1)] == row_uids
    surface_weights[matched] = weights[pos[matched]]
    return int(matched.sum())


class SuppressionHistory:
    def __init__(self, device="cpu", max_bytes: int = DEFAULT_HISTORY_BYTES):
        self.max_bytes = int(max_bytes)
        if self.max_bytes < EVENT_BYTES:
            raise ValueError("suppression history budget must hold at least one interval")
        self.uids = torch.empty(0, dtype=torch.int64, device=device)
        self.starts = torch.empty_like(self.uids)
        self.ends = torch.empty_like(self.uids)
        self.last_stamp = None
        self._open_count = 0
        self._version = 0
        self._index_key = self._indexed_uids = self._sorted_uids = self._order = None
        self._current_key = self._current_mask = None

    @property
    def nbytes(self) -> int:
        return len(self.uids) * EVENT_BYTES

    @property
    def open_count(self) -> int:
        return self._open_count

    def _ids(self, ids):
        ids = torch.as_tensor(ids, dtype=torch.int64, device=self.uids.device).reshape(-1)
        ids = torch.unique(ids, sorted=True)
        if len(ids) and bool((ids < 0).any()):
            raise ValueError("suppression UIDs must be nonnegative")
        return ids

    def _stamp(self, stamp: int) -> int:
        stamp = _finite_stamp(stamp)
        if self.last_stamp is not None and stamp < self.last_stamp:
            raise ValueError("suppression events must be applied in timestamp order")
        return stamp

    def ensure_capacity(self, additional_events: int) -> None:
        required = (len(self.uids) + int(additional_events)) * EVENT_BYTES
        if required > self.max_bytes:
            raise RuntimeError(f"suppression history budget exceeded: {required} bytes required, "
                               f"limit {self.max_bytes}; no history was discarded")

    def _changed(self) -> None:
        self._version += 1
        self._current_key = self._current_mask = None

    def append_events(self, events) -> None:
        uids, starts, ends = events
        self.ensure_capacity(len(uids))
        if not len(uids):
            return
        self.uids = torch.cat((self.uids, uids.to(self.uids.device)))
        self.starts = torch.cat((self.starts, starts.to(self.starts.device)))
        self.ends = torch.cat((self.ends, ends.to(self.ends.device)))
        self._open_count += int((ends == INT64_MAX).sum())
        latest = int(starts.max())
        finite_ends = ends[ends < INT64_MAX]
        if len(finite_ends):
            latest = max(latest, int(finite_ends.max()))
        self.last_stamp = latest if self.last_stamp is None else max(self.last_stamp, latest)
        self._changed()

    def suppress(self, ids, stamp: int) -> int:
        """Open an interval once; repeated suppression leaves its original start intact."""
        stamp, ids = self._stamp(stamp), self._ids(ids)
        if self._open_count and len(ids):
            ids = ids[~torch.isin(ids, self.uids[self.ends == INT64_MAX], assume_unique=True)]
        self.ensure_capacity(len(ids))
        if len(ids):
            self.append_events((ids, torch.full_like(ids, stamp), torch.full_like(ids, INT64_MAX)))
            self.last_stamp = stamp
        return len(ids)

    def reactivate(self, ids, stamp: int) -> int:
        """Close only open suppression intervals; permanent lifetimes are the caller's guard."""
        stamp, ids = self._stamp(stamp), self._ids(ids)
        if not len(ids) or not self._open_count:
            return 0
        mask = (self.ends == INT64_MAX) & torch.isin(self.uids, ids)
        changed = int(mask.sum())
        if changed:
            self.ends[mask] = stamp
            self._open_count -= changed
            empty = mask & (self.starts == stamp)
            if bool(empty.any()):
                # A low-resolution rejection and same-keyframe correction have no historical gap.
                keep = ~empty
                self.uids, self.starts, self.ends = self.uids[keep], self.starts[keep], self.ends[keep]
            self.last_stamp = stamp
            self._changed()
        return changed

    def close_for_permanent_end(self, ids, processing_stamp: int) -> int:
        """End the operational gap without backdating an event before its opening.

        A G5/state closure may set the permanent T1 end earlier than the observations
        that opened this gap. The base lifetime supplies that retroactive exclusion;
        this metadata is closed at the current processing clock and never reopens it.
        """
        stamp = _finite_stamp(processing_stamp)
        if self.last_stamp is not None:
            stamp = max(stamp, self.last_stamp)
        return self.reactivate(ids, stamp)

    def _index(self, row_uids):
        key = (id(row_uids), row_uids.data_ptr(), len(row_uids), row_uids._version)
        if key != self._index_key:
            self._index_key, self._indexed_uids = key, row_uids
            if len(row_uids) < 2 or bool((row_uids[1:] > row_uids[:-1]).all()):
                self._sorted_uids, self._order = row_uids, None
            else:
                self._sorted_uids, self._order = torch.sort(row_uids)
                if len(row_uids) > 1 and bool((self._sorted_uids[1:] == self._sorted_uids[:-1]).any()):
                    raise ValueError("one Gaussian container contains duplicate stable UIDs")
        return key

    def _mask_for(self, row_uids, selected_uids):
        mask = torch.zeros_like(row_uids, dtype=torch.bool)
        if not len(row_uids) or not len(selected_uids):
            return mask
        self._index(row_uids)
        pos = torch.searchsorted(self._sorted_uids, selected_uids)
        valid = pos < len(row_uids)
        valid &= self._sorted_uids[pos.clamp(max=len(row_uids) - 1)] == selected_uids
        pos = pos[valid]
        rows = pos if self._order is None else self._order[pos]
        mask[rows] = True
        return mask

    def current_mask(self, row_uids):
        key = ((id(row_uids), row_uids.data_ptr(), len(row_uids), row_uids._version), self._version)
        if key != self._current_key:
            self._current_mask = self._mask_for(row_uids, self.uids[self.ends == INT64_MAX])
            self._current_key = key
        return self._current_mask

    def mask_at(self, row_uids, stamp: int):
        stamp = _finite_stamp(stamp)
        if self.last_stamp is None or stamp >= self.last_stamp:
            return self.current_mask(row_uids)
        selected = (self.starts <= stamp) & (stamp < self.ends)
        return self._mask_for(row_uids, self.uids[selected])

    def prepare_inheritance(self, parent_uids, child_uids):
        """Copy parents' historical intervals to distinct child UIDs before a model mutation.

        A split may list the same parent several times. The vectorized range join
        produces one copy per child without an events-by-children matrix.
        """
        if len(parent_uids) != len(child_uids):
            raise ValueError("parent and child UID counts differ")
        empty = self.uids[:0]
        if not len(parent_uids) or not len(self.uids):
            return empty, empty, empty
        parents, order = torch.sort(parent_uids.to(self.uids.device))
        children = child_uids.to(self.uids.device)
        lo = torch.searchsorted(parents, self.uids)
        hi = torch.searchsorted(parents, self.uids, right=True)
        repeats = hi - lo
        total = int(repeats.sum())
        self.ensure_capacity(total)
        if not total:
            return empty, empty, empty
        event = torch.repeat_interleave(torch.arange(len(self.uids), device=self.uids.device), repeats)
        offsets = torch.arange(total, device=self.uids.device) - torch.repeat_interleave(
            torch.cumsum(repeats, 0) - repeats, repeats)
        child = order[lo[event] + offsets]
        return children[child], self.starts[event], self.ends[event]

    def selected_events(self, ids):
        ids = self._ids(ids)
        keep = torch.isin(self.uids, ids)
        return self.uids[keep], self.starts[keep], self.ends[keep]

    def drop_uids(self, ids) -> None:
        """Forget only physically removed rows; surviving UID histories need no remapping."""
        ids = self._ids(ids)
        if not len(ids) or not len(self.uids):
            return
        keep = ~torch.isin(self.uids, ids)
        if bool(keep.all()):
            return
        self.uids, self.starts, self.ends = self.uids[keep], self.starts[keep], self.ends[keep]
        self._open_count = int((self.ends == INT64_MAX).sum())
        self._changed()

    def state_dict(self):
        return dict(version=1, uids=self.uids.cpu(), starts=self.starts.cpu(), ends=self.ends.cpu(),
                    last_stamp=self.last_stamp, max_bytes=self.max_bytes)

    @classmethod
    def from_state(cls, state, device="cpu", max_bytes: int = DEFAULT_HISTORY_BYTES):
        history = cls(device=device, max_bytes=min(int(state.get("max_bytes", max_bytes)), int(max_bytes)))
        if int(state.get("version", 1)) != 1:
            raise ValueError("unsupported suppression history version")
        def column(name):
            value = state.get(name, [])
            if torch.is_tensor(value):
                if value.dtype != torch.int64 or value.ndim != 1:
                    raise ValueError(f"suppression history {name} must be a one-dimensional int64 tensor")
                return value.to(device=device)
            if not isinstance(value, (tuple, list)) or any(type(x) is not int or
                    not torch.iinfo(torch.int64).min <= x <= INT64_MAX for x in value):
                raise ValueError(f"suppression history {name} must contain int64 integers")
            return torch.tensor(value, dtype=torch.int64, device=device)
        uids, starts, ends = column("uids"), column("starts"), column("ends")
        if len(uids) != len(starts) or len(starts) != len(ends):
            raise ValueError("suppression history column lengths differ")
        if bool(((uids < 0) | (starts > ends) | (starts == INT64_MAX)).any()):
            raise ValueError("invalid suppression history interval")
        history.append_events((uids, starts, ends))
        latest_event = history.last_stamp
        saved_stamp = state.get("last_stamp", latest_event)
        if saved_stamp is not None:
            saved_stamp = _finite_stamp(saved_stamp)
        if latest_event is not None and (saved_stamp is None or saved_stamp < latest_event):
            raise ValueError("suppression history has an inconsistent latest event timestamp")
        history.last_stamp = saved_stamp
        if len(uids):
            # Refuse overlapping intervals of one UID, including duplicate open intervals.
            order = torch.argsort(starts, stable=True)
            order = order[torch.argsort(uids[order], stable=True)]
            same_uid = uids[order][1:] == uids[order][:-1]
            if bool((same_uid & (starts[order][1:] < ends[order][:-1])).any()):
                raise ValueError("overlapping suppression intervals for one UID")
        return history
