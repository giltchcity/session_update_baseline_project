"""A point map: surface elements hashed per (voxel, physical identity), the simplest backend.

Every integrated pixel (dynamic classes excepted) is back-projected and averaged into the element
of its 2 cm voxel and its physical identity (0 = background; objects never share an element with
background), with a normal from depth differences and the pixel's semantic label.

Own update (rows 1 and 3): free-space carving, the representation-level change handling of point
maps (DynaMem-style clearing): before a frame is added, every element it sees through (measured
range beyond the element by more than the 5 cm sensor tolerance, facing the camera within 60 deg
when it has a normal) is removed.

Across sessions end_session() hands over the alive elements; start_session() re-inserts them,
born at the previous session's final snapshot. A retired element's voxel starts a new element
(new id) when the surface is measured there again.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch

from ..eval.scenelist import ElementLifetimes
from ..frames import Frame, SessionSpec, dynamic_mask, torch_backproject, torch_pixel_normals
from ..interface import Backend, DatasetInfo, Elements

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OPEN = int(np.iinfo(np.int64).max)


class ElementStore:
    """Surface elements voxel-hashed per identity, with lifetimes (GPU)."""

    COLS = (("sum", (3,), torch.float64, 0), ("cnt", (), torch.float64, 0), ("nrm", (3,), torch.float64, 0),
            ("label", (), torch.int64, 0), ("birth", (), torch.int64, 0), ("last_seen", (), torch.int64, 0),
            ("death", (), torch.int64, OPEN), ("identity", (), torch.int64, 0))

    def __init__(self, voxel: float):
        self.voxel = voxel
        self.keys = torch.zeros(0, dtype=torch.int64, device=DEV)      # sorted
        self.order = torch.zeros(0, dtype=torch.int64, device=DEV)     # element index per sorted key
        self.n = 0
        for name, shape, dtype, fill in self.COLS:
            setattr(self, name, torch.full((1 << 16,) + shape, fill, dtype=dtype, device=DEV))

    def _grow(self, need: int) -> None:
        cap = len(self.cnt)
        if need <= cap:
            return
        new = max(need, cap * 2)
        for name, shape, dtype, fill in self.COLS:
            a = getattr(self, name)
            b = torch.full((new,) + shape, fill, dtype=dtype, device=DEV)
            b[:cap] = a
            setattr(self, name, b)

    def encode(self, pts: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        # 18 bits per axis (+-2.6 km at 2 cm) and the identity (< 512) in the top 9 bits
        k = torch.floor(pts / self.voxel).to(torch.int64) + (1 << 17)
        return (k[:, 0] << 36) | (k[:, 1] << 18) | k[:, 2] | (identity.to(torch.int64) << 54)

    def _lookup(self, keys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(element index or -1, position in the sorted key table) per key."""
        n = len(self.keys)
        if n == 0:
            return torch.full_like(keys, -1), torch.zeros_like(keys)
        pos = torch.searchsorted(self.keys, keys).clamp(max=n - 1)
        hit = self.keys[pos] == keys
        return torch.where(hit, self.order[pos], torch.full_like(keys, -1)), pos

    def add(self, pts: torch.Tensor, nrm: torch.Tensor, lab: torch.Tensor, identity: torch.Tensor,
            stamp: int) -> None:
        if len(pts) == 0:
            return
        if int(identity.max()) >= 512:
            raise ValueError("identity >= 512 does not fit the element key")
        keys = self.encode(pts, identity)
        uk, inv = torch.unique(keys, return_inverse=True)
        idx, pos = self._lookup(uk)
        known = idx >= 0
        dead = known.clone()
        dead[known] = self.death[idx[known]] != OPEN     # the surface came back: a new element
        new = ~known | dead
        n_new = int(new.sum())
        if n_new:
            self._grow(self.n + n_new)
            fresh = torch.arange(self.n, self.n + n_new, device=DEV)
            idx[new] = fresh
            self.birth[fresh] = stamp
            self.death[fresh] = OPEN
            self.n += n_new
            self.order[pos[dead]] = idx[dead]
            fresh_keys = ~known
            if fresh_keys.any():
                self.keys = torch.cat([self.keys, uk[fresh_keys]])
                self.order = torch.cat([self.order, idx[fresh_keys]])
                s = torch.argsort(self.keys)
                self.keys, self.order = self.keys[s], self.order[s]
        el = idx[inv]
        self.identity[el] = identity.to(torch.int64)
        ok = torch.isfinite(nrm).all(dim=1)
        self.sum.index_add_(0, el, pts.to(torch.float64))
        self.cnt.index_add_(0, el, torch.ones(len(el), dtype=torch.float64, device=DEV))
        self.nrm.index_add_(0, el[ok], nrm[ok].to(torch.float64))
        last = torch.full((len(uk),), -1, dtype=torch.int64, device=DEV)
        last.scatter_reduce_(0, inv, torch.arange(len(inv), device=DEV), reduce="amax")
        self.label[idx] = lab.to(torch.int64)[last]                  # the latest observation's label
        self.last_seen[idx] = stamp

    def alive(self) -> torch.Tensor:
        return torch.nonzero(self.death[: self.n] == OPEN).squeeze(1)

    def xyz(self, ids: torch.Tensor) -> torch.Tensor:
        return (self.sum[ids] / self.cnt[ids, None]).to(torch.float32)

    def normals(self, ids: torch.Tensor) -> torch.Tensor:
        v = self.nrm[ids]
        length = torch.linalg.norm(v, dim=1, keepdim=True)
        return torch.where(length > 1e-9, v / length, torch.full_like(v, float("nan"))).to(torch.float32)


class PointBackend(Backend):
    name = "points"
    CHANGES = ("own implementation, not a published mapper: 2 cm points per identity; own update = "
               "free-space carving",)

    def __init__(self, info: DatasetInfo, own_update: bool, voxel: float = 0.02,
                 tolerance: float = 0.05, max_incidence_deg: float = 60.0, work_dir=None):
        super().__init__(info, own_update, work_dir)
        self.voxel = voxel
        self.tolerance = tolerance                        # the sensor tolerance of the evidence
        self.min_cos = math.cos(math.radians(max_incidence_deg))
        self.store: Optional[ElementStore] = None
        self.stamps = []

    def start_session(self, spec: SessionSpec, prior: Optional[dict]) -> None:
        self.store = ElementStore(self.voxel)
        self.stamps = []
        self.carved = 0
        if prior is not None and len(prior["xyz"]):
            t = lambda a, d=torch.float32: torch.as_tensor(a, dtype=d, device=DEV)
            self.store.add(t(prior["xyz"]), t(prior["normal"]), t(prior["label"], torch.int64),
                           t(prior["identity"], torch.int64), int(prior["final_stamp"]))

    def integrate(self, frame: Frame) -> None:
        depth = torch.as_tensor(frame.depth, dtype=torch.float32, device=DEV)
        inst = torch.as_tensor(frame.instance, dtype=torch.int64, device=DEV)
        T = torch.as_tensor(frame.T_world_cam, dtype=torch.float32, device=DEV)
        if self.own_update:
            self._carve(frame, depth, T)
        keep = torch.isfinite(depth) & ~torch.as_tensor(dynamic_mask(frame, self.info.dynamic_semantics),
                                                        device=DEV)
        normals = torch_pixel_normals(depth, frame.K, T[:3, :3])
        pts, v, u = torch_backproject(depth, frame.K, T, keep)
        lab = torch.as_tensor(frame.semantic, dtype=torch.int64, device=DEV)[v, u] \
            if frame.semantic is not None else torch.zeros_like(v)
        self.store.add(pts, normals[v, u], lab, inst[v, u].clamp(min=0), frame.stamp_ns)

    def _carve(self, frame: Frame, depth: torch.Tensor, T: torch.Tensor) -> None:
        ids = self.store.alive()
        if not len(ids):
            return
        K = frame.K
        xyz = self.store.xyz(ids)
        R, t = T[:3, :3], T[:3, 3]
        cam = (xyz - t) @ R                                  # world -> camera
        z = cam[:, 2]
        uf = K.fx * cam[:, 0] / z + K.cx - K.offset + 0.5
        vf = K.fy * cam[:, 1] / z + K.cy - K.offset + 0.5
        ok = (z > 0) & torch.isfinite(uf) & torch.isfinite(vf)
        u = torch.where(ok, torch.floor(uf), torch.zeros_like(uf)).to(torch.int64)
        v = torch.where(ok, torch.floor(vf), torch.zeros_like(vf)).to(torch.int64)
        ok &= (u >= 0) & (u < K.width) & (v >= 0) & (v < K.height)
        d = torch.where(ok, depth[v.clamp(0, K.height - 1), u.clamp(0, K.width - 1)],
                        torch.full_like(z, float("nan")))
        query = torch.linalg.norm(cam, dim=1)
        measured = d * query / z                            # measured range along the element's ray
        nrm = self.store.normals(ids)
        has_n = torch.isfinite(nrm).all(dim=1)
        view = (xyz - t) / query[:, None]
        facing = ~has_n | (torch.abs((torch.nan_to_num(nrm) * view).sum(-1)) >= self.min_cos)
        through = ok & torch.isfinite(measured) & (measured - query > self.tolerance) & facing
        self.store.death[ids[through]] = frame.stamp_ns
        self.carved += int(through.sum())

    def elements(self) -> Elements:
        ids = self.store.alive()
        return Elements(ids, self.store.xyz(ids), self.store.normals(ids), self.store.identity[ids],
                        self.store.last_seen[ids])

    def retire(self, ids: torch.Tensor, stamp: int) -> None:
        if len(ids):
            ids = ids[self.store.death[ids] == OPEN]
            self.store.death[ids] = stamp

    def snapshot(self, stamp: int) -> None:
        self.stamps.append(int(stamp))       # element lifetimes carry the rest

    def timeline(self) -> ElementLifetimes:
        s, n = self.store, self.store.n
        ids = torch.arange(n, device=DEV)
        np_ = lambda a: a.detach().cpu().numpy()
        return ElementLifetimes(self.stamps, np_(s.xyz(ids)), np_(s.normals(ids)), np_(s.label[:n]),
                                np_(s.identity[:n]), np_(s.birth[:n]), np_(s.death[:n]))

    def end_session(self) -> dict:
        ids = self.store.alive()
        np_ = lambda a: a.detach().cpu().numpy()
        return dict(xyz=np_(self.store.xyz(ids)), normal=np_(self.store.normals(ids)),
                    label=np_(self.store.label[ids]), identity=np_(self.store.identity[ids]),
                    final_stamp=self.stamps[-1])
