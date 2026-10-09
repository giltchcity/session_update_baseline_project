"""[I1] The per-Gaussian label-weight table in sparse (COO) form.

The table holds, per Gaussian, the sum of the FlashSplat blending weights it received under each label
(_assign_identity). A Gaussian is seen under 1-2 labels (checkpoints 10-09: 1.75 non-zeros per row on the synthetic
map with 149 labels, 1.86 on real B with 18), so the dense N x L float32 table was 596 B/row on the synthetic map
(1.77 GB at 2.97 M Gaussians) and was copied whole at every densification and prune - both synthetic B runs that
died in the GPU driver (10-07 v5, 10-09 syn_row4i) died in exactly that copy. This class stores the non-zero
entries only (rows, cols, values; ~20 B per entry) and gives the same values: sums are formed from the same two
operands (the stored sum and the render's count), a row's best label is the first column of its largest entry (the
dense torch.max(dim=1) rule), a row with no entry has best 0 and label index 0 (the dense all-zeros row).
Checkpoints keep the CSR form _label_weight_store wrote since 10-08 (exact, unchanged format).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


class LabelTable:
    __slots__ = ("n", "m", "rows", "cols", "vals")

    def __init__(self, n: int, m: int, rows: Optional[torch.Tensor] = None, cols: Optional[torch.Tensor] = None,
                 vals: Optional[torch.Tensor] = None, device="cuda"):
        self.n, self.m = int(n), int(m)
        if vals is not None:
            device = vals.device
        self.rows = torch.empty(0, dtype=torch.int64, device=device) if rows is None else rows
        self.cols = torch.empty(0, dtype=torch.int64, device=device) if cols is None else cols
        self.vals = torch.empty(0, dtype=torch.float32, device=device) if vals is None else vals

    # --- shape protocol used by the backend -------------------------------------------------------------------------
    def __len__(self) -> int:
        return self.n

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.n, self.m)

    @property
    def device(self):
        return self.vals.device

    def new_zeros(self, shape) -> "LabelTable":
        return LabelTable(shape[0], shape[1], device=self.device)

    def nnz(self) -> int:
        return int(self.vals.numel())

    # --- construction / storage ---------------------------------------------------------------------------------------
    @classmethod
    def from_tensor(cls, t: torch.Tensor, device="cuda") -> "LabelTable":
        """From a dense or sparse (COO/CSR) 2-D tensor, zeros dropped; values exact."""
        if t.layout == torch.sparse_csr:
            t = t.to_sparse_coo()
        if t.layout == torch.sparse_coo:
            t = t.coalesce()
            idx, v = t.indices().to(device), t.values().to(device, torch.float32)
            keep = v != 0
            return cls._coalesced(t.shape[0], t.shape[1], idx[0][keep], idx[1][keep], v[keep])
        t = t.to(device)
        nz = t.nonzero(as_tuple=True)
        return cls._coalesced(t.shape[0], t.shape[1], nz[0], nz[1], t[nz].to(torch.float32))

    @classmethod
    def _coalesced(cls, n: int, m: int, rows, cols, vals) -> "LabelTable":
        """Entries sorted by (row, col) and unique: duplicates are summed (two operands at most per key when the
        callers keep the table coalesced and add a dense delta, so the sum equals the dense table's sum)."""
        if rows.numel() == 0:
            return cls(n, m, device=vals.device)
        key = rows * max(m, 1) + cols
        uniq, inv = torch.unique(key, return_inverse=True)
        v = torch.zeros(uniq.numel(), dtype=torch.float32, device=vals.device).index_add_(0, inv, vals)
        r = torch.div(uniq, max(m, 1), rounding_mode="floor")
        c = uniq - r * max(m, 1)
        keep = v != 0
        return cls(n, m, r[keep], c[keep], v[keep])

    def to_sparse_csr(self) -> torch.Tensor:
        """The checkpoint form (CSR float32 on the CPU), exact."""
        idx = torch.stack([self.rows, self.cols]).cpu()
        return torch.sparse_coo_tensor(idx, self.vals.cpu(), (self.n, self.m)).coalesce().to_sparse_csr()

    def to_dense(self) -> torch.Tensor:
        d = torch.zeros((self.n, self.m), dtype=torch.float32, device=self.device)
        if self.nnz():
            d[self.rows, self.cols] = self.vals
        return d

    # --- row operations (densify / prune) -----------------------------------------------------------------------------
    def __getitem__(self, sel) -> "LabelTable":
        """lw[keep] (bool mask: the kept rows, order preserved) or lw[par] (int64 rows, duplicates allowed: row i of
        the result is row par[i])."""
        sel = torch.as_tensor(sel, device=self.device)
        if sel.dtype == torch.bool:
            assert sel.numel() == self.n
            new_index = torch.cumsum(sel.to(torch.int64), 0) - 1
            k = sel[self.rows]
            return LabelTable(int(sel.sum()), self.m, new_index[self.rows[k]], self.cols[k], self.vals[k])
        par = sel.to(torch.int64)
        n_new = par.numel()
        if n_new == 0 or self.nnz() == 0:
            return LabelTable(n_new, self.m, device=self.device)
        count = torch.bincount(par, minlength=self.n)                    # how many new rows copy each old row
        order = torch.argsort(par, stable=True)                          # new rows grouped by their old row
        start = torch.cumsum(count, 0) - count
        rep = count[self.rows]                                           # each entry is copied rep times
        e_idx = torch.repeat_interleave(torch.arange(self.nnz(), device=self.device), rep)
        total = int(rep.sum())
        first = torch.repeat_interleave(torch.cumsum(rep, 0) - rep, rep)
        t = torch.arange(total, device=self.device) - first
        new_rows = order[start[self.rows[e_idx]] + t]
        return LabelTable._coalesced(n_new, self.m, new_rows, self.cols[e_idx], self.vals[e_idx])

    def append_rows(self, other: "LabelTable") -> "LabelTable":
        """torch.cat([lw, other], 0)."""
        assert other.m == self.m
        return LabelTable(self.n + other.n, self.m, torch.cat([self.rows, other.rows + self.n]),
                          torch.cat([self.cols, other.cols]), torch.cat([self.vals, other.vals]))

    def append_empty_rows(self, k: int) -> "LabelTable":
        return LabelTable(self.n + int(k), self.m, self.rows, self.cols, self.vals)

    def add_columns(self, k: int) -> "LabelTable":
        return LabelTable(self.n, self.m + int(k), self.rows, self.cols, self.vals)

    # --- votes --------------------------------------------------------------------------------------------------------
    def add_dense(self, delta: torch.Tensor) -> "LabelTable":
        """lw += delta for a dense (n, m) delta (the render's used_count): the result holds, per key, the stored sum
        plus the delta (one float32 addition, as the dense table does)."""
        assert tuple(delta.shape) == (self.n, self.m), (tuple(delta.shape), (self.n, self.m))
        nz = delta.nonzero(as_tuple=True)
        return LabelTable._coalesced(self.n, self.m, torch.cat([self.rows, nz[0]]), torch.cat([self.cols, nz[1]]),
                                     torch.cat([self.vals, delta[nz].to(torch.float32)]))

    def max(self, dim: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """(best, j) as torch.max(dense, dim=1): best = the largest entry of the row (0 for a row without entries,
        the dense zero), j = the first column holding it (0 for a row without entries)."""
        assert dim == 1 and self.m > 0
        best = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        if self.nnz():
            best.scatter_reduce_(0, self.rows, self.vals, "amax", include_self=True)
        j = torch.full((self.n,), self.m, dtype=torch.int64, device=self.device)
        if self.nnz():
            at = self.vals == best[self.rows]                             # entries holding their row's max
            j.scatter_reduce_(0, self.rows[at], self.cols[at], "amin", include_self=True)
        j = torch.where(j < self.m, j, torch.zeros_like(j))
        # a row whose entries are all below 0 would have dense max 0 at column 0 — entries are non-negative sums
        return best, j
