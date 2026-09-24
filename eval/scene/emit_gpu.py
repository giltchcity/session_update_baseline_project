"""GPU versions of the real-harness writers in emit_harness.py (same files, same values).

  python -m eval.scene.emit_gpu TIMELINE EXPORT_DIR STAGE RESIDUAL_PROBES MESH_PROBES BG OBJ

writes snapshots.jsonl, cleanup_probes.bin, cleanup_mesh.jsonl (not for stage a) and
surfels_final.ply like the CPU emitter in eval_real_scene.sh, with one scene per snapshot.
Nearest-sample queries run on a uniform grid on the GPU in float64 on the same float32
samples cKDTree reads as float64, so distances and chosen samples match the CPU writers
except for exact distance ties (the lowest sample index wins here).
"""
from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/jixian/Desktop/FT")
sys.path.insert(0, "/home/jixian/Desktop/FT/wt_layer_43c")
from eval.scene.emit_harness import _string, _u32, _u64, snapshot_record  # noqa: E402
from eval.scene.emit_synthetic import surfel_triangles  # noqa: E402
from eval.scene.scene import INVALID_LABEL  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_B = 1 << 20
_OFFS = torch.tensor([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
                     dtype=torch.int64, device=DEV)


def _cell_keys(cells: torch.Tensor) -> torch.Tensor:
    c = cells + _B
    return (c[:, 0] << 42) | (c[:, 1] << 21) | c[:, 2]


def first_per_voxel(points: np.ndarray, voxel: float) -> np.ndarray:
    """emit_harness.first_per_voxel: first index per voxel in input order, sorted (keys on host)."""
    if len(points) == 0:
        return np.zeros(0, dtype=np.int64)
    cells = torch.as_tensor(np.floor(points / voxel).astype(np.int64), device=DEV)
    _, inv = torch.unique(_cell_keys(cells), return_inverse=True)
    first = torch.full((int(inv.max()) + 1,), len(points), dtype=torch.int64, device=DEV)
    first.scatter_reduce_(0, inv, torch.arange(len(points), device=DEV), reduce="amin")
    return torch.sort(first).values.cpu().numpy()


class Nearest:
    """Exact nearest sample of query points (cKDTree.query semantics, float64 distances)."""

    def __init__(self, points: np.ndarray):
        self.P = torch.as_tensor(np.asarray(points, dtype=np.float64), device=DEV)
        self._grids = {}

    def _grid(self, cell: float):
        if cell not in self._grids:
            keys = _cell_keys(torch.floor(self.P / cell).long())
            self._grids[cell] = torch.sort(keys, stable=True)
        return self._grids[cell]

    def _search(self, Q: torch.Tensor, cell: float, best_d2, best_i, rows, budget: int = 1 << 24):
        """Update best (d2, index) of rows `rows` of Q from the 27 cells around each query."""
        sk, order = self._grid(cell)
        nk = _cell_keys((torch.floor(Q[rows] / cell).long()[:, None, :] + _OFFS[None]).reshape(-1, 3))
        lo = torch.searchsorted(sk, nk, right=False)
        cnt = torch.searchsorted(sk, nk, right=True) - lo
        per_row = torch.cumsum(cnt.view(-1, 27).sum(1), 0).cpu().numpy()
        if not len(per_row) or per_row[-1] == 0:
            return
        # row chunks of about `budget` candidate pairs each
        cuts = np.unique(np.searchsorted(per_row, np.arange(budget, per_row[-1], budget), side="right"))
        for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(per_row)]):
            if b <= a:
                continue
            c = cnt[a * 27:b * 27]
            total = int(per_row[b - 1] - (per_row[a - 1] if a else 0))
            if total == 0:
                continue
            slot = torch.repeat_interleave(torch.arange(len(c), device=DEV), c)
            start = torch.cumsum(c, 0) - c
            pi = order[lo[a * 27:b * 27][slot] + torch.arange(total, device=DEV) - start[slot]]
            qi = rows[a + slot // 27]
            self._reduce(Q, qi, pi, best_d2, best_i)

    def _reduce(self, Q, qi, pi, best_d2, best_i):
        diff = Q[qi] - self.P[pi]
        d2 = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1] + diff[:, 2] * diff[:, 2]
        cur = best_d2.clone()
        cur.scatter_reduce_(0, qi, d2, reduce="amin")
        # ties: the lowest sample index among the minima (kept best wins only if strictly lower index)
        at_min = d2 == cur[qi]
        cand = torch.full_like(best_i, torch.iinfo(torch.int64).max)
        keep_old = (best_d2 == cur) & (best_i >= 0)
        cand[keep_old] = best_i[keep_old]
        cand.scatter_reduce_(0, qi[at_min], pi[at_min], reduce="amin")
        hit = cand != torch.iinfo(torch.int64).max
        best_d2.copy_(torch.where(hit, cur, best_d2))
        best_i.copy_(torch.where(hit, cand, best_i))

    def query(self, queries: np.ndarray, radius: float | None = None):
        """(distance float64, index int64); inf / -1 when nothing lies closer than `radius`."""
        Q = torch.as_tensor(np.asarray(queries, dtype=np.float64), device=DEV)
        best_d2 = torch.full((len(Q),), math.inf, dtype=torch.float64, device=DEV)
        best_i = torch.full((len(Q),), -1, dtype=torch.int64, device=DEV)
        if len(Q) and len(self.P):
            rows = torch.arange(len(Q), device=DEV)
            if radius is not None:
                # every sample within `radius` lies in the 27 cells (slightly larger than radius) around the query
                self._search(Q, radius * 1.001, best_d2, best_i, rows)
            else:
                for cell in (0.1, 0.5):
                    self._search(Q, cell, best_d2, best_i, rows)
                    rows = rows[~(best_d2[rows] <= (0.999 * cell) ** 2)]   # exact once the minimum is inside
                    if not len(rows):
                        break
                if len(rows):
                    self._far(Q, rows, best_d2, best_i)
        if radius is not None:
            miss = ~(best_d2 < radius * radius)
            best_d2[miss] = math.inf
            best_i[miss] = -1
        return torch.sqrt(best_d2).cpu().numpy(), best_i.cpu().numpy()

    def _far(self, Q, rows, best_d2, best_i):
        """Remaining queries against all samples: float32 distances pick the candidates (within a
        margin far above their rounding error), float64 decides among them."""
        centre = self.P.mean(0)
        P32 = (self.P - centre).float()
        per = max(1, (1 << 27) // len(self.P))
        for s in range(0, len(rows), per):
            r = rows[s:s + per]
            D2 = torch.cdist((Q[r] - centre).float(), P32).square_()
            m = D2.min(dim=1).values
            qj, pi = torch.nonzero(D2 <= (m + 1e-3 + 1e-3 * m)[:, None], as_tuple=True)
            self._reduce(Q, r[qj], pi, best_d2, best_i)


def cleanup_block(f, ix: int, t: int, scene, groups, probes32) -> None:
    """One snapshot of cleanup_probes.bin (emit_harness.write_cleanup_binary)."""
    keep = first_per_voxel(scene.background, 0.03)
    g_pts, g_lab = scene.background[keep], scene.background_label[keep]
    owners, o_pts, o_own = [], [], []
    for o in scene.objects:
        if len(o.points) == 0:
            continue
        o_own.append(np.full(len(o.points), len(owners), dtype=np.int64))
        o_pts.append(o.points)
        owners.append((o.id, o.semantic))
    if o_pts:
        o_pts = np.concatenate(o_pts)
        o_own = np.concatenate(o_own)
        k = first_per_voxel(o_pts, 0.02)
        o_pts, o_own = o_pts[k], o_own[k]
    g_nn = Nearest(g_pts) if len(g_pts) else None
    o_nn = Nearest(o_pts) if len(o_pts) else None
    q_all = np.concatenate(probes32) if probes32 else np.zeros((0, 3), np.float32)
    if g_nn is not None and len(q_all):
        gd_all, gi_all = g_nn.query(q_all)
    if o_nn is not None and len(q_all):
        od_all, oi_all = o_nn.query(q_all)
    _u32(f, ix)
    _u64(f, int(t))
    _u32(f, len(groups))
    off = 0
    for g, q in zip(groups, probes32):
        _string(f, g["event_id"])
        _u32(f, len(q))
        sl = slice(off, off + len(q))
        off += len(q)
        gd = np.full(len(q), math.inf, np.float32)
        gl = np.full(len(q), INVALID_LABEL, np.uint32)
        od = np.full(len(q), math.inf, np.float32)
        oi = np.full(len(q), INVALID_LABEL, np.uint32)
        if g_nn is not None and len(q):
            gd, gl = gd_all[sl].astype(np.float32), g_lab[gi_all[sl]].astype(np.uint32)
        if o_nn is not None and len(q):
            od, oi = od_all[sl].astype(np.float32), o_own[oi_all[sl]].astype(np.uint32)
        f.write(np.rec.fromarrays([gd, gl, od, oi], formats="<f4,<u4,<f4,<u4").tobytes())
    _u32(f, len(owners))
    for oid, sem in owners:
        _string(f, oid)
        _u32(f, int(sem) & 0xFFFFFFFF)


def mesh_row(ix: int, t: int, scene, groups, probes64, radius: float = 0.10) -> dict:
    """One snapshot of cleanup_mesh.jsonl (emit_harness.write_cleanup_mesh)."""
    layers = [(scene.background.astype(np.float64), 0, "0")]
    layers += [(o.points.astype(np.float64), 1 if o.present else 2, o.id) for o in scene.objects]
    row = {"index": ix, "stamp_ns": int(t), "faces": 0,
           "vertices": int(sum(len(p) for p, _, _ in layers)), "invalid_faces": 0,
           "events": {}, "history_events": {}}
    q_all = np.concatenate(probes64) if probes64 else np.zeros((0, 3))
    found = {}
    for key, want in (("events", (0, 1)), ("history_events", (2,))):
        # trees in order; an earlier tree wins exact ties (strict < in the CPU writer)
        sel = [(p, layer, owner) for p, layer, owner in layers if len(p) and layer in want]
        if not sel or not len(q_all):
            found[key] = None
            continue
        pts = np.concatenate([p for p, _, _ in sel])
        src = np.concatenate([np.full(len(p), j) for j, (p, _, _) in enumerate(sel)])
        d, i = Nearest(pts).query(q_all, radius)
        found[key] = (d, i, pts, src, sel)
    off = 0
    for g, q in zip(groups, probes64):
        sl = slice(off, off + len(q))
        off += len(q)
        for key in ("events", "history_events"):
            rec = [None] * len(q)
            if found[key] is not None:
                d, i, pts, src, sel = found[key]
                hits = np.flatnonzero(np.isfinite(d[sl]))
                j = i[sl][hits]
                for k, dk, p, sj in zip(hits.tolist(), d[sl][hits].tolist(), pts[j].tolist(), src[j].tolist()):
                    _, layer, owner = sel[sj]
                    rec[k] = [math.sqrt(dk ** 2), p[0], p[1], p[2], layer, owner]
            row[key][g["event_id"]] = rec
    return row


def write_surfels(scene, bg: float, obj: float, path) -> int:
    parts = [(scene.background, scene.background_normal, bg)] + [(o.points, o.normals, obj) for o in scene.objects]
    verts, faces, off = [], [], 0
    for pts, nrm, sp in parts:
        k = first_per_voxel(pts, sp)
        v, f = surfel_triangles(pts[k], None if nrm is None else nrm[k], sp)
        verts.append(v)
        faces.append(f + off)
        off += len(v)
    v, f = np.concatenate(verts), np.concatenate(faces)
    with open(path, "wb") as out:
        out.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\nproperty float x\n"
                   f"property float y\nproperty float z\nelement face {len(f)}\n"
                   "property list uchar int vertex_indices\nend_header\n").encode())
        out.write(v.astype("<f4").tobytes())
        rec = np.zeros(len(f), dtype=[("n", "u1"), ("i", "<i4", 3)])
        rec["n"] = 3
        rec["i"] = f
        out.write(rec.tobytes())
    return len(f)


def main() -> None:
    tl_path, exp, stage, residual, probes, bg, obj = sys.argv[1:8]
    bg, obj, exp = float(bg), float(obj), Path(exp)
    exp.mkdir(parents=True, exist_ok=True)
    with open(tl_path, "rb") as fh:
        import pickle
        tl = pickle.load(fh)
    stamps = tl.stamps()
    cleanup = stage != "a"
    if cleanup:
        r_groups = json.loads(Path(residual).read_text())
        r_probes = [np.asarray(g["points"], dtype=np.float32).reshape(-1, 3) for g in r_groups]
        m_groups = json.loads(Path(probes).read_text())
        m_probes = [np.asarray(g["points"], dtype=np.float64).reshape(-1, 3) for g in m_groups]
    snap = (exp / "snapshots.jsonl").open("w")
    binf = open(exp / "cleanup_probes.bin", "wb") if cleanup else None
    mesh = (exp / "cleanup_mesh.jsonl").open("w") if cleanup else None
    if binf is not None:
        binf.write(b"ABCCLN1\0")
        _u32(binf, len(stamps))
    for ix, t in enumerate(stamps):
        scene = tl.scene(t)
        snap.write(json.dumps(snapshot_record(ix, scene)) + "\n")
        if cleanup:
            cleanup_block(binf, ix, t, scene, r_groups, r_probes)
            mesh.write(json.dumps(mesh_row(ix, t, scene, m_groups, m_probes)) + "\n")
    for fh in (snap, binf, mesh):
        if fh is not None:
            fh.close()
    n = write_surfels(tl.scene(stamps[-1]), bg, obj, exp / "surfels_final.ply")
    (exp / "COMPLETE.json").write_text(json.dumps(
        {"map": tl_path, "snapshots": len(stamps), "provenance": "evaluation_scene"}, indent=2))
    print("emitted", len(stamps), "snapshots;", n, "surfels")


if __name__ == "__main__":
    main()
