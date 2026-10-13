"""SurfelMeshing (Schöps, Sattler, Pollefeys, TPAMI 2019; ETH CVG) as a backend.

=====================================================================================================
CHANGES TO PUBLISHED SurfelMeshing (FT/baselines/surfelmeshing @bdee93e) -- all in
SurfelMeshing_bdee93e.patch (next to this file) except S6; labels go into every run.json.
  build   CUDA 12.8 / sm_120 fixes, Qt5X11Extras optional, --headless (no OpenGL context: WSL2).
  [S1] --command_mode: frames are processed only as far as 'run <i>' allows; 'dump' writes all surfel
       entries; 'retire' deletes surfels the way a merge deletes a surfel (the coupling).
  [S2] --no_conflict_update (rows 2 and 4 only): a measurement behind a surfel no longer lowers the
       surfel's confidence or replaces it (its own change handling off).
  [S3] deleted surfels (merged or retired) no longer enter the min-depth and association passes
       (all rows; published: a merged surfel still does, next to the one it merged into).
  [S4] the fixed-size surfel buffer is checked (all rows; published: silent overflow).
  [S5] --no_meshing: surfels only, no triangulation (all rows).
  [S6] --max_pose_interpolation_time_extent 1e9 (adapter argument, since 2026-09-30 13:00): the
       reader keeps every frame (published default 0.05 s drops each session's last frame of a
       chain and shifts all later frame indices).
Input: frames 960x540 (real) / 680x480 (synthetic) at 30 Hz, people as invalid depth.
=====================================================================================================

The program cannot load a map, so one run of it serves a chain of sessions: its input is one
TUM-format sequence of the sessions from the current one on, at the sensor's 30 Hz, each frame
written (to /dev/shm) just before the program reads it and removed after. A session started
without a prior (row 1, or the first session) starts a new run. Frames: the session's frames
subsampled to about the sensor's native depth resolution (real: 1920x1080 -> 960x540; synthetic:
680x480 as is), depth beyond the sensor range and dynamic pixels (people) written as invalid.

Elements: the live surfels; id = creation frame << 32 | entry index (an entry is reused when its
surfel is replaced). SurfelMeshing has no labels: a surfel's identity and semantic label are those
of the frame that last updated it, at its projection. last_update = that frame's stamp; extent =
the surfel's radius.
"""
from __future__ import annotations

import atexit
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from ...eval.scenelist import ElementLifetimes, SnapshotRecorder
from ...frames import FlatSession, Frame, SessionSpec, dynamic_mask
from ...interface import Backend, DatasetInfo, Elements

BINARY = Path("/home/jixian/Desktop/FT/baselines/surfelmeshing/build_cuda128/applications/surfel_meshing/SurfelMeshing")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STEP = {"synthetic": 1, "real": 2}
DEPTH_SCALING = 4000.0                 # PNG value per metre (16 bit: up to 16.38 m)
OUTLIER_FRAMES = 8                     # SurfelMeshing --outlier_filtering_frame_count default
COLUMNS = ("x", "y", "z", "nx", "ny", "nz", "r2", "confidence", "creation", "last_update")


class _Program:
    """One run of the patched SurfelMeshing over a chain of sessions."""

    def __init__(self, info: DatasetInfo, specs: List[SessionSpec], own_update: bool, log_path: Path,
                 max_surfels: int):
        self.info = info
        self.step = STEP[info.name]
        self.folder = Path(f"/dev/shm/update_layer_surfelmeshing_{os.getpid()}_{id(self)}")
        (self.folder / "rgb").mkdir(parents=True, exist_ok=True)
        (self.folder / "depth").mkdir(parents=True, exist_ok=True)
        # the frame table of the chain: global index -> (session, local index, stamp)
        self.sessions = {sp.name: FlatSession(sp, pixel_step=self.step) for sp in specs}
        self.offset: Dict[str, int] = {}
        self.table: List[Tuple[str, int]] = []
        stamps = []
        for sp in specs:
            fs = self.sessions[sp.name]
            self.offset[sp.name] = len(self.table)
            for i in range(len(fs.ids)):
                self.table.append((sp.name, i))
                stamps.append(fs.stamp_ns(i))
        self.stamps = np.asarray(stamps, dtype=np.int64)
        K = next(iter(self.sessions.values())).K
        self.fx, self.fy = K.fx, K.fy
        self.cx, self.cy = K.cx - K.offset, K.cy - K.offset          # origin at the top-left pixel centre
        self.poses = {}                                             # global index -> T_world_cam
        self.labels = {}                                            # global index -> (instance, semantic)
        self._write_index()
        self.written = -1
        self.pool = ThreadPoolExecutor(max_workers=4)
        self._write_until(0)
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        args = [str(BINARY), str(self.folder), "trajectory.txt", "--headless", "--command_mode", "--no_meshing",
                "--depth_scaling", str(DEPTH_SCALING), "--max_depth", str(info.depth_range[1]),
                "--depth_valid_region_radius", "100000", "--restrict_fps_to", "100000",
                "--max_surfel_count", str(max_surfels),
                # every frame has a trajectory pose at its own stamp; without this the reader drops
                # a frame whose next pose is > 0.05 s away (the last frame of each session)
                "--max_pose_interpolation_time_extent", "1e9"]
        if not own_update:
            args.append("--no_conflict_update")
        self.log = open(log_path, "a")
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                     text=True, bufsize=1, env=env, cwd=str(self.folder))
        self.done = -1                                              # all frames <= done handled

    # -- input ----------------------------------------------------------------------------------
    def _write_index(self) -> None:
        lines_assoc, lines_traj = [], []
        for g, (name, i) in enumerate(self.table):
            t = f"{self.stamps[g] / 1e9:.9f}"
            lines_assoc.append(f"{t} rgb/{g}.png {t} depth/{g}.png")
            T = self.sessions[name].pose(i)
            self.poses[g] = T
            q = _quaternion(T[:3, :3])
            lines_traj.append(f"{t} {T[0, 3]:.9f} {T[1, 3]:.9f} {T[2, 3]:.9f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}")
        (self.folder / "associated.txt").write_text("\n".join(lines_assoc) + "\n")
        (self.folder / "trajectory.txt").write_text("\n".join(lines_traj) + "\n")
        (self.folder / "calibration.txt").write_text(f"{self.fx} {self.fy} {self.cx} {self.cy}\n")

    def _write_frame(self, g: int):
        name, i = self.table[g]
        f = self.sessions[name].load(i, color=True)
        depth = np.nan_to_num(f.depth, nan=0.0)
        depth[dynamic_mask(f, self.info.dynamic_semantics)] = 0.0
        png = np.clip(np.round(depth * DEPTH_SCALING), 0, 65535).astype(np.uint16)
        cv2.imwrite(str(self.folder / f"depth/{g}.png"), png)
        cv2.imwrite(str(self.folder / f"rgb/{g}.png"), cv2.cvtColor(f.color, cv2.COLOR_RGB2BGR))
        sem = f.semantic.astype(np.uint16) if f.semantic is not None else np.zeros_like(png)
        return g, f.instance.astype(np.uint16), sem

    def _write_until(self, g: int) -> None:
        g = min(g, len(self.table) - 1)
        if g <= self.written:
            return
        for gg, inst, sem in self.pool.map(self._write_frame, range(self.written + 1, g + 1)):
            self.labels[gg] = (inst, sem)
        self.written = g

    def _release(self) -> None:
        """Remove frame files the program no longer reads (it keeps OUTLIER_FRAMES / 2 behind)."""
        for g in range(max(0, getattr(self, "_released", 0)), self.done - OUTLIER_FRAMES):
            for sub in ("rgb", "depth"):
                try:
                    (self.folder / f"{sub}/{g}.png").unlink()
                except FileNotFoundError:
                    pass
            self._released = g + 1

    # -- commands -------------------------------------------------------------------------------
    def call(self, line: str) -> str:
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        while True:
            out = self.proc.stdout.readline()
            if not out:
                raise RuntimeError(f"SurfelMeshing exited (see {self.log.name})")
            if out.startswith("SM "):
                reply = out[3:].strip()
                if reply.startswith("error"):
                    raise RuntimeError(f"SurfelMeshing: {reply}")
                return reply

    def run_to(self, g: int) -> None:
        if g <= self.done:
            return
        self._write_until(g + OUTLIER_FRAMES // 2 + 2)
        self.done = int(self.call(f"run {g}").split()[1])
        if not getattr(self, "_checked", False):          # its frame i must be our frame i
            self.log.flush()
            read = [l for l in open(self.log.name, errors="replace") if "Read dataset with" in l]
            n = int(read[-1].split("Read dataset with")[1].split()[0]) if read else -1
            if n != len(self.table):
                raise RuntimeError(f"SurfelMeshing read {n} frames, the chain has {len(self.table)}")
            self._checked = True
        self._release()

    def dump(self) -> Dict[str, np.ndarray]:
        path = self.folder / "dump.bin"
        n = int(self.call(f"dump {path}").split()[1])
        with open(path, "rb") as fh:
            assert fh.read(8) == b"SMDUMP01"
            entries, done = np.frombuffer(fh.read(8), dtype=np.int32)
            assert entries == n
            data = np.frombuffer(fh.read(), dtype=np.float32)
        path.unlink()
        cols = {c: data[k * n:(k + 1) * n] for k, c in enumerate(COLUMNS)}
        cols["creation"] = cols["creation"].view(np.uint32)
        cols["last_update"] = cols["last_update"].view(np.uint32)
        return cols

    def retire(self, entries: np.ndarray, creation: np.ndarray) -> int:
        path = self.folder / "retire.bin"
        with open(path, "wb") as fh:
            fh.write(np.uint32(len(entries)).tobytes())
            fh.write(entries.astype(np.uint32).tobytes())
            fh.write(creation.astype(np.uint32).tobytes())
        n = int(self.call(f"retire {path}").split()[1])
        path.unlink()
        return n

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.call("quit")
            except RuntimeError:
                pass
            self.proc.wait(timeout=60)
        self.pool.shutdown()
        self.log.close()
        for sub in ("rgb", "depth"):
            for p in (self.folder / sub).glob("*.png"):
                p.unlink()
            (self.folder / sub).rmdir()
        for p in self.folder.glob("*"):
            p.unlink()
        self.folder.rmdir()


def _quaternion(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> (qx, qy, qz, qw), qw >= 0."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s]
    q = np.asarray(q) / np.linalg.norm(q)
    return -q if q[3] < 0 else q


class SurfelMeshingBackend(Backend):
    name = "surfelmeshing"
    CHANGES = ("S1 command mode (run/dump/retire)", "S2 no conflict update in rows 2 and 4",
               "S3 deleted surfels out of min-depth and association", "S4 surfel buffer check",
               "S5 no meshing", "S6 reader keeps every frame (pose interpolation extent 1e9)")

    def __init__(self, info: DatasetInfo, own_update: bool, work_dir=None, max_surfels: int = 50_000_000):
        super().__init__(info, own_update, work_dir)
        self.max_surfels = max_surfels
        self.program: Optional[_Program] = None
        self.runs = 0
        atexit.register(self._close)

    def _close(self) -> None:
        if self.program is not None:
            self.program.close()
            self.program = None

    # -- session ------------------------------------------------------------------------------
    def start_session(self, spec: SessionSpec, prior: Optional[_Program]) -> None:
        if prior is None:
            self._close()
            names = [sp.name for sp in self.info.sessions]
            chain = list(self.info.sessions)[names.index(spec.name):] if spec.name in names else [spec]
            log_dir = Path(self.work_dir) if self.work_dir else Path("/dev/shm")
            self.runs += 1
            self.program = _Program(self.info, chain, self.own_update,
                                    log_dir / f"surfelmeshing_{self.runs}_{spec.name}.log", self.max_surfels)
            # per program: entry -> (creation, identity, semantic) of its last labelling
            self.lab_key = np.zeros(0, np.uint64)
            self.lab_identity = np.zeros(0, np.int64)
            self.lab_semantic = np.zeros(0, np.int64)
        else:
            self.program = prior
        self.spec = spec
        self.cache = None                                           # (done, dump)
        self.recorder = SnapshotRecorder()                          # this session's element lifetimes

    def end_session(self) -> _Program:
        return self.program

    # -- frames -------------------------------------------------------------------------------
    def integrate(self, frame: Frame) -> None:
        p = self.program
        p.run_to(p.offset[self.spec.name] + frame.index)

    # -- surfels ------------------------------------------------------------------------------
    def _surfels(self) -> Dict[str, np.ndarray]:
        """The live surfels now, labelled; cached until the program moves on."""
        p = self.program
        if self.cache is not None and self.cache[0] == p.done:
            return self.cache[1]
        d = p.dump()
        live = np.nonzero(d["r2"] > 0)[0].astype(np.int64)
        creation, last = d["creation"][live].astype(np.uint64), d["last_update"][live].astype(np.int64)
        s = dict(entry=live, creation=creation, last_update=last, radius=np.sqrt(d["r2"][live]),
                 ids=(creation << np.uint64(32)) | live.astype(np.uint64),
                 xyz=np.stack([d["x"][live], d["y"][live], d["z"][live]], 1),
                 normal=np.stack([d["nx"][live], d["ny"][live], d["nz"][live]], 1))
        s["identity"], s["semantic"] = self._label(s)
        self.cache = (p.done, s)
        return s

    def _label(self, s) -> Tuple[np.ndarray, np.ndarray]:
        """Identity and semantic label of each live surfel from the frame that last updated it."""
        p = self.program
        key = (s["creation"] << np.uint64(32)) | s["last_update"].astype(np.uint64)
        n = int(s["entry"].max()) + 1 if len(s["entry"]) else 0
        if len(self.lab_key) < n:
            grow = n - len(self.lab_key)
            self.lab_key = np.concatenate([self.lab_key, np.full(grow, np.iinfo(np.uint64).max, np.uint64)])
            self.lab_identity = np.concatenate([self.lab_identity, np.zeros(grow, np.int64)])
            self.lab_semantic = np.concatenate([self.lab_semantic, np.zeros(grow, np.int64)])
        e = s["entry"]
        stale = np.nonzero(self.lab_key[e] != key)[0]
        for g in np.unique(s["last_update"][stale]).tolist():
            rows = stale[s["last_update"][stale] == g]
            ident = np.zeros(len(rows), np.int64)
            sem = np.zeros(len(rows), np.int64)
            if g in p.labels:
                inst_img, sem_img = p.labels[g]
                T = p.poses[g]
                cam = (s["xyz"][rows].astype(np.float64) - T[:3, 3]) @ T[:3, :3]
                z = cam[:, 2]
                with np.errstate(divide="ignore", invalid="ignore"):
                    u = np.floor(p.fx * cam[:, 0] / z + p.cx + 0.5)
                    v = np.floor(p.fy * cam[:, 1] / z + p.cy + 0.5)
                h, w = inst_img.shape
                ok = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
                ident[ok] = inst_img[v[ok].astype(np.int64), u[ok].astype(np.int64)]
                sem[ok] = sem_img[v[ok].astype(np.int64), u[ok].astype(np.int64)]
            self.lab_identity[e[rows]] = ident
            self.lab_semantic[e[rows]] = sem
            self.lab_key[e[rows]] = key[rows]
        # label images of frames every live surfel has been labelled from are no longer needed
        for g in [g for g in p.labels if g < p.done - OUTLIER_FRAMES]:
            del p.labels[g]
        return self.lab_identity[e].copy(), self.lab_semantic[e].copy()

    def elements(self) -> Elements:
        s = self._surfels()
        t = lambda a, d: torch.as_tensor(a, dtype=d, device=DEV)
        return Elements(t(s["ids"].astype(np.int64), torch.int64), t(s["xyz"], torch.float32),
                        t(s["normal"], torch.float32), t(s["identity"], torch.int64),
                        t(self.program.stamps[s["last_update"]], torch.int64), t(s["radius"], torch.float32))

    def retire(self, ids: torch.Tensor, stamp: int) -> None:
        if not len(ids):
            return
        ids = ids.cpu().numpy().astype(np.uint64)
        entry = (ids & np.uint64(0xFFFFFFFF)).astype(np.int64)
        creation = (ids >> np.uint64(32)).astype(np.int64)
        self.program.retire(entry, creation)
        if self.cache is not None:                                  # keep the cached view current
            s = self.cache[1]
            keep = ~np.isin(s["ids"], ids)
            self.cache = (self.cache[0], {k: v[keep] for k, v in s.items()})

    # -- evaluation ---------------------------------------------------------------------------
    def snapshot(self, stamp: int) -> None:
        s = self._surfels()
        self.recorder.record(stamp, s["ids"].astype(np.int64), s["xyz"], s["normal"], s["semantic"], s["identity"])

    def timeline(self) -> ElementLifetimes:
        return self.recorder.timeline()
