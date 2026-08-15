#!/usr/bin/env python3
"""HTTP bridge: 4dmap mesh server -> browser WebGL viewer.

Zero-install: the heavy C++ reader (already built) runs here in WSL; the
Windows browser renders the meshes natively with WebGL. No files are ever
produced -- /frame streams binary straight from the resident server process.

Endpoints (http://localhost:8123):
    /                       -> viewer page (index.html)
    /meta                   -> JSON {a:{frames,stamps}, b:{...}, total, progress}
    /frame?s=a&i=5          -> binary: <nv u32><nf u32><ts u64> verts f32
                               colors u8 faces u32 (nv==0 => no mesh)
    /poses                  -> JSON {a/b: {positions: [[x,y,z]..], stamps: []}}
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import struct
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_SERVER = (
    Path(__file__).resolve().parent.parent
    / "build_canonical" / "4dmap_mesh_server"
)
_WEB_ROOT = Path(__file__).resolve().parent / "viewer_web"

# Runtime library roots required by the canonical build; order matters.
_LD_LIBRARY_PATH = [
    "/home/jixian/ros2_ws/install/hydra/lib",
    "/home/jixian/ros2_ws/install/teaserpp/lib",
    "/home/jixian/ros2_ws/install/spark_dsg/lib",
    "/home/jixian/ros2_ws/install/kimera_pgmo/lib",
    "/home/jixian/ros2_ws/install/config_utilities/lib",
    "/home/jixian/ros2_ws/install/kimera_rpgo/lib",
    "/opt/ros/jazzy/lib/x86_64-linux-gnu",
    "/home/jixian/ros2_ws/install/pose_graph_tools/lib",
    "/home/jixian/Desktop/FT/session_update_baseline/.canonical_mapping/install/lib",
]

_HEADER = struct.Struct("<IIIQ")  # magic, n_vertices, n_faces, timestamp_ns
_MAGIC = 0x314D4241

_proc = None
_meta = None
_progress_lines = []
_pipe_lock = threading.Lock()
_meta_lock = threading.Lock()
_frame_cache = {}  # (session, index) -> bytes
_cache_order = []
_FRAME_CACHE_MAX = 4


def _drain_stderr() -> None:
    for line in _proc.stderr:
        text = line.decode(errors="replace").rstrip()
        if text:
            _progress_lines.append(text)
            if len(_progress_lines) > 6:
                _progress_lines.pop(0)


def _ensure_meta() -> dict:
    global _meta
    with _meta_lock:
        if _meta is None:
            _meta = json.loads(_proc.stdout.readline().decode())
    return _meta


def _request(session: str, index: int) -> bytes:
    """One pipe round-trip; returns raw frame payload (verts/colors/faces)."""
    with _pipe_lock:
        _proc.stdin.write(f"{session} {index}\n".encode())
        _proc.stdin.flush()
        magic, nv, nf, ts = _HEADER.unpack(_proc.stdout.read(_HEADER.size))
        if magic != _MAGIC:
            raise RuntimeError(f"bad magic 0x{magic:x} from server")
        payload = b""
        if nv:
            payload = _proc.stdout.read(12 * nv + 3 * nv + 12 * nf)
    return struct.pack("<IIQ", nv, nf, ts) + payload


def _frame(session: str, index: int) -> bytes:
    key = (session, index)
    if key in _frame_cache:
        _cache_order.remove(key)
        _cache_order.append(key)
        return _frame_cache[key]
    data = _request(session, index)
    _frame_cache[key] = data
    _cache_order.append(key)
    while len(_frame_cache) > _FRAME_CACHE_MAX:
        _frame_cache.pop(_cache_order.pop(0))
    return data


def _load_poses(rgbd_dir: Path | None):
    """Camera positions per session; same format the desktop viewer uses."""
    if not rgbd_dir or not rgbd_dir.is_dir():
        return None
    rows = []
    with (rgbd_dir / "timestamps.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            rows.append(row)
    positions, stamps = [], []
    for row in rows:
        image_id = row.get("ImageID", row.get("image_id", ""))
        pose_file = rgbd_dir / f"{image_id}_pose.txt"
        if not pose_file.is_file():
            continue
        matrix = []
        with pose_file.open() as stream:
            for line in stream:
                matrix.append([float(v) for v in line.split()])
                if len(matrix) == 4:
                    positions.append([matrix[0][3], matrix[1][3], matrix[2][3]])
                    matrix = []
        stamps.append(int(row.get("TimeStamp", row.get("timestamp", ""))))
    if not positions:
        return None
    return {"positions": positions, "stamps": stamps}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # keep stderr clean
        pass

    def _send(self, body: bytes, ctype: str, cache: bool = False):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", "public, max-age=3600")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        try:
            if path == "/":
                index = _WEB_ROOT / "index.html"
                if index.is_file():
                    self._send(index.read_bytes(), "text/html; charset=utf-8")
                else:
                    self._send(b"missing viewer_web/index.html", "text/plain")
            elif path == "/meta":
                if _meta is None:
                    self.send_response(503)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = json.dumps(_meta).encode()
                self._send(body, "application/json", cache=True)
            elif path == "/progress":
                body = json.dumps({"lines": _progress_lines[-4:]}).encode()
                self._send(body, "application/json")
            elif path == "/frame":
                from urllib.parse import urlparse, parse_qs
                q = parse_qs(urlparse(self.path).query)
                session = (q.get("s") or ["a"])[0]
                index = int((q.get("i") or ["0"])[0])
                self._send(_frame(session, index), "application/octet-stream",
                           cache=True)
            elif path == "/poses":
                body = json.dumps(_poses).encode()
                self._send(body, "application/json", cache=True)
            else:
                self.send_error(404)
        except BrokenPipeError:
            pass
        except Exception as exc:  # surface server-side faults to the page
            try:
                self._send(json.dumps({"error": str(exc)}).encode(),
                           "application/json")
            except Exception:
                pass


def main() -> None:
    global _proc, _poses
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map-a", default=("/home/jixian/Desktop/FT/runs/"
                           "session_ab_1cm_20260815/session_a/final.4dmap"))
    parser.add_argument(
        "--map-b", default=("/home/jixian/Desktop/FT/runs/"
                           "session_ab_1cm_20260815/session_b/final.4dmap"))
    parser.add_argument("--pose-a", default=None)
    parser.add_argument("--pose-b", default=None)
    parser.add_argument("--port", type=int, default=8123)
    args = parser.parse_args()

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(_LD_LIBRARY_PATH) + ":" + env.get(
        "LD_LIBRARY_PATH", "")
    _proc = subprocess.Popen(
        [_SERVER, "--map_a", args.map_a, "--map_b", args.map_b, "--stride", "1"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=env)
    threading.Thread(target=_drain_stderr, daemon=True).start()

    _poses = {"a": _load_poses(Path(args.pose_a)) if args.pose_a else None,
              "b": _load_poses(Path(args.pose_b)) if args.pose_b else None}

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"4dmap bridge listening on http://localhost:{args.port} "
          f"(maps loading in background)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            with _pipe_lock:
                _proc.stdin.write(b"QUIT\n")
                _proc.stdin.flush()
            _proc.wait(timeout=10)
        except Exception:
            _proc.kill()


if __name__ == "__main__":
    main()
