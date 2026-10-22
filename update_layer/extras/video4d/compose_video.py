"""The 4D video (user 10-22: "主要要看到变化"): the synthetic mapping A then B from one fixed oblique camera, ceiling removed.
  left:  ground truth, Isaac's scene at time t (render_video_isaac.py; its background, no depth, white as ours), with
         what the dataset camera sees at t tinted (that frame's depth, back-projected and depth-tested against Isaac's
         view), its path so far and its frustum;
  right: ours, the 4D map of t (our latest round at or before t; render_4d_rounds.py; before the first round there is
         none), with the same path and frustum.
The furniture's changes (changes.py, from the ground truth and the dataset camera only) are outlined on both sides
(the outline of the object's surface in this view): red where it was, green where it is, an arrow from the one to the
other (set off to its right, so that two objects trading places keep two arrows); dashed while the move has not
finished in the ground truth, thin and dashed for a new place the video camera cannot see; from the move until the
dataset camera has seen it and our round after that is shown, and 10 s more. A bar below: the
session's time, the ground truth's changes (orange; at 0 in B: the changes between the sessions) and our rounds (blue).
One output frame per Isaac render (<session>_<t>_rgb.png) of the sessions given (default a and b):
OUT_DIR/frames_<session>/NNNNN.png, each with NNNNN.json (its time, our round, the marks drawn).
--encode: OUT_DIR/<name>.mp4 (H.264, FPS frames per second) from frames_a then frames_b.
  compose_video.py CAMERA.json ISAAC_DIR OURS_DIR CHANGES.json OUT_DIR [--sessions ab] [--encode] [--fps N] [--name NAME]"""
import json, subprocess, sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))                    # the repository's root
from update_layer.frames import FlatSession, backproject, synthetic_session  # noqa: E402

cam = json.load(open(sys.argv[1])); isaac, ours, out = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[5])
changes = json.load(open(sys.argv[4]))
opt = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
fps, name, sessions = int(opt("--fps", 24)), opt("--name", "syn_4d"), opt("--sessions", "ab")
FFMPEG = "/mnt/c/ProgramData/ScreenToGif/ffmpeg.exe"
H, W = int(cam["height"]), int(cam["width"])
K = np.array(cam["K"], dtype=np.float64); w2c = np.array(cam["T_world_to_camera"], dtype=np.float64)
TOP = float(cam["crop_box_world"]["z"][1])
TINT, TRAIL, FRUST, ARROW = (255, 170, 0), (40, 110, 230), (30, 30, 30), (100, 100, 100)            # RGB
RED, GREEN, ORANGE, BLUE = (230, 20, 20), (0, 170, 60), (255, 140, 0), (40, 90, 220)
# Both panels cut to the same window of the rendered view (user 10-23: "2个一样大 但是2个map都大点"): the balcony, the
# living room, the bedroom and the doors of the kitchen and the bathroom (u 66-1023 of 1280; the flat's rows 4-637, ours'
# floaters to 666), shown at their rendered size side by side, 1920 wide.
U0, U1, V0, V1, GAP = 66, 1023, 0, 676, 6
BAND, BAR = 84, 112
LEGEND = [("orange: seen now", (200, 110, 0)), ("blue: camera path", BLUE), ("red: furniture was here", RED),
          ("green: it is here now", GREEN), ("dashed: moving / hidden", (90, 90, 90))]
rounds = json.loads((ours / "rounds.json").read_text())["rounds"]


def project(P):
    c = P @ w2c[:3, :3].T + w2c[:3, 3]
    z = c[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([K[0, 0] * c[:, 0] / z + K[0, 2], K[1, 1] * c[:, 1] / z + K[1, 2]], 1)
    return uv, z


def outline(img, poly, colour, dashed=False, thick=2, sc=1.0):
    q = np.array(poly, np.float64) * sc
    if not dashed:
        cv2.polylines(img, [np.round(q).astype(np.int32).reshape(-1, 1, 2)], True, colour, thick, cv2.LINE_AA)
    else:                                                                       # 10 px dashes, 7 px gaps along the outline
        pos = 0.0
        for a, b in zip(q, np.roll(q, -1, 0)):
            L = np.linalg.norm(b - a)
            for d0 in np.arange(-pos, L, 17.0):
                d1 = min(d0 + 10.0, L)
                if d1 > max(d0, 0):
                    p0, p1 = a + (b - a) * max(d0, 0) / L, a + (b - a) * d1 / L
                    cv2.line(img, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), colour, thick, cv2.LINE_AA)
            pos = (pos + L) % 17.0
    m = cv2.moments(np.round(q).astype(np.int32))
    return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]]) if m["m00"] else q.mean(0)


def visible_mask(frame, depth_obl):
    """Pixels of the oblique view showing what the dataset camera sees in this frame: its depth pixels (every 2nd),
    back-projected, projected into the oblique view and kept where Isaac's oblique depth agrees (not hidden there)."""
    sub = np.zeros(frame.depth.shape, bool); sub[::2, ::2] = True
    P, _, _ = backproject(frame, sub)
    P = P[P[:, 2] <= TOP]                                                       # not the wall tops cut away
    uv, z = project(P)
    u, v = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
    ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[ok], v[ok], z[ok]
    ok = z <= depth_obl[v, u] + 0.12 + 0.01 * z
    m = np.zeros((H, W), np.uint8); m[v[ok], u[ok]] = 255
    m = cv2.morphologyEx(cv2.dilate(m, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return m > 0


def draw_camera(img, fs, i, trail, sc=1.0, frust=2):
    uv, z = project(trail)
    pts = np.round(uv[z > 0] * sc).astype(np.int32)
    if len(pts) > 1:
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, TRAIL, 1, cv2.LINE_AA)
    T = fs.pose(i); k = fs.K; d = 0.9
    corners = [((x + k.offset - k.cx) / k.fx * d, (y + k.offset - k.cy) / k.fy * d, d)
               for x, y in [(0, 0), (k.width, 0), (k.width, k.height), (0, k.height)]]
    P = np.array([[0, 0, 0]] + corners) @ T[:3, :3].T + T[:3, 3]
    uv, z = project(P)
    if (z > 0).all():
        q = np.round(uv * sc).astype(np.int32)
        for a in range(1, 5):
            cv2.line(img, tuple(q[0]), tuple(q[a]), FRUST, frust, cv2.LINE_AA)
            cv2.line(img, tuple(q[a]), tuple(q[1 + a % 4]), FRUST, frust, cv2.LINE_AA)
        cv2.circle(img, tuple(q[0]), 2 + 2 * frust, FRUST, -1, cv2.LINE_AA)


def draw_marks(img, live, t, sc=1.0):
    for mk in live:
        moving = t < mk["arrived"]
        c = {}
        for col, poly in mk["polygons"].items():
            hidden = col == "green" and mk["green_hidden"]
            c[col] = outline(img, poly, RED if col == "red" else GREEN, dashed=moving or hidden, thick=1, sc=sc)
        if len(c) == 2:
            a, b = c["red"], c["green"]
            n = np.array([-(b - a)[1], (b - a)[0]]) / max(np.linalg.norm(b - a), 1)       # to the arrow's right
            a, b = a + 4 * n, b + 4 * n
            cv2.arrowedLine(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), ARROW, 1,
                            cv2.LINE_AA, tipLength=min(0.3, 12 / max(np.linalg.norm(b - a), 1)))


def label(img, text, org, scale=1.0, color=(20, 20, 20), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick + 5, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def timeline(width, s, t, end_s, moved, between, mine, spec):
    bar = np.full((BAR, width, 3), 255, np.uint8)
    x0, x1, y = 190, width - 40, 34
    X = lambda v: int(x0 + (x1 - x0) * min(max(v / end_s, 0), 1))
    label(bar, f"session {s.upper()}", (20, y + 9), 0.9)
    cv2.line(bar, (x0, y), (x1, y), (200, 200, 200), 4)
    cv2.rectangle(bar, (x0, y - 3), (X(t), y + 3), (90, 90, 90), -1)
    for v in range(0, int(end_s) + 1, 10):
        cv2.line(bar, (X(v), y + 16), (X(v), y + 22), (120, 120, 120), 1)
        cv2.putText(bar, f"{v}s", (X(v) - 12, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (110, 110, 110), 1, cv2.LINE_AA)
    for r in mine:
        rt = (r["stamp"] - spec.start_ns) / 1e9
        if 0 <= rt <= end_s:
            cv2.line(bar, (X(rt), y - 12), (X(rt), y + 12), BLUE, 2)
    for c in moved + ([0.0] if between else []):
        cv2.fillPoly(bar, [np.array([[X(c) - 7, y - 26], [X(c) + 7, y - 26], [X(c), y - 14]], np.int32)], ORANGE)
    cv2.circle(bar, (X(t), y), 9, (20, 20, 20), -1)
    note = ("orange: furniture moved" + ("; at 0: between A and B" if between else "") + "    blue: our map's rounds"
            "    outlines (furniture of 0.55 m or more) stay until our map has a round after the camera saw the change, + 10 s")
    cv2.putText(bar, note, (x0, BAR - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
    return bar


for s in sessions:
    files = sorted(isaac.glob(f"{s}_*_rgb.png"), key=lambda p: float(p.name.split("_")[1]))
    if not files:
        continue
    fdir = out / f"frames_{s}"
    fdir.mkdir(parents=True, exist_ok=True)
    spec = synthetic_session(s); fs = FlatSession(spec)
    rel = np.array(fs.rel_ns) / 1e9
    centres = np.array([fs.pose(i)[:3, 3] for i in range(len(rel))])
    mine = [r for r in rounds if r["session"] == f"synthetic_{s}"]
    marks = changes[s]["marks"]
    moved = sorted({round(m["left"], 1) for m in marks if m["left"] > 0})
    between = any(m["left"] == 0 for m in marks)
    print(f"session {s}: {len(marks)} marks", flush=True)
    for n_out, f in enumerate(files):
        t = float(f.name.split("_")[1])
        i = int(np.argmin(np.abs(rel - t)))
        g = spec.start_ns + int(round(t * 1e9))
        r = max((r for r in mine if r["stamp"] <= g), key=lambda r: r["stamp"], default=None)   # the 4D map of t
        left = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        depth = np.load(str(f).replace("_rgb.png", "_depth.npz"))["depth"]
        left[~np.isfinite(depth)] = 255                                          # the sky dome and its grass
        depth = np.where(np.isfinite(depth) & (depth > 0), depth, 1e9)
        m = visible_mask(fs.load(i), depth)
        left[m] = (0.78 * left[m] + 0.22 * np.array(TINT)).astype(np.uint8)
        if r is not None:
            right = cv2.cvtColor(cv2.imread(str(ours / r["file"])), cv2.COLOR_BGR2RGB)
        else:                                                                   # before our first round: no map yet
            right = np.full((H, W, 3), 255, np.uint8)
            label(right, f"no map yet: our first round maps up to {(mine[0]['stamp'] - spec.start_ns) / 1e9:.1f} s", (230, H // 2), 0.8, (90, 90, 90))
        live = [mk for mk in marks if mk["start"] <= t <= mk["until"]]
        for img in (left, right):
            draw_camera(img, fs, i, centres[:i + 1], frust=2)
            draw_marks(img, live, t)
        pw, ph = U1 - U0, V1 - V0
        FW = 2 * pw + GAP
        frame = np.full((BAND + ph + BAR, FW, 3), 255, np.uint8)
        frame[BAND:BAND + ph, :pw] = left[V0:V1, U0:U1]
        frame[BAND:BAND + ph, pw + GAP:] = right[V0:V1, U0:U1]
        upto = (r["stamp"] - spec.start_ns) / 1e9 if r is not None else None
        label(frame, f"Ground truth (Isaac), t = {t:.1f} s", (14, 36), 0.8)
        label(frame, "Ours: the 4D map at t: no round yet" if r is None else
              f"Ours: the 4D map at t = round {r['k']} of {s.upper()}, mapped up to {max(upto, 0):.1f} s", (pw + GAP + 14, 36), 0.8)
        x = 14
        for text, col in LEGEND:
            label(frame, text, (x, 70), 0.5, col, 1)
            x += cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] + 16
        label(frame, "the 4D map of t is our latest round at or before t", (pw + GAP + 14, 70), 0.5, (60, 60, 60), 1)
        frame[BAND + ph:] = timeline(FW, s, t, changes[s]["end_s"], moved, between, mine, spec)
        cv2.imwrite(str(fdir / f"{n_out:05d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        (fdir / f"{n_out:05d}.json").write_text(json.dumps(dict(
            session=s, t=t, dataset_frame=i, round=r["k"] if r else None, round_mapped_up_to_s=upto,
            seen_px=int(m.sum()), marks=[{k: v for k, v in mk.items() if k != "polygons"} for mk in live]), indent=1))
        print("FRAME", n_out + 1, s, t, "round", r["k"] if r else None, "marks", len(live), flush=True)
if "--encode" in sys.argv:
    win = lambda p: subprocess.run(["wslpath", "-w", str(p)], capture_output=True, text=True).stdout.strip()
    parts = [d for d in (out / "frames_a", out / "frames_b") if d.is_dir() and any(d.glob("*.png"))]
    inputs = [a for d in parts for a in ("-framerate", str(fps), "-i", win(d) + "\\%05d.png")]
    concat = "".join(f"[{k}:v]" for k in range(len(parts))) + f"concat=n={len(parts)}:v=1[v]"
    subprocess.run([FFMPEG, "-y", *inputs, "-filter_complex", concat, "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "18", "-preset", "slow", win(out / f"{name}.mp4")], check=True)
    print("VIDEO", out / f"{name}.mp4", [len(list(d.glob("*.png"))) for d in parts], "frames", fps, "fps")
