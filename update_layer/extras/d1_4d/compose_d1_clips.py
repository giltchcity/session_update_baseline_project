"""The D1 4D clips (user 10-23: "做出来先在视频里看那两段是否连续、对得上真值"): for every watched move, in close-up and at the
scene's own speed (12 frames per second of renders 1/12 s apart), three panels side by side:
  ground truth (Isaac) | our 4D map, rounds only | our 4D map with the tracked furniture (render_d1_4d.py)
with the time, the round shown, and the ground truth's move (left .. arrived). Writes OUT_DIR/frames/NNNNN.png and
OUT_DIR/d1_4d_clips.mp4.
  compose_d1_clips.py D1_DIR OUT_DIR"""
import json, subprocess, sys
from pathlib import Path
import cv2
import numpy as np

D, out = Path(sys.argv[1]), Path(sys.argv[2])
I = Path("/mnt/d/3Study/ETH/FT/Synthetic_ABC/final_state_renders_20261022")
FFMPEG = "/mnt/c/ProgramData/ScreenToGif/ffmpeg.exe"
(out / "frames").mkdir(parents=True, exist_ok=True)
EVENTS = [("b_72", "B", "chair_0004", 41.2, 46.1), ("b_164", "B", "table_0003", 30.8, 32.4), ("b_109", "B", "flower_0003", 110.7, 115.5),
          ("a_164", "A", "table_0003", 116.2, 121.2)]
PW, PH, BAND, FOOT = 640, 360, 60, 44


def label(img, text, org, scale=0.6, color=(20, 20, 20), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick + 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


n = 0
for key, S, name, left, arrived in EVENTS:
    ours = json.loads((D / f"ours_{key}" / "renders.json").read_text())["renders"]
    plain = {r["t"]: r for r in json.loads((D / f"rounds_{key}" / "renders.json").read_text())["renders"]}
    gt = sorted((I / f"d1_{key}").glob("*_rgb.png"), key=lambda p: float(p.name.split("_")[1]))
    gt_at = {round(float(p.name.split("_")[1]), 2): p for p in gt}
    for r in ours:
        t = r["t"]
        g = gt_at.get(round(t, 2))
        if g is None or t not in plain:
            continue
        a = cv2.imread(str(g)); dep = np.load(str(g).replace("_rgb.png", "_depth.npz"))["depth"]
        a[~np.isfinite(dep)] = 255
        b = cv2.imread(str(D / f"rounds_{key}" / plain[t]["file"])); c = cv2.imread(str(D / f"ours_{key}" / r["file"]))
        panels = [cv2.resize(x, (PW, PH), interpolation=cv2.INTER_AREA) for x in (a, b, c)]
        frame = np.full((BAND + PH + FOOT, 3 * PW, 3), 255, np.uint8)
        for j, p in enumerate(panels):
            frame[BAND:BAND + PH, j * PW:(j + 1) * PW] = p
        moving = left <= t <= arrived
        label(frame, f"Ground truth (Isaac), {S} t = {t:.2f} s", (10, 24), 0.6, (20, 20, 20), 1)
        label(frame, f"Ours, rounds only: round {plain[t]['round']}", (PW + 10, 24), 0.6, (20, 20, 20), 1)
        label(frame, f"Ours with D1 4D Gaussians: round {r['round']}" + (" + tracked" if r["tracked"] else ""), (2 * PW + 10, 24), 0.6, (20, 20, 20), 1)
        label(frame, f"{name}: moved {left:.1f} - {arrived:.1f} s (ground truth)" + ("   MOVING" if moving else ""), (10, 50), 0.55,
              (0, 0, 200) if moving else (90, 90, 90), 1)
        label(frame, "the moving piece in the right panel: its Gaussians posed by tracking the frames that watch it (yaw + translation per frame)",
              (10, BAND + PH + 28), 0.5, (70, 70, 70), 1)
        cv2.imwrite(str(out / "frames" / f"{n:05d}.png"), frame); n += 1
    print(key, "frames so far", n, flush=True)
win = lambda p: subprocess.run(["wslpath", "-w", str(p)], capture_output=True, text=True).stdout.strip()
subprocess.run([FFMPEG, "-y", "-framerate", "12", "-i", win(out / "frames") + "\\%05d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "18", win(out / "d1_4d_clips.mp4")], check=True)
print("VIDEO", out / "d1_4d_clips.mp4", n, "frames at 12 fps")
