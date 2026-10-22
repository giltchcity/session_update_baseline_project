"""Hindsight against the incremental 4D map (user 10-23: "如果我们的4DGS 能把这个东西补回去呢"): for each close-up, at the
scene's own speed (12 frames per second of renders 1/12 s apart), three panels side by side:
  ground truth (Isaac) | ours, incremental (the latest round at or before t) | ours in hindsight (hindsight_render.py)
Writes OUT_DIR/frames/NNNNN.png and OUT_DIR/hindsight_clips.mp4.
  compose_hindsight_clips.py D1_DIR HINDSIGHT_DIR OUT_DIR KEY:SESSION:NAME ..."""
import json, subprocess, sys
from pathlib import Path
import cv2
import numpy as np

D, Hd, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
I = Path("/mnt/d/3Study/ETH/FT/Synthetic_ABC/final_state_renders_20261022")
FFMPEG = "/mnt/c/ProgramData/ScreenToGif/ffmpeg.exe"
(out / "frames").mkdir(parents=True, exist_ok=True)
PW, PH, BAND = 640, 360, 40


def label(img, text, org, scale=0.6, color=(20, 20, 20), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick + 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


n = 0
for spec in sys.argv[4:]:
    key, S, name = spec.split(":")
    inc = {r["t"]: r for r in json.loads((D / f"rounds_{key}" / "renders.json").read_text())["renders"]}
    hin = json.loads((Hd / f"hind_{key}" / "renders.json").read_text())["renders"]
    gt = {round(float(p.name.split("_")[1]), 2): p for p in (I / f"d1_{key}").glob("*_rgb.png")}
    for r in hin:
        t = r["t"]; g = gt.get(round(t, 2))
        if g is None or t not in inc:
            continue
        a = cv2.imread(str(g)); dep = np.load(str(g).replace("_rgb.png", "_depth.npz"))["depth"]; a[~np.isfinite(dep)] = 255
        b = cv2.imread(str(D / f"rounds_{key}" / inc[t]["file"])); c = cv2.imread(str(Hd / f"hind_{key}" / r["file"]))
        frame = np.full((BAND + PH, 3 * PW, 3), 255, np.uint8)
        for j, x in enumerate((a, b, c)):
            frame[BAND:, j * PW:(j + 1) * PW] = cv2.resize(x, (PW, PH), interpolation=cv2.INTER_AREA)
        label(frame, f"Ground truth (Isaac), {S} t = {t:.2f} s   ({name})", (10, 27))
        label(frame, f"Ours, incremental: round {inc[t]['round']} (known at t)", (PW + 10, 27))
        label(frame, "Ours in hindsight (filled back)", (2 * PW + 10, 27))
        cv2.imwrite(str(out / "frames" / f"{n:05d}.png"), frame); n += 1
    print(key, "frames so far", n, flush=True)
win = lambda p: subprocess.run(["wslpath", "-w", str(p)], capture_output=True, text=True).stdout.strip()
subprocess.run([FFMPEG, "-y", "-framerate", "12", "-i", win(out / "frames") + "\\%05d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "18", win(out / "hindsight_clips.mp4")], check=True)
print("VIDEO", out / "hindsight_clips.mp4", n, "frames at 12 fps")
