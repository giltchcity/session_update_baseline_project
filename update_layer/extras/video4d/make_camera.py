"""A virtual camera JSON for the paper figures: eye, look-at, world up, image size, focal length, the crop box (world
metres: the ceiling and the cut-away near wall), the objects whose ground-truth states are boxed.
  make_camera.py OUT.json NAME FRAME_NOTE EYE(x,y,z) LOOKAT(x,y,z) UP(x,y,z) W H F CROP(json) OBJECTS(json) [NOTE]"""
import json, sys
import numpy as np
out, name, frame_note = sys.argv[1:4]
E, T, U = (np.array([float(v) for v in a.split(",")]) for a in sys.argv[4:7])
W, H, f = int(sys.argv[7]), int(sys.argv[8]), float(sys.argv[9])
crop, objects = json.loads(sys.argv[10]), json.loads(sys.argv[11])
z = (T - E) / np.linalg.norm(T - E); x = np.cross(z, U); x /= np.linalg.norm(x); y = np.cross(z, x)
c2w = np.eye(4); c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, E
cam = dict(name=name, frame=frame_note, width=W, height=H, K=[[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]],
           T_camera_to_world=c2w.tolist(), T_world_to_camera=np.linalg.inv(c2w).tolist(), eye=E.tolist(), look_at=T.tolist(),
           world_up=U.tolist(), crop_box_world=crop, objects=objects, note=sys.argv[12] if len(sys.argv) > 12 else "")
json.dump(cam, open(out, "w"), indent=1); print("written", out)
