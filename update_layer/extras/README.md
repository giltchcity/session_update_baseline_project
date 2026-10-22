# Extras: the 4D map, shown and extended (synthetic A/B)

Extra features on top of the update layer's 4D map (`backends/game/snapshots.py`: every round's live map; the map of
time t is our latest round at or before t). None of them changes the mapping or a reported metric. They read a mapped
run (`syn_snap_c298ed5`), the dataset's frames and, for the comparison panels only, Isaac's ground truth. Data and
output paths in the scripts are this machine's (`runs/update_layer_game_T1_20261006/...`, `D:\3Study\ETH\FT\Synthetic_ABC\...`).
GPU jobs go through `video4d/gpu_run.sh` (`runs/GPU.lock`, one job at a time, stopped above 14.5 GB).

## video4d: the whole mapping as a video

Left: Isaac's scene at time t from one fixed oblique camera (ceiling hidden, walls cut to the map's 2.3 m); right: our
4D map of t from the same camera. The dataset camera's view, path and frustum are drawn; furniture changes are outlined
(red where it was, green where it is), from the move until the camera has seen it and our map has a round after that,
plus 10 s. Both panels are cut to the same window (balcony, living room, bedroom, the kitchen and bathroom doors).

| script | does |
|---|---|
| `render_video_isaac.py` | Isaac (run with Isaac's `python.bat`): a session at a series of times from a camera JSON. `--history cut --subframes 24`: each image is its time's alone (the renderer otherwise keeps a fading ghost of removed objects while the camera stands still). Also writes the hidden lamps' boxes. |
| `render_4d_rounds.py` | every round of our run from the same camera (lamps out, ceiling cut) |
| `changes.py` | the outlines: furniture changes from the ground-truth states and when the dataset camera saw them |
| `compose_video.py` | the frames and the H.264 video |
| `isaac_run.sh`, `isaac_full2.sh`, `full7.sh` | how v7 was made: Isaac every 0.25 s, composed at 24 fps (6x) |
| `make_camera.py`, `camera_video_v2.json` | the camera |

## d1_4d: watched moves as 4D Gaussians

A piece of furniture the camera watches being moved (D1, the visible changes) is tracked frame by frame as a rigid body
(`track_d1.py`: its Gaussians in our latest round before the move; per frame a yaw and a 3D translation by trimmed ICP to
the depth of its instance-labelled pixels). In the 4D map (`render_d1_4d.py`) its rows are replaced from the move on by
the posed model, held at its last pose afterwards: the move is continuous, and only the final state stays (the round
maps smear it). Synthetic: B chair_0004, B table_0003, B flower_0003, A table_0003; tracking residual 0.9-3.6 cm,
the last pose 1-4.5 cm from the ground truth's state after the move.

| script | does |
|---|---|
| `track_d1.py` | one watched move's track |
| `render_d1_4d.py` | the 4D map with the tracked furniture (or `--rounds-only`) at given times |
| `compose_d1_clips.py` | the close-up clips: ground truth, rounds only, with the tracked furniture |
| `render_all.sh`, `render_plant.sh`, `rerender_ours.sh`, `render_closeup_gt.sh`, `cam_*.json` | how the clips were made |

## hindsight: the 4D map filled back

Once a session is mapped, every object state of our rounds gets the interval it existed in, filled back where our own
observations (depth and instance labels of the dataset's frames) allow it (`hindsight_intervals.py`): a state first
seen at t whose place was never seen empty before is there from the session's start (a change between the sessions, or
an object seen late); a state inherited from the session before whose place is seen empty and never with it is gone
from the start; an unwatched move keeps the time it was seen (when it happened in between is not known); one object is
in one place; watched moves follow their track. `hindsight_render.py` draws the session's last background with the
states of each moment; `compose_hindsight_clips.py`: ground truth, incremental (known at t), in hindsight.
Not yet: the background is the session's last, so it can carry unlabelled parts of objects that came later.

| script | does |
|---|---|
| `hindsight_intervals.py` | the intervals of a session's object states |
| `hindsight_render.py` | the 4D map in hindsight at given times |
| `compose_hindsight_clips.py`, `run_hindsight.sh` | the clips (B table_0003, chair_0004) |
