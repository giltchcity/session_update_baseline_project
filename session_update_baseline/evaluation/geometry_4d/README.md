# Home temporal geometry: Mesh and Mesh_4D (scheme C, 11-17)

The main table's Home (synthetic A -> B) columns *Mesh* and *Mesh_4D*.

## Metric

Every map slice is compared with the Isaac GT mesh of its own moment: every entity at its pose in that frame. F1 at 5 cm.

- **Precision**: 150k area samples of the map slice (seed 20260917), distance to the complete GT mesh.
- **Recall**: 150k area samples of the GT reference (seed 20261115), distance to the map slice.
- **Reference (scheme C)**: every GT entity whole, except walls, floors and ceilings, which are replaced by their
  room-facing part (`room_facing_shell.npz`). Furniture backs and undersides stay in, so they count as missed.
- **Mesh_4D**: the mean F1 over the method's own slices. Each slice is queried at the first GT frame at or after its stamp.
  B's inherited slice is queried at B's frame 0. Panoptic is queried at the 25 frames of `panoptic_query_times.csv`
  (Khronos's slices), as in its first score.
- **Mesh**: the F1 of the slice at B's last frame (frame 3554), i.e. the continued map at the end.

### Why scheme C

The Isaac shell is modelled with thickness. The complete GT mesh is 1835 m², and about half of it can never be seen:

| entity | area | what it is |
|---|---|---|
| `ceiling_0007` | 243 m² | a slab at 3.35-3.85 m, above the suspended ceiling (`/Root/Meshes/ceiling/ceiling_0007`) |
| `floor_0005` | 243 m² | a slab at -0.55 to -0.05 m, under the floor (`/Root/Meshes/floor/floor_0005`) |
| walls | 458 m² | outer faces and wall thickness |

Scene file: `D:\3Study\ETH\FT\Synthetic_ABC\assets\InteriorAgent\kujiale_0021\kujiale_0021.usda`.
Against the whole mesh, every method's recall is capped at the share any camera ever sees, about 42%. That keeps every
F1 near 50-60. It also credits thick surfaces: wavemap's occupied cells reach unseen slab and wall faces, and its recall
of 43-47% is higher than that seen share. Only the shell is reduced; nothing else is.

### How the room-facing shell is made (`room_facing_shell.py`)

1. Split walls, floors and ceilings into pieces with edges of at most 10 cm.
2. Keep a piece when a straight line from some camera position reaches its centroid within 2 cm. The camera positions
   are every 30th frame of A and B, 259 in total. Only structure blocks the line: walls, floors, ceilings, doors,
   windows, curtains and lights. Furniture does not block, so shell behind or under furniture stays and counts.

## Reproduce (in this order)

Set up the environment before every step:
`PYTHONPATH=/mnt/d/3Study/ETH/FT/Synthetic_ABC/scripts`, using the python of
`/home/jixian/Desktop/miniconda3/envs/3d_vsg`. The default output folder is
`/home/jixian/Desktop/FT/wt_takeover_runs/geometry_4d_20261114`, with results in its `room/` subfolder.

1. **`room_facing_shell.py`**: writes `room_facing_shell.npz` (45 MB, not in git; about 90 s). It must print:
   - `259 camera positions, shell 1285.9 m2 in 4769301 pieces, room-facing 442.1 m2 (34.4%)`
   - `ceiling_0007 ... kept 0.0%`, `floor_0005 ... kept 5.3%`, `floor_0002 ... kept 99.0%`
2. **`check_shell.py`**: the two checks against what the cameras saw. It must print:
   - wall / floor / ceiling: 100.0 / 99.8 / 100.0% of the observed shell points lie on the kept shell, within 5 cm.
   - 517 views: 0.112% of the visible shell pixels land on removed pieces. The worst view is a 330, at 1.28%; these
     pixels are thin strips where a 10 cm piece is only partly visible.
   - For a visual check, `view_shell.py` opens an Open3D window (WSLg). It lists every entity under its Isaac name, kept
     parts green and removed parts red. `show_shell.py` writes still pictures of the same split.
3. **`run_4d.sh`**: TSDF ours and Khronos.
4. **`run_wavemap.sh`**: Wavemap (row 3) and Occupancy ours (row 4).
5. **`run_panoptic.sh`**: Panoptic. It first rescores the complete mesh at the 25 frames; the mean F1 must be 41.6442
   (A 42.4271, B 40.9216). Then it runs scheme C at those frames and at the last frames.
6. **3DGS rows** (GaME, ours): not exported as meshes. The code is pinned in the 3DGS repo `code_mine_20261014`, branch
   `eval-slice-4d-20261117`, tag `home-geometry-3dgs-20261117` (commits cb88ad7 -> 3e8065c -> 3065368). The scripts
   are in `update_layer/eval/slice_4d/` (`slice_geometry_4d.py`, `run_geometry_4d.sh`, `room_export.py`), and its
   README lists the inputs and expected values.
   - Per slice it renders median depth at the 1,228 keyframe views (where the keyframe measured depth), fuses a 2 cm
     TSDF, and scores in memory with this reference and these seeds.
   - Slices follow the query rule above: 25 slices (13 A + 12 B). B's inherited slice is scored at B's frame 0.
   - GaME is read with each round's own parameters (its archived per-round history). 3DGS ours is read from the stored
     4D map (`snapshots.py` at 674b871).
   - It writes `room/map_vs_gt_{game,3dgs_ours}/MAP_VS_GT.csv`, plus `MAP_VS_GT_complete.csv` for the whole mesh.
7. **`summarize.py`**: the table. Expected:

| Map | Method | Mesh | Mesh_4D (A / B) |
|---|---|---|---|
| TSDF | Khronos | 76.7 | 69.5 (66.2 / 72.6) |
| TSDF | Panoptic Mapping | 76.1 | 60.3 (62.0 / 58.7) |
| TSDF | Ours | 83.0 | 75.3 (70.7 / 80.3) |
| 3DGS | GaME | 77.2 | 71.6 (68.1 / 75.4) |
| 3DGS | Ours | 81.2 | 76.9 (74.3 / 79.6) |
| Occupancy | Wavemap | 79.9 | 72.5 (67.9 / 77.5) |
| Occupancy | Ours | 80.8 | 72.8 (68.0 / 78.0) |

Run one job at a time: each uses 4 cores at nice 19, with exports in /dev/shm that are removed at the end.

## Maps used

| row | maps |
|---|---|
| TSDF ours | `G:/我的云端硬盘/FT_archive/ablation_vd_20261111/on/synthetic_{a,b}/final.4dmap.zpk`, the A -> B continued chain; its synthetic metrics equal the main table's |
| Khronos | `wt_takeover_runs/khronos_continuous/synthetic_5cm/sessions/session_{a,b}.4dmap.zpk`, official Khronos run continuously A -> B at 5 cm |
| Wavemap / Occupancy ours | timelines of `wm5_row3_22e68d6` (in `KEEP_baseline_runs_wavemap_rows13`) and `wm_synth_row4_7b63cc1` |
| Panoptic | `G:/我的云端硬盘/FT_archive/eval/panoptic_rerun_synthetic_5cm/{a,b}/native` |
| 3DGS | ours: tag `reg2_syn_a`, the stored 4D map (each session's kept parameters); GaME: published `94c0435`, each round's own parameters |

## Do not get it wrong again

- **The default reference is scheme C.** `GEO4D_RECALL=complete` gives the whole-mesh numbers of 11-15: Mesh_4D TSDF
  53.1, Khronos 48.2, Wavemap 55.0, Occupancy 55.2, Panoptic 41.6, GaME 53.2, 3DGS ours 56.7. Any other value is an error.
- **Do not use the observed-surface variants** ("observed by now", "observed by the end of the visit"). They are
  superseded, and their results are archived in `G:/我的云端硬盘/FT_archive/geometry_4d_20261114_observed_reference`.
- **Khronos must be the continuous A -> B run.** The separately run Khronos of September starts B from an empty map. It
  is archived in `G:/我的云端硬盘/FT_archive/synthetic_eval_complete_20260917_khronos`.
- **TSDF must be the continued A -> B chain**, not a single-run map such as `wt_takeover_runs/synthetic_b_single_obs`,
  which starts B empty and lacks everything only A saw.
- **The final Mesh is the slice at B's last frame.** A Khronos row at the last frame is not one of its slices: its last
  B slice is at 118.3 s. That row counts for Mesh, not for Mesh_4D.
- **3DGS ours 56.0** (complete mesh, old per-round format) cannot be reproduced. The per-round parameters were deleted
  in the 11-16 history conversion. The stored 4D map gives 56.7.
- **GaME must be read with each round's own parameters**, as the map it had at that moment. Read with the parameters
  each session kept, its early slices lose precision (11.8 s: 0.60 vs 0.84), which penalises it.
- **GaME's earlier Home Mesh 80.9 has no source.** `wt_takeover_runs/RESULTS_TABLE_20261029.md` records GaME Home
  A 94.32 / B 91.25 under the observed-surface protocol.
