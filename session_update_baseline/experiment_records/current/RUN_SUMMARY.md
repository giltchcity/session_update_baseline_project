# Khronos Office Reversed A/B 30 s

Date: 2026-07-30

## Protocol

- Session A uses the first 30 seconds of the official Office sequence in forward order.
- Session B uses the last 30 seconds (217.27 to 187.27 seconds) in reverse order.
- Both derived bags use a new monotonic session clock starting at 1 second.
- Khronos uses the same mapper configuration and `MESH_ONLY` mode for A and B.
- Playback rate is fixed to `1`. The earlier `A_khronos_v2` run at rate `5` is invalid because only about 18 frames were processed.

## Khronos Input Gates

| Run | Source images | Processed timing rows | Saved map times | Final map |
|---|---:|---:|---:|---:|
| A | 600 | 570 including header | 13 | 79 MB |
| B from scratch | 601 | 586 including header | 13 | 150 MB |

Khronos reported a clean experiment finish for both runs. A FastDDS shutdown
segmentation fault may occur after all map files have been written; it is not used
as evidence of mapping success. Success is established from the clean-finish flag,
saved map count, processing counts, and successful map reload.

## Legacy Object-Memory-Only Results

The following results predate global-map continuation. They loaded A object memory,
but still used B's from-scratch global mesh as the output map body. Keep them only
for traceability; they do not satisfy the cross-session global-map gate.

| Output | Initial vertices | Removed | Injected | Final vertices |
|---|---:|---:|---:|---:|
| A within-session | 28,260 | 501 | 8,835 | 36,594 |
| B without A memory | 31,904 | 1,521 | 10,467 | 40,850 |
| B with A memory | 31,904 | 1,521 | 10,752 | 41,135 |

The two B controls used byte-identical original maps:

`8b0e5918340bf748cd9739738e546f470dee9f456aa9c119555b272cf56c0535`

The improved outputs differ:

- B without memory: `ad665929207977abb2cdd60b0799208948a39a9bd0cfe972733acbb0cf0b31a6`
- B with A memory: `cf3a71d001f55a74edd0cc1bf982d19d3054375dfc52b747a82093b94399eec3`

Cross-session evidence for B with A memory:

- 11 prior objects loaded
- 6 prior/current matches
- 13 unmatched current objects
- 1 prior object classified absent
- 4 prior objects classified unobserved
- 4 prior objects restored
- 285 additional vertices relative to the no-memory result

The B with-memory map passed a no-op reload test with 41,135 vertices unchanged.

## Unified Global-Map Continuation

The corrected implementation uses A's improved global mesh as the Session B map
body. B's Khronos reconstruction is treated as current-session observation
evidence:

1. A vertices supported by nearby B surface are persistent.
2. Unsupported A vertices are removed only with reliable Khronos ray/free-space
   absence evidence.
3. Unsupported vertices without reliable evidence stay unobserved and are kept.
4. B surfaces farther than 8 cm from the retained A mesh are inserted as new
   current geometry.
5. The existing object-guided cleanup and repair pass is then applied to this
   reconciled current map.

Formal output:

`B_ours_with_A_global_memory_unified/improved_final.4dmap`

| Quantity | Vertices |
|---|---:|
| A prior global mesh | 36,594 |
| B current evidence mesh | 31,904 |
| A prior classified absent | 2,441 |
| A prior classified persistent | 23,678 |
| A prior classified unobserved and retained | 10,475 |
| New B surface inserted before object reconciliation | 14,851 |
| Object-guided vertices removed | 6,932 |
| Object private-mesh vertices injected | 7,395 |
| Final unified current map | 49,467 |

The saved map reloads through Khronos map IO and exports a mesh with 49,467
vertices and 58,599 faces.

### Continuity Gates

- A-to-A idempotence: 36,594 -> 36,594 vertices, zero absent vertices, zero
  inserted current vertices. The binary PLY geometry payload is byte-identical.
- First B checkpoint: starts from all 36,594 A vertices, not the 5,156-vertex B
  partial reconstruction. It classifies A as 31 absent, 7,531 persistent, and
  29,032 unobserved, inserts 412 B vertices, and outputs 36,975 vertices.
- Full A/B sequence: 26 loadable visualization frames. Every B checkpoint is
  independently reconciled from the immutable A final map plus cumulative B
  evidence, so update errors are not accumulated from one visualization
  checkpoint to the next.

Artifacts:

- `process_visualization_unified/sequence_manifest.csv`
- `process_visualization_unified/frames/`
- `B_ours_with_A_global_memory_unified/evidence_summary.json`
- `B_ours_with_A_global_memory_unified/mesh_update_summary.csv`
- `B_ours_with_A_global_memory_unified/object_update_summary.csv`

## Welded Surface And Dynamic-Track Extension

The second unified implementation replaces vertex-only B insertion with
nearest-neighbor vertex welding while retaining B's mapped triangle topology.
This addresses holes where a B triangle previously disappeared whenever one of
its vertices was considered too close to A.

Formal welded output:

`B_ours_with_A_global_memory_unified_welded/improved_final.4dmap`

The vertex decisions are unchanged from the first unified result, while surface
connectivity changes as follows:

| Topology quantity | Vertex-only merge | Welded merge |
|---|---:|---:|
| Final vertices | 49,467 | 49,467 |
| Final faces | 58,599 | 63,546 |
| Boundary edges | 25,977 | 24,789 |
| Low-height faces | 15,016 | 15,774 |

The welded merge adds 4,947 mapped faces, removes 1,188 open boundary edges,
and adds 758 low-height faces without adding extra vertices. A-to-A strict
idempotence remains 36,594 vertices and 43,565 faces with zero insertions.

The process exporter now reads dynamic objects directly from each current
Khronos snapshot:

- A checkpoints 0-1: zero confirmed dynamic tracks.
- A checkpoints 2-5: one 30-point track.
- A checkpoints 6-12: two tracks, including a human-sized 113-point track.
- B checkpoints 0-12: zero dynamic tracks detected in this selected segment.

Dynamic trajectories and current bounding boxes are rendered as magenta
overlays. They are not inserted into the static global mesh. B overlays are
read from B's observation snapshot, not restored A history.

Updated process artifacts:

- `process_visualization_unified_welded_dynamic/sequence_manifest.csv`
- `process_visualization_unified_welded_dynamic/frames/*.ply`
- `process_visualization_unified_welded_dynamic/frames/*.json`

## Time-Aware Dynamic Surface Replay

The first dynamic viewer always placed a bounding box at the last trajectory
position and drew the complete trajectory at every query time. That was a
visualization error. The corrected viewer:

- selects the current dynamic sample from `session_time_s`,
- truncates the trajectory to the current query time,
- places the bounding box at the selected trajectory position,
- hides the current box after the track ends while retaining its history, and
- renders the stored dynamic point frame when one exists.

The original A run used `store_visualization_details: false`, so its
`dynamic_object_points` arrays were empty even though trajectory centroids were
stored. A controlled A rerun changed only this recording switch to `true` and
saved:

`A_khronos_dynamic_details/final.4dmap`

The resulting dynamic data are:

| Track | Label | Trajectory frames | Non-empty point frames | Stored points |
|---|---:|---:|---:|---:|
| Small motion track | 2 | 32 | 32 | 34,137 |
| Human-sized geometry track | -1 | 111 | 111 | 149,507 |

Label `-1` means the human-sized track was obtained from geometric motion
evidence rather than a semantic `Human` classification. It must not be reported
as semantic human recognition.

The unified process now has 26 real map snapshots and a 0.1 s display manifest
with 554 rows. Densification holds the nearest real map snapshot and only
interpolates display/query time; it does not fabricate intermediate map states.

Updated artifacts:

- `process_visualization_unified_welded_dynamic_v2/sequence_manifest.csv`
- `process_visualization_unified_welded_dynamic_v2/sequence_manifest_dense.csv`
- `process_visualization_unified_welded_dynamic_v2/diagnostics/A_dynamic_history.json`
- `process_visualization_unified_welded_dynamic_v2/view_process.py`

## Session-History and Dual-Trajectory Visualization

The process viewer now separates the saved current map from comparison-only
history layers:

- the current reconciled map is rendered as the solid mesh,
- after entering Session B, Session A's final mesh can be overlaid as a
  translucent blue history/reference layer,
- Session A's robot trajectory is cyan,
- Session B's robot trajectory is orange, and
- the configured 5 m sensor maximum-range envelope is a yellow horizontal ring, and
- dynamic-entity trajectories remain magenta.

During Session A, the A robot trajectory is truncated at the current query
time. During Session B, the complete A trajectory remains visible and the B
trajectory grows with B query time. The trajectories are read directly from
`AgentNodeAttributes` in the two frontend DSG files, not inferred from mesh
geometry:

| Session | Direct DSG agent poses |
|---|---:|
| A | 36 |
| B | 56 |

The translucent A mesh is a viewer-only comparison layer. It is not inserted
again into `improved_final.4dmap`; the saved B output remains one reconciled
current map containing retained A memory and accepted B updates.

Artifacts:

- `process_visualization_unified_welded_dynamic_v2/session_trajectories.json`
- `process_visualization_unified_welded_dynamic_v2/session_ab_trajectories_16x9.png`
- `scripts/extract_khronos_session_trajectories.py`
- `scripts/plot_session_ab_trajectories.py`

## Synchronized Sensor-View Insets

The A/B process viewer now also shows the actual RGB observations preserved in
the two retimed Office bags:

- during Session A, one inset shows the current Visit 1 RGB frame;
- during Session B, the right inset shows the current Visit 2 RGB frame and the
  left inset shows the spatially nearest Visit 1 RGB frame;
- the Visit 1 inset reports the camera-position distance to avoid presenting a
  weakly overlapping view as an exact correspondence.

The extracted view index contains 240 Visit 1 frames and 241 Visit 2 frames at
approximately 0.1 s spacing. At the first Session B checkpoint, the selected
Visit 1 and Visit 2 camera positions differ by 0.034 m. The frames are decoded
directly from `/tesse/left_cam/rgb/image_raw`; they are not rendered from the
map.

Artifacts:

- `process_visualization_unified_welded_dynamic_v2/sensor_views.json`
- `process_visualization_unified_welded_dynamic_v2/sensor_views/A/`
- `process_visualization_unified_welded_dynamic_v2/sensor_views/B/`
- `scripts/khronos/extract_office_ab_sensor_views.py`

The viewer's Session A dynamic-history overlay was subsequently upgraded from
the geometric-only Human track to the controlled semantic-assisted track:

| Track | Label | Frames | Session-time interval |
|---|---:|---:|---:|
| Geometric-only Human | -1 | 111 | 5.97-11.62 s |
| Semantic-assisted Human | 17 | 170 | 5.92-14.52 s |

At dense viewer frame 99 (session 13.37 s / source 12.37 s), the new Human
track is active and contains 1,331 current dynamic points. The replaced
geometric-only overlay remains available as
`diagnostics/A_dynamic_history_geometric_only.json`.

## Dynamic Residue Diagnostic

The human-sized trajectory's swept bounding boxes contain 147 final global-mesh
vertices; 82 have mesh timestamps within the dynamic track interval. Bounding
boxes are too coarse to justify deletion, so the audit also measures distance
to the actual stored dynamic point frames:

| Dynamic-surface radius | Nearby global vertices | Also time-overlapping |
|---|---:|---:|
| 0.05 m | 0 | 0 |
| 0.10 m | 20 | 18 |
| 0.20 m | 114 | 75 |

This supports the presence of nearby residual geometry at relaxed tolerances,
but does not prove every nearby vertex is a stale human surface. Automatic
deletion remains disabled. A safe deletion rule must combine exact dynamic
support, temporal support, later free-space contradiction, and structural
surface protection.

Artifacts:

- `A_dynamic_residue_diagnostic_exact/object_audit.csv`
- `A_dynamic_residue_diagnostic_10cm/object_audit.csv`
- `A_dynamic_residue_diagnostic_20cm/object_audit.csv`

## Dynamic-Residue Cleaner

Base1 now has an isolated `dynamic_cleanup` mode. It does not run the static
private-object cleanup, so all reported changes are attributable to dynamic
residue handling.

The per-vertex gate is:

```text
near a stored dynamic point frame
AND mesh timestamp overlaps the dynamic track
AND not a protected semantic or horizontal structural surface
AND (
  later free-space confirms absence
  OR transient quarantine is explicitly enabled and no later persistent ray exists
)
```

The current static mesh may exclude quarantined vertices, while the dynamic
object trajectory and point-frame history remain in the saved DSG.

### Failed distance-only ablation

At 0.20 m, 119 vertices met the dynamic-distance and time gates. A diagnostic
run without geometric plane protection removed all 119 and slightly reduced
the fixed-GT final-snapshot score:

| Metric | Original | Distance-only quarantine |
|---|---:|---:|
| F1 @ 0.05 m | 0.1614 | 0.1607 |
| F1 @ 0.10 m | 0.2423 | 0.2415 |
| F1 @ 0.20 m | 0.2709 | 0.2704 |

Spatial inspection showed that all 119 vertices lie on a 1.1 x 1.2 m
horizontal sheet at approximately `z=1.00 m`, only 1.4 cm thick. They are floor
support under the moving person, not a reconstructed human body. This ablation
is retained as a failure case and must not be reported as an improvement.

### Safe result

The final run adds an incident-face-normal structural guard:

```text
abs(normal.z) >= 0.85 -> protect horizontal surface
```

with Office semantic labels `5,6,12` also protected. The result is:

| Quantity | Count |
|---|---:|
| Dynamic tracks | 2 |
| Stored dynamic support points in `.4dmap` | 781,997 |
| Distance/time candidates | 119 |
| Geometrically protected floor vertices | 119 |
| Later-absence confirmed vertices | 0 |
| Quarantined vertices | 0 |
| Removed vertices | 0 |

This only characterizes vertices near the dynamic point frames that Khronos
actually stored. It shows that the 119 near-track candidates are floor, but it
does not rule out human geometry fused after geometric tracking was lost and
the person moved beyond the recorded trajectory. The corrected viewer still
hides stale dynamic points and the current bounding box 0.3 s after the track
ends, while retaining the trajectory as history; that viewer fix is independent
of global-mesh residue.

Artifacts:

- `A_dynamic_cleanup_safe_final/improved_final.4dmap`
- `A_dynamic_cleanup_safe_final/mesh_vertex_update_summary.csv`
- `A_dynamic_cleanup_safe_final/evidence_summary.json`
- `A_dynamic_cleanup_10cm_quarantine/`
- `A_dynamic_cleanup_20cm_quarantine/`

## Dynamic Semantic Mask After Track Loss

The main human-sized geometric track ends at `11.620002 s`, even though the
input segmentation continues to contain Human pixels after that time. A direct
bag audit found:

| Quantity | Value |
|---|---:|
| Segmentation frames | 600 |
| Frames containing raw Human IDs 131 or 201 | 339 |
| Human frames after 10.62 s | 126 |
| Last Human segmentation time | 25.550002 s |

Khronos previously masked only the current geometric `dynamic_image` before
TSDF integration. Its source also contained the unresolved comment
`TODO(nathan) also add semantic masking`. The controlled patch now combines:

```text
configured dynamic semantic labels
OR current geometric dynamic-image pixels
-> static TSDF integration mask
```

The implementation is in
`khronos/src/active_window/active_window.cpp::ActiveWindow::updateMap()` and the
portable patch is:

`session_update_baseline/ports/khronos_core/0001-mask-configured-dynamic-semantics.patch`

The control and treatment runs use the same deduplicated 2978-pose CSV. The
three input files have the identical SHA-256:

`9faf2986caf94fd2a85c518115e702794edd7bbc466472f116c046ecbb3aa9d1`

Both runs finished cleanly. The main trajectory still ends at `11.620002 s`;
the patch does not invent motion after tracking is lost. A 5 cm directed
old-versus-patched mesh comparison finds one dominant old-only component:

| Quantity | Value |
|---|---:|
| Old-only component vertices | 138 |
| Center [m] | `[-11.318, 26.019, 1.981]` |
| Extent [m] | `[0.369, 0.446, 1.331]` |

This component has human scale and lies beyond the last recorded track position
in the person's continued direction of travel. It is absent from the
semantic-mask reconstruction. The patched-only differences contain no
human-sized connected component.

This satisfies the conservative branch of the requested behavior:

```text
geometric tracking may end
but Human-labeled pixels remain excluded from the static global mesh
so no untracked human-shaped residual is fused
```

Artifacts:

- `controlled_semantic_mask_compare/fixed_pose_0_29p95.csv`
- `controlled_semantic_mask_compare/unpatched_dynamic_only/final.4dmap`
- `controlled_semantic_mask_compare/patched_semantic_dynamic/final.4dmap`
- `controlled_semantic_mask_compare/mesh_difference/comparison_overlay.ply`
- `controlled_semantic_mask_compare/mesh_difference/summary.json`
- `scripts/khronos/audit_semantic_human_frames.py`
- `scripts/khronos/compare_dynamic_semantic_mask_meshes.py`

## Semantic-Assisted Dynamic Tracking

The semantic TSDF mask prevents Human pixels from entering the static mesh, but
the original tracker still records only geometric dynamic clusters. Base1 now
optionally promotes connected components of configured dynamic semantic labels
to Khronos dynamic measurements:

```text
connected Human semantic component
-> dynamic cluster
-> existing MaxIoUTracker
-> trajectory + temporal bbox + per-frame point cloud
```

The option is disabled by default in the Khronos class and enabled only in the
Base1 Office mapper configuration. A geometric dynamic cluster substantially
covered by a promoted semantic component is replaced to prevent duplicate
tracks. Unrelated geometry-only dynamics remain unchanged. A promoted dynamic
measurement can also initialize the track's semantic label.

The final controlled run uses the same fixed 2,978-pose CSV as the previous
runs. Its SHA-256 is again:

`9faf2986caf94fd2a85c518115e702794edd7bbc466472f116c046ecbb3aa9d1`

| Human-track quantity | Geometric-only Khronos | Semantic-assisted |
|---|---:|---:|
| Semantic label | unknown (`-1`) | Human (`17`) |
| Stored samples | 114 | 170 |
| First sample | 5.970002 s | 5.920002 s |
| Last sample | 11.620002 s | 14.520002 s |
| Last centroid x | -14.789 m | -11.244 m |

The magenta dynamic surface in the process viewer is the stored point cloud for
the selected trajectory frame. It is not a private object mesh. The magenta
line is the centroid trajectory and the magenta wireframe is the temporal
bounding box.

The remaining Human segmentation intervals are:

```text
0.00-13.50 s
16.85-19.85 s
25.25-25.55 s
```

Only eight frames of the second interval (`16.85-17.20 s`) contain at least 50
Human depth pixels inside the configured 5 m sensing range. The rest are about
5.2-6.4 m away, and the final interval is about 10-11.6 m away. Eight valid
frames are below the existing object-allocation confidence requirement, so the
second interval is not exported as a reliable trajectory. Base1 does not invent
identity or motion outside the reliable sensor range.

Increasing the tracker timeout from 3 s to 4 s did not bridge this gap and
increased the risk of associating an old dynamic track with a different nearby
target. The final configuration therefore retains the original 3 s timeout.

A fresh 5 cm directed mesh comparison against the geometric-only run reproduces
the same dominant old-only Human-scale component:

| Quantity | Value |
|---|---:|
| Old-only component vertices | 138 |
| Center [m] | `[-11.318, 26.019, 1.981]` |
| Extent [m] | `[0.369, 0.446, 1.331]` |

Thus the final behavior is:

```text
reliably visible Human -> longer semantic dynamic track
lost or out-of-range Human -> no fabricated continuation
all configured Human pixels -> excluded from static TSDF
```

Artifacts:

- `controlled_semantic_mask_compare/semantic_assisted_segmented/final.4dmap`
- `controlled_semantic_mask_compare/semantic_assisted_segmented/dsg.json`
- `controlled_semantic_mask_compare/semantic_assisted_segmented_control/`
- `controlled_semantic_mask_compare/semantic_assisted_segmented_mesh_difference/`
- `ports/khronos_core/0002-promote-dynamic-semantic-clusters.patch`
- `ports/khronos_core/0003-preserve-promoted-dynamic-semantics.patch`
- `ports/khronos_core/0004-accept-semantic-dynamic-tracks.patch`

## Structural Change Extension

The three temporal modes are represented by one generic scene-element state
machine. A scene element may be an object or a locally connected structural
surface patch:

`present -> dynamic | persistent | absent | unobserved | new`

Wall/floor/ceiling semantics are optional evidence, not a prerequisite for
geometric change detection. Geometry and re-observation determine the state;
semantics improve patch grouping, structural protection, and class-dependent
priors. The detailed design and evidence gates are recorded in:

`session_update_baseline/THREE_MODE_UNIFIED_MODEL.md`

This design has not yet been implemented as a full structural-patch updater.
The welded A/B result only verifies that cross-session surface topology can
repair part of the floor seam without duplicating it.

## Interpretation Boundary

This run proves that the complete adapter -> Khronos -> Base1 A/B pipeline executes,
that A global geometry is the actual Session B map starting point, and that B
evidence changes both inherited geometry and object decisions.

It does not yet prove a metric improvement. The artificial reverse session has a
retimed clock, so the stock Khronos temporal evaluator would compare B against the
wrong Office ground-truth time unless source-time mapping is added to the evaluator.
Do not report Background/Object/Change P/R/F1 from this run as official metrics.

This is reconciliation-level continuation. Khronos still reconstructs B
observations from scratch, while Base1 starts the output current map from A and
updates it with B evidence. It does not restore Khronos's live PGMO graph, ray hash,
active window, or asynchronous backend threads. Therefore this result is not a
native Khronos backend hot restart and must not be described as one.

## Consistent Semantic A/B Process Visualization

The earlier `process_visualization_unified_welded_dynamic_v2` mixed the newer
semantic-assisted dynamic overlay with mesh checkpoints generated from the older
geometric-only A run. This made the Human trajectory look correct while the old
gray Human-shaped TSDF residual remained visible.

The fully consistent replacement is:

`process_visualization_semantic_unified_v3/`

Its complete data chain is:

```text
semantic-assisted A checkpoints
-> semantic-assisted A improved final map and object memory
-> B checkpoint reconciliation using that exact A memory
-> dense 0.1 s visualization timeline
```

The dense timeline contains 554 rows: 277 for A and 277 for B. It holds the most
recent real map checkpoint between checkpoint times; it interpolates only the
display/query time and does not synthesize intermediate mesh geometry.

Direct ghost check at A checkpoint 5:

| Check | Result |
|---|---:|
| v3 mesh vertices | 12,964 |
| semantic-assisted source vertices | 12,964 |
| Maximum corresponding vertex delta | 0 m |
| v3/source triangle indices equal | yes |
| Old geometric-only vertices farther than 5 cm in Human swept region | 139 |

The old-only component is therefore absent from the v3 main mesh rather than
hidden by the magenta overlay. The dynamic Human is represented by the existing
Khronos trajectory, temporal bbox, and per-frame point cloud; Human label 17 has
no private static object mesh and is not injected by Base1.

The synchronized inset set contains 240 Visit A RGB frames and 241 Visit B RGB
frames. The trajectories are extracted directly from the DSG agent layers:
37 A poses and 56 B poses.

Launch:

```bash
/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python \
  /home/jixian/Desktop/FT/session_update_baseline/rounds/khronos_office_reversed_ab_30s_20260730/process_visualization_semantic_unified_v3/view_process.py \
  --manifest /home/jixian/Desktop/FT/session_update_baseline/rounds/khronos_office_reversed_ab_30s_20260730/process_visualization_semantic_unified_v3/sequence_manifest_dense.csv \
  --frame-seconds 0.1
```

## Unified Object-Layer Display Fix

At Session B checkpoint 3, the process viewer appeared to remove a chair even
though the current observation still contained it. The object audit showed:

```text
semantic label: 11
state: persistent_prior_matched
prior/current centroid distance: 0.017798 m
private object mesh vertices: 30,126
global vertices removed near the private surface: 2,870
```

The chair was therefore not classified absent. Base1 had separated its private
object mesh from overlapping global/background geometry, while the old process
PLY exporter displayed only the global mesh. This created a visualization-only
false deletion.

The exporter now has an opt-in `--include_object_meshes true` mode. The process
visualization generator enables it by default and retains
`--global-mesh-only` for layer diagnostics. This composition happens only in
the generated display PLY; it does not weld high-density private meshes back
into `.4dmap`, alter reconciliation state, or change evaluator input.

At the affected chair bbox:

| Display | Vertices in bbox |
|---|---:|
| global-only v3 | 156 |
| global + valid object layer v4 | 33,867 |

The complete corrected visualization is:

`process_visualization_semantic_unified_v4_objects/`

It contains 554 dense rows with no missing frame references, synchronized
Visit A/B RGB views, and 148,936,999 bytes of artifacts. Dynamic Human objects
remain trajectories/point frames because they have no private static mesh.

Launch:

```bash
/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python \
  /home/jixian/Desktop/FT/session_update_baseline/rounds/khronos_office_reversed_ab_30s_20260730/process_visualization_semantic_unified_v4_objects/view_process.py \
  --manifest /home/jixian/Desktop/FT/session_update_baseline/rounds/khronos_office_reversed_ab_30s_20260730/process_visualization_semantic_unified_v4_objects/sequence_manifest_dense.csv \
  --frame-seconds 0.1
```

## Re-observation-Gated Cross-Session Fusion

The v4 run exposed a separate map-update error near the chair: object cleanup
removed global vertices solely because they were close to a private object mesh
or inside its expanded bbox. This could also remove unchanged wall or floor
geometry near the object.

Cross-session object cleanup now uses the following evidence gate:

```text
current B surface within 8 cm -> keep
ray reports presence          -> keep
no reliable ray observation   -> unobserved, keep
ray reports absence           -> removal is allowed
```

This gate is enabled only for the cross-session visualization run. It does not
disable the existing prior-map change detector. The prior map is still split
into absent, persistent, and unobserved vertices using B's ray evidence; B
geometry not already represented within 8 cm is then added to the retained A
geometry.

Controlled B checkpoint 3 comparison:

| Quantity | v4 | v5 gated |
|---|---:|---:|
| Initial A vertices | 39,610 | 39,610 |
| Prior vertices ray-confirmed absent | 2,786 | 2,786 |
| Current B vertices added | 1,981 | 1,981 |
| Object-cleanup vertices removed | 2,939 | 0 |
| Object-cleanup vertices protected by current surface | 0 | 2,938 |
| Object-cleanup vertices protected as unobserved | 0 | 1 |
| Final global vertices | 35,878 | 38,817 |

Across all 13 B checkpoints, prior-map ray-confirmed deletion remained active
(600 to 3,403 vertices), while current B observations added 170 to 14,702
vertices. No object-cleanup candidate in this sequence had ray-confirmed
absence; therefore object cleanup correctly deleted none. At the final
checkpoint, 6,524 candidates were protected by current B surfaces and 181 were
protected as unobserved.

The complete replacement visualization is:

`process_visualization_semantic_unified_v5_reobservation/`

It contains 554 dense rows: 277 for A and 277 for B, with synchronized Visit
A/B RGB insets.

Launch:

```bash
/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python \
  /home/jixian/Desktop/FT/session_update_baseline/rounds/khronos_office_reversed_ab_30s_20260730/process_visualization_semantic_unified_v5_reobservation/view_process.py \
  --manifest /home/jixian/Desktop/FT/session_update_baseline/rounds/khronos_office_reversed_ab_30s_20260730/process_visualization_semantic_unified_v5_reobservation/sequence_manifest_dense.csv \
  --frame-seconds 0.1
```
