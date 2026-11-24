# 4D maps kept as piece histories

The mapper keeps a 4D map's history as versioned pieces (`khronos::PieceHistory`, backend `piece_history: true`, the
default): a mesh block (the active window's voxel blocks), a node or an edge is stored again only when its content
changed, as a zstd delta on its previous version. A saved map starts with `KHRONOS-PIECE-MAP-1`.

- Every snapshot restores with the same content: mesh vertices and faces (unwelded vertices included), nodes, edges and
  attributes. The order of mesh vertices and faces is not kept: a restored mesh lists them by first-seen time and
  block. Scores that sample a mesh in face order (e.g. G1's seeded area samples) can differ from those of the same
  map kept as keyframe/delta snapshots at the sampling-noise level (synthetic A/B: <= 0.013 points).
- Readers built against this library read both formats. For tools built before it:
  - one moment at a time: `eval_fast/vendor/pack/zpk_snapshot.py` cuts a piece map with `piece_map_cut` (env
    `PIECE_MAP_CUT`, `PIECE_MAP_LIB`);
  - a whole map (eval/chain_fill's chain_fill, change_fill, motion_fill, timeline_export; the official Khronos
    evaluator): `piece_map_plain MAP OUT.4dmap` writes it as a plain serialization-version-1 map with its own time
    range and flags (OUT may be a pipe; a plain map is large, about 30x the piece map).

Tools (build.sh, against `KHRONOS_BUILD`): `piece_map_cut`, `piece_map_plain`, `piece_map_check` (converts a map and
compares every snapshot and query), `hist_convert`, `hist_measure`, `mesh_dupes`.
