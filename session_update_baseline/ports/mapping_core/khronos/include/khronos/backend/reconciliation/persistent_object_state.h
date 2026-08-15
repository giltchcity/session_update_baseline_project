/** -----------------------------------------------------------------------------
 * Copyright (c) 2024 Massachusetts Institute of Technology.
 * All Rights Reserved.
 *
 * AUTHORS:      Lukas Schmid <lschmid@mit.edu>, Marcus Abate <mabate@mit.edu>,
 *               Yun Chang <yunchang@mit.edu>, Luca Carlone <lcarlone@mit.edu>
 * AFFILIATION:  MIT SPARK Lab, Massachusetts Institute of Technology
 * YEAR:         2024
 * SOURCE:       https://github.com/MIT-SPARK/Khronos
 * LICENSE:      BSD 3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * -------------------------------------------------------------------------- */

#pragma once

#include <cstddef>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "khronos/common/common_types.h"

namespace khronos {

/**
 * @brief Backend-domain persistent state for physical objects, keyed by
 * `physical_instance_id`.
 *
 * Khronos' short-term tracker produces a new DSG node every time a physical
 * object's visibility segment is broken (tracking-window timeout, session
 * boundary). `UpdateKhronosObjectsFunctor::canonicalizePhysicalObjects`
 * already reduces every segment carrying the same `instance_id` to one
 * current DSG node per canonicalization round using
 * `mergeObjectAttributes()` for presence/trajectory. That function alone,
 * however, treats each round's set of segments as the *entire* geometric
 * history: a re-observation is a fresh election among the segments present
 * in that one call, so an object's current mesh is winner-takes-all rather
 * than a persistent, accumulating identity.
 *
 * `PersistentObjectState` layers a real physical-object memory on top:
 * track segments become *observations* of one persistent canonical
 * geometry rather than competing authorities for it. It intentionally does
 * not touch presence/trajectory merging (`mergeObjectAttributes` is call
 * -then-overwritten, not replaced) -- it only overwrites the
 * geometry-authoritative fields of the already-merged attributes: mesh,
 * bounding box, position, and the geometry-support details.
 *
 * Semantics per canonicalization round for one physical ID:
 *  - STATIC observation (no tracker motion evidence, bounding box not
 *    displaced from the current canonical pose by >= the D2/D3 physical
 *    move threshold): the segment's mesh is appended to the canonical mesh
 *    (world-frame union, MVP pure concatenation -- no welding/TSDF
 *    reintegration) and the canonical bounding box grows to the union.
 *  - MOVED observation (tracker motion evidence, or a stationary segment
 *    whose box does not intersect and is displaced far enough from the
 *    canonical pose established in a *previous* canonicalization round):
 *    the canonical mesh's local (bounding-box-frame) vertices are left
 *    untouched -- only the canonical bounding box/position move to the new
 *    observation's pose. A moved object keeps its shape; it does not adopt
 *    whatever partial geometry the first re-observation at the new site
 *    happened to reconstruct.
 *  - The very first time a physical ID is ever established in this
 *    registry (registry has no canonical geometry for it yet when the
 *    round starts), there is no established shape to protect from a move:
 *    that round instead performs the same accumulate-if-static /
 *    replace-if-moved reduction as above, but "moved" fully replaces the
 *    running composite (matching the existing degenerate single-round
 *    winner-takes-all behavior) instead of only translating it. Every
 *    subsequent round for that ID is protected by the "moved keeps shape"
 *    rule above.
 *
 * Idempotence: `canonicalizePhysicalObjects` re-feeds the same (now
 * already-merged) target node back into this function on every
 * canonicalization pass. Segments already folded into the canonical
 * geometry must never be re-ingested, or geometry would double on every
 * backend update. Two redundant, purely-integer locks guard this:
 *   1. Anchor lock: the earliest segment this round (by
 *      (observationFirstStamp, observationLastStamp, node_id)) is skipped
 *      if its observationFirstStamp equals the stamp recorded the last time
 *      this ID was processed (i.e. it is last round's merge target).
 *   2. Interval lock: every segment's exact (observationFirstStamp,
 *      observationLastStamp) pair is checked against the set of pairs
 *      already ingested; an exact repeat is skipped.
 * A physical ID with no registry state yet ingests every segment presented
 * (fresh semantics).
 */
class PersistentObjectState {
 public:
  PersistentObjectState() = default;

  /**
   * @brief Overwrite the geometry-authoritative fields of `merged` (mesh,
   * bounding box, position, geometry-support details) with this physical
   * object's persistent canonical geometry, first folding in any
   * not-yet-ingested segments from `nodes` (read from `graph`).
   *
   * `merged` must already carry a valid `instance_id` detail (i.e. it is
   * the output of `UpdateKhronosObjectsFunctor::mergeObjectAttributes` for
   * the same `nodes`); this is a no-op if it does not.
   */
  void applyPhysicalGeometry(const DynamicSceneGraph& graph,
                             const std::vector<NodeId>& nodes,
                             KhronosObjectAttributes& merged);

  /**
   * @brief Seed the registry from an already-materialized DSG's OBJECTS
   * layer, e.g. the inherited seed snapshot loaded at the start of a new
   * session (D3 cross-session restore). Each object node with a valid
   * `instance_id` detail becomes that ID's canonical geometry, exactly as
   * if it had just been produced by `applyPhysicalGeometry`. Node IDs are
   * irrelevant to registry identity (which is `physical_instance_id`), so
   * this is unaffected by any node-ID rewriting (e.g. an 'M' prefix)
   * applied while reseeding the new session's working DSG.
   *
   * Does not clear any existing state; call `clear()` first if a full
   * reset is desired.
   */
  void initializeFromObjects(const DynamicSceneGraph& dsg);

  /** Drop all registry state. */
  void clear();

  /** Number of physical IDs currently tracked. Exposed for tests/debugging. */
  size_t numStates() const;

  /** Whether the registry has canonical geometry for this physical ID. */
  bool hasState(size_t physical_instance_id) const;

 private:
  struct State {
    // Canonical accumulated mesh. Vertex positions are relative to
    // `canonical_bbox`'s origin, exactly like KhronosObjectAttributes::mesh.
    spark_dsg::Mesh canonical_mesh{false, true, false, true};
    // Canonical/current bounding box (also the frame `canonical_mesh` is
    // expressed in).
    BoundingBox canonical_bbox;
    // Canonical/current position (mirrors KhronosObjectAttributes::position).
    Eigen::Vector3d canonical_position = Eigen::Vector3d::Zero();

    // Anchor lock: observationFirstStamp of the merged node the last time
    // this ID was processed.
    TimeStamp canonical_observation_first = 0;
    // Interval lock: exact (observationFirstStamp, observationLastStamp)
    // pairs already folded into canonical_mesh/bbox (or explicitly skipped
    // as empty trajectory-only observations).
    std::set<std::pair<TimeStamp, TimeStamp>> ingested_intervals;

    // Accumulated reconstruction support carried in the geometry-support
    // detail (kReconstructionFramesDetail); reset to the new observation's
    // support on a move, summed while statically accumulating.
    size_t reconstruction_frames = 0;
    // Sticky: once a real move has been observed for this ID, stays true.
    bool has_dynamic_history = false;
  };

  std::map<size_t, State> states_;
};

}  // namespace khronos
