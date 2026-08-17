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
#include <optional>
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
 * current DSG node per canonicalization round using `mergeObjectAttributes()`
 * for presence/trajectory. That function alone, however, treats each round's
 * set of segments as the *entire* geometric history: a re-observation is a
 * fresh election among the segments present in that one call, so an object's
 * current mesh is winner-takes-all rather than a persistent identity.
 *
 * `PersistentObjectState` layers a physical-object memory on top. It does not
 * touch presence/trajectory merging (`mergeObjectAttributes` is
 * call-then-overwritten, not replaced) -- it only overwrites the
 * geometry-authoritative fields of the already-merged attributes: mesh,
 * bounding box, position, and the geometry-support details.
 *
 * ## State model
 *
 * State is a sequence of temporal *fragments* per physical ID. One fragment is
 * the object as it was at one place, over one stretch of time. At most one
 * fragment is CURRENT, and only the CURRENT fragment is ever written back into
 * the DSG node:
 *
 *     PhysicalState(i)
 *       fragments   F0 [closed] -> F1 [closed] -> F2 [open]     <- history
 *       current     index of the one open fragment, if any
 *       unresolved  observations that are neither supported nor contradicted
 *
 * Physical identity *links* fragments across time. It never means
 * `current = union(all historical world-space meshes)`, and it never by itself
 * proves that a new temporal state has begun. Those two conflations are what
 * put two spatial clusters under one ID.
 *
 * ## What an already-ingested observation does to CURRENT
 *
 * The question asked of every geometry-bearing segment is not "did the object
 * move?" but "does this observation support, contradict, or say nothing about
 * the state we currently hold?":
 *
 *  - SAME_STATE -- the segment's surface and the CURRENT fragment's surface
 *    occupy common space (`surfacesShareSpace`), i.e. this is another view of
 *    the same site. The segment is folded into the CURRENT fragment (pure
 *    concatenation -- no welding/TSDF reintegration), the fragment's bounding
 *    box grows to the union, and its support time extends. Any unresolved
 *    candidate that the grown fragment now reaches is absorbed with it, which
 *    is how two disjoint views of one static object (a wardrobe's front and
 *    its back) get reunited once a third view bridges them.
 *
 *  - NEW_STATE -- the observation is accompanied by direct evidence that the
 *    state we hold no longer holds: tracker motion evidence on the segment
 *    (the object was *watched* moving, D1). The CURRENT fragment is closed and
 *    the segment opens a new fragment as CURRENT.
 *
 *  - UNRESOLVED -- neither. This is the ordinary case for an observation that
 *    simply does not overlap what we hold, and it must not be resolved by
 *    guessing: no overlap is not evidence of a move (a new viewpoint of a
 *    static object routinely shares no surface with the old one), and it is
 *    not evidence of a second copy either. The observation is kept as a
 *    separate candidate, materialized nowhere, until later evidence resolves
 *    it -- either support that absorbs it into CURRENT, or a contradiction of
 *    CURRENT that promotes it (see `reportCurrentContradicted`).
 *
 * A trajectory-only segment (empty mesh) contributes no geometry and leaves
 * the CURRENT fragment exactly as established.
 *
 * Nothing here consults a distance: no "moved far enough" bound, no bounding
 * box separation, no connectivity radius. `surfacesShareSpace` answers "is
 * this the same surface" at the map's own reconstruction scale, and its only
 * possible answers are SAME_STATE or "not enough information".
 *
 * ## Where contradiction comes from
 *
 * The only evidence that ends a fragment without watching the object move is
 * a real measurement that passed through the surface the fragment claims is
 * there. That evidence is produced against actual RGB-D endpoints, outside
 * this class, and delivered through `reportCurrentContradicted`. Because
 * promotion of a candidate and closure of CURRENT are the same operation,
 * the result does not depend on which of the two arrives first: a relocation
 * seen new-site-first and one seen old-site-empty-first converge on the same
 * CURRENT and the same history.
 *
 * ## Idempotence
 *
 * `canonicalizePhysicalObjects` re-feeds the same (now already-merged) target
 * node back into this function on every canonicalization pass. Segments
 * already folded in must never be re-ingested, or geometry would double on
 * every backend update. Two redundant, purely-integer locks guard this:
 *   1. Anchor lock: the earliest segment this round (by
 *      (observationFirstStamp, observationLastStamp, node_id)) is skipped if
 *      its observationFirstStamp equals the stamp recorded the last time this
 *      ID was processed (i.e. it is last round's merge target).
 *   2. Interval lock: every segment's exact (observationFirstStamp,
 *      observationLastStamp) pair is checked against the set of pairs already
 *      ingested; an exact repeat is skipped.
 * A physical ID with no registry state yet ingests every segment presented.
 */
class PersistentObjectState {
 public:
  PersistentObjectState() = default;

  /** Read-only view of one temporal fragment. Pointers are owned by the registry. */
  struct FragmentView {
    const spark_dsg::Mesh* geometry = nullptr;
    const BoundingBox* bbox = nullptr;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    TimeStamp birth_time = 0;
    TimeStamp last_support_time = 0;
    // Unset while the fragment is CURRENT; set once it has been closed.
    std::optional<TimeStamp> death_time;
    size_t reconstruction_frames = 0;
  };

  /**
   * @brief Overwrite the geometry-authoritative fields of `merged` (mesh,
   * bounding box, position, geometry-support details) with this physical
   * object's CURRENT fragment, first folding in any not-yet-ingested segments
   * from `nodes` (read from `graph`).
   *
   * `merged` must already carry a valid `instance_id` detail (i.e. it is the
   * output of `UpdateKhronosObjectsFunctor::mergeObjectAttributes` for the
   * same `nodes`); this is a no-op if it does not. If the ID has no CURRENT
   * fragment (none established yet, or the last one was contradicted and no
   * candidate has been promoted) the merge result is left untouched: node-level
   * absence, not this function, decides that an object is gone.
   */
  void applyPhysicalGeometry(const DynamicSceneGraph& graph,
                             const std::vector<NodeId>& nodes,
                             KhronosObjectAttributes& merged);

  /**
   * @brief Report that a real measurement contradicted the CURRENT geometry of
   * this physical ID: the surface it claims is occupied was seen through.
   *
   * Closes the CURRENT fragment with `stamp` as its death time and promotes the
   * most recent unresolved candidate, if any, to CURRENT. The closed fragment
   * stays in the history; its geometry is never transported to the promoted
   * candidate, which keeps the geometry it was actually observed with.
   *
   * `stamp` is an upper bound on when the object left, not a measured instant:
   * the true departure lies in (last_support, stamp]. Losing CURRENT ownership
   * is immediate regardless, so that an object is never CURRENT in two places
   * while its exact departure time is still being narrowed down.
   *
   * @return true if a CURRENT fragment was closed by this call.
   */
  bool reportCurrentContradicted(size_t physical_instance_id, TimeStamp stamp);

  /**
   * @brief Report that a real measurement still lands on the CURRENT geometry of this physical ID
   * at `stamp`: the state we hold is confirmed to still hold at that moment.
   *
   * This is what lets a disjoint observation be recognised as *more of the same object* rather
   * than a second copy of it. One physical ID cannot be in two places at one instant, so if
   * CURRENT is confirmed present at or after the moment a non-overlapping observation began, the
   * two surfaces are parts of one object seen from different sides, and the observation is folded
   * in. Without such a confirmation the observation stays UNRESOLVED: with no evidence that the
   * old surface is still there, merging would fabricate a union and splitting would fabricate a
   * move.
   *
   * @return true if this ID has a CURRENT fragment that was confirmed.
   */
  bool reportCurrentSupported(size_t physical_instance_id, TimeStamp stamp);

  /**
   * @brief Seed the registry from an already-materialized DSG's OBJECTS layer,
   * e.g. the inherited seed snapshot loaded at the start of a new session (D3
   * cross-session restore). Each object node with a valid `instance_id` detail
   * becomes that ID's CURRENT fragment.
   *
   * A DSG node carries only the CURRENT materialization, so a seeded ID starts
   * with exactly one fragment: whatever history the previous session held is
   * not recoverable from the node and is not invented here. Node IDs are
   * irrelevant to registry identity (which is `physical_instance_id`), so this
   * is unaffected by any node-ID rewriting (e.g. an 'M' prefix) applied while
   * reseeding the new session's working DSG.
   *
   * IDs not present in `dsg` keep whatever state they had; a seeded ID's prior
   * in-memory state is replaced. Call `clear()` first for a full reset.
   */
  void initializeFromObjects(const DynamicSceneGraph& dsg);

  /** Drop all registry state. */
  void clear();

  /** Number of physical IDs currently tracked. Exposed for tests/debugging. */
  size_t numStates() const;

  /** Whether the registry holds any state for this physical ID. */
  bool hasState(size_t physical_instance_id) const;

  /** Every physical ID the registry holds state for, ascending. */
  std::vector<size_t> trackedIds() const;

  /** The ID's CURRENT fragment, or nullopt if it currently has none. */
  std::optional<FragmentView> currentFragment(size_t physical_instance_id) const;

  /** Every fragment of this ID, oldest first, closed ones included. */
  std::vector<FragmentView> historyFragments(size_t physical_instance_id) const;

  /** Observations held as unresolved candidates for this ID, oldest first. */
  std::vector<FragmentView> unresolvedCandidates(size_t physical_instance_id) const;

 private:
  /**
   * @brief One temporal state of a physical object: the object as it was at one
   * place, over one stretch of time. A relocation never edits a fragment -- it
   * closes one and opens another.
   *
   * `geometry` is stored in `bbox`'s frame, exactly like
   * KhronosObjectAttributes::mesh, and always comes from the observations that
   * were actually made of this state. Geometry is never carried over from an
   * earlier fragment.
   */
  struct Fragment {
    spark_dsg::Mesh geometry{false, true, false, true};
    BoundingBox bbox;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();

    // Observation bounds of the segment that opened this fragment, and of the
    // most recent segment that supported it.
    TimeStamp birth_time = 0;
    TimeStamp last_support_time = 0;
    // Set when the fragment stops being CURRENT. A closed fragment is history:
    // it is never materialized and never becomes CURRENT again.
    std::optional<TimeStamp> death_time;

    // Reconstruction support (kReconstructionFramesDetail) accumulated over the
    // segments folded into *this* fragment. A new fragment starts from its own
    // opening observation rather than inheriting the closed fragment's count.
    size_t reconstruction_frames = 0;
  };

  /** @brief Every temporal state of one physical_instance_id. */
  struct PhysicalState {
    // History, oldest first. Closed fragments are retained, never destroyed.
    std::vector<Fragment> fragments;
    // Index into `fragments` of the one open fragment, if there is one.
    std::optional<size_t> current;
    // Observations that neither supported nor contradicted CURRENT. Never
    // materialized; promoted only when CURRENT is contradicted, absorbed only
    // when CURRENT grows to reach them.
    std::vector<Fragment> unresolved;

    // Anchor lock: observationFirstStamp of the merged node the last time this
    // ID was processed.
    TimeStamp last_merged_observation_first = 0;
    // Interval lock: exact (observationFirstStamp, observationLastStamp) pairs
    // already ingested, including trajectory-only segments that carried no
    // geometry.
    std::set<std::pair<TimeStamp, TimeStamp>> ingested_intervals;

    // Sticky: stays true once this ID has been observed to change state.
    bool has_dynamic_history = false;
  };

  static FragmentView viewOf(const Fragment& fragment);
  static std::vector<FragmentView> viewsOf(const std::vector<Fragment>& fragments);

  /** A fragment holding exactly the geometry of `attrs`, and nothing inherited. */
  static Fragment makeFragment(const KhronosObjectAttributes& attrs,
                               TimeStamp first,
                               TimeStamp last);

  /** Reduce one geometry-bearing observation against `state`: support, new state, or unresolved. */
  static void ingestObservation(PhysicalState& state,
                                const KhronosObjectAttributes& attrs,
                                TimeStamp first,
                                TimeStamp last);

  /**
   * Fold in every unresolved candidate the CURRENT fragment now reaches, repeating until none is
   * left: absorbing one candidate grows CURRENT, which can bring a further one within reach. This
   * is transitive support through actually observed surface, not a radius -- every step still
   * requires two surfaces that share space.
   */
  static void absorbReachableCandidates(PhysicalState& state);

  /** Close the CURRENT fragment, leaving the ID with no CURRENT. */
  static void closeCurrent(PhysicalState& state, TimeStamp stamp);

  /** Make the newest unresolved candidate CURRENT, if there is no CURRENT and a candidate exists. */
  static void promoteNewestCandidate(PhysicalState& state);

  std::map<size_t, PhysicalState> states_;
};

}  // namespace khronos
