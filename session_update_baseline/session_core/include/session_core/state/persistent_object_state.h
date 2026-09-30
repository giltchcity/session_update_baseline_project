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
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "khronos/common/common_types.h"

namespace khronos {

/** Physical identity and temporal fragments. README equations (3)--(5b).
 * Khronos supplies observations; this registry owns their state association,
 * geometric materialization, and cross-session handoff.
 */
class PersistentObjectState {
 public:
  PersistentObjectState() = default;

  /** One whole round of surface evidence of one placement, README (7), (7u). */
  struct SurfaceEvidence {
    size_t support_rays = 0;        // S_t: actual (frame,pixel) sources that support the surface
    size_t contradiction_rays = 0;  // F_t: sources that pass it
    size_t surface_samples = 0;
    TimeStamp latest_support_stamp = 0;  // Actual sensor time, never reducer/check time.
    // Actual sensor time of the earliest penetrating source of the round (0: none), README (5b).
    TimeStamp first_contradiction_stamp = 0;
    // Predictive likelihoods of the round under the normal and the change model. Without a valid
    // source the round is the unit factor and leaves the placement statistics unchanged.
    bool informative = false;
    double l_in = 1.0;
    double l_out = 1.0;
    // Diagnostic votes.
    size_t supported_votes = 0;
    size_t free_space_votes = 0;
    size_t occluded_votes = 0;
    size_t unobserved_samples = 0;
    // The measured geometry and the input cutoff belong to this result, not its call-site slot.
    uint64_t evidence_key = 0;
    uint64_t geometry_revision = 0;
    TimeStamp measured_through = 0;
  };

  // README (5p), (5h): result of one POCD soft update of a placement's stationarity Beta.
  struct PlacementUpdate {
    double alpha = 0, beta = 0;  // moment-matched Beta of the posterior
    double omega = 0;            // probability that this round is compatible with the placement
  };
  // Initial stationarity: mean 0.67 with strength 2, README s3.2 (fixed weak prior).
  static constexpr double kInitialMean = 0.67;
  static constexpr double kInitialStrength = 2.0;
  static constexpr double kInitialAlpha = kInitialMean * kInitialStrength;
  static constexpr double kInitialBeta = (1.0 - kInitialMean) * kInitialStrength;
  // README (5), (5p), (5h): chi(c) in {0,1}, the round's likelihoods L_in and L_out.
  // chi is empty for a class the scene's table does not declare: no semantic weight.
  static PlacementUpdate updatePlacement(double alpha, double beta, std::optional<int> chi,
                                         double l_in, double l_out);

  // What one decision round did to a placement pair, for the calibration of README (7).
  struct RoundResult {
    bool closed = false;            // the top-level current was closed
    bool inherited_normal = false;  // the measured top-level current was judged in place
    bool session_normal = false;    // the measured session placement was judged in place
  };

  /** Read-only view of one temporal fragment. Pointers are owned by the registry. */
  struct FragmentView {
    const spark_dsg::Mesh* geometry = nullptr;
    const BoundingBox* bbox = nullptr;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    uint64_t evidence_key = 0;  // Immutable while observations refine this fragment.
    uint64_t geometry_revision = 0;
    bool inherited = false;
    TimeStamp birth_time = 0;
    // First tracker sighting of the observations folded into this fragment.
    TimeStamp track_first_seen = 0;
    TimeStamp last_support_time = 0;
    TimeStamp input_boundary = 0;  // Processed input boundary, independent of support.
    TimeStamp last_confirmed_support = 0;
    // Unset while the fragment is CURRENT; set once it has been closed.
    std::optional<TimeStamp> death_time;
    size_t reconstruction_frames = 0;
    // README (5): stationarity v ~ Beta(alpha, beta) of this placement.
    double alpha = kInitialAlpha;
    double beta = kInitialBeta;
    // README (5e): the stationarity this placement had when the session started; it is the
    // persistence prior of the placement's historical faces, independent of this session's rounds.
    double prior_alpha = kInitialAlpha;
    double prior_beta = kInitialBeta;
    // README (5b): the earliest valid contradiction after the last support (0: none). The change
    // time lies between the latest actual support and this time.
    TimeStamp first_contradiction = 0;
  };

  // README (3), (5): ingest new observation intervals and materialize current.
  // Geometry is empty once every established fragment has closed.
  void ingestObjects(const DynamicSceneGraph& graph);
  void applyPhysicalGeometry(const DynamicSceneGraph& graph,
                             const std::vector<NodeId>& nodes,
                             KhronosObjectAttributes& merged);

  // Direct measured reports. stamp is the sensor time in the support report.
  bool reportCurrentContradicted(size_t physical_instance_id, TimeStamp stamp);
  bool reportCurrentSupported(size_t physical_instance_id, TimeStamp stamp);

  // README (5b), P13: settle the independent session states at terminal drain.
  size_t finalizePendingAbsences(TimeStamp stamp);

  // README (5), (5e): consume a frozen evidence pair, update the measured placements' stationarity
  // and close a placement whose expected stationarity fell below 1/2.
  RoundResult resolveCurrentEvidence(size_t physical_instance_id,
                                     const SurfaceEvidence& inherited_evidence,
                                     const SurfaceEvidence& session_evidence,
                                     TimeStamp stamp);

  void setMapResolution(float resolution);
  // README (3.2): the scene's declared class table chi(c). A movable class has chi = 0, a static
  // class chi = 1; a class in neither list is undeclared and receives no semantic weight.
  void setHighMobilitySemanticLabels(const std::vector<int>& labels);
  void setStaticSemanticLabels(const std::vector<int>& labels);

  // Seed each physical ID's current from OBJECTS; reset that ID's local evidence.
  // The boundary is the seed snapshot time, distinct from its last observation.
  // Cross-session persistence is specified by README section 7.
  void initializeFromObjects(const DynamicSceneGraph& dsg, TimeStamp boundary);

  void materialize(DynamicSceneGraph& graph) const;

  struct BackgroundObligation {
    Point point = Point::Zero();
    TimeStamp reconstructed = 0, supported = 0;
  };
  const std::vector<BackgroundObligation>& backgroundObligations() const {
    return background_obligations_;
  }

  // README (15): terminal live state, bound to the exact chain map file.
  void saveCheckpoint(const std::string& path, const std::string& chain_path,
                      TimeStamp boundary, const DynamicSceneGraph& chain) const;
  void loadCheckpoint(const std::string& path, const std::string& chain_path,
                      const DynamicSceneGraph& chain, TimeStamp boundary);

  /** Drop all registry state. */
  void clear();

  // README (5b), (4.0) P5: per identity, the latest actual support of its closed placements
  // (the successor watermark w_closed). Observations at or before it belong to a closed placement.
  std::map<size_t, TimeStamp> successionFloors() const;

  /** Number of physical IDs currently tracked. Exposed for tests/debugging. */
  size_t numStates() const;

  /** Whether the registry holds any state for this physical ID. */
  bool hasState(size_t physical_instance_id) const;

  /** Every physical ID the registry holds state for, ascending. */
  std::vector<size_t> trackedIds() const;
  std::set<std::pair<size_t,uint64_t>> liveEvidenceKeys() const;

  /** The ID's CURRENT fragment, or nullopt if it currently has none. */
  std::optional<FragmentView> currentFragment(size_t physical_instance_id) const;

  /** The ID's independent B-session CURRENT fragment, if one exists. */
  std::optional<FragmentView> sessionCurrentFragment(
      size_t physical_instance_id) const;

  /** Every fragment of this ID, oldest first, closed ones included. */
  std::vector<FragmentView> historyFragments(size_t physical_instance_id) const;

  /** Latest unresolved observation, if present. */
  std::optional<FragmentView> observedNew(size_t physical_instance_id) const;

  /**
   * Every unresolved observation, retaining its geometry and time.
   */
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
    // most recent segment or ray measurement that supported it.
    uint64_t evidence_key = 0;  // Immutable while observations refine this fragment.
    uint64_t geometry_revision = 1;  // Incremented whenever this fragment's geometry changes.
    TimeStamp birth_time = 0;
    TimeStamp last_support_time = 0;
    TimeStamp input_boundary = 0;  // Processed input boundary, independent of support.

    // Earliest tracker first sighting (kTrackFirstSeenDetail, falling back to
    // the observation start) of the segments folded into this fragment.
    // Earliest state-authorized input for session-end re-integration.
    TimeStamp track_first_seen = 0;

    // Last time a real measurement confirmed this state was still present at
    // its CURRENT site. This is distinct from `last_support_time`: a direct
    // segment sets both. Checkpoints retain this actual support time across
    // sessions; the independent input boundary controls new-session processing.
    TimeStamp last_confirmed_support = 0;

    // Semantic class of this fragment, used only for the generic mobility
    // prior (movable object categories vs static furniture categories).
    int semantic_label = -1;

    // Restored geometry has an independent session reconstruction until
    // inheritedRelation and terminal settlement associate the two (README 5).
    bool requires_current_session_support = false;

    // Set when the fragment stops being CURRENT. A closed fragment is history:
    // it is never materialized and never becomes CURRENT again.
    std::optional<TimeStamp> death_time;

    // Reconstruction support (kReconstructionFramesDetail) accumulated over the
    // segments folded into *this* fragment. A new fragment starts from its own
    // opening observation rather than inheriting the closed fragment's count.
    size_t reconstruction_frames = 0;

    // README (5): stationarity v ~ Beta(alpha, beta), updated once per measured round.
    double alpha = kInitialAlpha;
    double beta = kInitialBeta;
    // README (5e): the stationarity this placement had when the session started (the
    // persistence prior of its historical faces, independent of this session's rounds).
    double prior_alpha = kInitialAlpha;
    double prior_beta = kInitialBeta;
    // README (5b): the earliest valid contradiction after the last support (0: none).
    TimeStamp first_contradiction = 0;
    // README (4): the distinct capture-frame keys (acquisition times) of this placement's
    // geometry, sorted; the frame count is their number. Empty for an older map, which then
    // keeps its stored count.
    std::vector<TimeStamp> frame_keys;
  };

  /** @brief Every temporal state of one physical_instance_id. */
  struct PhysicalState {
    // Insertion-ordered fragments; historyFragments exposes chronological views.
    std::vector<Fragment> fragments;
    // Index into `fragments` of the one open fragment, if there is one.
    std::optional<size_t> current;
    // README (5b): unresolved observations retain separate geometry and time.
    std::vector<Fragment> observed_new;

    // Independent B-session state. While CURRENT is an inherited fragment,
    // B observations run their own mini D1/D2 state machine here so that a
    // B-internal move (cabinet X->Y) is resolved without touching A history.
    std::unique_ptr<PhysicalState> b_session;

    // README (6b): whether the last round judged the inherited current compatible (omega > 1/2).
    bool inherited_compatible = false;

    TimeStamp succession_floor = 0;  // Latest actual support of a closed predecessor.
    TimeStamp last_motion_consumed = 0;  // Accepted native trajectory watermark.
    // Exactly-once raw extraction versions; observation intervals are not identities.
    std::set<std::string> ingested_sources;

    // Sticky: stays true once this ID has been observed to change state.
    bool has_dynamic_history = false;

  };

  static TimeStamp latestSupport(const Fragment& fragment);
  static bool isEligibleSuccessor(const PhysicalState& state, const Fragment& candidate);
  static bool ownsEvidence(const Fragment& fragment, const SurfaceEvidence& evidence);
  void ingestSegments(const DynamicSceneGraph& graph, const std::vector<NodeId>& nodes, size_t id);
  void materializeState(const PhysicalState& state, KhronosObjectAttributes& attrs) const;
  static void reserveEvidenceKeys(uint64_t maximum);
  static FragmentView viewOf(const Fragment& fragment);
  static std::vector<FragmentView> viewsOf(const std::vector<Fragment>& fragments);

  /** A fragment holding exactly the geometry of `attrs`, and nothing inherited. */
  static Fragment makeFragment(const KhronosObjectAttributes& attrs,
                               TimeStamp first,
                               TimeStamp last);

  /** Reduce one geometry-bearing observation against `state`: current, observed_new, or motion. */
  void ingestObservation(PhysicalState& state,
                                const KhronosObjectAttributes& attrs,
                                TimeStamp first,
                                TimeStamp last,
                                size_t physical_instance_id);

  /** Retain one observation without assuming a same-state relationship. */
  static void mergeObservedNew(PhysicalState& state,
                               const KhronosObjectAttributes& attrs,
                               TimeStamp first,
                               TimeStamp last);

  /**
   * Fold the accumulated observed_new slot into CURRENT. Precondition: a real
   * measurement confirmed CURRENT present through `stamp`.
   */
  void absorbObservedThrough(PhysicalState& state, TimeStamp stamp);

  /** Close the CURRENT fragment, leaving the ID with no CURRENT. */
  static bool consumeMotion(PhysicalState& state, const KhronosObjectAttributes& attrs);
  static void closeCurrent(PhysicalState& state, TimeStamp stamp);

  // README (5), (6b): one relationship for materialization and settlement.
  enum class StateRelation { kSeparate, kRefine, kReplace };
  bool canAbsorb(const Fragment& observation, TimeStamp stamp) const;
  StateRelation inheritedRelation(const PhysicalState& state) const;
  // The outcome of one measured round on one placement.
  struct LocalOutcome {
    bool applied = false;     // the round was owned by the fragment and informative
    bool closed = false;      // expected stationarity fell below 1/2
    bool compatible = false;  // omega > 1/2
  };
  LocalOutcome applyRound(Fragment& fragment, const SurfaceEvidence& evidence,
                          TimeStamp stamp) const;
  LocalOutcome resolveLocalEvidence(PhysicalState& state, const SurfaceEvidence& evidence,
                                    TimeStamp stamp);
  std::optional<int> chiOf(const Fragment& fragment) const;
  // README (4), (5b): common geometry reduction and lossless history handoff.
  static void mergeFragments(Fragment& target, const Fragment& observation);
  static bool settleSession(PhysicalState& state, StateRelation relation,
                            TimeStamp stamp);

  /** Promote the latest pending observation, preserving the other observations. */
  static size_t latestPendingIndex(const PhysicalState& state);
  static void closePending(PhysicalState& state, TimeStamp stamp);
  static void promoteObservedNew(PhysicalState& state);

  std::map<size_t, PhysicalState> states_;
  std::vector<BackgroundObligation> background_obligations_;
  float map_resolution_ = 0.05f;
  // README (3.2): classes with chi(c) = 0 and chi(c) = 1.
  std::set<int> movable_labels_;
  std::set<int> static_labels_;
};

}  // namespace khronos
