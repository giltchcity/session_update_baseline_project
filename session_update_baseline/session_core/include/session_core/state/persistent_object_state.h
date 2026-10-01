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
#include <unordered_map>
#include <utility>
#include <vector>

#include "khronos/common/common_types.h"
#include "session_core/evidence/element_round.h"
#include "session_core/model/change_filter.h"
#include "session_core/model/persistence_prior.h"
#include "session_core/model/round_model.h"

namespace khronos {

/** Physical identity and temporal fragments. README equations (3)--(5t), (7), (15b).
 * Khronos supplies observations; this registry owns their state association (the posterior of
 * each placement by the Persistence Filter recursion (5r) under the prior Pi and the round
 * likelihood of principle 6), the commitments of (5e), the geometric materialization and the
 * cross-session handoff.
 */
class PersistentObjectState {
 public:
  PersistentObjectState() = default;

  /** One round of one placement: the latest verdicts of its elements in the new frames. */
  struct RoundInput {
    uint64_t evidence_key = 0;
    uint64_t geometry_revision = 0;
    TimeStamp window_begin = 0;     // first new input of the round
    TimeStamp measured_through = 0; // last input of the round
    TimeStamp session_start = 0;    // first frame of this session
    ElementRound elements;
  };

  /** README (5o), E_{h->o}: what the frames (or the free space) while the current placement was
   * in place say about the elements of one pending candidate. */
  struct PairInput {
    uint64_t evidence_key = 0;      // the candidate
    uint64_t geometry_revision = 0;
    uint64_t current_key = 0;       // the placement it was evaluated against
    ElementRound exclusion;         // verdicts of the candidate's elements while h was in place
  };

  /** What one decision round did to an identity. */
  struct RoundResult {
    bool closed = false;     // the current placement was committed changed and closed
    bool confirmed = false;  // the current placement was committed in place
    bool absorbed = false;   // a pending candidate was committed to belong to the placement
  };

  /** Read-only view of one temporal fragment. Pointers are owned by the registry. */
  struct FragmentView {
    const spark_dsg::Mesh* geometry = nullptr;
    const BoundingBox* bbox = nullptr;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    uint64_t evidence_key = 0;  // Immutable while observations refine this fragment.
    uint64_t geometry_revision = 0;
    bool inherited = false;
    int semantic_label = -1;
    TimeStamp birth_time = 0;    // the first observation of the placement
    TimeStamp presence_begin = 0;  // b_hat of (5f)
    // First tracker sighting of the observations folded into this fragment.
    TimeStamp track_first_seen = 0;
    TimeStamp last_support_time = 0;
    TimeStamp input_boundary = 0;  // Processed input boundary, independent of support.
    TimeStamp last_confirmed_support = 0;
    // Unset while the fragment is CURRENT; set once it has been closed: the right end of (5t).
    std::optional<TimeStamp> death_time;
    TimeStamp change_left = 0;  // left end of the (1 - alpha) interval of (5t), once closed
    double closure_odds = 0.0;  // the odds of the commitment that closed it (>= (1-alpha)/alpha)
    size_t reconstruction_frames = 0;
    double odds = 0.0;  // Lambda of (5r)
    TimeStamp ended_since = 0;   // the stamp of the commitment that closed it
    TimeStamp ended_processed = 0;
  };

  // README (3), (5): ingest new observation intervals and materialize current.
  void ingestObjects(const DynamicSceneGraph& graph);
  void applyPhysicalGeometry(const DynamicSceneGraph& graph,
                             const std::vector<NodeId>& nodes,
                             KhronosObjectAttributes& merged);

  // README (5e), (5r), (5o): consume the measured round of the current placement and of the
  // pending candidates of one identity; commit or defer.
  RoundResult resolveRound(size_t physical_instance_id, const RoundInput& current,
                           const std::vector<PairInput>& pending, TimeStamp stamp);

  // README principle 6: rounds of a placement after a committed end teach the ended distribution.
  void addEndedRound(size_t physical_instance_id, uint64_t evidence_key, const ElementRound& round,
                     TimeStamp stamp);

  // README (5b), (13): terminal drain; undecided candidates stay deferred.
  void finalizePendingAbsences(TimeStamp stamp);

  void setMapResolution(float resolution);
  /** README (12), (7s): hits of the construction of an element (the minimum mesh weight). */
  void setConstructionHits(double hits);

  // Seed each physical ID's current from OBJECTS; the boundary is the seed snapshot time.
  void initializeFromObjects(const DynamicSceneGraph& dsg, TimeStamp boundary);

  void materialize(DynamicSceneGraph& graph) const;

  // A background element that coincides with the surface of a placement committed changed
  // (README principle 9): its prior change odds are that placement's closure odds, at
  // the distance of the coincidence.
  struct BackgroundObligation {
    Point point = Point::Zero();
    TimeStamp reconstructed = 0, supported = 0;
    double odds = 0.0;
    float distance = 0.f;
  };
  const std::vector<BackgroundObligation>& backgroundObligations() const {
    return background_obligations_;
  }

  // README (15b): terminal live state, bound to the exact chain map file.
  void saveCheckpoint(const std::string& path, const std::string& chain_path,
                      TimeStamp boundary, const DynamicSceneGraph& chain) const;
  void loadCheckpoint(const std::string& path, const std::string& chain_path,
                      const DynamicSceneGraph& chain, TimeStamp boundary);

  /** Drop all registry state. */
  void clear();

  // README (8): per identity the right end of (5t) of its closed placements; frames at or before
  // it belong to a closed placement.
  std::map<size_t, TimeStamp> successionFloors() const;

  double constructionHits() const { return construction_hits_; }

  /** README principle 9: the committed-round histories of the elements of the current inherited
   * placement at the start of the session (the same data is not multiplied twice, (5g)). */
  struct StartOfSessionPrior {
    std::unordered_map<uint64_t, std::pair<float, float>> histories;  // cell -> (hits, see-throughs)
  };
  std::optional<StartOfSessionPrior> startOfSessionPrior(size_t physical_instance_id) const;

  /** The world-frame surfaces of the placements committed changed, with their closure odds. */
  struct ClosedSurface {
    std::vector<Eigen::Vector3f> vertices;
    std::vector<std::array<uint32_t, 3>> faces;
    double odds = 0.0;
  };
  std::vector<ClosedSurface> closedSurfaces() const;

  /** The hazard of each identity for the tracker's visible-motion recursion (principle 5). */
  std::map<size_t, model::PersistencePrior::Hazard> motionPriors() const;

  size_t numStates() const;
  bool hasState(size_t physical_instance_id) const;
  std::vector<size_t> trackedIds() const;
  std::set<std::pair<size_t, uint64_t>> liveEvidenceKeys() const;

  /** The ID's CURRENT fragment, or nullopt if it currently has none. */
  std::optional<FragmentView> currentFragment(size_t physical_instance_id) const;

  /** Every fragment of this ID, oldest first, closed ones included. */
  std::vector<FragmentView> historyFragments(size_t physical_instance_id) const;

  /** The pending candidates of the identity whose exclusion evidence E_{h->o} against the current
   * placement has not been evaluated for their present geometry. */
  std::vector<FragmentView> pendingNeedingExclusion(size_t physical_instance_id) const;

  /** Latest unresolved observation, if present. */
  std::optional<FragmentView> observedNew(size_t physical_instance_id) const;

  /** Every unresolved observation, retaining its geometry and time. */
  std::vector<FragmentView> unresolvedCandidates(size_t physical_instance_id) const;

  /** Closed placements still being watched for the ended distribution of principle 6. */
  std::vector<std::pair<size_t, FragmentView>> endedWatch() const;

  const model::PersistencePrior& persistencePrior() const { return prior_; }
  const model::RoundModel& roundModel() const { return rounds_; }

  /** README principle 2 (transfer): load the statistics of earlier data as the hyper-prior of
   * this registry (each source counted once by the caller). Placements restored before this call
   * keep the odds they started with. */
  void importStatistics(model::PersistencePrior prior, model::RoundModel rounds) {
    prior_ = std::move(prior);
    rounds_ = std::move(rounds);
  }

 private:
  /** The history of one element of a placement, README (7s): committed-round hits and see-throughs. */
  struct ElementHistory {
    float hits = 0.f, through = 0.f;
  };

  /**
   * @brief One placement of a physical object: the object at one place over one stretch of time.
   * A relocation never edits a fragment -- it closes one and opens another.
   * `geometry` is stored in `bbox`'s frame and always comes from the observations actually made of
   * this placement.
   */
  struct Fragment {
    spark_dsg::Mesh geometry{false, true, false, true};
    BoundingBox bbox;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();

    uint64_t evidence_key = 0;       // Immutable while observations refine this fragment.
    uint64_t geometry_revision = 1;  // Incremented whenever this fragment's geometry changes.
    TimeStamp birth_time = 0;        // first observation
    TimeStamp presence_begin = 0;    // b_hat of (5f)
    TimeStamp last_support_time = 0;
    TimeStamp input_boundary = 0;    // Processed input boundary, independent of support.
    TimeStamp track_first_seen = 0;
    // Last time a committed round (5e) placed this placement in place.
    TimeStamp last_confirmed_support = 0;

    // Semantic class of this fragment: the grouping key of the prior (principle 2), nothing else.
    int semantic_label = -1;

    // A placement restored from a previous session; its odds start at q^g / (1 - q^g).
    bool inherited = false;
    // README (5g): the element histories as they were at the start of the session.
    std::unordered_map<uint64_t, ElementHistory> elements_at_start;
    // The odds of the commitment that closed the placement (elements that coincide with its
    // surface in the background take it as their prior change probability).
    double closure_odds = 0.0;
    // The gap outcome of an inherited placement is judged once (Pi, principle 2).
    bool gap_pending = false;

    // README (5r): the posterior of "changed", and the clock of the recursion and of the exposure.
    model::ChangeFilter filter;
    TimeStamp filter_time = 0;     // the recursion has advanced through this stamp
    TimeStamp exposure_clock = 0;  // committed-in-place time is accounted through this stamp

    // Set when the placement is committed changed: the right end of the (1 - alpha) interval (5t).
    std::optional<TimeStamp> death_time;
    TimeStamp change_left = 0;
    TimeStamp ended_since = 0;       // the stamp of the commitment; ended rounds follow it
    TimeStamp ended_processed = 0;

    size_t reconstruction_frames = 0;
    // README (4): the distinct capture-frame keys of this placement's geometry, sorted.
    std::vector<TimeStamp> frame_keys;

    // README (7s): per-element histories, keyed by the element's cell.
    std::unordered_map<uint64_t, ElementHistory> elements;

    // README (5o), E_{h->o}: the exclusion evidence of a pending candidate against the placement
    // that was current when it was evaluated (ln LR of H_new : H_same).
    bool exclusion_known = false;
    uint64_t exclusion_current = 0, exclusion_revision = 0;
    double exclusion_log_lr = 0.0;
  };

  /** Every temporal state of one physical_instance_id. */
  struct PhysicalState {
    std::vector<Fragment> fragments;       // insertion-ordered
    std::optional<size_t> current;         // the one open placement
    std::vector<Fragment> observed_new;    // README (13): unresolved candidates keep geometry/time

    TimeStamp succession_floor = 0;        // Latest actual support of a closed predecessor.
    TimeStamp closed_through = 0;          // Right end of (5t) of the closed placements.
    TimeStamp last_motion_consumed = 0;    // Accepted native trajectory watermark.
    // Exactly-once raw extraction versions; observation intervals are not identities.
    std::set<std::string> ingested_sources;

    // Sticky: stays true once this ID has been observed to change state.
    bool has_dynamic_history = false;
  };

  static TimeStamp latestSupport(const Fragment& fragment);
  static bool isEligibleSuccessor(const PhysicalState& state, const Fragment& candidate);
  static bool ownsEvidence(const Fragment& fragment, const RoundInput& evidence);
  void ingestSegments(const DynamicSceneGraph& graph, const std::vector<NodeId>& nodes, size_t id);
  void materializeState(const PhysicalState& state, KhronosObjectAttributes& attrs) const;
  static void reserveEvidenceKeys(uint64_t maximum);
  static FragmentView viewOf(const Fragment& fragment);
  static std::vector<FragmentView> viewsOf(const std::vector<Fragment>& fragments);

  /** A fragment holding exactly the geometry of `attrs`, and nothing inherited. */
  static Fragment makeFragment(const KhronosObjectAttributes& attrs, TimeStamp first, TimeStamp last);

  /** Reduce one geometry-bearing observation against `state`: current, pending, or motion. */
  void ingestObservation(PhysicalState& state, const KhronosObjectAttributes& attrs, TimeStamp first,
                         TimeStamp last, size_t physical_instance_id);

  /** README (13) (a): fold a pending candidate into the current placement. */
  void absorb(PhysicalState& state, size_t pending_index);
  /** The pending candidates the current placement is already committed to contain (13 (a)). */
  void absorbCommitted(PhysicalState& state);

  /** Close the CURRENT fragment with its change interval, leaving the ID with no CURRENT. */
  void closeCurrent(size_t id, PhysicalState& state, TimeStamp commit_stamp, TimeStamp left,
                    TimeStamp right, double odds);
  bool consumeMotion(size_t id, PhysicalState& state, const KhronosObjectAttributes& attrs);
  void promoteObservedNew(size_t id, PhysicalState& state);
  static size_t latestPendingIndex(const PhysicalState& state);
  static void closePending(PhysicalState& state, TimeStamp stamp);

  /** The filter of a fresh placement (session-born) or of an inherited one; an inherited
   * placement also records its q^g and the histories it starts the session with. */
  model::ChangeFilter newFilter(size_t id, Fragment& fragment) const;

  /** README (7s), (7): counts of the stable elements of a round. */
  model::RoundModel::Counts roundCounts(size_t id, const Fragment& fragment,
                                        const ElementRound& round) const;
  void foldRound(size_t id, Fragment& fragment, const ElementRound& round,
                 const model::RoundModel::Counts& counts);

  // README (4): common geometry reduction.
  static void mergeFragments(Fragment& target, const Fragment& observation);

  // Prior Pi (principle 2): statistics of the committed outcomes of all placements.
  model::PersistencePrior prior_;
  // Round model of principle 6, learned from committed rounds.
  model::RoundModel rounds_;

  std::map<size_t, PhysicalState> states_;
  std::vector<BackgroundObligation> background_obligations_;
  float map_resolution_ = 0.05f;
  double construction_hits_ = 0.0;
};

}  // namespace khronos
