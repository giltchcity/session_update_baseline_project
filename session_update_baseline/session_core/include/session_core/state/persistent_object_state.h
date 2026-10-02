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
#include <unordered_set>
#include <utility>
#include <vector>

#include "khronos/common/common_types.h"
#include "session_core/evidence/element_round.h"
#include "session_core/model/change_filter.h"
#include "session_core/model/persistence_prior.h"
#include "session_core/model/round_model.h"

namespace khronos {

/** Physical identity and temporal fragments. README equations (3)--(5t), (7), (15b).
 * Khronos supplies observations; this registry owns their state association (the CUSUM (5r) of
 * each placement over the looks of principle 6, with the ray majority), the commitments of (5e),
 * the geometric materialization and the cross-session handoff (principle 13).
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

  /** README (5o), principle 13: what the frames say about one pending candidate o against the
   * current placement h. */
  struct PairInput {
    uint64_t evidence_key = 0;      // the candidate
    uint64_t geometry_revision = 0;
    uint64_t current_key = 0;       // the placement it was evaluated against
    ElementRound exclusion;         // E_{h->o}: the candidate's elements in the frames while h was in place
    ElementRound support;           // E_{o->h}: h's elements in the frames of the candidate
    // (g): the elements of the candidate, the share of them farther from h's surface than the
    // cross-session band delta_* (principle 4: sigma_x and the scale displacement), and whether
    // that band exists (sigma_x estimated).
    size_t candidate_elements = 0;
    size_t candidate_away = 0;
    bool cross_session_band = false;
  };

  /** What one decision round did to an identity. */
  struct RoundResult {
    bool closed = false;     // the current placement was committed ended and closed
    bool confirmed = false;  // the current placement was directly seen in place
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
    TimeStamp change_left = 0;  // left end of the interval (5t), once closed
    size_t reconstruction_frames = 0;
    double cusum = 0.0;  // C of (5r)
    TimeStamp ended_since = 0;   // the stamp of the commitment that closed it
    double zeta_e = 0.0;  // the depth scale of the session that made its elements
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

  // README (5b), (13): terminal drain; undecided candidates stay deferred.
  void finalizePendingAbsences(TimeStamp stamp);

  void setMapResolution(float resolution);

  /** zeta of the previous session: the zeta_e of placements restored without a record of their own. */
  void setPreviousScale(double zeta);
  /** zeta of this session, recorded as zeta_e of the placements this session made. */
  void setSessionScale(double zeta) { session_zeta_ = zeta; }

  // Seed each physical ID's current from OBJECTS; the boundary is the seed snapshot time.
  void initializeFromObjects(const DynamicSceneGraph& dsg, TimeStamp boundary);

  void materialize(DynamicSceneGraph& graph) const;

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

  /** The world-frame surfaces of the placements committed ended. */
  struct ClosedSurface {
    std::vector<Eigen::Vector3f> vertices;
    std::vector<std::array<uint32_t, 3>> faces;
    size_t physical_id = 0;
    int semantic_label = -1;
  };
  std::vector<ClosedSurface> closedSurfaces() const;

  /** README principle 12: the recorded distance rho_e of the elements of the current placement of
   * an identity, by element cell; null where it has no current placement. */
  const std::unordered_map<uint64_t, float>* elementRanges(size_t physical_instance_id) const;

  size_t numStates() const;
  bool hasState(size_t physical_instance_id) const;
  std::vector<size_t> trackedIds() const;
  std::set<std::pair<size_t, uint64_t>> liveEvidenceKeys() const;

  /** The ID's CURRENT fragment, or nullopt if it currently has none. */
  std::optional<FragmentView> currentFragment(size_t physical_instance_id) const;

  /** Every fragment of this ID, oldest first, closed ones included. */
  std::vector<FragmentView> historyFragments(size_t physical_instance_id) const;

  /** The pending candidates of the identity whose evidence against the current placement has not
   * been evaluated for their present geometry. */
  std::vector<FragmentView> pendingNeedingExclusion(size_t physical_instance_id) const;

  /** Latest unresolved observation, if present. */
  std::optional<FragmentView> observedNew(size_t physical_instance_id) const;

  /** Every unresolved observation, retaining its geometry and time. */
  std::vector<FragmentView> unresolvedCandidates(size_t physical_instance_id) const;

  const model::PersistencePrior& persistencePrior() const { return prior_; }
  const model::RoundModel& roundModel() const { return rounds_; }

  /** README principle 2 (transfer): load the statistics of earlier data as the hyper-prior of
   * this registry (each source counted once by the caller). Placements restored before this call
   * keep the state they started with. */
  void importStatistics(model::PersistencePrior prior, model::RoundModel rounds) {
    prior_ = std::move(prior);
    rounds_ = std::move(rounds);
  }

 private:
  /** What one sample of a placement showed in this session (README (7s)) and the distance rho_e
   * of the measurement that made it (principle 12). */
  struct ElementState {
    float hits = 0.f, through = 0.f;  // h_e, v_e in the frames that directly saw the placement in place
    float range = 0.f;                // rho_e
    float range_count = 0.f;          // hits that built the running mean of rho_e (session-born)
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
    TimeStamp last_support_time = 0; // newest frame whose ray landed on the placement's identity
    TimeStamp input_boundary = 0;    // Processed input boundary, independent of support.
    TimeStamp track_first_seen = 0;
    // Newest frame that directly saw the placement in place (README principle 6 (2)).
    TimeStamp last_confirmed_support = 0;

    // Semantic class of this fragment: the grouping key of the prior (principle 2), nothing else.
    int semantic_label = -1;

    // A placement restored from a previous session; its CUSUM starts at 0 (principle 12).
    bool inherited = false;
    // The scale of the session that made its elements (b of (6s)); `zeta_recorded` is false for a
    // placement restored without a record of it.
    double zeta_e = 0.0;
    bool zeta_recorded = false;

    // README (5r): the CUSUM statistic, the samples already judged in the accumulation, and the
    // time through which the recursion has advanced.
    model::Cusum cusum;
    std::unordered_set<uint64_t> counted;
    TimeStamp looked_through = 0;
    // README principle 6 (7s): the placement was directly seen in place in this session; until then
    // all of its samples vote.
    bool recognized = false;
    // The rays since the last look that directly saw it in place (principle 3).
    size_t support_rays = 0, contradict_rays = 0;
    // README principle 2: the gap before the next decisive look, and the time of the last one.
    bool gap_session = false;
    TimeStamp last_decisive = 0;

    // Set when the placement is committed ended: the right end of the interval (5t).
    std::optional<TimeStamp> death_time;
    TimeStamp change_left = 0;
    TimeStamp ended_since = 0;       // the stamp of the commitment

    size_t reconstruction_frames = 0;
    // README (4): the distinct capture-frame keys of this placement's geometry, sorted.
    std::vector<TimeStamp> frame_keys;

    // README (7s): per-sample records, keyed by the sample's cell.
    std::unordered_map<uint64_t, ElementState> elements;
    mutable std::unordered_map<uint64_t, float> range_cache;

    // README (5o), E_{h->o}, E_{o->h} of a pending candidate against the placement that was
    // current when it was evaluated (ln LR of H_new : H_same), and (g).
    bool exclusion_known = false;
    uint64_t exclusion_current = 0, exclusion_revision = 0;
    double exclusion_log_lr = 0.0;
    size_t away_elements = 0, total_elements = 0;
    bool cross_band = false;
    double cold_mean = 0.0;  // m_0 of the readings that judged it (0: none)
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
  /** The pending candidates the current placement was directly seen in place after (13 (a)). */
  void absorbCommitted(PhysicalState& state);

  /** Close the CURRENT fragment with its interval (5t), leaving the ID with no CURRENT. */
  void closeCurrent(size_t id, PhysicalState& state, TimeStamp commit_stamp, TimeStamp left,
                    TimeStamp right);
  bool consumeMotion(size_t id, PhysicalState& state, const KhronosObjectAttributes& attrs);
  void promoteObservedNew(size_t id, PhysicalState& state);
  static size_t latestPendingIndex(const PhysicalState& state);
  static void closePending(PhysicalState& state, TimeStamp stamp);

  /** The type of the gap before the next decisive look of a placement at `stamp` (principle 2). */
  model::Gap gapOf(const Fragment& fragment, TimeStamp stamp) const;
  /** README principle 2 (6): the outcome of the gap before a decisive look. */
  void decideGap(size_t id, Fragment& fragment, TimeStamp stamp, bool changed);

  /** README (7s): which samples vote, and the counts n, F, the mean predicted rate m_0 and the
   * share of first-time judged samples of one look. */
  struct Look {
    double n = 0.0, f = 0.0, predicted = 0.0, reliable_total = 0.0;
    std::vector<uint64_t> fresh;  // the samples not yet judged in this accumulation
  };
  Look lookOf(const Fragment& fragment, const ElementRound& round, size_t k_min) const;
  /** ln LR of (7) for the look of an object (neutral 0 for n = 0). */
  double lookLogRatio(size_t id, const Look& look) const;

  // README (4): common geometry reduction.
  static void mergeFragments(Fragment& target, const Fragment& observation);

  // Prior Pi (principle 2): statistics of the decided gaps of all placements.
  model::PersistencePrior prior_;
  // Look statistics of principle 6, learned from the looks that directly saw a placement in place.
  model::RoundModel rounds_;

  std::map<size_t, PhysicalState> states_;
  float map_resolution_ = 0.05f;
  double previous_zeta_ = 0.0, session_zeta_ = 0.0;
  TimeStamp previous_round_ = 0, round_stamp_ = 0;  // the stamps of the two latest rounds
};

}  // namespace khronos
