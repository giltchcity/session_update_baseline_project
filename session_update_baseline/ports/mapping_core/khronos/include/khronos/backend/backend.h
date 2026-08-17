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

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <hydra/backend/backend_module.h>
#include <hydra/common/shared_module_state.h>

#include "khronos/backend/change_detection/sequential_change_detector.h"
#include "khronos/backend/change_state.h"
#include "khronos/backend/latest_only_worker.h"
#include "khronos/backend/reconciliation/persistent_object_state.h"
#include "khronos/backend/reconciliation/reconciler.h"
#include "khronos/backend/update_khronos_objects_functor.h"
#include "khronos/common/common_types.h"
#include "khronos/spatio_temporal_map/spatio_temporal_map.h"

namespace khronos {

class Backend : public hydra::BackendModule {
 public:
  // Config.
  struct Config : public hydra::BackendModule::Config {
    int verbosity;

    double max_dt_merge_proposal = 3.0;
    bool optimize_on_new_merge = true;
    bool add_merge_factor = true;

    double pose_object_covariance = 0;
    double object_merge_covariance = 0;
    double pose_object_consistency_threshold = 0;

    double fix_input_pose_variance = 1e-2;
    bool fix_input_poses = false;

    // How often to run change detection in BACKEND UPDATES, not camera frames.
    // One backend update is emitted by the frontend after ActiveWindow's
    // min_output_separation. A value of 0 runs only on loop closures. A value of
    // -1 or lower disables online change detection. Terminal finalization still
    // creates an exact final map snapshot.
    // TODO(lschmid): Refactor this together with asynchronous 4D-map updates.
    int run_change_detection_every_n_frames = 0;

    // Member configs.
    UpdateKhronosObjectsFunctor::Config update_objects;
    SpatioTemporalMap::Config spatio_temporal_map;
    SequentialChangeDetector::Config change_detection;
    Reconciler::Config reconciler;
  } const config;

  // Types.
  using Ptr = std::shared_ptr<Backend>;
  using ConstPtr = std::shared_ptr<const Backend>;
  using ChangeSink = hydra::OutputSink<TimeStamp, const Changes&>;

  // Construction.
  Backend(const Config& config,
          const hydra::SharedDsgInfo::Ptr& dsg,
          const hydra::SharedModuleState::Ptr& state);
  ~Backend();

  // Save data.
  void save(const hydra::DataDirectory& log_setup) override;
  // Production checkpoint: serialize only the authoritative 4D map and compact
  // change tables. Full module DSG/mesh/debug snapshots belong to save().
  void saveFinalMap(const hydra::DataDirectory& log_setup);
  bool saveProposedMerges(const hydra::DataDirectory& log_setup);

  // Spinning.
  void start() override;
  void spin();
  void spinCallback(const hydra::BackendInput& input);

  // Interaction.
  /**
   * @brief Run a final optimization after all input data has been received.
   */
  void finishProcessing();

  /**
   * @brief Wait until the most recently requested change-detection update has
   * completed.
   *
   * This is primarily exposed for pipeline finalization and deterministic
   * saving. Normal callers should use finishProcessing(), which also performs a
   * final synchronous update at the last received timestamp.
   */
  void waitForChangeDetection();

  void addChangeSink(const ChangeSink::Ptr& sink);

  /** Forward the shared session-local endpoint evidence store to change detection. */
  void setPhysicalEvidenceStore(PhysicalEvidenceStore::Ptr store);

  /** Inherit the active map resolution for surface correspondence checks. */
  void setObjectSurfaceResolution(float resolution);

  // Accessors.
  const RPGOMerges& getProposedMerges() const { return proposed_merges_; }
  const Changes& getChanges() const { return change_detector_->getChanges(); }
  const DynamicSceneGraph& getDsg() const { return *private_dsg_->graph; }

 protected:
  using hydra::BackendModule::optimize;  // disables hidden virtual warning
  void optimize(size_t timestamp_ns, bool force_find_merge_proposals = false);

  size_t findClosestNode(size_t timestamp_ns);

  void fixInputPoses(const hydra::BackendInput& input);

  void runChangeDetection(bool had_loopclosure);

  void runChangeDetectionThread(DynamicSceneGraph::Ptr dsg,
                                RPGOMerges rpgo_merges,
                                TimeStamp stamp,
                                bool had_loopclosure,
                                bool finalize_pending = false);

  void saveMapAndChanges(const hydra::DataDirectory& log_setup,
                         bool save_individual_dsgs);

 protected:
  // Members.
  SpatioTemporalMap map_;
  std::unique_ptr<SequentialChangeDetector> change_detector_;
  std::unique_ptr<Reconciler> reconciler_;

  // Persistent physical-object geometry registry, keyed by
  // physical_instance_id. Track segments become observations of one
  // accumulating physical object instead of each canonicalization round
  // re-electing a geometry winner from scratch (see PersistentObjectState).
  // Only ever touched by the single change-detection worker thread, inside
  // map_mutex_ (see runChangeDetectionThread). A derived SessionBackend
  // seeds this from the inherited prior-session state (D3 cross-session
  // restore) before the pipeline starts.
  PersistentObjectState persistent_objects_;

  /**
   * @brief Test every CURRENT object fragment against the measurements gathered this round, and
   * close the ones a real measurement passed through.
   *
   * This is the object half of the same visibility semantics the background mesh already gets:
   * SUPPORT / FREE_SPACE / OCCLUDED / UNOBSERVED, evaluated per surface point. A fragment is
   * contradicted only when nothing of it is still being seen and something of it was seen
   * through, so support always dominates -- a partially occluded or partially re-observed object
   * is never closed, and an object nobody looked at is never closed. Contradiction closes the
   * fragment; it never edits a mesh, so the geometry survives in the history.
   *
   * Must run before Reconciler::reconcile: rays index the background mesh by vertex number and
   * ChangeMerger::merge erases from it, which invalidates those indices for the rest of the round.
   *
   * @returns The number of fragments closed.
   */
  size_t verifyCurrentObjectStates(TimeStamp stamp);

  // One level-triggered worker. While change detection is busy, requests are
  // coalesced and the next execution snapshots only the newest backend state.
  std::mutex map_mutex_;
  std::unique_ptr<LatestOnlyWorker> change_detection_worker_;
  // Sticky OR-latch for coalesced asynchronous requests. A loop closure seen
  // by any request must survive until the worker snapshots that generation.
  std::atomic<bool> pending_change_detection_had_loopclosure_{false};
  std::atomic<uint64_t> change_detection_clone_time_ns_{0};
  std::atomic<uint64_t> change_detection_compute_time_ns_{0};
  std::mutex finalization_mutex_;
  bool final_processing_complete_ = false;

  // Variables.
  std::set<NodeId> objects_added_;
  // Mutex when accessing 'proposed_merges_'.
  std::mutex proposed_merges_mutex_;
  // All merge proposals generated and to be used in the optimization.
  hydra::MergeList new_proposed_merges_;
  // Validated result of 'new_proposed_merges_' after optimization, ready to be used in change
  // detection. TODO: Since we no longer get merges from optimization (or add objects directly to
  // optimization), this is currently broken (nothing is proposing merges).
  RPGOMerges proposed_merges_;
  uint64_t last_merge_proposal_t_ = 0;
  uint64_t last_timestamp_received_ = 0;
  int num_backend_updates_since_last_change_detection_ = 0;

  ChangeSink::List change_sinks_;
  
  // Registration for factory
  inline static const auto registration_ = config::RegistrationWithConfig<
      hydra::BackendModule,
      Backend,
      Config,
      hydra::SharedDsgInfo::Ptr,
      hydra::SharedModuleState::Ptr>("Backend");
};

void declare_config(Backend::Config& config);

}  // namespace khronos
