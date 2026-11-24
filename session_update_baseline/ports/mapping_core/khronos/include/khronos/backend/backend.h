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

#include <array>
#include <atomic>
#include <deque>
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
#include <optional>


#include "khronos/backend/reconciliation/reconciler.h"
#include "khronos/backend/update_khronos_objects_functor.h"
#include "khronos/common/common_types.h"
#include "khronos/active_window/object_extraction/move_trajectories.h"
#include "khronos/spatio_temporal_map/spatio_temporal_map.h"

namespace khronos {

class PhysicalEvidenceStore;
class PresentSurface;

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


    // Keep the 4D map's history as versioned pieces (SpatioTemporalMap::usePieceHistory), split by the active
    // window's own voxel blocks: a mesh block, node or edge is stored again only when it changed. False: one
    // keyframe/delta snapshot per update, as before.
    bool piece_history = true;

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

  /** The 4D map keeps its history as versioned pieces; block_edge: the active window's voxel block edge. */
  void usePieceHistory(double block_edge) { map_.usePieceHistory(block_edge); }

  // A mapping session built on this backend (session_update's SessionBackend) takes the pipeline's
  // shared evidence store and present surface; the plain backend has no session layer.
  struct SessionInputs {
    std::shared_ptr<PhysicalEvidenceStore> evidence;
    std::shared_ptr<PresentSurface> present;
    float object_surface_resolution = 0.05f;
  };
  virtual bool hasSessionLayer() const { return false; }
  virtual void connectSession(const SessionInputs& /*inputs*/) {}

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

  // One round on the state of its request. A mapping session (session_update's SessionBackend) runs
  // its own round in place of Khronos's change detection and reconciliation.
  virtual void runChangeDetectionThread(DynamicSceneGraph::Ptr dsg,
                                        RPGOMerges rpgo_merges,
                                        TimeStamp stamp,
                                        bool had_loopclosure,
                                        bool finalize_pending = false);

  void saveMapAndChanges(const hydra::DataDirectory& log_setup,
                         bool save_individual_dsgs);

 protected:
  // Extension points of a mapping session (session_update's SessionBackend), each called by the
  // change-detection round at its place; the plain backend does nothing there.
  virtual void beforeChangeDetection(const DynamicSceneGraph::Ptr& /*dsg*/) {}
  virtual void beforeReconcile(const DynamicSceneGraph::Ptr& /*dsg*/, Changes& /*changes*/,
                               TimeStamp /*stamp*/, bool /*finalize_pending*/) {}
  virtual void afterReconcile(const DynamicSceneGraph::Ptr& /*dsg*/, TimeStamp /*stamp*/,
                              bool /*finalize_pending*/) {}
  virtual void afterSnapshotStored(const DynamicSceneGraph& /*dsg*/, TimeStamp /*stamp*/) {}
  virtual void afterTerminalRound() {}
  virtual void afterFinalMapSaved(const hydra::DataDirectory& /*log_setup*/) {}

  // The trajectories of the session's moves (MoveTrajectories), written to the snapshots of the
  // 4D map only: a snapshot as it is stored, the snapshots stored before a move was told as soon
  // as it is told (the latest few), and before the map is saved those of the moves told last.
  // The graph the mapping works on never carries them.
  void storeSnapshot(const DynamicSceneGraph::Ptr& dsg, TimeStamp stamp);
  void fillMoveTrajectories();
  // Writes the moves told by now to the stored snapshots they belong to; returns how many snapshots were stored anew.
  size_t writeToldMoves(const std::map<int, std::vector<MoveTrajectories::Sample>>& moves);
  // Snapshots stored anew while the session ran, as moves were told.
  size_t snapshots_stored_anew_ = 0;
  // Keeps one record of move_samples_ per snapshot of the map: a snapshot stored without storeSnapshot (a chained
  // session's seed, its first) carries no sample of this session's moves.
  void alignMoveSamples();
  // Per snapshot and instance: how many samples of its moves the snapshot carries.
  std::vector<std::map<int, size_t>> move_samples_;

  // Members.
  SpatioTemporalMap map_;
  std::unique_ptr<SequentialChangeDetector> change_detector_;
  std::unique_ptr<Reconciler> reconciler_;

  // One worker runs every requested change detection, in order, each on the
  // backend state of its request (cloned when requested): the 4D map gets a
  // snapshot at every requested time, however long one execution takes.
  struct ChangeDetectionJob {
    DynamicSceneGraph::Ptr dsg;
    RPGOMerges merges;
    TimeStamp stamp = 0;
    bool had_loopclosure = false;
  };
  std::mutex map_mutex_;
  std::unique_ptr<LatestOnlyWorker> change_detection_worker_;
  std::mutex change_detection_jobs_mutex_;
  std::deque<ChangeDetectionJob> change_detection_jobs_;
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
