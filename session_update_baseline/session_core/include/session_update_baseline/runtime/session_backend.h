#pragma once

#include <array>
#include <map>
#include <optional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <atomic>
#include <deque>
#include <mutex>

#include <hydra/backend/backend_module.h>
#include <hydra/common/global_info.h>
#include <hydra/common/shared_module_state.h>
#include <hydra/utils/nearest_neighbor_utilities.h>
#include "session_core/objects/move_trajectories.h"
#include "session_core/runtime/latest_only_worker.h"
#include <khronos/spatio_temporal_map/spatio_temporal_map.h>

#include "session_core/evidence/physical_evidence_store.h"
#include "session_core/evidence/surface_evidence.h"
#include "session_core/runtime/mapping_backend.h"
#include "session_core/state/persistent_object_state.h"
#include "session_core/surface/present_surface.h"

namespace session_update::runtime {

/**
 * The mapping session's backend, on Hydra's backend module (the frontend's scene graph and mesh,
 * the pose graph): every N backend updates it runs one round on the state of that moment, which
 * decides the objects' states across and within visits (PersistentObjectState), dates their
 * presence, composes the present surface with the memory of the previous visit (PresentSurface)
 * and stores the round's snapshot in the 4D map; at the end it saves the map and the session state
 * the next visit starts from.
 */
class SessionBackend : public hydra::BackendModule, public khronos::session::MappingBackend {
 public:
  struct Config : public hydra::BackendModule::Config {
    int verbosity = hydra::GlobalInfo::instance().getConfig().default_verbosity;
    // A round every n backend updates (one update per frontend output); 0: only on loop closures and
    // the terminal round; negative: the terminal round alone.
    int run_change_detection_every_n_frames = 0;
    // The 4D map keeps its history as versioned pieces (SpatioTemporalMap::usePieceHistory).
    bool piece_history = true;
    khronos::SpatioTemporalMap::Config spatio_temporal_map;
    // The session's measurement of its objects' surfaces (the keys of change_detection/ray_verificator).
    khronos::session::SurfaceEvidence::Config evidence;
    // The one recursive state input, produced by a previous run_session call.
    std::string input_state;
    // Config-driven semantic ontology prior for the generic moveability decision: semantic
    // categories whose members are generally movable (chairs, bags, fans, monitors, ...). A weak
    // prior used only to decide whether surface overlap between fragments is trustworthy
    // co-observation evidence; it never deletes or merges anything by itself. Empty = ontology
    // disabled (only observed D1 history and past relocation frequency are used).
    std::vector<int> high_mobility_semantic_labels;
  } const config;

  SessionBackend(const Config& config,
                 const hydra::SharedDsgInfo::Ptr& dsg,
                 const hydra::SharedModuleState::Ptr& state);
  // No round may still run while this layer's state goes away.
  ~SessionBackend() override;

  // Hydra's module: the backend's own spin loop, and its save (the final map).
  void start() override;
  void save(const hydra::DataDirectory& log_setup) override;

  // The pipeline's view (MappingBackend).
  bool usesPieceHistory() const override { return config.piece_history; }
  void usePieceHistory(double block_edge) override { map_.usePieceHistory(block_edge); }
  void connectSession(const SessionInputs& inputs) override;
  void finishProcessing() override;
  void saveFinalMap(const hydra::DataDirectory& log_setup) override;

  // Until the latest requested round has run.
  void waitForRounds();

 private:
  // The backend loop: each frontend output advances the scene graph and the mesh; every N updates a
  // round is requested on a clone of the state of that moment.
  void spinLoop();
  void spinCallback(const hydra::BackendInput& input);
  void requestRound(bool had_loopclosure);

  // One round on the state of its request: identity merges, the state layer's decision on every
  // CURRENT state from the stored pixels, presence, one logical node per identity, the composed
  // surface, the snapshot.
  void runRound(khronos::DynamicSceneGraph::Ptr dsg, khronos::TimeStamp stamp, bool finalize_pending);
  void mergeIdentities(const khronos::DynamicSceneGraph::Ptr& dsg);
  void decideStates(const khronos::DynamicSceneGraph::Ptr& dsg, khronos::TimeStamp stamp, bool finalize_pending);
  void composeRound(const khronos::DynamicSceneGraph::Ptr& dsg, khronos::TimeStamp stamp, bool finalize_pending);
  void showStatesFromStart(const khronos::DynamicSceneGraph& dsg, khronos::TimeStamp stamp);
  void afterTerminalRound();
  void saveSessionState(const hydra::DataDirectory& log_setup);

  // The trajectories of the session's moves (MoveTrajectories), written to the snapshots of the
  // 4D map only: a snapshot as it is stored, the snapshots stored before a move was told as soon
  // as it is told (the latest few), and before the map is saved those of the moves told last.
  // The graph the mapping works on never carries them.
  void storeSnapshot(const khronos::DynamicSceneGraph::Ptr& dsg, khronos::TimeStamp stamp);
  void fillMoveTrajectories();
  // Writes the moves told by now to the stored snapshots they belong to; returns how many snapshots
  // were stored anew.
  size_t writeToldMoves(const std::map<int, std::vector<khronos::MoveTrajectories::Sample>>& moves);
  // Keeps one record of move_samples_ per snapshot of the map: a snapshot stored without
  // storeSnapshot (a chained session's seed, its first) carries no sample of this session's moves.
  void alignMoveSamples();

  void loadInputState(const std::string& state_path);

  /** Surface (world vertices and triangles) of the inherited state this session started from. */
  void setLoadedMemory(std::vector<Eigen::Vector3f> points,
                       std::vector<std::array<uint32_t, 3>> faces = {});
  /**
   * Memory as the previous session's final map showed it (its saved shown state): the surface
   * every round composes its memory from (PresentSurface::Memory::shown).
   */
  void setShownMemory(khronos::PresentSurface::Surface shown);

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
   * @returns The number of fragments closed.
   */
  size_t verifyCurrentObjectStates(khronos::TimeStamp stamp);

  /**
   * @brief Opens every physical object node's presence and dates its start from the session's own
   * pixels (session::presenceStart: the observed-absence test run backwards over the rounds before
   * the node's first observation). The presence stays open: the state layer ends it when the
   * identity's last state closes (applyPhysicalGeometry), so no segment of an identity ends on its own.
   */
  void estimatePhysicalPresence(khronos::DynamicSceneGraph& dsg, khronos::TimeStamp stamp);

  // The 4D map and its rounds: one worker runs every requested round, in order, each on the backend
  // state of its request (cloned when requested), so the map gets a snapshot at every requested time.
  khronos::SpatioTemporalMap map_;
  std::mutex map_mutex_;
  struct Round {
    khronos::DynamicSceneGraph::Ptr dsg;
    khronos::TimeStamp stamp = 0;
    bool had_loopclosure = false;
  };
  std::unique_ptr<khronos::LatestOnlyWorker> round_worker_;
  std::mutex rounds_mutex_;
  std::deque<Round> rounds_;
  std::atomic<uint64_t> round_clone_time_ns_{0};
  std::atomic<uint64_t> round_compute_time_ns_{0};
  std::mutex finalization_mutex_;
  bool final_processing_complete_ = false;
  uint64_t last_timestamp_received_ = 0;
  int num_backend_updates_since_last_round_ = 0;
  // Per snapshot and instance: how many samples of its moves the snapshot carries.
  std::vector<std::map<int, size_t>> move_samples_;
  // Snapshots stored anew while the session ran, as moves were told.
  size_t snapshots_stored_anew_ = 0;

  const std::vector<int> high_mobility_semantic_labels_;
  // The session's own measurement of its objects' surfaces (stored pixels, observed absence).
  khronos::session::SurfaceEvidence surface_evidence_;
  // The session's round stamps: each ends one look of the stored frames.
  std::vector<uint64_t> round_ends_;
  khronos::PhysicalEvidenceStore::Ptr physical_evidence_store_;
  khronos::PresentSurface::Ptr present_surface_;
  // The previous session's surfaces.
  std::vector<Eigen::Vector3f> loaded_memory_;
  std::vector<std::array<uint32_t, 3>> loaded_memory_faces_;
  std::unique_ptr<hydra::PointNeighborSearch> loaded_memory_search_;
  float memory_copy_tolerance_ = 0.f;
  std::unique_ptr<khronos::PresentSurface::Surface> shown_memory_;
  // The terminal round's report.
  std::string refusion_report_;
  // Persistent physical-object geometry registry, keyed by physical_instance_id. Track segments
  // become observations of one accumulating physical object instead of each canonicalization
  // round re-electing a geometry winner from scratch. Only ever touched by the single
  // change-detection worker thread, inside the backend's map mutex; seeded from the inherited
  // prior-session state (D3 cross-session restore) before the pipeline starts.
  khronos::PersistentObjectState persistent_objects_;
  float object_surface_resolution_ = 0.05f;
  // A state begun in this session is shown from its start (its first sighting), as the 3DGS map
  // dates its rows: the round that makes it CURRENT notes it (composeRound) and, once that
  // round's snapshot is stored, the snapshots stored since its start take its surface.
  std::map<size_t, uint64_t> session_current_uid_;
  std::vector<std::pair<size_t, khronos::TimeStamp>> session_new_states_;

  inline static const auto registration_ = config::RegistrationWithConfig<
      hydra::BackendModule,
      SessionBackend,
      Config,
      hydra::SharedDsgInfo::Ptr,
      hydra::SharedModuleState::Ptr>("SessionBackend");
};

void declare_config(SessionBackend::Config& config);

}  // namespace session_update::runtime
