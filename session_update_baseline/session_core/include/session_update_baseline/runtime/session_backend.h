#pragma once

#include <array>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <hydra/common/shared_module_state.h>
#include <hydra/utils/nearest_neighbor_utilities.h>
#include <khronos/backend/backend.h>
#include <khronos/backend/change_detection/ray_change_detector.h>

#include "session_core/evidence/physical_evidence_store.h"
#include "session_core/evidence/surface_evidence.h"
#include "session_core/state/persistent_object_state.h"
#include "session_core/surface/present_surface.h"

namespace session_update::runtime {

/**
 * The mapping session: Khronos's backend runs the per-visit reconstruction, this layer decides
 * the objects' states across and within visits (PersistentObjectState), composes every round's
 * present surface with the memory of the previous visit (PresentSurface), dates the history and
 * saves the session state the next visit starts from. It enters the backend's rounds through the
 * backend's extension points only.
 */
class SessionBackend : public khronos::Backend {
 public:
  struct Config : public khronos::Backend::Config {
    // The one recursive state input, produced by a previous run_session call.
    std::string input_state;
    // Config-driven semantic ontology prior for the generic moveability decision: semantic
    // categories whose members are generally movable (chairs, bags, fans, monitors, ...). A weak
    // prior used only to decide whether surface overlap between fragments is trustworthy
    // co-observation evidence; it never deletes or merges anything by itself. Empty = ontology
    // disabled (only observed D1 history and past relocation frequency are used).
    std::vector<int> high_mobility_semantic_labels;
  };

  SessionBackend(const Config& config,
                 const hydra::SharedDsgInfo::Ptr& dsg,
                 const hydra::SharedModuleState::Ptr& state);
  // No round may still run while this layer's state goes away (the base stops its worker only
  // after the derived members are destroyed).
  ~SessionBackend() override;

  bool hasSessionLayer() const override { return true; }
  void connectSession(const SessionInputs& inputs) override;

 protected:
  void beforeChangeDetection(const khronos::DynamicSceneGraph::Ptr& dsg) override;
  void beforeReconcile(const khronos::DynamicSceneGraph::Ptr& dsg, khronos::Changes& changes,
                       khronos::TimeStamp stamp, bool finalize_pending) override;
  void afterReconcile(const khronos::DynamicSceneGraph::Ptr& dsg, khronos::TimeStamp stamp,
                      bool finalize_pending) override;
  void afterSnapshotStored(const khronos::DynamicSceneGraph& dsg, khronos::TimeStamp stamp) override;
  void afterTerminalRound() override;
  void afterFinalMapSaved(const hydra::DataDirectory& log_setup) override;

 private:
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
   * Must run before Reconciler::reconcile: rays index the background mesh by vertex number and
   * ChangeMerger::merge erases from it, which invalidates those indices for the rest of the round.
   *
   * @returns The number of fragments closed.
   */
  size_t verifyCurrentObjectStates(khronos::TimeStamp stamp);

  /**
   * @brief Estimate every physical object node's presence interval from the session's own pixels.
   *
   * Before its first and after its last real observation, each sampled surface point of the node is
   * looked up in the stored frames (SurfaceEvidence::checkProjected, through the round's stamp);
   * the change detection's temporal estimator (its ray_change_detector settings, physical coverage)
   * gives the closest absence and the furthest persistence on either side, and the reconciler's rule
   * turns them into the interval: an observed absence ends (starts) presence halfway between it
   * and the nearest evidence of presence (minimum expected risk); without one, an unobserved site
   * is not an absent object (time_estimates_conservative: the evidence itself). The object's state
   * then ends a closed identity's presence no later than its own decision (applyPhysicalGeometry).
   */
  void estimatePhysicalPresence(khronos::DynamicSceneGraph& dsg, khronos::TimeStamp stamp);

  const std::vector<int> high_mobility_semantic_labels_;
  // The session's own measurement of its objects' surfaces (stored pixels, observed absence).
  khronos::session::SurfaceEvidence surface_evidence_;
  // The temporal estimator and the object query settings of the change detection (the same keys).
  khronos::RayChangeDetector presence_estimator_;
  khronos::TimeStamp presence_query_margin_ns_ = 0;
  int presence_query_subsampling_ = 100;
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
  // dates its rows: the round that makes it CURRENT notes it (afterReconcile) and, once that
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
