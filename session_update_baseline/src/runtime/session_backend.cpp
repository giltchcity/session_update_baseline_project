#include "session_update_baseline/runtime/session_backend.h"

#include <stdexcept>

#include <config_utilities/config_utilities.h>
#include <glog/logging.h>
#include <kimera_pgmo/mesh_offset_info.h>

#include "session_update_baseline/runtime/session_state.h"

namespace session_update::runtime {

void declare_config(SessionBackend::Config& config) {
  using namespace config;
  name("SessionBackend");
  base<khronos::Backend::Config>(config);
  field(config.input_state, "input_state");
}

SessionBackend::SessionBackend(const Config& config,
                               const hydra::SharedDsgInfo::Ptr& dsg,
                               const hydra::SharedModuleState::Ptr& state)
    : khronos::Backend(config, dsg, state) {
  if (!config.input_state.empty()) {
    loadInputState(config.input_state);
  }
}

void SessionBackend::loadInputState(const std::string& state_path) {
  auto seed_map = khronos::SpatioTemporalMap::load(state_path);
  if (!seed_map || seed_map->numTimeSteps() == 0) {
    throw std::runtime_error("Failed to load prior session seed map: " + state_path);
  }

  const auto seed = latestSessionSeed(*seed_map);
  const auto prior_stamp = seed.stamp;
  auto prior_dsg = seed.dsg;
  if (!prior_dsg || !prior_dsg->hasMesh()) {
    throw std::runtime_error("Prior session seed map has no mesh");
  }

  // This session inherits exactly the prior session's latest reconciled state,
  // not its timeline. Keep that state as the one initial time step and append
  // only this session's reconciled history. Thus P_B = latest(P_A) + B and
  // P_C = latest(P_B) + C, without recursively serializing A, A+B, A+B+C.
  initializeSessionTimeline(map_, seed);

  // D3 crosses a process/serialization boundary, but it is not a second
  // change algorithm. Import the prior current scene into the same working DSG
  // used for D2, then let the ordinary detector/reconciler consume this
  // session's observations.
  initializeHiddenChangeWorkingDsgPair(
      seed, *private_dsg_->graph, *unmerged_graph_);
  CHECK_EQ(private_dsg_->graph->mesh().get(), unmerged_graph_->mesh().get())
      << "Session reseed must preserve Hydra's shared live-mesh invariant";

  const auto mesh = private_dsg_->graph->mesh();

  const auto num_vertices = mesh->numVertices();
  original_vertices_->resize(num_vertices);
  vertex_stamps_->resize(num_vertices);
  for (std::size_t i = 0; i < num_vertices; ++i) {
    const auto& point = mesh->pos(i);
    (*original_vertices_)[i].x = point.x();
    (*original_vertices_)[i].y = point.y();
    (*original_vertices_)[i].z = point.z();
    (*vertex_stamps_)[i] =
        mesh->has_timestamps && i < mesh->stamps.size() ? mesh->stamps[i] : prior_stamp;
  }
  mesh_offsets_ = kimera_pgmo::MeshOffsetInfo(
      num_vertices, num_vertices, mesh->numFaces());
  last_deformed_vertices_ = num_vertices;

  change_detector_->setDsg(unmerged_graph_);

  // D3 cross-session restore: reconstruct the in-memory persistent physical-
  // object registry from the inherited OBJECTS layer, exactly as if this
  // session's backend had produced that geometry itself. Registry identity is
  // physical_instance_id, not DSG node ID, so any 'M'-prefix node-ID rewriting
  // applied while reseeding the working DSG does not affect this.
  //
  // KNOWN GAP. A DSG node carries only the CURRENT materialization, so this
  // restores exactly one fragment per physical ID. The registry's temporal
  // history (closed fragments and their birth/death bounds) and its unresolved
  // candidates do NOT survive the process boundary: .4dmap has nowhere to put
  // them. D3 is therefore only "D2 after a restart" for CURRENT, not yet for
  // history, and an object relocated in A is indistinguishable after restart
  // from one that has always been where A last saw it. Closing this needs the
  // registry serialized alongside the map, not a change to the seeding rule.
  persistent_objects_.initializeFromObjects(*unmerged_graph_);

  LOG(INFO) << "Loaded previous session state '" << state_path << "' with "
            << num_vertices << " mesh vertices into the live B backend.";
}

}  // namespace session_update::runtime
