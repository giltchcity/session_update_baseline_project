#include <filesystem>
#include <fstream>
#include "khronos/backend/change_detection/ray_verificator.h"
#include "session_update_baseline/runtime/session_backend.h"

#include <stdexcept>

#include <config_utilities/config_utilities.h>
#include <glog/logging.h>
#include <kimera_pgmo/mesh_offset_info.h>

#include "session_update_baseline/runtime/session_state.h"
#include <khronos/backend/update_khronos_objects_functor.h>
#include <khronos/utils/khronos_attribute_utils.h>

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
  // The state a predecessor hands over is its final map (state_path): the
  // surface of its latest snapshot, with the errors saved next to it, is the
  // previous surface of this session's end-of-session update (a map saved
  // without errors records none). This session reasons on the object
  // reasoning's final state saved next to it (chain_state) when the predecessor
  // saved one, else on that same latest snapshot.
  const auto state_dir = std::filesystem::path(state_path).parent_path();
  const auto chain_path = state_dir / "chain_state.4dmap.zpk";
  const bool chained = std::filesystem::exists(chain_path);
  auto final_map = khronos::SpatioTemporalMap::load(state_path);
  if (!final_map || final_map->numTimeSteps() == 0) {
    throw std::runtime_error("Failed to load prior session map: " + state_path);
  }
  std::unique_ptr<khronos::SpatioTemporalMap> chain_map;
  if (chained) {
    chain_map = khronos::SpatioTemporalMap::load(chain_path.string());
    if (!chain_map || chain_map->numTimeSteps() == 0) {
      throw std::runtime_error("Failed to load prior session chain state: " + chain_path.string());
    }
  }
  auto& seed_map = chained ? chain_map : final_map;
  khronos::SessionSurface previous =
      khronos::SessionSurface::fromDsg(*final_map->rawDsg(final_map->numTimeSteps() - 1));
  const auto error_path = state_dir / khronos::SessionSurface::kErrorFileName;
  const bool recorded = std::filesystem::exists(error_path);
  if (recorded && !previous.loadErrors(error_path.string())) {
    throw std::runtime_error("The surface errors do not match the prior session map: " + error_path.string());
  }
  LOG(INFO) << "[SessionRefusion] reasoning on " << (chained ? chain_path.string() : state_path)
            << "; previous surface: the latest snapshot of " << state_path << " with "
            << previous.numFaces() << " faces" << (recorded ? " and their errors." : " (no errors recorded).");

  {
    const auto sidecar = std::filesystem::path(state_path).parent_path() / "sensor_statistics.txt";
    if (std::filesystem::exists(sidecar) && khronos::loadAbsenceSensorStatistics(sidecar.string())) {
      LOG(INFO) << "Loaded absence sensor statistics from " << sidecar;
    }
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

  // Every vertex position of the loaded state (background and object meshes,
  // world frame): the geometry the object reasoning carries over.
  std::vector<Eigen::Vector3f> carried;
  const auto prior_mesh = prior_dsg->mesh();
  carried.reserve(prior_mesh->numVertices());
  for (std::size_t i = 0; i < prior_mesh->numVertices(); ++i) {
    carried.push_back(prior_mesh->pos(i));
  }
  if (prior_dsg->hasLayer(khronos::DsgLayers::OBJECTS)) {
    for (const auto& [id, node] : prior_dsg->getLayer(khronos::DsgLayers::OBJECTS).nodes()) {
      const auto* attrs = node->tryAttributes<khronos::KhronosObjectAttributes>();
      if (!attrs) continue;
      for (std::size_t i = 0; i < attrs->mesh.numVertices(); ++i) {
        carried.push_back(attrs->bounding_box.pointToWorldFrame(attrs->mesh.pos(i)));
      }
    }
  }
  const auto num_carried = carried.size();
  setCarriedGeometry(std::move(carried));
  LOG(INFO) << "[SessionRefusion] carried vertices: " << num_carried
            << ", previous surface faces: " << previous.numFaces();
  setPreviousSurface(std::move(previous));

  LOG(INFO) << "Loaded previous session state '" << state_path << "' with "
            << num_vertices << " mesh vertices into the live B backend.";
}

}  // namespace session_update::runtime
