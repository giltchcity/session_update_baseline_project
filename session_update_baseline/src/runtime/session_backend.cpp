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
  // A predecessor stores its updated final map (what is evaluated and shown,
  // also saved as shown_state) and, next to it, the object reasoning's final
  // state (chain_state). Reason on the chain state, so object and change
  // reasoning is exactly that of the plain map; show the shown state as memory.
  const auto state_dir = std::filesystem::path(state_path).parent_path();
  const auto chain_path = state_dir / "chain_state.4dmap.zpk";
  const bool chained = std::filesystem::exists(chain_path);
  auto seed_map = khronos::SpatioTemporalMap::load(chained ? chain_path.string() : state_path);
  if (chained) {
    LOG(INFO) << "[SessionRefusion] reasoning on the chain state " << chain_path;
  }
  if (!seed_map || seed_map->numTimeSteps() == 0) {
    throw std::runtime_error("Failed to load prior session seed map: " + state_path);
  }

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

  // Every surface point of the loaded state (background and object meshes,
  // world frame): which surface of this session's final map is memory.
  std::vector<Eigen::Vector3f> memory;
  const auto prior_mesh = prior_dsg->mesh();
  memory.reserve(prior_mesh->numVertices());
  for (std::size_t i = 0; i < prior_mesh->numVertices(); ++i) {
    memory.push_back(prior_mesh->pos(i));
  }
  if (prior_dsg->hasLayer(khronos::DsgLayers::OBJECTS)) {
    for (const auto& [id, node] : prior_dsg->getLayer(khronos::DsgLayers::OBJECTS).nodes()) {
      const auto* attrs = node->tryAttributes<khronos::KhronosObjectAttributes>();
      if (!attrs) continue;
      for (std::size_t i = 0; i < attrs->mesh.numVertices(); ++i) {
        memory.push_back(attrs->bounding_box.pointToWorldFrame(attrs->mesh.pos(i)));
      }
    }
  }
  const auto num_memory = memory.size();
  setLoadedMemory(std::move(memory));
  LOG(INFO) << "[SessionRefusion] loaded surface points: " << num_memory;

  // Memory as the previous session's final map showed it (its shown state, one
  // snapshot saved next to the chain state): the surface this session's final
  // map composes its memory from.
  {
    std::vector<float> scales;
    std::ifstream in(state_dir / "depth_scales.txt");
    for (float s; in >> s;) scales.push_back(s);
    LOG(INFO) << "[SessionRefusion] depth scales of the earlier sessions: " << scales.size();
    setPreviousDepthScales(std::move(scales));
  }
  const auto shown_path = state_dir / "shown_state.4dmap.zpk";
  if (chained && std::filesystem::exists(shown_path)) {
    const auto shown_map = khronos::SpatioTemporalMap::load(shown_path.string());
    if (shown_map && shown_map->numTimeSteps() > 0) {
      const auto shown_dsg = shown_map->rawDsg(shown_map->numTimeSteps() - 1);
      khronos::SessionRefusion::Surface shown;
      auto add = [&](const spark_dsg::Mesh& mesh, const khronos::KhronosObjectAttributes* attrs,
                     uint32_t physical) {
        const auto base = static_cast<uint32_t>(shown.vertices.size());
        const std::size_t n = mesh.numVertices();
        for (std::size_t i = 0; i < n; ++i) {
          shown.vertices.push_back(attrs ? attrs->bounding_box.pointToWorldFrame(mesh.pos(i))
                                         : mesh.pos(i));
        }
        for (const auto& f : mesh.faces) {
          if (f[0] >= n || f[1] >= n || f[2] >= n) continue;
          shown.faces.push_back({base + static_cast<uint32_t>(f[0]), base + static_cast<uint32_t>(f[1]),
                                 base + static_cast<uint32_t>(f[2])});
          shown.face_physical.push_back(physical);
        }
      };
      if (shown_dsg->hasMesh() && shown_dsg->mesh()) add(*shown_dsg->mesh(), nullptr, 0);
      if (shown_dsg->hasLayer(khronos::DsgLayers::OBJECTS)) {
        for (const auto& [id, node] : shown_dsg->getLayer(khronos::DsgLayers::OBJECTS).nodes()) {
          const auto* attrs = node->tryAttributes<khronos::KhronosObjectAttributes>();
          if (!attrs || !khronos::hasCurrentObjectMesh(*attrs)) continue;
          add(attrs->mesh, attrs,
              static_cast<uint32_t>(
                  khronos::UpdateKhronosObjectsFunctor::physicalInstanceId(*attrs).value_or(0)));
        }
      }
      LOG(INFO) << "[SessionRefusion] memory as the previous final map showed it: "
                << shown.faces.size() << " faces from " << shown_path;
      setShownMemory(std::move(shown));
    }
  }

  LOG(INFO) << "Loaded previous session state '" << state_path << "' with "
            << num_vertices << " mesh vertices into the live B backend.";
}

}  // namespace session_update::runtime
