#include <filesystem>
#include <cmath>
#include "session_core/runtime/session_bundle.h"
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
  session_extensions_enabled_ = true;
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
  std::set<std::string> bound_members;
  uint64_t bundle_stamp = 0;
  const bool bound_bundle = khronos::session_io::verifyBundle(state_path, &bound_members, &bundle_stamp);
  const auto present = [&](const std::string& name) {
    return bound_bundle ? bound_members.count(name) != 0
                        : std::filesystem::exists(state_dir / name);
  };
  const auto chain_path = state_dir / "chain_state.4dmap.zpk";
  const bool chained = present("chain_state.4dmap.zpk");
  auto seed_map = khronos::SpatioTemporalMap::load(chained ? chain_path.string() : state_path);
  if (chained) {
    LOG(INFO) << "[SessionRefusion] reasoning on the chain state " << chain_path;
  }
  if (!seed_map || seed_map->numTimeSteps() == 0) {
    throw std::runtime_error("Failed to load prior session seed map: " + state_path);
  }

  const auto seed = latestSessionSeed(*seed_map);
  const auto prior_stamp = seed.stamp;
  if (bound_bundle && prior_stamp != bundle_stamp)
    throw std::invalid_argument("Session reasoning boundary differs from bundle");
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

  // Seed CURRENT from its own materialized geometry and finite state metadata.
  // Observation-interval metadata remains the native input provenance.
  const auto registry_path = state_dir / "registry_state.cbor";
  if (chained && present("registry_state.cbor")) {
    persistent_objects_.loadCheckpoint(registry_path.string(), chain_path.string(),
                                      *unmerged_graph_, prior_stamp);
  } else {
    persistent_objects_.initializeFromObjects(*unmerged_graph_, prior_stamp);
  }
  persistent_objects_.materialize(*unmerged_graph_);
  persistent_objects_.materialize(*private_dsg_->graph);
  change_detector_->setDsg(unmerged_graph_);
  const auto verificator = change_detector_->getRayVerificator();
  if (!verificator) throw std::logic_error("Session evidence model is unavailable");
  auto& absence = verificator->observedAbsenceModel();
  if (present("evidence_state.cbor")) {
    absence.load((state_dir / "evidence_state.cbor").string(),prior_stamp,
                 persistent_objects_.liveEvidenceKeys());
    const auto motion = absence.motionState();
    if (motion.is_object()) frame_attribution_->motion().fromJson(motion);
  } else {
    // README s7.1: a map without the finite evidence state starts from the default range model of
    // the appendix and the cold-start statistics of principles 2 and 6.
    LOG(WARNING) << "No evidence state next to " << state_path << "; starting from the default model";
  }


  // Memory as the previous session's final map showed it (its shown state, one
  // snapshot saved next to the chain state): the surface this session's final
  // map composes its memory from.
  {
    std::vector<float> scales;
    std::ifstream in(state_dir / "depth_scales.txt");
    if (present("depth_scales.txt")) {
      if (!in) throw std::runtime_error("Cannot read depth scale history");
      for (float scale; in >> scale;) {
        // README (9b): multiplicative range correction is signed; negative
        // calibration remains valid across a session boundary.
        if (!std::isfinite(scale))
          throw std::invalid_argument("Invalid prior depth scale");
        scales.push_back(scale);
      }
      if (!in.eof()) throw std::invalid_argument("Malformed depth scale history");
    }
    LOG(INFO) << "[SessionRefusion] depth scales of the earlier sessions: " << scales.size();
    setPreviousDepthScales(std::move(scales));
  }
  const auto shown_path = state_dir / "shown_state.4dmap.zpk";
  const auto memory_map = chained && present("shown_state.4dmap.zpk")
      ? khronos::SpatioTemporalMap::load(shown_path.string())
      : khronos::SpatioTemporalMap::load(state_path);
  if (!memory_map || memory_map->numTimeSteps() == 0) {
    throw std::runtime_error("Failed to load prior session surface: " + state_path);
  }
  if (bound_bundle && memory_map->stamps().back() != bundle_stamp)
    throw std::invalid_argument("Session shown boundary differs from bundle");
  auto shown = khronos::SessionRefusion::fromDsg(
      *memory_map->rawDsg(memory_map->numTimeSteps() - 1));
  if (present("surface_error.bin"))
    khronos::SessionRefusion::loadSurfaceError((state_dir / "surface_error.bin").string(), shown);
  LOG(INFO) << "[SessionRefusion] inherited surface faces=" << shown.faces.size();
  // README (15b), principle 9: the memory elements the last map did not show but did not delete
  // are memory again; their records bring the evidence collected so far.
  {
    const auto hidden = absence.hiddenRecords();
    if (!hidden.is_null()) {
      const auto records = khronos::SessionRefusion::Surface::fromJson(hidden);
      shown.append(records);
      LOG(INFO) << "[SessionRefusion] hidden memory records=" << records.faces.size();
    }
  }
  setShownMemory(std::move(shown));

  LOG(INFO) << "Loaded previous session state '" << state_path << "' with "
            << num_vertices << " mesh vertices into the live B backend.";
}

}  // namespace session_update::runtime
