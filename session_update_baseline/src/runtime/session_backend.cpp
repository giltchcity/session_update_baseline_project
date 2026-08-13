#include "session_update_baseline/runtime/session_backend.h"

#include <stdexcept>

#include <config_utilities/config_utilities.h>
#include <glog/logging.h>
#include <kimera_pgmo/mesh_offset_info.h>
#include <spark_dsg/node_symbol.h>

namespace session_update::runtime {

void declare_config(SessionBackend::Config& config) {
  using namespace config;
  name("SessionBackend");
  base<khronos::Backend::Config>(config);
  field(config.prior_map, "prior_map");
  field(config.prior_seed_map, "prior_seed_map");
}

SessionBackend::SessionBackend(const Config& config,
                               const hydra::SharedDsgInfo::Ptr& dsg,
                               const hydra::SharedModuleState::Ptr& state)
    : khronos::Backend(config, dsg, state) {
  if (!config.prior_map.empty()) {
    loadPriorMap(config.prior_map, config.prior_seed_map);
  }
}

void SessionBackend::save(const hydra::DataDirectory& log_setup) {
  {
    std::lock_guard<std::mutex> lock(map_mutex_);
    map_.update(private_dsg_->graph->clone(), last_timestamp_received_);
  }
  khronos::Backend::save(log_setup);
}

void SessionBackend::loadPriorMap(const std::string& history_path,
                                  const std::string& seed_path) {
  const auto state_path = seed_path.empty() ? history_path : seed_path;
  auto seed_map = khronos::SpatioTemporalMap::load(state_path);
  if (!seed_map || seed_map->numTimeSteps() == 0) {
    throw std::runtime_error("Failed to load prior session seed map: " + state_path);
  }

  const auto prior_stamp = seed_map->latest();
  auto prior_dsg = seed_map->getDsgPtr(prior_stamp);
  if (!prior_dsg || !prior_dsg->hasMesh()) {
    throw std::runtime_error("Prior session seed map has no mesh");
  }

  // Session B inherits only Session A's latest reconciled state. Keeping A's full
  // temporal history here duplicates it in B and makes final serialization scale
  // with both sessions. The full A map remains a separate visualization artifact.
  map_ = *seed_map;
  auto mesh = prior_dsg->mesh()->clone();
  private_dsg_->graph->setMesh(mesh);
  unmerged_graph_->setMesh(mesh);

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

  if (prior_dsg->hasLayer(khronos::DsgLayers::OBJECTS)) {
    const auto& objects = prior_dsg->getLayer(khronos::DsgLayers::OBJECTS);
    for (const auto& [node_id, node] : objects.nodes()) {
      spark_dsg::NodeSymbol source(node_id);
      spark_dsg::NodeSymbol memory_id('M', source.categoryId());
      while (private_dsg_->graph->hasNode(memory_id)) {
        ++memory_id;
      }
      private_dsg_->graph->emplaceNode(
          khronos::DsgLayers::OBJECTS, memory_id, node->attributes().clone());
      unmerged_graph_->emplaceNode(
          khronos::DsgLayers::OBJECTS, memory_id, node->attributes().clone());
    }
  }

  change_detector_->setDsg(unmerged_graph_);
  LOG(INFO) << "Loaded prior session final-state seed '" << state_path
            << "' (history retained separately at '" << history_path << "') with "
            << num_vertices << " mesh vertices into the live B backend.";
}

}  // namespace session_update::runtime
