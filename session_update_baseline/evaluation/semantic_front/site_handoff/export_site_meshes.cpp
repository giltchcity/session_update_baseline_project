#include <filesystem>
#include <fstream>
#include <iostream>

#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <khronos/utils/khronos_attribute_utils.h>
#include <nlohmann/json.hpp>

// Read the completed visit once. Keep triangle connectivity and every vertex
// in world coordinates so identity evidence can use the actual mesh surface.
int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: export_site_meshes INPUT.4dmap OUTPUT_DIRECTORY\n";
    return 2;
  }
  const auto map = khronos::SpatioTemporalMap::load(argv[1]);
  if (!map || !map->numTimeSteps()) return 3;
  const auto index = map->numTimeSteps() - 1;
  const auto stamp = map->stamps()[index];
  const auto graph = map->rawDsg(index);
  using Json = nlohmann::json;
  Json rows = Json::array();
  if (graph->hasLayer(khronos::DsgLayers::OBJECTS)) {
    for (const auto& [id, node] : graph->getLayer(khronos::DsgLayers::OBJECTS).nodes()) {
      const auto* object = node->tryAttributes<khronos::KhronosObjectAttributes>();
      if (!object || !khronos::isPresent(*object, stamp) || !object->mesh.numVertices()) continue;
      const auto ids = object->details.find("instance_id");
      if (ids == object->details.end() || ids->second.empty()) continue;
      Json vertices = Json::array();
      for (const auto& vertex : object->mesh.points) {
        const auto world = object->bounding_box.pointToWorldFrame(vertex);
        vertices.push_back({world.x(), world.y(), world.z()});
      }
      rows.push_back({{"id", std::to_string(id)}, {"semantic", object->semantic_label},
                      {"details", {{"instance_id", ids->second}}}, {"present", true},
                      {"mesh_vertices", object->mesh.numVertices()}, {"surface", std::move(vertices)},
                      {"faces", object->mesh.faces}});
    }
  }
  std::filesystem::create_directories(argv[2]);
  std::ofstream out(std::filesystem::path(argv[2]) / "snapshots.jsonl");
  out << Json({{"index", index}, {"stamp_ns", stamp}, {"objects", std::move(rows)}}).dump() << '\n';
  out.close();
  return out ? 0 : 4;
}
