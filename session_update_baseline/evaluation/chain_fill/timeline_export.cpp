// Evaluation only: a 4D map's snapshots as the object timeline that eval/timeline_to_4dmap reads back (the format
// another method exports to use the same official 3RScan evaluation). One JSON line per snapshot:
//   {index, stamp_ns, mesh: {map, index}, objects: [{id, semantic, first_observed_ns, last_observed_ns, present,
//    bbox_center, bbox_dimensions, surface: [[x,y,z],...] (world frame), details: {NAME: [...], ...}}]}
// mesh names this map's snapshot, whose global mesh (with the vertices' first-seen times, which the official evaluator
// uses to time objects first observed at 0) and layers the snapshot is written back with; details are all the
// node's details (e.g. the relocations the map counts, session_mobility_changes, and the first observation at the
// new place, observation_first_stamp_ns). MAX_POINTS caps an object's surface (every k-th mesh point); 0 = all.
// usage: timeline_export MAP OUT.jsonl [MAX_POINTS]
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>
#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <khronos/utils/khronos_attribute_utils.h>

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "usage: timeline_export MAP OUT.jsonl [MAX_POINTS]\n";
    return 2;
  }
  const size_t max_points = argc == 4 ? std::stoul(argv[3]) : 0;
  const auto map = khronos::SpatioTemporalMap::load(argv[1]);
  if (!map || !map->numTimeSteps()) return 3;
  const std::string source = std::filesystem::absolute(argv[1]).string();
  std::ofstream out(argv[2]);
  for (size_t ix = 0; ix < map->numTimeSteps(); ++ix) {
    const auto dsg = map->rawDsg(ix);
    const auto stamp = map->stamps()[ix];
    nlohmann::json snap = {{"index", ix}, {"stamp_ns", stamp}, {"mesh", {{"map", source}, {"index", ix}}},
                           {"objects", nlohmann::json::array()}};
    if (dsg->hasLayer(khronos::DsgLayers::OBJECTS)) {
      for (const auto& [id, node] : dsg->getLayer(khronos::DsgLayers::OBJECTS).nodes()) {
        const auto* a = node->tryAttributes<khronos::KhronosObjectAttributes>();
        if (!a) continue;
        const auto& box = a->bounding_box;
        nlohmann::json row = {{"id", std::to_string(id)}, {"semantic", a->semantic_label},
                              {"first_observed_ns", a->first_observed_ns}, {"last_observed_ns", a->last_observed_ns},
                              {"present", khronos::isPresent(*a, stamp)},
                              {"bbox_center", {box.world_P_center.x(), box.world_P_center.y(), box.world_P_center.z()}},
                              {"bbox_dimensions", {box.dimensions.x(), box.dimensions.y(), box.dimensions.z()}},
                              {"surface", nlohmann::json::array()}, {"details", a->details}};
        const size_t n = a->mesh.points.size();
        const size_t stride = max_points && n > max_points ? (n + max_points - 1) / max_points : 1;
        for (size_t i = 0; i < n; i += stride) {
          const Eigen::Vector3f w = box.pointToWorldFrame(a->mesh.points[i]);
          row["surface"].push_back({w.x(), w.y(), w.z()});
        }
        snap["objects"].push_back(std::move(row));
      }
    }
    out << snap.dump() << "\n";
    if (!out) return 4;
  }
  return 0;
}
