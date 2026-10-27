// Evaluation only: an object timeline (eval/timeline_complete.py's output, the timeline_export format) written as a
// 4D map for Khronos's official evaluator. Each line becomes one snapshot: when its "mesh" names a map snapshot
// ({map, index}, timeline_export), that snapshot with its objects removed (its layers and its global mesh); else the
// TEMPLATE map's first snapshot (any map of the room: its layer layout) with its objects removed and the global mesh
// of the "mesh" file (timeline_complete.py: vertices with their first-seen times, which the official evaluator uses
// to time objects first observed at 0). Then one object node per timeline object, with its id when the id is a
// number (else numbered in order): its box (bbox_center and bbox_dimensions, else bbox_min and bbox_max, else the
// axis-aligned box around its surface points), the surface points box-local as its mesh, its semantic class,
// first/last observed times and details. An object without surface points keeps an empty mesh when it has a box
// and is left out otherwise.
// usage: timeline_to_4dmap TEMPLATE_MAP TIMELINE.jsonl OUT.4dmap
#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <iostream>
#include <memory>
#include <string>

#include <nlohmann/json.hpp>
#include <khronos/spatio_temporal_map/spatio_temporal_map.h>

namespace {
Eigen::Vector3f vec(const nlohmann::json& v) { return {v[0].get<float>(), v[1].get<float>(), v[2].get<float>()}; }

void removeObjects(spark_dsg::DynamicSceneGraph& dsg) {
  std::vector<spark_dsg::NodeId> old;
  for (const auto& [id, node] : dsg.getLayer(khronos::DsgLayers::OBJECTS).nodes()) old.push_back(id);
  for (const auto id : old) dsg.removeNode(id);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: timeline_to_4dmap TEMPLATE_MAP TIMELINE.jsonl OUT.4dmap\n";
    return 2;
  }
  const auto templ = khronos::SpatioTemporalMap::load(argv[1]);
  if (!templ || !templ->numTimeSteps()) return 3;
  auto base = templ->rawDsg(0)->clone();
  const auto objects = base->getLayerKey(khronos::DsgLayers::OBJECTS);
  if (!objects) return 3;
  removeObjects(*base);
  base->setMesh(std::make_shared<spark_dsg::Mesh>());

  khronos::SpatioTemporalMap::Config config;
  config.finalize_incrementally = false;
  khronos::SpatioTemporalMap out(config);
  std::ifstream in(argv[2]);
  std::string line;
  size_t snapshots = 0, nodes = 0;
  std::map<std::string, std::shared_ptr<spark_dsg::Mesh>> meshes;
  std::string map_path;                                   // the map named last (the lines come visit by visit)
  std::unique_ptr<khronos::SpatioTemporalMap> map;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    const auto snap = nlohmann::json::parse(line);
    std::shared_ptr<spark_dsg::DynamicSceneGraph> dsg;
    if (snap.contains("mesh") && snap.at("mesh").is_object()) {
      const auto path = snap.at("mesh").at("map").get<std::string>();
      if (path != map_path) {
        map.reset();
        map = khronos::SpatioTemporalMap::load(path);
        if (!map) { std::cerr << "cannot read map " << path << "\n"; return 5; }
        map_path = path;
      }
      dsg = map->rawDsg(snap.at("mesh").at("index").get<size_t>())->clone();
      removeObjects(*dsg);
    } else {
      dsg = base->clone();
    }
    if (snap.contains("mesh") && snap.at("mesh").is_string()) {
      const auto path = snap.at("mesh").get<std::string>();
      if (!meshes.count(path)) {
        std::ifstream f(path, std::ios::binary);
        uint64_t count = 0;
        f.read(reinterpret_cast<char*>(&count), sizeof(count));
        auto mesh = std::make_shared<spark_dsg::Mesh>(true, true, true, true);  // the official finalizeMesh writes colors and timestamps too
        mesh->resizeVertices(count);
        for (uint64_t i = 0; i < count; ++i) {
          float xyz[3]; uint64_t first = 0;
          f.read(reinterpret_cast<char*>(xyz), sizeof(xyz)); f.read(reinterpret_cast<char*>(&first), sizeof(first));
          mesh->points[i] = Eigen::Vector3f(xyz[0], xyz[1], xyz[2]); mesh->first_seen_stamps[i] = first;
        }
        if (!f) { std::cerr << "cannot read mesh " << path << "\n"; return 5; }
        meshes[path] = mesh;
      }
      dsg->setMesh(std::make_shared<spark_dsg::Mesh>(*meshes[path]));
    }
    size_t n = 0;
    for (const auto& o : snap.at("objects")) {
      const auto& surface = o.at("surface");
      Eigen::Vector3f center, size;
      if (o.contains("bbox_center")) {
        center = vec(o.at("bbox_center")); size = vec(o.at("bbox_dimensions"));
      } else if (o.contains("bbox_min")) {
        const Eigen::Vector3f lo = vec(o.at("bbox_min")), hi = vec(o.at("bbox_max"));
        center = (lo + hi) / 2; size = hi - lo;
      } else if (!surface.empty()) {
        Eigen::Vector3f lo = Eigen::Vector3f::Constant(1e30f), hi = Eigen::Vector3f::Constant(-1e30f);
        for (const auto& p : surface) { lo = lo.cwiseMin(vec(p)); hi = hi.cwiseMax(vec(p)); }
        center = (lo + hi) / 2; size = hi - lo;
      } else {
        continue;
      }
      auto attrs = std::make_unique<khronos::KhronosObjectAttributes>();
      attrs->bounding_box = spark_dsg::BoundingBox(size, center);
      attrs->position = center.cast<double>();
      attrs->semantic_label = o.at("semantic").get<uint32_t>();
      attrs->first_observed_ns = o.at("first_observed_ns").get<std::vector<uint64_t>>();
      attrs->last_observed_ns = o.at("last_observed_ns").get<std::vector<uint64_t>>();
      if (o.contains("details")) {
        for (const auto& [key, values] : o.at("details").items()) attrs->details[key] = values.get<std::vector<size_t>>();
      }
      attrs->mesh.resizeVertices(surface.size());
      for (size_t i = 0; i < surface.size(); ++i) attrs->mesh.points[i] = vec(surface[i]) - center;
      const auto id = o.at("id").get<std::string>();
      const bool numeric = !id.empty() && std::all_of(id.begin(), id.end(), ::isdigit);
      dsg->emplaceNode(objects->layer, numeric ? spark_dsg::NodeId(std::stoull(id)) : spark_dsg::NodeId(spark_dsg::NodeSymbol('o', n)),
                       std::move(attrs));
      ++n;
    }
    out.update(dsg, snap.at("stamp_ns").get<uint64_t>());
    ++snapshots; nodes += n;
  }
  if (!out.save(argv[3])) return 4;
  std::cout << "TIMELINE_TO_4DMAP snapshots=" << snapshots << " object_nodes=" << nodes << "\n";
  return 0;
}
