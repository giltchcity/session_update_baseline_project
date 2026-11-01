// Evaluation only, as chain_fill: our 4D map with the motion the run itself recorded (user 11-01: an object is dynamic
// while it moves; once it stops the map builds it as usual). TRAJECTORY.csv (instance, stamp_ns, x, y, z: from the run's
// own static-surface boundary and tracker, analysis/ours_motion_from_run.py) becomes the trajectory of the object node
// of that physical instance (details "instance_id"); each snapshot carries the points up to its own stamp. Nothing else
// in the map changes (meshes, presence, the other objects).
// usage: motion_fill OUT.4dmap IN.4dmap TRAJECTORY.csv
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <khronos/utils/khronos_attribute_utils.h>

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: motion_fill OUT.4dmap IN.4dmap TRAJECTORY.csv\n";
    return 2;
  }
  std::map<size_t, std::vector<std::pair<uint64_t, Eigen::Vector3f>>> motion;   // instance -> (stamp, position)
  std::ifstream csv(argv[3]);
  std::string line;
  std::getline(csv, line);
  while (std::getline(csv, line)) {
    std::stringstream in(line);
    std::string f[5];
    for (auto& x : f) std::getline(in, x, ',');
    motion[std::stoul(f[0])].push_back({std::stoull(f[1]), Eigen::Vector3f(std::stof(f[2]), std::stof(f[3]), std::stof(f[4]))});
  }
  for (auto& [inst, points] : motion) std::sort(points.begin(), points.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  const auto map = khronos::SpatioTemporalMap::load(argv[2]);
  if (!map || !map->numTimeSteps()) {
    std::cerr << "cannot read " << argv[2] << "\n";
    return 3;
  }
  khronos::SpatioTemporalMap::Config config;
  config.finalize_incrementally = false;  // store the snapshots exactly as read
  khronos::SpatioTemporalMap out(config);
  size_t written = 0, nodes = 0;
  for (size_t ix = 0; ix < map->numTimeSteps(); ++ix) {
    auto dsg = map->rawDsg(ix)->clone();
    const auto stamp = map->stamps()[ix];
    if (dsg->hasLayer(khronos::DsgLayers::OBJECTS)) {
      std::map<size_t, khronos::KhronosObjectAttributes*> owner;   // one node per instance: the present one, else the first
      for (const auto& [id, node] : dsg->getLayer(khronos::DsgLayers::OBJECTS).nodes()) {
        auto* attrs = node->tryAttributes<khronos::KhronosObjectAttributes>();
        if (!attrs) continue;
        const auto it = attrs->details.find("instance_id");
        if (it == attrs->details.end() || it->second.empty()) continue;
        const size_t inst = static_cast<size_t>(it->second.front());
        if (!motion.count(inst)) continue;
        if (!owner.count(inst) || (khronos::isPresent(*attrs, stamp) && !khronos::isPresent(*owner[inst], stamp))) owner[inst] = attrs;
      }
      for (auto& [inst, attrs] : owner) {
        for (const auto& [t, p] : motion[inst]) {
          if (t > stamp) break;
          attrs->trajectory_timestamps.push_back(t);
          attrs->trajectory_positions.push_back(p);
          ++written;
        }
        ++nodes;
      }
    }
    out.update(dsg, stamp);
  }
  if (!out.save(argv[1])) {
    std::cerr << "cannot write " << argv[1] << "\n";
    return 4;
  }
  std::cout << "MOTION_FILL snapshots=" << map->numTimeSteps() << " instances=" << motion.size() << " node_snapshots=" << nodes
            << " trajectory_points_written=" << written << "\n";
  return 0;
}
