// Evaluation only: one 4D map of a chained run's visits for Khronos's official evaluator, from the snapshots the
// run already holds (no mapping is rerun, no map is changed). The visits' maps are read in order and their
// snapshots written, in time order, into one map, completed from the earlier snapshots:
//   mesh: an object node whose mesh is empty (an object the run took away keeps its node, without a mesh) gets the
//     last non-empty mesh, with its bounding box, of the same physical object (details "instance_id", else the
//     node id) in an earlier snapshot, so the evaluator can associate it by surface;
//   first_observed_ns: an object first present in a later visit than the first gets, while its first observed
//     time is 0, the stamp of the first snapshot it is present in, so it counts as appeared (objects of the first
//     visit keep 0, present from the start as the ground truth's);
//   moves: none declared. The run keeps a moved object's node and counts its state closures (details
//     "session_mobility_changes"), which also counts a removal and a state reopened in place, so a count is not a
//     relocation; as the GaME and wavemap exports (10-30), only the method's appearances are written (until 10-30 a
//     rise of that count closed the earlier record as a move: chain_fill.cpp.before_appearances).
// usage: chain_fill OUT.4dmap VISIT_0_MAP [VISIT_1_MAP ...]
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <khronos/utils/khronos_attribute_utils.h>

namespace {
size_t objectKey(const khronos::KhronosObjectAttributes& attrs, spark_dsg::NodeId id) {
  const auto it = attrs.details.find("instance_id");
  return it != attrs.details.end() && !it->second.empty() ? static_cast<size_t>(it->second.front()) : id;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: chain_fill OUT.4dmap VISIT_0_MAP [VISIT_1_MAP ...]\n";
    return 2;
  }
  khronos::SpatioTemporalMap::Config config;
  config.finalize_incrementally = false;  // store the snapshots exactly as read
  khronos::SpatioTemporalMap out(config);
  std::map<size_t, std::pair<khronos::Mesh, khronos::BoundingBox>> last_mesh;
  std::map<size_t, std::pair<int, khronos::TimeStamp>> first_present;  // visit, stamp
  size_t filled_meshes = 0, set_appeared = 0, snapshots = 0;
  for (int visit = 0; visit + 2 < argc; ++visit) {
    const auto map = khronos::SpatioTemporalMap::load(argv[visit + 2]);
    if (!map || !map->numTimeSteps()) {
      std::cerr << "cannot read " << argv[visit + 2] << "\n";
      return 3;
    }
    for (size_t ix = 0; ix < map->numTimeSteps(); ++ix) {
      auto dsg = map->rawDsg(ix)->clone();
      const auto stamp = map->stamps()[ix];
      if (dsg->hasLayer(khronos::DsgLayers::OBJECTS)) {
        for (const auto& [id, node] : dsg->getLayer(khronos::DsgLayers::OBJECTS).nodes()) {
          auto* attrs = node->tryAttributes<khronos::KhronosObjectAttributes>();
          if (!attrs) continue;
          const size_t key = objectKey(*attrs, id);
          if (attrs->mesh.numVertices() > 0) {
            last_mesh[key] = {attrs->mesh, attrs->bounding_box};
          } else if (const auto it = last_mesh.find(key); it != last_mesh.end()) {
            attrs->mesh = it->second.first;
            attrs->bounding_box = it->second.second;
            ++filled_meshes;
          }
          if (khronos::isPresent(*attrs, stamp) && !first_present.count(key)) {
            first_present[key] = {visit, stamp};
          }
          const auto first = first_present.find(key);
          if (first != first_present.end() && first->second.first > 0 && !attrs->first_observed_ns.empty() &&
              attrs->first_observed_ns.front() == 0) {
            attrs->first_observed_ns.front() = first->second.second;
            ++set_appeared;
          }
        }
      }
      if (const auto objects = dsg->getLayerKey(khronos::DsgLayers::OBJECTS)) {
      }
      out.update(dsg, stamp);
      ++snapshots;
    }
  }
  if (!out.save(argv[1])) {
    std::cerr << "cannot write " << argv[1] << "\n";
    return 4;
  }
  std::cout << "CHAIN_FILL snapshots=" << snapshots << " filled_meshes=" << filled_meshes
            << " appeared_set=" << set_appeared << " objects=" << last_mesh.size() << "\n";
  return 0;
}
