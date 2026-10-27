// Evaluation only: one 4D map of a chained run's visits for Khronos's official evaluator, from the snapshots the
// run already holds (no mapping is rerun, no map is changed). The visits' maps are read in order and their
// snapshots written, in time order, into one map, completed from the earlier snapshots:
//   mesh: an object node whose mesh is empty (an object the run took away keeps its node, without a mesh) gets the
//     last non-empty mesh, with its bounding box, of the same physical object (details "instance_id", else the
//     node id) in an earlier snapshot, so the evaluator can associate it by surface;
//   first_observed_ns: an object first present in a later visit than the first gets, while its first observed
//     time is 0, the stamp of the first snapshot it is present in, so it counts as appeared (objects of the first
//     visit keep 0, present from the start as the ground truth's);
//   moves: the run keeps a moved object's node and counts its relocations (details "session_mobility_changes"),
//     with the first observation at the new place (details "observation_first_stamp_ns"). When the count rises,
//     the object's last mesh is held; at the node's next mesh, if that mesh no longer represents the held one
//     (under 50 % of its points within 0.10 m, the criterion of the ground truth's states) the object moved: the
//     held mesh is kept as a closed node at the old place (last observed = the first observation at the new
//     place), present in every later snapshot, and the moved node is first observed then (it appeared there);
//     otherwise it was re-observed in place. The old node and old mesh stored apart, as the ground truth has a
//     state per place.
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
double detail(const khronos::KhronosObjectAttributes& attrs, const std::string& name, double fallback) {
  const auto it = attrs.details.find(name);
  return it != attrs.details.end() && !it->second.empty() ? it->second.front() : fallback;
}
// World points of an object's mesh, at most 2000.
std::vector<Eigen::Vector3f> worldPoints(const khronos::KhronosObjectAttributes& attrs) {
  std::vector<Eigen::Vector3f> out;
  const size_t n = attrs.mesh.points.size(), stride = std::max<size_t>(1, (n + 1999) / 2000);
  for (size_t i = 0; i < n; i += stride) out.push_back(attrs.bounding_box.pointToWorldFrame(attrs.mesh.points[i]));
  return out;
}
// Share of a's points within 0.10 m of b's.
double represented(const std::vector<Eigen::Vector3f>& a, const std::vector<Eigen::Vector3f>& b) {
  if (a.empty() || b.empty()) return 0.0;
  size_t near = 0;
  for (const auto& p : a) {
    for (const auto& q : b) {
      if ((p - q).squaredNorm() <= 0.01f) { ++near; break; }
    }
  }
  return static_cast<double>(near) / a.size();
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
  std::map<size_t, double> moves;                                      // last seen move count
  std::map<size_t, khronos::KhronosObjectAttributes> last_attrs;       // last attributes with a mesh
  std::vector<khronos::KhronosObjectAttributes> places;                // closed old places
  std::map<size_t, khronos::TimeStamp> moved_at;                       // the latest move of an object
  std::map<size_t, khronos::KhronosObjectAttributes> held;             // last mesh before a relocation count rose
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
          const double count = detail(*attrs, "session_mobility_changes", 0);
          if (const auto seen = moves.find(key); seen != moves.end() && count > seen->second) {
            if (const auto old = last_attrs.find(key); old != last_attrs.end() && !held.count(key)) {
              held[key] = old->second;
            }
          }
          moves[key] = count;
          if (const auto old = held.find(key); old != held.end() && attrs->mesh.numVertices() > 0) {
            if (represented(worldPoints(old->second), worldPoints(*attrs)) < 0.5) {
              auto place = old->second;
              const auto at = static_cast<khronos::TimeStamp>(detail(*attrs, "observation_first_stamp_ns", stamp));
              place.last_observed_ns = {at};
              places.push_back(std::move(place));
              moved_at[key] = at;
            }
            held.erase(old);
          }
          if (const auto at = moved_at.find(key); at != moved_at.end() && !attrs->first_observed_ns.empty() &&
              attrs->first_observed_ns.front() < at->second) {
            attrs->first_observed_ns.front() = at->second;      // appeared at its new place
          }
          if (attrs->mesh.numVertices() > 0) {
            last_mesh[key] = {attrs->mesh, attrs->bounding_box};
            last_attrs[key] = *attrs;
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
        for (size_t n = 0; n < places.size(); ++n) {
          dsg->emplaceNode(objects->layer, spark_dsg::NodeSymbol('h', n), places[n].clone());
        }
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
            << " appeared_set=" << set_appeared << " moves=" << places.size() << " objects=" << last_mesh.size() << "\n";
  return 0;
}
