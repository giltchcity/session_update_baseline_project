#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <khronos/common/common_types.h>
#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <khronos/utils/khronos_attribute_utils.h>
#include <nlohmann/json.hpp>
#include <spark_dsg/serialization/graph_binary_serialization.h>

#include "session_update_baseline/runtime/session_state_fingerprint.h"

namespace {

using Json = nlohmann::json;

std::optional<std::size_t> physicalInstanceId(
    const khronos::KhronosObjectAttributes& attrs) {
  const auto iter = attrs.details.find("instance_id");
  if (iter == attrs.details.end() || iter->second.size() != 1 ||
      iter->second.front() == 0) {
    return std::nullopt;
  }
  return iter->second.front();
}

Json summarizeDsg(const khronos::DynamicSceneGraph::Ptr& dsg) {
  Json result = {
      {"global_mesh_vertices", 0},
      {"global_mesh_faces", 0},
      {"current_object_nodes", 0},
      {"current_private_mesh_vertices", 0},
      {"current_private_mesh_faces", 0},
      {"current_trajectory_objects", 0},
      {"current_physical_ids", Json::array()},
      {"current_physical_id_node_counts", Json::object()},
      {"current_physical_id_semantic_labels", Json::object()},
      {"current_physical_id_private_mesh_vertices", Json::object()},
      {"duplicate_current_physical_ids", Json::array()},
      {"current_semantic_labels", Json::array()},
  };
  if (!dsg) {
    return result;
  }
  std::vector<std::uint8_t> serialized;
  spark_dsg::io::binary::writeGraph(*dsg, serialized, true);
  // Keep the raw serializer fingerprint as a diagnostic only. DSG containers
  // may serialize equivalent nodes in a different order after load/reseed, so
  // production equivalence must use the canonical scene fingerprint below.
  std::uint64_t fingerprint = 1469598103934665603ULL;
  for (const auto byte : serialized) {
    fingerprint ^= byte;
    fingerprint *= 1099511628211ULL;
  }
  result["dsg_binary_bytes"] = serialized.size();
  result["dsg_fingerprint_fnv1a64"] = fingerprint;
  const auto canonical =
      session_update::runtime::canonicalCurrentSceneFingerprint(*dsg);
  result["canonical_current_scene_schema"] =
      "session_update_current_scene/v1";
  result["canonical_current_scene_bytes"] = canonical.encoded_bytes;
  result["canonical_current_scene_objects"] = canonical.object_records;
  result["canonical_current_scene_fingerprint_fnv1a64"] = canonical.fnv1a64;
  if (dsg->hasMesh() && dsg->mesh()) {
    result["global_mesh_vertices"] = dsg->mesh()->numVertices();
    result["global_mesh_faces"] = dsg->mesh()->numFaces();
    result["global_mesh_label_counts"] = Json::object();
    std::map<std::uint32_t, std::size_t> label_counts;
    for (const auto label : dsg->mesh()->labels) {
      ++label_counts[label];
    }
    for (const auto& [label, count] : label_counts) {
      result["global_mesh_label_counts"][std::to_string(label)] = count;
    }
  }
  if (!dsg->hasLayer(khronos::DsgLayers::OBJECTS)) {
    return result;
  }

  std::map<std::size_t, std::size_t> physical_counts;
  std::map<std::size_t, std::set<std::size_t>> physical_semantics;
  std::map<std::size_t, std::size_t> physical_mesh_vertices;
  std::set<std::size_t> semantic_labels;
  const auto& objects = dsg->getLayer(khronos::DsgLayers::OBJECTS);
  result["current_object_nodes"] = objects.numNodes();
  for (const auto& [unused, node] : objects.nodes()) {
    (void)unused;
    const auto* attrs = node->tryAttributes<khronos::KhronosObjectAttributes>();
    if (!attrs) {
      continue;
    }
    semantic_labels.insert(attrs->semantic_label);
    result["current_private_mesh_vertices"] =
        result["current_private_mesh_vertices"].get<std::size_t>() +
        attrs->mesh.numVertices();
    result["current_private_mesh_faces"] =
        result["current_private_mesh_faces"].get<std::size_t>() +
        attrs->mesh.numFaces();
    if (khronos::hasTrajectoryHistory(*attrs)) {
      result["current_trajectory_objects"] =
          result["current_trajectory_objects"].get<std::size_t>() + 1;
    }
    const auto instance_id = physicalInstanceId(*attrs);
    if (instance_id) {
      ++physical_counts[*instance_id];
      physical_semantics[*instance_id].insert(attrs->semantic_label);
      physical_mesh_vertices[*instance_id] += attrs->mesh.numVertices();
    }
  }

  for (const auto& [id, count] : physical_counts) {
    result["current_physical_ids"].push_back(id);
    result["current_physical_id_node_counts"][std::to_string(id)] = count;
    result["current_physical_id_semantic_labels"][std::to_string(id)] =
        Json::array();
    for (const auto semantic : physical_semantics[id]) {
      result["current_physical_id_semantic_labels"][std::to_string(id)].push_back(
          semantic);
    }
    result["current_physical_id_private_mesh_vertices"][std::to_string(id)] =
        physical_mesh_vertices[id];
    if (count > 1) {
      result["duplicate_current_physical_ids"].push_back(id);
    }
  }
  for (const auto label : semantic_labels) {
    result["current_semantic_labels"].push_back(label);
  }
  return result;
}

}  // namespace

namespace {

Json dumpGeometry(const khronos::DynamicSceneGraph::Ptr& dsg) {
  Json objects = Json::array();
  if (!dsg) {
    return objects;
  }
  if (dsg->hasLayer(khronos::DsgLayers::OBJECTS)) {
    const auto& layer = dsg->getLayer(khronos::DsgLayers::OBJECTS);
    for (const auto& [node_id, node] : layer.nodes()) {
      (void)node_id;
      const auto* attrs = node->tryAttributes<khronos::KhronosObjectAttributes>();
      if (!attrs) {
        continue;
      }
      Json record = {
          {"node_id", node_id},
          {"semantic_label", attrs->semantic_label},
          {"position", {attrs->position.x(), attrs->position.y(), attrs->position.z()}},
          {"bbox_center",
           {attrs->bounding_box.world_P_center.x(),
            attrs->bounding_box.world_P_center.y(),
            attrs->bounding_box.world_P_center.z()}},
          {"bbox_dims",
           {attrs->bounding_box.dimensions.x(),
            attrs->bounding_box.dimensions.y(),
            attrs->bounding_box.dimensions.z()}},
          {"last_update_time_ns", attrs->last_update_time_ns},
          {"is_active", attrs->is_active},
          {"mesh_vertices", attrs->mesh.numVertices()},
          {"name", attrs->name},
      };
      const auto physical = attrs->details.find("instance_id");
      if (physical != attrs->details.end() && !physical->second.empty()) {
        record["physical_id"] = physical->second.front();
      } else {
        record["physical_id"] = 0;
      }
      // World-frame vertices of the current private mesh. Needed to decide whether one
      // physical ID's CURRENT geometry is a single spatial cluster at its newest position
      // or still carries a second cluster left behind at an older position.
      Json mesh_world = Json::array();
      for (const auto& local_point : attrs->mesh.points) {
        const auto world = attrs->bounding_box.pointToWorldFrame(local_point);
        mesh_world.push_back({world.x(), world.y(), world.z()});
      }
      record["mesh_world_points"] = std::move(mesh_world);
      Json first = Json::array();
      for (const auto stamp : attrs->first_observed_ns) {
        first.push_back(stamp);
      }
      Json last = Json::array();
      for (const auto stamp : attrs->last_observed_ns) {
        last.push_back(stamp);
      }
      record["first_observed_ns"] = first;
      record["last_observed_ns"] = last;
      objects.push_back(std::move(record));
    }
  }
  return objects;
}

}  // namespace

int main(int argc, char** argv) {
  const bool dump_geometry =
      argc >= 2 && std::string(argv[1]) == "--dump-geometry";
  if ((!dump_geometry && argc != 2) || (dump_geometry && argc != 3)) {
    std::cerr << "usage: inspect_session_state [--dump-geometry] MAP.4dmap\n";
    return 2;
  }
  const char* map_path = dump_geometry ? argv[2] : argv[1];
  try {
    auto map = khronos::SpatioTemporalMap::load(map_path);
    if (!map || map->numTimeSteps() == 0) {
      std::cerr << "state is unreadable or empty\n";
      return 3;
    }
    const auto first = map->stamps().front();
    const auto latest = map->stamps().back();
    Json result = {
        {"schema", "session_update_state_summary/v1"},
        {"map", argv[1]},
        {"time_steps", map->numTimeSteps()},
        {"first_stamp_ns", first},
        {"latest_stamp_ns", latest},
        {"strictly_increasing_stamps", true},
    };
    for (std::size_t i = 1; i < map->stamps().size(); ++i) {
      if (map->stamps()[i] <= map->stamps()[i - 1]) {
        result["strictly_increasing_stamps"] = false;
      }
    }
    result["initial"] = summarizeDsg(map->getDsgPtr(first));
    result["current"] = summarizeDsg(map->getDsgPtr(latest));
    if (dump_geometry) {
      std::cout << Json{{"objects", dumpGeometry(map->getDsgPtr(latest))}}.dump(2)
                << "\n";
    } else {
      std::cout << result.dump(2) << "\n";
    }
    return result["strictly_increasing_stamps"].get<bool>() ? 0 : 4;
  } catch (const std::exception& e) {
    std::cerr << "state inspection failed: " << e.what() << "\n";
    return 5;
  }
}
