// Final-map surface adapter (background plus the objects present at the final stamp), with separately named native diagnostic.
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <khronos/utils/khronos_attribute_utils.h>
#include <nlohmann/json.hpp>

using Json = nlohmann::json;
using Point = std::array<float, 3>;
using Face = std::array<uint32_t, 3>;

int main(int argc, char** argv) {
  try {
    if (argc < 3 || argc > 5) throw std::runtime_error("Usage: geometry_export MAP OUTPUT.ply [viewer|native] [QUERY_NS]");
    const std::string policy = argc >= 4 ? argv[3] : "viewer";
    if (policy != "viewer" && policy != "native") throw std::runtime_error("Unknown view policy");
    std::cerr << "Loading " << argv[1] << '\n';
    const auto map = khronos::SpatioTemporalMap::load(argv[1]);
    if (!map || !map->numTimeSteps()) throw std::runtime_error("No snapshots");
    const size_t index = map->numTimeSteps()-1;
    const auto final_stamp = map->stamps()[index];
    // QUERY_NS: the final map asked about an earlier time; objects present then, background seen by then.
    const auto stamp = argc == 5 ? static_cast<decltype(final_stamp)>(std::stoull(argv[4])) : final_stamp;
    const auto graph = policy == "viewer" ? map->getDsgPtr(final_stamp) : map->rawDsg(index);
    std::vector<Point> points;
    std::vector<Face> faces;
    Json metadata = {{"map", argv[1]}, {"output", argv[2]}, {"stamp_ns", stamp}, {"final_stamp_ns", final_stamp},
      {"snapshot_index", index}, {"snapshot_count", map->numTimeSteps()},
      {"policy_name", policy},
      {"policy", policy == "viewer" ? "The final map: the final viewer slice via getDsgPtr(stamp), background plus the objects present at the final stamp (isPresent). Object private meshes transformed with bounding_box.pointToWorldFrame. No GT clipping or semantic filtering." : "Native diagnostic only: rawDsg final background and native isPresent objects."},
      {"frame", "aligned session-A world"}, {"layers", Json::array()},
      {"native_nonpresent_objects", Json::array()}};
    auto append = [&](const spark_dsg::Mesh& mesh, const std::string& layer, uint64_t owner,
                      const khronos::KhronosObjectAttributes* attrs) {
      const size_t first = points.size(), face_first = faces.size();
      const bool filter = !attrs && !mesh.first_seen_stamps.empty() &&
                          mesh.first_seen_stamps.size() == mesh.numVertices();
      std::vector<int64_t> index_map(mesh.numVertices(), -1);
      for (size_t i=0; i<mesh.numVertices(); ++i) {
        if (filter && mesh.first_seen_stamps[i] > stamp) continue;
        const Eigen::Vector3f p = attrs ? attrs->bounding_box.pointToWorldFrame(mesh.pos(i)) : mesh.pos(i);
        if (!p.allFinite()) continue;
        index_map[i] = points.size(); points.push_back({p.x(), p.y(), p.z()});
      }
      size_t invalid=0;
      for (size_t i=0; i<mesh.numFaces(); ++i) {
        const auto& f=mesh.face(i);
        if (f[0]>=index_map.size() || f[1]>=index_map.size() || f[2]>=index_map.size() ||
            index_map[f[0]]<0 || index_map[f[1]]<0 || index_map[f[2]]<0) { ++invalid; continue; }
        faces.push_back({uint32_t(index_map[f[0]]), uint32_t(index_map[f[1]]), uint32_t(index_map[f[2]])});
      }
      metadata["layers"].push_back({{"layer",layer},{"owner",std::to_string(owner)},
        {"semantic",attrs ? int(attrs->semantic_label) : -1},{"vertices",points.size()-first},
        {"faces",faces.size()-face_first},{"vertex_start",first},{"face_start",face_first},
        {"invalid_or_future_faces",invalid},{"future_vertex_filter",filter},
        {"native_present",attrs ? khronos::isPresent(*attrs,stamp) : true}});
    };
    if (graph->hasMesh() && graph->mesh()) append(*graph->mesh(), "background", 0, nullptr);
    if (graph->hasLayer(khronos::DsgLayers::OBJECTS)) {
      for (const auto& [id,node] : graph->getLayer(khronos::DsgLayers::OBJECTS).nodes()) {
        const auto* attrs=node->tryAttributes<khronos::KhronosObjectAttributes>();
        if (!attrs) continue;
        if (!khronos::isPresent(*attrs, stamp)) {
          metadata["native_nonpresent_objects"].push_back({{"owner",std::to_string(id)},
            {"semantic",attrs->semantic_label},{"vertices",attrs->mesh.numVertices()},
            {"faces",attrs->mesh.numFaces()}});
          continue;
        }
        append(attrs->mesh,"object",id,attrs);
      }
    }
    std::filesystem::create_directories(std::filesystem::path(argv[2]).parent_path());
    std::ofstream output(argv[2],std::ios::binary);
    output << "ply\nformat binary_little_endian 1.0\nelement vertex " << points.size()
      << "\nproperty float x\nproperty float y\nproperty float z\nelement face " << faces.size()
      << "\nproperty list uchar uint vertex_indices\nend_header\n";
    for (const auto& p:points) output.write(reinterpret_cast<const char*>(p.data()),12);
    const uint8_t count=3;
    for (const auto& f:faces) {
      output.write(reinterpret_cast<const char*>(&count),1);
      output.write(reinterpret_cast<const char*>(f.data()),12);
    }
    output.close();
    if (!output) throw std::runtime_error("Write failed");
    metadata["vertices"]=points.size(); metadata["faces"]=faces.size();
    std::ofstream(std::string(argv[2])+".json") << metadata.dump(2);
    std::cerr << "Exported " << points.size() << " vertices " << faces.size() << " faces\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
