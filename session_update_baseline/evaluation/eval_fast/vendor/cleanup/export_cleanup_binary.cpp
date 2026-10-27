#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>
#include <Eigen/Core>

#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <khronos/utils/khronos_attribute_utils.h>
#include <hydra/utils/nearest_neighbor_utilities.h>

using J = nlohmann::json;

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr std::uint32_t kInvalid = std::numeric_limits<std::uint32_t>::max();

struct Owner {
  std::string id;
  std::uint32_t semantic = 0;
};

struct VoxelKey {
  std::int64_t x, y, z;
  bool operator==(const VoxelKey& o) const { return x == o.x && y == o.y && z == o.z; }
};

struct VoxelHash {
  std::size_t operator()(const VoxelKey& k) const {
    std::size_t h = 1469598103934665603ULL;
    auto mix = [&](std::int64_t v) {
      h ^= static_cast<std::size_t>(v);
      h *= 1099511628211ULL;
    };
    mix(k.x); mix(k.y); mix(k.z);
    return h;
  }
};

VoxelKey keyFor(const Eigen::Vector3f& p, float voxel) {
  return {static_cast<std::int64_t>(std::floor(p.x() / voxel)),
          static_cast<std::int64_t>(std::floor(p.y() / voxel)),
          static_cast<std::int64_t>(std::floor(p.z() / voxel))};
}

void writeU32(std::ofstream& out, std::uint32_t v) {
  out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}
void writeU64(std::ofstream& out, std::uint64_t v) {
  out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}
void writeF32(std::ofstream& out, float v) {
  out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}
void writeString(std::ofstream& out, const std::string& s) {
  writeU32(out, static_cast<std::uint32_t>(s.size()));
  out.write(s.data(), static_cast<std::streamsize>(s.size()));
}

std::vector<Eigen::Vector3f> downsample(const std::vector<Eigen::Vector3f>& in,
                                        float voxel) {
  if (voxel <= 0.0f || in.empty()) return in;
  std::unordered_map<VoxelKey, std::size_t, VoxelHash> seen;
  seen.reserve(in.size() / 2 + 1);
  std::vector<Eigen::Vector3f> out;
  out.reserve(in.size());
  for (const auto& p : in) {
    const auto key = keyFor(p, voxel);
    if (seen.emplace(key, out.size()).second) out.push_back(p);
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: export_cleanup_binary MAP PROBES_JSON OUT_BIN\n";
    return 2;
  }
  const std::string map_path = argv[1];
  const std::string probes_path = argv[2];
  const std::string out_path = argv[3];

  auto map = khronos::SpatioTemporalMap::load(map_path);
  if (!map || !map->numTimeSteps()) {
    std::cerr << "failed to load map: " << map_path << "\n";
    return 3;
  }

  std::ifstream probes_in(probes_path);
  if (!probes_in) {
    std::cerr << "failed to open probes: " << probes_path << "\n";
    return 4;
  }
  J probes;
  probes_in >> probes;

  std::filesystem::create_directories(std::filesystem::path(out_path).parent_path());
  std::ofstream out(out_path, std::ios::binary);
  if (!out) {
    std::cerr << "failed to open output: " << out_path << "\n";
    return 5;
  }
  const char magic[8] = {'A', 'B', 'C', 'C', 'L', 'N', '1', '\0'};
  out.write(magic, 8);
  const std::uint32_t n_steps = static_cast<std::uint32_t>(map->numTimeSteps());
  writeU32(out, n_steps);

  for (std::size_t ix = 0; ix < map->numTimeSteps(); ++ix) {
    auto dsg = map->rawDsg(ix);
    const auto stamp = map->stamps()[ix];

    std::vector<Eigen::Vector3f> global_pts;
    std::vector<std::uint32_t> global_labels;
    if (dsg && dsg->hasMesh() && dsg->mesh()) {
      const auto& mesh = *dsg->mesh();
      global_pts.reserve(mesh.numVertices());
      global_labels.reserve(mesh.numVertices());
      for (std::size_t i = 0; i < mesh.numVertices(); ++i) {
        const auto p = mesh.pos(i);
        global_pts.emplace_back(static_cast<float>(p.x()),
                                static_cast<float>(p.y()),
                                static_cast<float>(p.z()));
        global_labels.push_back(i < mesh.labels.size() ? mesh.labels[i] : kInvalid);
      }
    }
    global_pts = downsample(global_pts, 0.03f);

    std::vector<Eigen::Vector3f> object_pts;
    std::vector<std::uint32_t> object_owner;
    std::vector<Owner> owners;
    if (dsg && dsg->hasLayer(khronos::DsgLayers::OBJECTS)) {
      const auto& layer = dsg->getLayer(khronos::DsgLayers::OBJECTS);
      for (const auto& [node_id, node] : layer.nodes()) {
        auto attrs = node->tryAttributes<khronos::KhronosObjectAttributes>();
        if (!attrs || attrs->mesh.numVertices() == 0) continue;
        const std::uint32_t owner_idx = static_cast<std::uint32_t>(owners.size());
        owners.push_back({std::to_string(node_id), attrs->semantic_label});
        for (const auto& raw : attrs->mesh.points) {
          const auto world = attrs->bounding_box.pointToWorldFrame(raw);
          object_pts.emplace_back(static_cast<float>(world.x()),
                                  static_cast<float>(world.y()),
                                  static_cast<float>(world.z()));
          object_owner.push_back(owner_idx);
        }
      }
    }
    // Downsample object points but retain owner for the retained point.
    if (!object_pts.empty()) {
      std::unordered_map<VoxelKey, std::size_t, VoxelHash> seen;
      seen.reserve(object_pts.size() / 2 + 1);
      std::vector<Eigen::Vector3f> ds_pts;
      std::vector<std::uint32_t> ds_owner;
      ds_pts.reserve(object_pts.size());
      ds_owner.reserve(object_pts.size());
      for (std::size_t i = 0; i < object_pts.size(); ++i) {
        const auto key = keyFor(object_pts[i], 0.02f);
        if (seen.emplace(key, ds_pts.size()).second) {
          ds_pts.push_back(object_pts[i]);
          ds_owner.push_back(object_owner[i]);
        }
      }
      object_pts.swap(ds_pts);
      object_owner.swap(ds_owner);
    }

    std::unique_ptr<hydra::PointNeighborSearch> global_search;
    if (!global_pts.empty()) {
      global_search = std::make_unique<hydra::PointNeighborSearch>(global_pts);
    }
    std::unique_ptr<hydra::PointNeighborSearch> object_search;
    if (!object_pts.empty()) {
      object_search = std::make_unique<hydra::PointNeighborSearch>(object_pts);
    }

    writeU32(out, static_cast<std::uint32_t>(ix));
    writeU64(out, static_cast<std::uint64_t>(stamp));
    writeU32(out, static_cast<std::uint32_t>(probes.size()));

    for (const auto& group : probes) {
      const std::string event_id = group.at("event_id").get<std::string>();
      const auto& points = group.at("points");
      const std::uint32_t n = static_cast<std::uint32_t>(points.size());
      writeString(out, event_id);
      writeU32(out, n);
      for (const auto& point : points) {
        const Eigen::Vector3f q(static_cast<float>(point[0].get<double>()),
                                static_cast<float>(point[1].get<double>()),
                                static_cast<float>(point[2].get<double>()));
        float gd = kInf;
        std::uint32_t gidx = kInvalid;
        if (global_search) {
          float dsq = kInf;
          std::size_t idx = 0;
          if (global_search->search(q, dsq, idx)) {
            gd = std::sqrt(dsq);
            gidx = static_cast<std::uint32_t>(idx);
          }
        }
        float od = kInf;
        std::uint32_t oidx = kInvalid;
        if (object_search) {
          float dsq = kInf;
          std::size_t idx = 0;
          if (object_search->search(q, dsq, idx)) {
            od = std::sqrt(dsq);
            oidx = static_cast<std::uint32_t>(object_owner[idx]);
          }
        }
        const std::uint32_t glabel = (gidx < global_labels.size()) ? global_labels[gidx] : kInvalid;
        writeF32(out, gd);
        writeU32(out, glabel);
        writeF32(out, od);
        writeU32(out, oidx);
      }
    }

    writeU32(out, static_cast<std::uint32_t>(owners.size()));
    for (const auto& owner : owners) {
      writeString(out, owner.id);
      writeU32(out, owner.semantic);
    }

    std::cout << "snapshot " << ix + 1 << "/" << map->numTimeSteps()
              << " global=" << global_pts.size()
              << " object=" << object_pts.size()
              << " owners=" << owners.size() << std::endl;
  }
  return 0;
}
