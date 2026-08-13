#include <cstdint>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <khronos/spatio_temporal_map/spatio_temporal_map.h>

namespace fs = std::filesystem;

namespace {

struct Args {
  std::string map_file;
  std::string dsg_file;
  std::string output_ply;
  std::string output_view_json;
  std::string output_sequence_dir;
  std::string map_time = "latest";
  std::size_t vertex_stride = 1;
  double sequence_period_s = 0.0;
  std::uint64_t sequence_start_ns = 0;
  std::uint64_t sequence_end_ns = 0;
  bool include_faces = true;
  bool include_object_meshes = false;
};

struct AgentPose {
  std::uint64_t node_id = 0;
  std::int64_t timestamp_ns = 0;
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Eigen::Quaterniond world_R_body = Eigen::Quaterniond::Identity();
};

bool parseBool(const std::string& value) {
  if (value == "1" || value == "true" || value == "yes") {
    return true;
  }
  if (value == "0" || value == "false" || value == "no") {
    return false;
  }
  throw std::runtime_error("Expected boolean value, got: " + value);
}

Args parseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto value = [&]() {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + key);
      }
      return std::string(argv[++i]);
    };

    if (key == "--map_file") {
      args.map_file = value();
    } else if (key == "--dsg_file") {
      args.dsg_file = value();
    } else if (key == "--output_ply") {
      args.output_ply = value();
    } else if (key == "--output_view_json") {
      args.output_view_json = value();
    } else if (key == "--output_sequence_dir") {
      args.output_sequence_dir = value();
    } else if (key == "--map_time") {
      args.map_time = value();
    } else if (key == "--vertex_stride" || key == "--point_stride") {
      args.vertex_stride = std::stoull(value());
    } else if (key == "--sequence_period_s") {
      args.sequence_period_s = std::stod(value());
    } else if (key == "--sequence_start_ns") {
      args.sequence_start_ns = std::stoull(value());
    } else if (key == "--sequence_end_ns") {
      args.sequence_end_ns = std::stoull(value());
    } else if (key == "--include_faces") {
      args.include_faces = parseBool(value());
    } else if (key == "--include_object_meshes") {
      args.include_object_meshes = parseBool(value());
    } else {
      throw std::runtime_error("Unknown argument: " + key);
    }
  }

  if ((args.map_file.empty() && args.dsg_file.empty()) ||
      (args.output_ply.empty() && args.output_view_json.empty() &&
       args.output_sequence_dir.empty())) {
    throw std::runtime_error(
        "--map_file/--dsg_file and at least one output destination are required");
  }
  if (!args.dsg_file.empty() &&
      (!args.output_ply.empty() || args.output_view_json.empty() ||
       !args.output_sequence_dir.empty())) {
    throw std::runtime_error(
        "--dsg_file currently supports only --output_view_json");
  }
  if (args.vertex_stride == 0) {
    throw std::runtime_error("--vertex_stride must be >= 1");
  }
  return args;
}

khronos::TimeStamp selectMapTime(const khronos::SpatioTemporalMap& map,
                                 const std::string& selector) {
  if (selector == "latest") {
    return map.latest();
  }
  if (selector == "earliest") {
    return map.earliest();
  }

  const std::string index_prefix = "index:";
  if (selector.rfind(index_prefix, 0) == 0) {
    const auto index = static_cast<std::size_t>(std::stoul(selector.substr(index_prefix.size())));
    const auto& stamps = map.stamps();
    if (index >= stamps.size()) {
      throw std::runtime_error("--map_time index out of range");
    }
    return stamps[index];
  }

  throw std::runtime_error("Unsupported --map_time selector: " + selector);
}

std::size_t stridedVertexCount(const spark_dsg::Mesh& mesh, std::size_t stride) {
  if (mesh.numVertices() == 0) {
    return 0;
  }
  return (mesh.numVertices() + stride - 1) / stride;
}

bool validFace(const spark_dsg::Mesh& mesh, const spark_dsg::Mesh::Face& face) {
  return face[0] < mesh.numVertices() && face[1] < mesh.numVertices() &&
         face[2] < mesh.numVertices();
}

std::size_t validFaceCount(const spark_dsg::Mesh& mesh) {
  std::size_t count = 0;
  for (std::size_t i = 0; i < mesh.numFaces(); ++i) {
    if (validFace(mesh, mesh.face(i))) {
      ++count;
    }
  }
  return count;
}

template <typename T>
void writeBinary(std::ofstream& out, const T& value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

void writePly(const spark_dsg::Mesh& mesh,
              const fs::path& path,
              const std::string& source,
              std::size_t stride,
              bool include_faces) {
  fs::create_directories(path.parent_path());

  const bool write_faces = include_faces && stride == 1;
  const std::size_t vertex_count = stridedVertexCount(mesh, stride);
  const std::size_t face_count = write_faces ? validFaceCount(mesh) : 0;

  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("Failed to open output PLY: " + path.string());
  }

  out << "ply\n";
  out << "format binary_little_endian 1.0\n";
  out << "comment exported_from " << source << "\n";
  out << "comment map_mesh_vertices " << mesh.numVertices() << "\n";
  out << "comment map_mesh_faces " << mesh.numFaces() << "\n";
  out << "comment vertex_stride " << stride << "\n";
  out << "element vertex " << vertex_count << "\n";
  out << "property float x\n";
  out << "property float y\n";
  out << "property float z\n";
  out << "property uchar red\n";
  out << "property uchar green\n";
  out << "property uchar blue\n";
  if (write_faces) {
    out << "element face " << face_count << "\n";
    out << "property list uchar uint vertex_indices\n";
  }
  out << "end_header\n";

  for (std::size_t i = 0; i < mesh.numVertices(); i += stride) {
    const auto& p = mesh.pos(i);
    const float x = p.x();
    const float y = p.y();
    const float z = p.z();
    writeBinary(out, x);
    writeBinary(out, y);
    writeBinary(out, z);

    const auto color = mesh.has_colors && i < mesh.colors.size() ? mesh.colors[i]
                                                                 : spark_dsg::Color(180, 180, 180);
    writeBinary(out, color.r);
    writeBinary(out, color.g);
    writeBinary(out, color.b);
  }

  if (write_faces) {
    for (std::size_t i = 0; i < mesh.numFaces(); ++i) {
      const auto& face = mesh.face(i);
      if (!validFace(mesh, face)) {
        continue;
      }
      const std::uint8_t n = 3;
      const std::uint32_t a = static_cast<std::uint32_t>(face[0]);
      const std::uint32_t b = static_cast<std::uint32_t>(face[1]);
      const std::uint32_t c = static_cast<std::uint32_t>(face[2]);
      writeBinary(out, n);
      writeBinary(out, a);
      writeBinary(out, b);
      writeBinary(out, c);
    }
  }

  if (!out.good()) {
    throw std::runtime_error("Failed while writing PLY: " + path.string());
  }

  std::cout << "wrote " << path << " vertices=" << vertex_count
            << " faces=" << face_count << " has_colors=" << mesh.has_colors << "\n";
}

std::pair<std::size_t, std::size_t> appendObjectMeshes(
    const khronos::DynamicSceneGraph& dsg,
    spark_dsg::Mesh& display_mesh) {
  if (!dsg.hasLayer(khronos::DsgLayers::OBJECTS)) {
    return {};
  }

  std::size_t added_vertices = 0;
  std::size_t added_faces = 0;
  for (const auto& [unused_id, node] :
       dsg.getLayer(khronos::DsgLayers::OBJECTS).nodes()) {
    (void)unused_id;
    const auto* attrs = node->tryAttributes<khronos::KhronosObjectAttributes>();
    if (!attrs || attrs->mesh.numVertices() == 0 ||
        !attrs->trajectory_positions.empty()) {
      continue;
    }

    spark_dsg::Mesh object_world(display_mesh.has_colors,
                                 display_mesh.has_timestamps,
                                 display_mesh.has_labels,
                                 display_mesh.has_first_seen_stamps);
    object_world.resizeVertices(attrs->mesh.numVertices());
    for (std::size_t i = 0; i < attrs->mesh.numVertices(); ++i) {
      object_world.setPos(
          i, attrs->bounding_box.pointToWorldFrame(attrs->mesh.pos(i)));
      if (object_world.has_colors) {
        object_world.setColor(
            i,
            attrs->mesh.has_colors && i < attrs->mesh.colors.size()
                ? attrs->mesh.colors[i]
                : spark_dsg::Color(180, 180, 180));
      }
      if (object_world.has_timestamps) {
        object_world.setTimestamp(i, 0);
      }
      if (object_world.has_first_seen_stamps) {
        object_world.setFirstSeenTimestamp(i, 0);
      }
      if (object_world.has_labels) {
        object_world.setLabel(
            i, static_cast<spark_dsg::Mesh::Label>(attrs->semantic_label));
      }
    }

    std::vector<spark_dsg::Mesh::Face> valid_faces;
    valid_faces.reserve(attrs->mesh.numFaces());
    for (std::size_t i = 0; i < attrs->mesh.numFaces(); ++i) {
      const auto face = attrs->mesh.face(i);
      if (validFace(attrs->mesh, face)) {
        valid_faces.push_back(face);
      }
    }
    object_world.resizeFaces(valid_faces.size());
    for (std::size_t i = 0; i < valid_faces.size(); ++i) {
      object_world.face(i) = valid_faces[i];
    }

    if (!display_mesh.append(object_world)) {
      throw std::runtime_error("Failed to append private object mesh for display.");
    }
    added_vertices += object_world.numVertices();
    added_faces += object_world.numFaces();
  }
  return {added_vertices, added_faces};
}

std::vector<AgentPose> collectAgentPoses(const khronos::DynamicSceneGraph& dsg) {
  std::vector<AgentPose> poses;
  const auto agent_key = dsg.getLayerKey(khronos::DsgLayers::AGENTS);
  if (!agent_key) {
    return poses;
  }

  const auto& partitions = dsg.layer_partition(agent_key->layer);
  for (const auto& [partition_id, layer] : partitions) {
    if (!layer) {
      continue;
    }
    for (const auto& [node_id, node] : layer->nodes()) {
      if (!node) {
        continue;
      }

      AgentPose pose;
      pose.node_id = node_id;
      const auto& base_attrs = node->attributes<spark_dsg::NodeAttributes>();
      pose.position = base_attrs.position;
      if (const auto* agent_attrs = node->tryAttributes<spark_dsg::AgentNodeAttributes>()) {
        pose.timestamp_ns = agent_attrs->timestamp.count();
        pose.world_R_body = agent_attrs->world_R_body;
      }
      poses.push_back(pose);
    }
  }

  std::sort(poses.begin(), poses.end(), [](const auto& lhs, const auto& rhs) {
    if (lhs.timestamp_ns != rhs.timestamp_ns) {
      return lhs.timestamp_ns < rhs.timestamp_ns;
    }
    return lhs.node_id < rhs.node_id;
  });
  return poses;
}

Eigen::Vector3d startDirection(const std::vector<AgentPose>& poses) {
  if (poses.empty()) {
    return Eigen::Vector3d::UnitX();
  }

  const auto start = poses.front().position;
  for (std::size_t i = 1; i < poses.size(); ++i) {
    Eigen::Vector3d delta = poses[i].position - start;
    if (delta.norm() > 1.0e-3) {
      delta.normalize();
      return delta;
    }
  }

  Eigen::Vector3d forward = poses.front().world_R_body * Eigen::Vector3d::UnitX();
  if (forward.norm() <= 1.0e-6) {
    return Eigen::Vector3d::UnitX();
  }
  forward.normalize();
  return forward;
}

void writeVec3Json(std::ofstream& out, const Eigen::Vector3d& value) {
  out << "[" << value.x() << "," << value.y() << "," << value.z() << "]";
}

void writeVec3Json(std::ofstream& out, const Eigen::Vector3f& value) {
  out << "[" << value.x() << "," << value.y() << "," << value.z() << "]";
}

void writeTrajectoryStartJson(const khronos::DynamicSceneGraph& dsg, const fs::path& path) {
  fs::create_directories(path.parent_path());
  const auto poses = collectAgentPoses(dsg);
  if (poses.empty()) {
    throw std::runtime_error("Selected DSG has no agent poses");
  }

  const auto& start = poses.front();
  const Eigen::Vector3d forward = startDirection(poses);
  const Eigen::Vector3d next = start.position + forward;

  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Failed to open output view JSON: " + path.string());
  }

  out << "{\n";
  out << "  \"source\": \"dsg_agents_layer\",\n";
  out << "  \"agent_pose_count\": " << poses.size() << ",\n";
  out << "  \"start_node_id\": " << start.node_id << ",\n";
  out << "  \"start_timestamp_ns\": " << start.timestamp_ns << ",\n";
  out << "  \"start_position\": ";
  writeVec3Json(out, start.position);
  out << ",\n";
  out << "  \"start_direction\": ";
  writeVec3Json(out, forward);
  out << ",\n";
  out << "  \"look_at\": ";
  writeVec3Json(out, next);
  out << ",\n";
  out << "  \"dynamic_tracks\": [";
  bool first_track = true;
  if (dsg.hasLayer(khronos::DsgLayers::OBJECTS)) {
    for (const auto& [node_id, node] :
         dsg.getLayer(khronos::DsgLayers::OBJECTS).nodes()) {
      const auto* attrs = node->tryAttributes<khronos::KhronosObjectAttributes>();
      if (!attrs || attrs->trajectory_positions.empty()) {
        continue;
      }
      if (!first_track) {
        out << ",";
      }
      first_track = false;
      out << "\n    {\"object_id\":" << node_id
          << ",\"semantic_label\":" << static_cast<int>(attrs->semantic_label)
          << ",\"bbox_dimensions\":";
      writeVec3Json(out, attrs->bounding_box.dimensions);
      out << ",\"timestamps_ns\":[";
      for (std::size_t i = 0; i < attrs->trajectory_timestamps.size(); ++i) {
        if (i > 0) {
          out << ",";
        }
        out << attrs->trajectory_timestamps[i];
      }
      out << "],\"positions\":[";
      for (std::size_t i = 0; i < attrs->trajectory_positions.size(); ++i) {
        if (i > 0) {
          out << ",";
        }
        writeVec3Json(out, attrs->trajectory_positions[i]);
      }
      out << "],\"point_frames\":[";
      for (std::size_t frame_index = 0;
           frame_index < attrs->dynamic_object_points.size();
           ++frame_index) {
        if (frame_index > 0) {
          out << ",";
        }
        out << "[";
        const auto& frame_points = attrs->dynamic_object_points[frame_index];
        const std::size_t stride = std::max<std::size_t>(
            1, (frame_points.size() + 1499) / 1500);
        bool first_point = true;
        for (std::size_t point_index = 0;
             point_index < frame_points.size();
             point_index += stride) {
          if (!first_point) {
            out << ",";
          }
          first_point = false;
          writeVec3Json(out, frame_points[point_index]);
        }
        out << "]";
      }
      out << "]}";
    }
  }
  if (!first_track) {
    out << "\n  ";
  }
  out << "]\n";
  out << "}\n";

  if (!out.good()) {
    throw std::runtime_error("Failed while writing view JSON: " + path.string());
  }

  std::cout << "wrote " << path << " agent_pose_count=" << poses.size()
            << " start=[" << start.position.transpose() << "] direction=["
            << forward.transpose() << "]\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parseArgs(argc, argv);

    if (!args.dsg_file.empty()) {
      auto dsg = spark_dsg::DynamicSceneGraph::load(args.dsg_file);
      if (!dsg) {
        throw std::runtime_error("Failed to load DSG: " + args.dsg_file);
      }
      writeTrajectoryStartJson(*dsg, args.output_view_json);
      return 0;
    }

    auto map = khronos::SpatioTemporalMap::load(args.map_file);
    if (!map) {
      throw std::runtime_error("Failed to load map: " + args.map_file);
    }

    if (!args.output_sequence_dir.empty()) {
      const fs::path output_dir(args.output_sequence_dir);
      fs::create_directories(output_dir);
      std::ofstream manifest(output_dir / "map_sequence.csv");
      if (!manifest) {
        throw std::runtime_error("Failed to open sequence manifest");
      }
      manifest << "index,timestamp_ns,ply,overlay\n";
      std::vector<khronos::TimeStamp> query_stamps;
      if (args.sequence_period_s > 0.0) {
        const auto period_ns = static_cast<khronos::TimeStamp>(
            args.sequence_period_s * 1.0e9);
        if (period_ns == 0) {
          throw std::runtime_error("--sequence_period_s is too small");
        }
        const auto start = args.sequence_start_ns > 0
                               ? args.sequence_start_ns
                               : map->earliest();
        const auto end = args.sequence_end_ns > 0
                             ? args.sequence_end_ns
                             : map->latest();
        if (start > end) {
          throw std::runtime_error("sequence start is after sequence end");
        }
        for (auto stamp = start; stamp < end;) {
          query_stamps.push_back(stamp);
          if (end - stamp < period_ns) {
            break;
          }
          stamp += period_ns;
        }
        if (query_stamps.empty() || query_stamps.back() != end) {
          query_stamps.push_back(end);
        }
      } else {
        query_stamps = map->stamps();
      }
      std::cout << "sequence_query_range earliest=" << query_stamps.front()
                << " latest=" << query_stamps.back()
                << " samples=" << query_stamps.size() << "\n";
      for (std::size_t index = 0; index < query_stamps.size(); ++index) {
        const auto stamp = query_stamps[index];
        auto dsg = map->getDsgPtr(stamp);
        if (!dsg || !dsg->hasMesh() || !dsg->mesh()) {
          continue;
        }
        const auto stem = "frame_" + std::to_string(index);
        const auto ply_path = output_dir / (stem + ".ply");
        const auto overlay_path = output_dir / (stem + ".json");
        if (args.include_object_meshes) {
          auto display_mesh = dsg->mesh()->clone();
          appendObjectMeshes(*dsg, *display_mesh);
          writePly(*display_mesh,
                   ply_path,
                   args.map_file,
                   args.vertex_stride,
                   args.include_faces);
        } else {
          writePly(*dsg->mesh(),
                   ply_path,
                   args.map_file,
                   args.vertex_stride,
                   args.include_faces);
        }
        try {
          writeTrajectoryStartJson(*dsg, overlay_path);
        } catch (const std::exception&) {
          std::ofstream empty_overlay(overlay_path);
          empty_overlay << "{\"dynamic_tracks\": []}\n";
        }
        manifest << index << "," << stamp << "," << ply_path.filename().string()
                 << "," << overlay_path.filename().string() << "\n";
        std::cout << "sequence " << (index + 1) << "/" << query_stamps.size()
                  << " stamp=" << stamp << "\n";
      }
      return 0;
    }

    const auto stamp = selectMapTime(*map, args.map_time);
    auto dsg = map->getDsgPtr(stamp);
    if (!dsg || !dsg->hasMesh() || !dsg->mesh()) {
      throw std::runtime_error("Selected DSG has no mesh");
    }

    const auto& mesh = *dsg->mesh();
    std::cout << "loaded " << args.map_file << " stamp=" << stamp
              << " mesh_vertices=" << mesh.numVertices() << " mesh_faces=" << mesh.numFaces()
              << " colors=" << mesh.colors.size() << "\n";

    if (!args.output_ply.empty()) {
      if (args.include_object_meshes) {
        auto display_mesh = mesh.clone();
        const auto [object_vertices, object_faces] =
            appendObjectMeshes(*dsg, *display_mesh);
        std::cout << "display_object_meshes vertices=" << object_vertices
                  << " faces=" << object_faces << "\n";
        writePly(*display_mesh,
                 args.output_ply,
                 args.map_file,
                 args.vertex_stride,
                 args.include_faces);
      } else {
        writePly(mesh, args.output_ply, args.map_file, args.vertex_stride, args.include_faces);
      }
    }
    if (!args.output_view_json.empty()) {
      writeTrajectoryStartJson(*dsg, args.output_view_json);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
