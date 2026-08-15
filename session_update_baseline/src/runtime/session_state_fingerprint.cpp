#include "session_update_baseline/runtime/session_state_fingerprint.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <khronos/common/common_types.h>
#include <spark_dsg/mesh.h>
#include <spark_dsg/node_attributes.h>

namespace session_update::runtime {
namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

class ByteBuffer {
 public:
  void byte(const std::uint8_t value) { bytes_.push_back(value); }
  const std::vector<std::uint8_t>& bytes() const { return bytes_; }

 private:
  std::vector<std::uint8_t> bytes_;
};

class FnvSink {
 public:
  void byte(const std::uint8_t value) {
    value_ ^= value;
    value_ *= kFnvPrime;
    ++bytes_;
  }
  std::uint64_t value() const { return value_; }
  std::size_t bytes() const { return bytes_; }

 private:
  std::uint64_t value_ = kFnvOffset;
  std::size_t bytes_ = 0;
};

template <typename Sink>
void appendUnsigned(Sink& sink, std::uint64_t value, std::size_t width) {
  for (std::size_t i = 0; i < width; ++i) {
    sink.byte(static_cast<std::uint8_t>((value >> (8 * i)) & 0xffU));
  }
}

template <typename Sink>
void appendBool(Sink& sink, bool value) {
  sink.byte(value ? 1U : 0U);
}

template <typename Sink>
void appendSize(Sink& sink, std::size_t value) {
  appendUnsigned(sink, static_cast<std::uint64_t>(value), 8);
}

template <typename Sink>
void appendFloat(Sink& sink, float value) {
  static_assert(sizeof(float) == sizeof(std::uint32_t));
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  appendUnsigned(sink, bits, 4);
}

template <typename Sink>
void appendDouble(Sink& sink, double value) {
  static_assert(sizeof(double) == sizeof(std::uint64_t));
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  appendUnsigned(sink, bits, 8);
}

template <typename Sink>
void appendString(Sink& sink, const std::string& value) {
  appendSize(sink, value.size());
  for (const auto character : value) {
    sink.byte(static_cast<std::uint8_t>(character));
  }
}

template <typename Sink, typename Scalar, int Rows, int Cols>
void appendEigen(Sink& sink, const Eigen::Matrix<Scalar, Rows, Cols>& value) {
  appendSize(sink, static_cast<std::size_t>(value.rows()));
  appendSize(sink, static_cast<std::size_t>(value.cols()));
  for (Eigen::Index row = 0; row < value.rows(); ++row) {
    for (Eigen::Index col = 0; col < value.cols(); ++col) {
      if constexpr (std::is_same_v<Scalar, float>) {
        appendFloat(sink, value(row, col));
      } else {
        appendDouble(sink, value(row, col));
      }
    }
  }
}

template <typename Sink>
void appendColor(Sink& sink, const spark_dsg::Color& color) {
  sink.byte(color.r);
  sink.byte(color.g);
  sink.byte(color.b);
  sink.byte(color.a);
}

template <typename Sink>
void appendBytes(Sink& sink, const std::vector<std::uint8_t>& bytes) {
  appendSize(sink, bytes.size());
  for (const auto byte : bytes) {
    sink.byte(byte);
  }
}

std::vector<std::uint8_t> encodeVertex(const spark_dsg::Mesh& mesh,
                                       std::size_t index) {
  ByteBuffer vertex;
  appendEigen(vertex, mesh.points.at(index));
  appendBool(vertex, index < mesh.colors.size());
  if (index < mesh.colors.size()) {
    appendColor(vertex, mesh.colors.at(index));
  }
  appendBool(vertex, index < mesh.stamps.size());
  if (index < mesh.stamps.size()) {
    appendUnsigned(vertex, mesh.stamps.at(index), 8);
  }
  appendBool(vertex, index < mesh.first_seen_stamps.size());
  if (index < mesh.first_seen_stamps.size()) {
    appendUnsigned(vertex, mesh.first_seen_stamps.at(index), 8);
  }
  appendBool(vertex, index < mesh.labels.size());
  if (index < mesh.labels.size()) {
    appendUnsigned(vertex, mesh.labels.at(index), 4);
  }
  return vertex.bytes();
}

std::vector<std::uint8_t> encodeFace(
    const std::array<std::vector<std::uint8_t>, 3>& vertices) {
  auto sorted = vertices;
  std::sort(sorted.begin(), sorted.end());
  ByteBuffer face;
  for (const auto& vertex : sorted) {
    appendBytes(face, vertex);
  }
  return face.bytes();
}

template <typename Sink>
void appendMesh(Sink& sink, const spark_dsg::Mesh& mesh) {
  appendString(sink, "spark_dsg_mesh/v1");
  appendBool(sink, mesh.has_colors);
  appendBool(sink, mesh.has_timestamps);
  appendBool(sink, mesh.has_labels);
  appendBool(sink, mesh.has_first_seen_stamps);

  const auto vertex_count = mesh.points.size();
  // Preserve the exact availability of every attribute array. Khronos private
  // meshes legitimately retain geometry while omitting timestamp arrays even
  // when the generic mesh capability flag is true.
  appendSize(sink, mesh.colors.size());
  appendSize(sink, mesh.stamps.size());
  appendSize(sink, mesh.first_seen_stamps.size());
  appendSize(sink, mesh.labels.size());
  if (mesh.colors.size() > vertex_count || mesh.stamps.size() > vertex_count ||
      mesh.first_seen_stamps.size() > vertex_count ||
      mesh.labels.size() > vertex_count) {
    throw std::runtime_error("mesh attribute array exceeds its vertex count");
  }

  // Mesh finalization and DSG reseeding may reorder vertex and face arrays.
  // Canonicalize the same complete per-vertex records and oriented triangles
  // as multisets, preserving geometry, all enabled attributes, and topology
  // while excluding container/index order and irrelevant triangle winding.
  std::vector<std::vector<std::uint8_t>> vertices;
  vertices.reserve(vertex_count);
  for (std::size_t i = 0; i < vertex_count; ++i) {
    vertices.push_back(encodeVertex(mesh, i));
  }
  auto sorted_vertices = vertices;
  std::sort(sorted_vertices.begin(), sorted_vertices.end());
  appendSize(sink, sorted_vertices.size());
  for (const auto& vertex : sorted_vertices) {
    appendBytes(sink, vertex);
  }

  std::vector<std::vector<std::uint8_t>> faces;
  faces.reserve(mesh.faces.size());
  for (const auto& face : mesh.faces) {
    std::array<std::vector<std::uint8_t>, 3> face_vertices;
    for (std::size_t i = 0; i < face.size(); ++i) {
      if (face[i] >= vertices.size()) {
        throw std::runtime_error("mesh face contains an invalid vertex index");
      }
      face_vertices[i] = vertices[face[i]];
    }
    faces.push_back(encodeFace(face_vertices));
  }
  std::sort(faces.begin(), faces.end());
  appendSize(sink, faces.size());
  for (const auto& face : faces) {
    appendBytes(sink, face);
  }
}

std::optional<std::size_t> physicalInstanceId(
    const spark_dsg::KhronosObjectAttributes& attrs) {
  const auto iter = attrs.details.find("instance_id");
  if (iter == attrs.details.end() || iter->second.size() != 1 ||
      iter->second.front() == 0) {
    return std::nullopt;
  }
  return iter->second.front();
}

template <typename Sink>
void appendObject(Sink& sink,
                  const spark_dsg::KhronosObjectAttributes& attrs) {
  appendString(sink, "khronos_current_object/v1");
  const auto physical_id = physicalInstanceId(attrs);
  appendBool(sink, physical_id.has_value());
  appendSize(sink, physical_id.value_or(0));
  appendUnsigned(sink, attrs.semantic_label, 4);

  appendEigen(sink, attrs.position);
  appendUnsigned(sink, attrs.last_update_time_ns, 8);
  appendBool(sink, attrs.is_active);
  appendBool(sink, attrs.is_predicted);
  appendString(sink, attrs.metadata.get().dump());
  appendString(sink, attrs.name);
  appendColor(sink, attrs.color);
  appendUnsigned(sink, static_cast<std::uint32_t>(attrs.bounding_box.type), 4);
  appendEigen(sink, attrs.bounding_box.dimensions);
  appendEigen(sink, attrs.bounding_box.world_P_center);
  appendEigen(sink, attrs.bounding_box.world_R_center);
  appendEigen(sink, attrs.semantic_feature);

  appendSize(sink, attrs.mesh_connections.size());
  for (const auto connection : attrs.mesh_connections) {
    appendSize(sink, connection);
  }
  appendBool(sink, attrs.registered);
  appendDouble(sink, attrs.world_R_object.w());
  appendDouble(sink, attrs.world_R_object.x());
  appendDouble(sink, attrs.world_R_object.y());
  appendDouble(sink, attrs.world_R_object.z());

  appendSize(sink, attrs.first_observed_ns.size());
  for (const auto stamp : attrs.first_observed_ns) {
    appendUnsigned(sink, stamp, 8);
  }
  appendSize(sink, attrs.last_observed_ns.size());
  for (const auto stamp : attrs.last_observed_ns) {
    appendUnsigned(sink, stamp, 8);
  }
  appendMesh(sink, attrs.mesh);

  appendSize(sink, attrs.trajectory_timestamps.size());
  for (const auto stamp : attrs.trajectory_timestamps) {
    appendUnsigned(sink, stamp, 8);
  }
  appendSize(sink, attrs.trajectory_positions.size());
  for (const auto& position : attrs.trajectory_positions) {
    appendEigen(sink, position);
  }
  appendSize(sink, attrs.dynamic_object_points.size());
  for (const auto& points : attrs.dynamic_object_points) {
    appendSize(sink, points.size());
    for (const auto& point : points) {
      appendEigen(sink, point);
    }
  }

  appendSize(sink, attrs.details.size());
  for (const auto& [name, values] : attrs.details) {
    appendString(sink, name);
    appendSize(sink, values.size());
    for (const auto value : values) {
      appendSize(sink, value);
    }
  }
}

struct ObjectRecord {
  bool has_physical_id = false;
  std::size_t physical_id = 0;
  std::uint32_t semantic_label = 0;
  std::vector<std::uint8_t> bytes;
};

}  // namespace

CanonicalSceneFingerprint canonicalCurrentSceneFingerprint(
    const spark_dsg::DynamicSceneGraph& dsg) {
  FnvSink sink;
  appendString(sink, "session_update_current_scene/v1");
  appendBool(sink, dsg.hasMesh() && static_cast<bool>(dsg.mesh()));
  if (dsg.hasMesh() && dsg.mesh()) {
    appendMesh(sink, *dsg.mesh());
  }

  std::vector<ObjectRecord> records;
  if (dsg.hasLayer(khronos::DsgLayers::OBJECTS)) {
    const auto& objects = dsg.getLayer(khronos::DsgLayers::OBJECTS);
    records.reserve(objects.numNodes());
    for (const auto& [unused, node] : objects.nodes()) {
      (void)unused;
      const auto* attrs =
          node->tryAttributes<spark_dsg::KhronosObjectAttributes>();
      if (!attrs) {
        throw std::runtime_error(
            "current OBJECTS layer contains non-Khronos attributes");
      }
      ByteBuffer buffer;
      appendObject(buffer, *attrs);
      const auto physical_id = physicalInstanceId(*attrs);
      records.push_back(ObjectRecord{physical_id.has_value(),
                                     physical_id.value_or(0),
                                     attrs->semantic_label,
                                     buffer.bytes()});
    }
  }

  std::sort(records.begin(), records.end(), [](const auto& lhs, const auto& rhs) {
    return std::tie(lhs.has_physical_id,
                    lhs.physical_id,
                    lhs.semantic_label,
                    lhs.bytes) <
           std::tie(rhs.has_physical_id,
                    rhs.physical_id,
                    rhs.semantic_label,
                    rhs.bytes);
  });
  appendSize(sink, records.size());
  for (const auto& record : records) {
    appendBytes(sink, record.bytes);
  }

  return {sink.value(), sink.bytes(), records.size()};
}

}  // namespace session_update::runtime
