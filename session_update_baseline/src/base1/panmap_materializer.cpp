#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#include <panoptic_mapping/common/common.h>
#include <panoptic_mapping/map/submap.h>
#include <panoptic_mapping/map/submap_collection.h>
#include <voxblox/core/block.h>
#include <voxblox/core/common.h>
#include <voxblox/core/layer.h>
#include <voxblox/core/voxel.h>

namespace {

using panoptic_mapping::ChangeState;
using panoptic_mapping::PanopticLabel;
using panoptic_mapping::Point;
using panoptic_mapping::Submap;
using panoptic_mapping::SubmapCollection;
using panoptic_mapping::TsdfLayer;
using panoptic_mapping::TsdfVoxel;

struct ObjectRow {
  int label_id = -1;
  std::string name;
  int class_id = -1;
  int panoptic_id = 0;
  int mesh_id = -1;
  int points = 0;
  std::string size;
  std::string points_file;
  std::string session_state;
};

struct SurfaceSample {
  Point point = Point::Zero();
  Point ray_origin = Point::Zero();
  Point normal = Point::Zero();
  voxblox::Color color = voxblox::Color(180u, 180u, 180u);
  bool has_ray_origin = false;
  bool has_normal = false;
  bool has_color = false;
};

struct LoadedCloud {
  std::vector<SurfaceSample> samples;
  bool has_ray_samples = false;
  bool has_color_samples = false;
};

struct AuditRow {
  int label_id = -1;
  std::string name;
  int class_id = -1;
  int panoptic_id = 0;
  int input_points = 0;
  int loaded_points = 0;
  int normal_points = 0;
  int submap_id = -1;
  float voxel_size = 0.0f;
  float truncation_distance = 0.0f;
  float surface_support_radius = 0.0f;
  std::string mesh_mode;
  std::string tsdf_mode;
  int color_r = 180;
  int color_g = 180;
  int color_b = 180;
  int min_tsdf_support = 1;
  int allocated_blocks = 0;
  long long mesh_vertices = 0;
  std::string panoptic_label;
  std::string decision;
};

struct CellKey {
  int x = 0;
  int y = 0;
  int z = 0;

  bool operator==(const CellKey& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct CellKeyHash {
  size_t operator()(const CellKey& key) const {
    uint64_t x = static_cast<uint32_t>(key.x);
    uint64_t y = static_cast<uint32_t>(key.y);
    uint64_t z = static_cast<uint32_t>(key.z);
    uint64_t h = x * 73856093ull;
    h ^= y * 19349663ull;
    h ^= z * 83492791ull;
    return static_cast<size_t>(h);
  }
};

struct CellSdf {
  float sdf = 0.0f;
  int support = 0;
  int color_support = 0;
  int color_r = 0;
  int color_g = 0;
  int color_b = 0;
};

uint8_t clampColor(float value) {
  if (!std::isfinite(value)) {
    return 180u;
  }
  return static_cast<uint8_t>(
      std::max(0.0f, std::min(255.0f, std::round(value))));
}

void addSampleColor(const SurfaceSample& sample, CellSdf* cell) {
  if (!sample.has_color || !cell) {
    return;
  }
  cell->color_support += 1;
  cell->color_r += static_cast<int>(sample.color.r);
  cell->color_g += static_cast<int>(sample.color.g);
  cell->color_b += static_cast<int>(sample.color.b);
}

voxblox::Color colorForCell(const CellSdf& cell,
                            const voxblox::Color& fallback) {
  if (cell.color_support <= 0) {
    return fallback;
  }
  return voxblox::Color(
      static_cast<uint8_t>(cell.color_r / cell.color_support),
      static_cast<uint8_t>(cell.color_g / cell.color_support),
      static_cast<uint8_t>(cell.color_b / cell.color_support));
}

uint32_t hashString(const std::string& value) {
  uint32_t hash = 2166136261u;
  for (const unsigned char c : value) {
    hash ^= c;
    hash *= 16777619u;
  }
  return hash;
}

voxblox::Color colorForObject(const ObjectRow& row) {
  static const std::vector<voxblox::Color> palette = {
      voxblox::Color(230u,  57u,  70u),
      voxblox::Color( 42u, 157u, 143u),
      voxblox::Color( 69u, 123u, 157u),
      voxblox::Color(244u, 162u,  97u),
      voxblox::Color(131u,  56u, 236u),
      voxblox::Color( 38u,  70u,  83u),
      voxblox::Color(255u, 183u,   3u),
      voxblox::Color(  0u, 150u, 199u),
      voxblox::Color(188u, 108u,  37u),
      voxblox::Color( 90u,  24u, 154u),
      voxblox::Color( 60u, 179u, 113u),
      voxblox::Color(255u, 127u,  80u),
  };
  const std::string key =
      row.name + "#" + std::to_string(row.label_id) + "#" +
      std::to_string(row.class_id);
  return palette[hashString(key) % palette.size()];
}

std::vector<std::string> splitCsvLine(const std::string& line) {
  std::vector<std::string> out;
  std::string current;
  bool in_quotes = false;
  for (size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (c == '"') {
      if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
        current.push_back('"');
        ++i;
      } else {
        in_quotes = !in_quotes;
      }
    } else if (c == ',' && !in_quotes) {
      out.push_back(current);
      current.clear();
    } else {
      current.push_back(c);
    }
  }
  out.push_back(current);
  return out;
}

std::string csvEscape(const std::string& value) {
  if (value.find_first_of(",\"\n") == std::string::npos) {
    return value;
  }
  std::string out = "\"";
  for (const char c : value) {
    if (c == '"') {
      out += "\"\"";
    } else {
      out.push_back(c);
    }
  }
  out += '"';
  return out;
}

std::string trimCsvToken(std::string value) {
  while (!value.empty() &&
         (value.back() == '\r' || value.back() == '\n' ||
          value.back() == ' ' || value.back() == '\t')) {
    value.pop_back();
  }
  size_t start = 0;
  while (start < value.size() &&
         (value[start] == ' ' || value[start] == '\t')) {
    ++start;
  }
  if (start > 0) {
    value.erase(0, start);
  }
  return value;
}

std::string lowerAscii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  return value;
}

int parseInt(const std::string& raw, int fallback = 0) {
  if (raw.empty()) {
    return fallback;
  }
  try {
    return std::stoi(raw);
  } catch (...) {
    return fallback;
  }
}

std::string dirnameOf(const std::string& path) {
  const size_t pos = path.find_last_of('/');
  if (pos == std::string::npos) {
    return ".";
  }
  if (pos == 0) {
    return "/";
  }
  return path.substr(0, pos);
}

std::string joinPath(const std::string& base, const std::string& rel) {
  if (rel.empty()) {
    return rel;
  }
  if (rel[0] == '/') {
    return rel;
  }
  if (base.empty() || base == ".") {
    return rel;
  }
  if (base.back() == '/') {
    return base + rel;
  }
  return base + "/" + rel;
}

std::vector<ObjectRow> loadObjectSummary(const std::string& path) {
  std::ifstream in(path);
  if (!in.is_open()) {
    throw std::runtime_error("failed to open object summary: " + path);
  }
  std::string header_line;
  if (!std::getline(in, header_line)) {
    throw std::runtime_error("empty object summary: " + path);
  }
  const auto header = splitCsvLine(header_line);
  std::unordered_map<std::string, size_t> col;
  for (size_t i = 0; i < header.size(); ++i) {
    col[trimCsvToken(header[i])] = i;
  }
  auto get = [&](const std::vector<std::string>& row,
                 const std::string& key) -> std::string {
    auto it = col.find(key);
    if (it == col.end() || it->second >= row.size()) {
      return "";
    }
    return trimCsvToken(row[it->second]);
  };

  std::vector<ObjectRow> rows;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    const auto fields = splitCsvLine(line);
    ObjectRow row;
    row.label_id = parseInt(get(fields, "label_id"), -1);
    row.name = get(fields, "name");
    row.class_id = parseInt(get(fields, "class_id"), -1);
    row.panoptic_id = parseInt(get(fields, "panoptic_id"), 0);
    row.mesh_id = parseInt(get(fields, "mesh_id"), -1);
    row.points = parseInt(get(fields, "points"), 0);
    row.size = get(fields, "size");
    row.points_file = get(fields, "points_file");
    row.session_state = get(fields, "session_state");
    rows.push_back(row);
  }
  return rows;
}

ChangeState changeStateForSessionState(const std::string& state) {
  if (state.find("new_current") != std::string::npos) {
    return ChangeState::kNew;
  }
  return ChangeState::kPersistent;
}

template <typename T>
T readPod(std::istream& in) {
  T value;
  in.read(reinterpret_cast<char*>(&value), sizeof(T));
  return value;
}

float readScalar(std::istream& in, const std::string& type) {
  if (type == "float" || type == "float32") {
    return readPod<float>(in);
  }
  if (type == "double" || type == "float64") {
    return static_cast<float>(readPod<double>(in));
  }
  if (type == "uchar" || type == "uint8") {
    return static_cast<float>(readPod<uint8_t>(in));
  }
  if (type == "char" || type == "int8") {
    return static_cast<float>(readPod<int8_t>(in));
  }
  if (type == "ushort" || type == "uint16") {
    return static_cast<float>(readPod<uint16_t>(in));
  }
  if (type == "short" || type == "int16") {
    return static_cast<float>(readPod<int16_t>(in));
  }
  if (type == "uint" || type == "uint32") {
    return static_cast<float>(readPod<uint32_t>(in));
  }
  if (type == "int" || type == "int32") {
    return static_cast<float>(readPod<int32_t>(in));
  }
  throw std::runtime_error("unsupported PLY scalar type: " + type);
}

LoadedCloud loadPlyXYZ(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("failed to open ply: " + path);
  }
  std::string line;
  std::string format;
  size_t vertex_count = 0;
  bool in_vertex_element = false;
  std::vector<std::pair<std::string, std::string>> properties;
  size_t header_bytes = 0;
  while (std::getline(in, line)) {
    header_bytes += line.size() + 1;
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.rfind("format ", 0) == 0) {
      std::istringstream ss(line);
      std::string keyword;
      ss >> keyword >> format;
    } else if (line.rfind("element vertex ", 0) == 0) {
      vertex_count =
          static_cast<size_t>(std::stoll(line.substr(std::string("element vertex ").size())));
      in_vertex_element = true;
    } else if (line.rfind("element ", 0) == 0) {
      in_vertex_element = false;
    } else if (in_vertex_element && line.rfind("property ", 0) == 0) {
      std::istringstream ss(line);
      std::string keyword;
      std::string type;
      std::string name;
      ss >> keyword >> type >> name;
      if (!type.empty() && !name.empty()) {
        properties.emplace_back(type, name);
      }
    } else if (line == "end_header") {
      break;
    }
  }

  auto propertyIndex = [&](const std::string& name) -> int {
    for (size_t i = 0; i < properties.size(); ++i) {
      if (properties[i].second == name) {
        return static_cast<int>(i);
      }
    }
    return -1;
  };
  const int x_idx = propertyIndex("x");
  const int y_idx = propertyIndex("y");
  const int z_idx = propertyIndex("z");
  int ox_idx = propertyIndex("origin_x");
  int oy_idx = propertyIndex("origin_y");
  int oz_idx = propertyIndex("origin_z");
  if (ox_idx < 0) {
    ox_idx = propertyIndex("ox");
  }
  if (oy_idx < 0) {
    oy_idx = propertyIndex("oy");
  }
  if (oz_idx < 0) {
    oz_idx = propertyIndex("oz");
  }
  int red_idx = propertyIndex("red");
  int green_idx = propertyIndex("green");
  int blue_idx = propertyIndex("blue");
  if (red_idx < 0) {
    red_idx = propertyIndex("r");
  }
  if (green_idx < 0) {
    green_idx = propertyIndex("g");
  }
  if (blue_idx < 0) {
    blue_idx = propertyIndex("b");
  }
  if (x_idx < 0 || y_idx < 0 || z_idx < 0) {
    throw std::runtime_error("PLY is missing x/y/z properties: " + path);
  }

  LoadedCloud cloud;
  cloud.samples.reserve(vertex_count);
  if (format == "ascii") {
    for (size_t i = 0; i < vertex_count && std::getline(in, line); ++i) {
      std::istringstream ss(line);
      std::vector<float> values;
      float value = 0.0f;
      while (ss >> value) {
        values.push_back(value);
      }
      const int max_idx = std::max(x_idx, std::max(y_idx, z_idx));
      if (static_cast<int>(values.size()) <= max_idx) {
        continue;
      }
      const float x = values[x_idx];
      const float y = values[y_idx];
      const float z = values[z_idx];
      if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
        SurfaceSample sample;
        sample.point = Point(x, y, z);
        const int max_origin_idx = std::max(ox_idx, std::max(oy_idx, oz_idx));
        if (ox_idx >= 0 && oy_idx >= 0 && oz_idx >= 0 &&
            static_cast<int>(values.size()) > max_origin_idx) {
          const float ox = values[ox_idx];
          const float oy = values[oy_idx];
          const float oz = values[oz_idx];
          if (std::isfinite(ox) && std::isfinite(oy) && std::isfinite(oz)) {
            sample.ray_origin = Point(ox, oy, oz);
            sample.has_ray_origin =
                (sample.point - sample.ray_origin).norm() > 1e-6f;
            cloud.has_ray_samples = cloud.has_ray_samples || sample.has_ray_origin;
          }
        }
        const int max_color_idx = std::max(red_idx, std::max(green_idx, blue_idx));
        if (red_idx >= 0 && green_idx >= 0 && blue_idx >= 0 &&
            static_cast<int>(values.size()) > max_color_idx) {
          sample.color = voxblox::Color(
              clampColor(values[red_idx]),
              clampColor(values[green_idx]),
              clampColor(values[blue_idx]));
          sample.has_color = true;
          cloud.has_color_samples = true;
        }
        cloud.samples.push_back(sample);
      }
    }
    return cloud;
  }

  if (format != "binary_little_endian") {
    throw std::runtime_error("unsupported PLY format: " + format);
  }
  in.clear();
  in.seekg(static_cast<std::streamoff>(header_bytes), std::ios::beg);
  for (size_t i = 0; i < vertex_count; ++i) {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float ox = 0.0f;
    float oy = 0.0f;
    float oz = 0.0f;
    float red = 180.0f;
    float green = 180.0f;
    float blue = 180.0f;
    for (size_t prop = 0; prop < properties.size(); ++prop) {
      const float value = readScalar(in, properties[prop].first);
      if (static_cast<int>(prop) == x_idx) {
        x = value;
      } else if (static_cast<int>(prop) == y_idx) {
        y = value;
      } else if (static_cast<int>(prop) == z_idx) {
        z = value;
      } else if (static_cast<int>(prop) == ox_idx) {
        ox = value;
      } else if (static_cast<int>(prop) == oy_idx) {
        oy = value;
      } else if (static_cast<int>(prop) == oz_idx) {
        oz = value;
      } else if (static_cast<int>(prop) == red_idx) {
        red = value;
      } else if (static_cast<int>(prop) == green_idx) {
        green = value;
      } else if (static_cast<int>(prop) == blue_idx) {
        blue = value;
      }
    }
    if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
      SurfaceSample sample;
      sample.point = Point(x, y, z);
      if (ox_idx >= 0 && oy_idx >= 0 && oz_idx >= 0 &&
          std::isfinite(ox) && std::isfinite(oy) && std::isfinite(oz)) {
        sample.ray_origin = Point(ox, oy, oz);
        sample.has_ray_origin =
            (sample.point - sample.ray_origin).norm() > 1e-6f;
        cloud.has_ray_samples = cloud.has_ray_samples || sample.has_ray_origin;
      }
      if (red_idx >= 0 && green_idx >= 0 && blue_idx >= 0) {
        sample.color =
            voxblox::Color(clampColor(red), clampColor(green), clampColor(blue));
        sample.has_color = true;
        cloud.has_color_samples = true;
      }
      cloud.samples.push_back(sample);
    }
  }
  return cloud;
}

float quantileValue(std::vector<float> values, float q) {
  if (values.empty()) {
    return 0.0f;
  }
  q = std::max(0.0f, std::min(1.0f, q));
  std::sort(values.begin(), values.end());
  const float scaled = q * static_cast<float>(values.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(std::floor(scaled));
  const std::size_t hi = static_cast<std::size_t>(std::ceil(scaled));
  if (lo == hi) {
    return values[lo];
  }
  const float t = scaled - static_cast<float>(lo);
  return (1.0f - t) * values[lo] + t * values[hi];
}

LoadedCloud canonicalizeRoomStructuralPlaneCloud(const LoadedCloud& input) {
  LoadedCloud output = input;
  if (output.samples.empty()) {
    return output;
  }

  std::vector<float> xs;
  std::vector<float> ys;
  xs.reserve(output.samples.size());
  ys.reserve(output.samples.size());
  for (const SurfaceSample& sample : output.samples) {
    xs.push_back(sample.point.x());
    ys.push_back(sample.point.y());
  }

  const float x_lo = quantileValue(xs, 0.02f);
  const float x_hi = quantileValue(xs, 0.98f);
  const float y_lo = quantileValue(ys, 0.02f);
  const float y_hi = quantileValue(ys, 0.98f);

  for (SurfaceSample& sample : output.samples) {
    const float distances[4] = {
        std::abs(sample.point.x() - x_lo),
        std::abs(sample.point.x() - x_hi),
        std::abs(sample.point.y() - y_lo),
        std::abs(sample.point.y() - y_hi)};
    int best = 0;
    for (int i = 1; i < 4; ++i) {
      if (distances[i] < distances[best]) {
        best = i;
      }
    }
    if (best == 0) {
      sample.point.x() = x_lo;
    } else if (best == 1) {
      sample.point.x() = x_hi;
    } else if (best == 2) {
      sample.point.y() = y_lo;
    } else {
      sample.point.y() = y_hi;
    }
    sample.has_normal = false;
  }
  return output;
}

long long countMeshVertices(const Submap& submap) {
  voxblox::BlockIndexList block_list;
  submap.getMeshLayer().getAllAllocatedMeshes(&block_list);
  long long vertices = 0;
  for (const auto& block_index : block_list) {
    vertices += submap.getMeshLayer().getMeshByIndex(block_index).vertices.size();
  }
  return vertices;
}

std::size_t estimateLocalPlaneNormals(LoadedCloud* cloud, float normal_radius) {
  if (!cloud || cloud->samples.size() < 6 || normal_radius <= 0.0f) {
    return 0;
  }

  std::unordered_map<CellKey, std::vector<int>, CellKeyHash> grid;
  grid.reserve(cloud->samples.size());
  for (std::size_t i = 0; i < cloud->samples.size(); ++i) {
    const Point& point = cloud->samples[i].point;
    const CellKey key{
        static_cast<int>(std::floor(point.x() / normal_radius)),
        static_cast<int>(std::floor(point.y() / normal_radius)),
        static_cast<int>(std::floor(point.z() / normal_radius))};
    grid[key].push_back(static_cast<int>(i));
  }

  const float radius_sq = normal_radius * normal_radius;
  std::size_t normals = 0;
  std::vector<int> neighbors;
  neighbors.reserve(128);
  for (std::size_t i = 0; i < cloud->samples.size(); ++i) {
    const Point& point = cloud->samples[i].point;
    const CellKey base{
        static_cast<int>(std::floor(point.x() / normal_radius)),
        static_cast<int>(std::floor(point.y() / normal_radius)),
        static_cast<int>(std::floor(point.z() / normal_radius))};
    neighbors.clear();
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dz = -1; dz <= 1; ++dz) {
          const CellKey key{base.x + dx, base.y + dy, base.z + dz};
          auto it = grid.find(key);
          if (it == grid.end()) {
            continue;
          }
          for (const int index : it->second) {
            const Point diff = cloud->samples[index].point - point;
            if (diff.squaredNorm() <= radius_sq) {
              neighbors.push_back(index);
            }
          }
        }
      }
    }
    if (neighbors.size() < 6) {
      continue;
    }

    Eigen::Vector3f mean = Eigen::Vector3f::Zero();
    for (const int index : neighbors) {
      mean += cloud->samples[index].point;
    }
    mean /= static_cast<float>(neighbors.size());

    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    for (const int index : neighbors) {
      const Eigen::Vector3f diff = cloud->samples[index].point - mean;
      covariance += diff * diff.transpose();
    }
    covariance /= static_cast<float>(neighbors.size());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    if (solver.info() != Eigen::Success) {
      continue;
    }
    Point normal = solver.eigenvectors().col(0);
    const float normal_norm = normal.norm();
    if (!std::isfinite(normal_norm) || normal_norm <= 1e-6f) {
      continue;
    }
    normal /= normal_norm;
    if (cloud->samples[i].has_ray_origin) {
      const Point to_camera = cloud->samples[i].ray_origin - point;
      if (normal.dot(to_camera) < 0.0f) {
        normal = -normal;
      }
    }
    cloud->samples[i].normal = normal;
    cloud->samples[i].has_normal = true;
    ++normals;
  }
  return normals;
}

void integrateSurfaceCloud(const LoadedCloud& cloud,
                           Submap* submap,
                           float surface_support_radius,
                           int min_tsdf_support,
                           const std::string& tsdf_mode,
                           const voxblox::Color& visual_color) {
  TsdfLayer* layer = submap->getTsdfLayerPtr().get();
  const float voxel_size = submap->getConfig().voxel_size;
  const float truncation = submap->getConfig().truncation_distance;
  const float support_radius = voxel_size;
  const float lateral_support_radius =
      surface_support_radius > 0.0f ? surface_support_radius : truncation;
  const int band_voxels =
      std::max(1, static_cast<int>(std::ceil(
                      (support_radius + truncation) / voxel_size)));
  std::unordered_map<CellKey, CellSdf, CellKeyHash> cell_sdf;
  cell_sdf.reserve(cloud.samples.size() * 32);

  auto cellCenter = [&](const CellKey& key) -> Point {
    return Point((static_cast<float>(key.x) + 0.5f) * voxel_size,
                 (static_cast<float>(key.y) + 0.5f) * voxel_size,
                 (static_cast<float>(key.z) + 0.5f) * voxel_size);
  };
  auto updateCell = [&](const CellKey& key,
                        float sdf,
                        const SurfaceSample* sample) {
    const float clamped_sdf =
        std::max(-truncation, std::min(truncation, sdf));
    auto it = cell_sdf.find(key);
    if (it == cell_sdf.end()) {
      CellSdf cell;
      cell.sdf = clamped_sdf;
      cell.support = 1;
      if (sample) {
        addSampleColor(*sample, &cell);
      }
      cell_sdf[key] = cell;
    } else {
      it->second.support += 1;
      if (std::abs(clamped_sdf) < std::abs(it->second.sdf)) {
        it->second.sdf = clamped_sdf;
      }
      if (sample) {
        addSampleColor(*sample, &it->second);
      }
    }
  };

  const bool use_raywise_tsdf = tsdf_mode == "raywise_tsdf";
  const bool use_ray_footprint_tsdf = tsdf_mode == "ray_footprint_tsdf";
  const bool use_free_space_carving = tsdf_mode == "surface_band_free_carve";
  const bool use_evidence_mask = tsdf_mode == "evidence_masked_surface_band";
  const bool use_surfel_splat = tsdf_mode == "surfel_splat_tsdf";
  const bool use_anisotropic_surface_band =
      tsdf_mode == "anisotropic_surface_band";
  std::unordered_set<CellKey, CellKeyHash> evidence_mask;
  if (use_evidence_mask) {
    evidence_mask.reserve(cloud.samples.size() * 27);
    for (const SurfaceSample& sample : cloud.samples) {
      const Point& point = sample.point;
      const CellKey base{
          static_cast<int>(std::floor(point.x() / voxel_size)),
          static_cast<int>(std::floor(point.y() / voxel_size)),
          static_cast<int>(std::floor(point.z() / voxel_size))};
      for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dz = -1; dz <= 1; ++dz) {
            evidence_mask.insert(CellKey{base.x + dx, base.y + dy, base.z + dz});
          }
        }
      }
    }
  }
  std::unordered_map<CellKey, int, CellKeyHash> free_space_support;
  if (use_free_space_carving) {
    free_space_support.reserve(cloud.samples.size() * 16);
  }
  auto updateFreeCell = [&](const CellKey& key) {
    if (!use_free_space_carving) {
      return;
    }
    auto it = free_space_support.find(key);
    if (it == free_space_support.end()) {
      free_space_support[key] = 1;
    } else {
      it->second += 1;
    }
  };
  if (use_surfel_splat) {
    std::unordered_map<CellKey, int, CellKeyHash> surfel_support;
    surfel_support.reserve(cloud.samples.size());
    for (const SurfaceSample& sample : cloud.samples) {
      const Point& point = sample.point;
      const CellKey key{
          static_cast<int>(std::floor(point.x() / voxel_size)),
          static_cast<int>(std::floor(point.y() / voxel_size)),
          static_cast<int>(std::floor(point.z() / voxel_size))};
      auto it = surfel_support.find(key);
      if (it == surfel_support.end()) {
        surfel_support[key] = 1;
      } else {
        it->second += 1;
      }
    }
    for (const auto& item : surfel_support) {
      const int support = item.second;
      if (support < min_tsdf_support) {
        continue;
      }
      auto center_it = cell_sdf.find(item.first);
      if (center_it == cell_sdf.end()) {
        CellSdf cell;
        cell.sdf = -0.5f * voxel_size;
        cell.support = support;
        cell_sdf[item.first] = cell;
      } else {
        center_it->second.sdf = -0.5f * voxel_size;
        center_it->second.support += support;
      }
      for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dz = -1; dz <= 1; ++dz) {
            if (dx == 0 && dy == 0 && dz == 0) {
              continue;
            }
            const CellKey neighbor{
                item.first.x + dx, item.first.y + dy, item.first.z + dz};
            if (surfel_support.find(neighbor) != surfel_support.end()) {
              continue;
            }
            auto neighbor_it = cell_sdf.find(neighbor);
            if (neighbor_it == cell_sdf.end()) {
                CellSdf cell;
                cell.sdf = 0.5f * voxel_size;
                cell.support = support;
                cell_sdf[neighbor] = cell;
            } else if (neighbor_it->second.sdf > 0.0f) {
              neighbor_it->second.support += support;
            }
          }
        }
      }
    }
  }

  if (!use_surfel_splat) {
  for (const SurfaceSample& sample : cloud.samples) {
    const Point& point = sample.point;
    if (sample.has_ray_origin) {
      const Point ray = point - sample.ray_origin;
      const float surface_depth = ray.norm();
      if (surface_depth <= 1e-6f) {
        continue;
      }
      const Point ray_direction = ray / surface_depth;
      if (use_free_space_carving) {
        const float free_step = std::max(voxel_size, 0.01f);
        const float free_end = surface_depth - 0.5f * voxel_size;
        std::unordered_set<CellKey, CellKeyHash> ray_free_cells;
        ray_free_cells.reserve(static_cast<size_t>(
            std::max(0.0f, std::ceil(free_end / free_step)) + 4.0f));
        for (float depth = voxel_size; depth < free_end; depth += free_step) {
          const Point probe = sample.ray_origin + ray_direction * depth;
          const CellKey key{
              static_cast<int>(std::floor(probe.x() / voxel_size)),
              static_cast<int>(std::floor(probe.y() / voxel_size)),
              static_cast<int>(std::floor(probe.z() / voxel_size))};
          if (ray_free_cells.insert(key).second) {
            updateFreeCell(key);
          }
        }
      }
      if (use_raywise_tsdf || use_ray_footprint_tsdf) {
        const float ray_step = std::max(0.5f * voxel_size, 0.01f);
        const float start_depth = std::max(voxel_size, surface_depth - truncation);
        const float end_depth = surface_depth + truncation;
        const float footprint_radius =
            use_ray_footprint_tsdf ? lateral_support_radius : 0.0f;
        const int lateral_steps =
            use_ray_footprint_tsdf
                ? std::max(1, static_cast<int>(
                                  std::ceil(footprint_radius / voxel_size)))
                : 0;
        Point tangent_u = Point::Zero();
        Point tangent_v = Point::Zero();
        if (use_ray_footprint_tsdf) {
          const Point helper =
              std::abs(ray_direction.z()) < 0.9f ? Point(0.0f, 0.0f, 1.0f)
                                                 : Point(0.0f, 1.0f, 0.0f);
          tangent_u = ray_direction.cross(helper);
          const float tangent_u_norm = tangent_u.norm();
          if (tangent_u_norm <= 1e-6f) {
            continue;
          }
          tangent_u /= tangent_u_norm;
          tangent_v = ray_direction.cross(tangent_u);
          const float tangent_v_norm = tangent_v.norm();
          if (tangent_v_norm <= 1e-6f) {
            continue;
          }
          tangent_v /= tangent_v_norm;
        }
        std::unordered_set<CellKey, CellKeyHash> ray_cells;
        ray_cells.reserve(static_cast<size_t>(
            (std::ceil((end_depth - start_depth) / ray_step) + 4.0f) *
            (2 * lateral_steps + 1) * (2 * lateral_steps + 1)));
        for (float depth = start_depth; depth <= end_depth; depth += ray_step) {
          const Point ray_point = sample.ray_origin + ray_direction * depth;
          for (int du = -lateral_steps; du <= lateral_steps; ++du) {
            for (int dv = -lateral_steps; dv <= lateral_steps; ++dv) {
              Point lateral = Point::Zero();
              if (use_ray_footprint_tsdf) {
                lateral = tangent_u * (du * voxel_size) +
                          tangent_v * (dv * voxel_size);
              }
              if (use_ray_footprint_tsdf &&
                  lateral.norm() > footprint_radius) {
                continue;
              }
              const Point probe = ray_point + lateral;
              const CellKey key{
                  static_cast<int>(std::floor(probe.x() / voxel_size)),
                  static_cast<int>(std::floor(probe.y() / voxel_size)),
                  static_cast<int>(std::floor(probe.z() / voxel_size))};
              if (!ray_cells.insert(key).second) {
                continue;
              }
              const Point center = cellCenter(key);
              const Point from_origin = center - sample.ray_origin;
              const float voxel_depth = from_origin.dot(ray_direction);
              if (voxel_depth <= 0.0f) {
                continue;
              }
              if (use_ray_footprint_tsdf) {
                const Point perpendicular =
                    from_origin - ray_direction * voxel_depth;
                if (perpendicular.norm() >
                    footprint_radius + 0.5f * voxel_size) {
                  continue;
                }
              }
              const float signed_distance = surface_depth - voxel_depth;
              if (std::abs(signed_distance) > truncation) {
                continue;
              }
              updateCell(key, signed_distance, &sample);
            }
          }
        }
        continue;
      }
      const Point direction = sample.has_normal ? sample.normal : ray_direction;
      const Point helper =
          std::abs(direction.z()) < 0.9f ? Point(0.0f, 0.0f, 1.0f)
                                         : Point(0.0f, 1.0f, 0.0f);
      Point tangent_u = direction.cross(helper);
      const float tangent_u_norm = tangent_u.norm();
      if (tangent_u_norm <= 1e-6f) {
        continue;
      }
      tangent_u /= tangent_u_norm;
      Point tangent_v = direction.cross(tangent_u);
      const float tangent_v_norm = tangent_v.norm();
      if (tangent_v_norm <= 1e-6f) {
        continue;
      }
      tangent_v /= tangent_v_norm;
      const int depth_steps =
          use_anisotropic_surface_band
              ? 1
              : std::max(1, static_cast<int>(std::ceil(truncation / voxel_size)));
      const float normal_support_radius =
          use_anisotropic_surface_band
              ? std::max(voxel_size, 0.5f * truncation)
              : truncation;
      const int lateral_steps = std::max(
          1, static_cast<int>(std::ceil(lateral_support_radius / voxel_size)));
      for (int ds = -depth_steps; ds <= depth_steps; ++ds) {
        for (int du = -lateral_steps; du <= lateral_steps; ++du) {
          for (int dv = -lateral_steps; dv <= lateral_steps; ++dv) {
            const Point lateral =
                tangent_u * (du * voxel_size) +
                tangent_v * (dv * voxel_size);
            if (lateral.norm() > lateral_support_radius) {
              continue;
            }
            const Point probe =
                point + direction * (ds * voxel_size) + lateral;
            const CellKey key{
                static_cast<int>(std::floor(probe.x() / voxel_size)),
                static_cast<int>(std::floor(probe.y() / voxel_size)),
                static_cast<int>(std::floor(probe.z() / voxel_size))};
            const Point center = cellCenter(key);
            const float voxel_depth =
                (center - sample.ray_origin).dot(ray_direction);
            if (voxel_depth <= 0.0f) {
              continue;
            }
            const float signed_distance =
                sample.has_normal ? (center - point).dot(direction)
                                  : surface_depth - voxel_depth;
            if (std::abs(signed_distance) > normal_support_radius) {
              continue;
            }
            const Point diff = center - point;
            Point tangent = Point::Zero();
            if (sample.has_normal) {
              tangent = diff - signed_distance * direction;
            } else {
              tangent = diff + signed_distance * direction;
            }
            if (tangent.norm() > lateral_support_radius) {
              continue;
            }
            updateCell(key, signed_distance, &sample);
          }
        }
      }
      continue;
    }
    for (int dx = -band_voxels; dx <= band_voxels; ++dx) {
      for (int dy = -band_voxels; dy <= band_voxels; ++dy) {
        for (int dz = -band_voxels; dz <= band_voxels; ++dz) {
          const Point probe =
              point + Point(dx * voxel_size, dy * voxel_size, dz * voxel_size);
          const CellKey key{
              static_cast<int>(std::floor(probe.x() / voxel_size)),
              static_cast<int>(std::floor(probe.y() / voxel_size)),
              static_cast<int>(std::floor(probe.z() / voxel_size))};
          const Point voxel_center = cellCenter(key);
          const float sdf = (voxel_center - point).norm() - support_radius;
          if (sdf > truncation) {
            continue;
          }
          updateCell(key, sdf, &sample);
        }
      }
    }
  }
  }

  for (const auto& item : cell_sdf) {
    if (use_evidence_mask && evidence_mask.find(item.first) == evidence_mask.end()) {
      continue;
    }
    float final_sdf = item.second.sdf;
    int final_support = item.second.support;
    if (use_free_space_carving) {
      const auto free_it = free_space_support.find(item.first);
      const int free_support =
          free_it == free_space_support.end() ? 0 : free_it->second;
      if (free_support > item.second.support) {
        final_sdf = truncation;
        final_support = free_support;
      }
    }
    if (final_support < min_tsdf_support) {
      continue;
    }
    const Point center = cellCenter(item.first);
    auto block = layer->allocateBlockPtrByCoordinates(center);
    const voxblox::VoxelIndex voxel_index =
        block->computeTruncatedVoxelIndexFromCoordinates(center);
    TsdfVoxel& voxel = block->getVoxelByVoxelIndex(voxel_index);
    if (voxel.weight <= 0.0f ||
        std::abs(final_sdf) < std::abs(voxel.distance)) {
      voxel.distance = final_sdf;
    }
    voxel.weight =
        std::max(voxel.weight, static_cast<float>(final_support));
    voxel.color = colorForCell(item.second, visual_color);
    block->set_has_data(true);
    block->setUpdatedAll();
  }
}

long long replaceMeshWithSurfaceSamples(const LoadedCloud& cloud,
                                        Submap* submap,
                                        float mesh_voxel_size,
                                        const voxblox::Color& visual_color) {
  if (!submap || mesh_voxel_size <= 0.0f) {
    return 0;
  }

  auto mesh_layer = submap->getMeshLayerPtr();
  voxblox::BlockIndexList allocated;
  mesh_layer->getAllAllocatedMeshes(&allocated);
  for (const auto& block_index : allocated) {
    mesh_layer->removeMesh(block_index);
  }

  std::unordered_set<CellKey, CellKeyHash> used_cells;
  used_cells.reserve(cloud.samples.size());
  long long vertices = 0;
  for (const SurfaceSample& sample : cloud.samples) {
    const Point& point = sample.point;
    const CellKey cell{
        static_cast<int>(std::floor(point.x() / mesh_voxel_size)),
        static_cast<int>(std::floor(point.y() / mesh_voxel_size)),
        static_cast<int>(std::floor(point.z() / mesh_voxel_size))};
    if (!used_cells.insert(cell).second) {
      continue;
    }

    auto mesh = mesh_layer->allocateMeshPtrByCoordinates(point);
    for (int i = 0; i < 3; ++i) {
      mesh->vertices.push_back(point);
      mesh->colors.push_back(sample.has_color ? sample.color : visual_color);
      mesh->normals.push_back(sample.has_normal ? sample.normal
                                                : Point(0.0f, 0.0f, 1.0f));
    }
    mesh->updated = true;
    vertices += 3;
  }
  return vertices;
}

long long projectMeshVerticesToSurfaceSamples(const LoadedCloud& cloud,
                                              Submap* submap,
                                              float max_projection_distance) {
  if (!submap || cloud.samples.empty() || max_projection_distance <= 0.0f) {
    return 0;
  }

  const float cell_size = max_projection_distance;
  const float max_distance_sq =
      max_projection_distance * max_projection_distance;
  std::unordered_map<CellKey, std::vector<int>, CellKeyHash> grid;
  grid.reserve(cloud.samples.size());
  for (std::size_t i = 0; i < cloud.samples.size(); ++i) {
    const Point& point = cloud.samples[i].point;
    const CellKey key{
        static_cast<int>(std::floor(point.x() / cell_size)),
        static_cast<int>(std::floor(point.y() / cell_size)),
        static_cast<int>(std::floor(point.z() / cell_size))};
    grid[key].push_back(static_cast<int>(i));
  }

  auto nearestSample = [&](const Point& vertex,
                           Point* nearest,
                           voxblox::Color* color) -> bool {
    const CellKey base{
        static_cast<int>(std::floor(vertex.x() / cell_size)),
        static_cast<int>(std::floor(vertex.y() / cell_size)),
        static_cast<int>(std::floor(vertex.z() / cell_size))};
    float best_distance_sq = std::numeric_limits<float>::infinity();
    int best_index = -1;
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dz = -1; dz <= 1; ++dz) {
          const CellKey key{base.x + dx, base.y + dy, base.z + dz};
          auto it = grid.find(key);
          if (it == grid.end()) {
            continue;
          }
          for (const int index : it->second) {
            const float distance_sq =
                (cloud.samples[index].point - vertex).squaredNorm();
            if (distance_sq < best_distance_sq) {
              best_distance_sq = distance_sq;
              best_index = index;
            }
          }
        }
      }
    }
    if (best_index < 0 || best_distance_sq > max_distance_sq) {
      return false;
    }
    *nearest = cloud.samples[best_index].point;
    if (color && cloud.samples[best_index].has_color) {
      *color = cloud.samples[best_index].color;
    }
    return true;
  };

  auto mesh_layer = submap->getMeshLayerPtr();
  voxblox::BlockIndexList allocated;
  mesh_layer->getAllAllocatedMeshes(&allocated);
  long long projected = 0;
  for (const auto& block_index : allocated) {
    auto& mesh = mesh_layer->getMeshByIndex(block_index);
    for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
      Point nearest = mesh.vertices[i];
      voxblox::Color color =
          i < mesh.colors.size() ? mesh.colors[i] : voxblox::Color();
      if (nearestSample(mesh.vertices[i], &nearest, &color)) {
        mesh.vertices[i] = nearest;
        if (i < mesh.colors.size()) {
          mesh.colors[i] = color;
        }
        ++projected;
      }
    }
    mesh.updated = true;
  }
  return projected;
}

PanopticLabel panopticLabelFor(const ObjectRow& row) {
  if (row.panoptic_id == 0) {
    return PanopticLabel::kBackground;
  }
  return PanopticLabel::kInstance;
}

float chooseVoxelSize(const ObjectRow& row, float fallback_voxel_size) {
  if (fallback_voxel_size > 0.0f) {
    return fallback_voxel_size;
  }
  if (row.panoptic_id == 0) {
    return 0.05f;
  }
  if (row.size == "S") {
    return 0.02f;
  }
  if (row.size == "L") {
    return 0.04f;
  }
  return 0.03f;
}

float chooseTruncationDistance(float voxel_size, float fallback_truncation) {
  if (fallback_truncation > 0.0f) {
    return fallback_truncation;
  }
  return 2.0f * voxel_size;
}

bool isRoomStructuralPlane(const ObjectRow& row) {
  if (row.panoptic_id != 0) {
    return false;
  }
  const std::string name = lowerAscii(row.name);
  if (name.find("tv_wall") != std::string::npos ||
      name.find("tv wall") != std::string::npos) {
    return false;
  }
  return name.find("walls") != std::string::npos ||
         name.find("room_wall") != std::string::npos ||
         name.find("structural_wall") != std::string::npos;
}

std::string labelToString(PanopticLabel label) {
  return panoptic_mapping::panopticLabelToString(label);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 6 || argc > 10) {
    std::cerr << "usage: panmap_materializer OBJECT_SUMMARY.csv OUT.panmap "
                 "DIAG_DIR VOXEL_SIZE_M TRUNCATION_DISTANCE_M "
                 "[SURFACE_SUPPORT_RADIUS_M] [MESH_MODE] "
                 "[MIN_TSDF_SUPPORT] [TSDF_MODE]\n";
    return 2;
  }
  const std::string object_summary_path = argv[1];
  const std::string out_map_path = argv[2];
  const std::string diag_dir = argv[3];
  const float fallback_voxel_size = std::stof(argv[4]);
  const float fallback_truncation_distance = std::stof(argv[5]);
  const float fallback_surface_support_radius =
      argc >= 7 ? std::stof(argv[6]) : fallback_truncation_distance;
  const std::string mesh_mode = argc >= 8 ? argv[7] : "marching_cubes";
  const int min_tsdf_support = argc >= 9 ? std::stoi(argv[8]) : 1;
  const std::string tsdf_mode = argc >= 10 ? argv[9] : "surface_band";
  if (tsdf_mode != "surface_band" && tsdf_mode != "raywise_tsdf" &&
      tsdf_mode != "ray_footprint_tsdf" &&
      tsdf_mode != "surface_band_free_carve" &&
      tsdf_mode != "evidence_masked_surface_band" &&
      tsdf_mode != "anisotropic_surface_band" &&
      tsdf_mode != "structural_plane_anisotropic_surface_band" &&
      tsdf_mode != "room_plane_canonicalized_anisotropic" &&
      tsdf_mode != "surfel_splat_tsdf") {
    std::cerr << "unsupported TSDF_MODE: " << tsdf_mode << "\n";
    return 2;
  }
  const std::string map_dir = dirnameOf(object_summary_path);

  const auto objects = loadObjectSummary(object_summary_path);
  SubmapCollection output;
  std::vector<AuditRow> audit;
  int created = 0;
  int skipped = 0;

  for (const ObjectRow& object : objects) {
    AuditRow row;
    row.label_id = object.label_id;
    row.name = object.name;
    row.class_id = object.class_id;
    row.panoptic_id = object.panoptic_id;
    row.input_points = object.points;
    const float voxel_size = chooseVoxelSize(object, fallback_voxel_size);
    const float truncation_distance =
        chooseTruncationDistance(voxel_size, fallback_truncation_distance);
    float surface_support_radius =
        fallback_surface_support_radius > 0.0f
            ? fallback_surface_support_radius
            : truncation_distance;
    std::string object_tsdf_mode = tsdf_mode;
    if (tsdf_mode == "structural_plane_anisotropic_surface_band") {
      object_tsdf_mode = "anisotropic_surface_band";
      if (isRoomStructuralPlane(object)) {
        surface_support_radius =
            std::min(surface_support_radius, std::max(voxel_size, 0.03f));
      }
    } else if (tsdf_mode == "room_plane_canonicalized_anisotropic") {
      object_tsdf_mode = "anisotropic_surface_band";
      if (isRoomStructuralPlane(object)) {
        surface_support_radius =
            std::min(surface_support_radius, std::max(voxel_size, 0.03f));
      }
    }
    row.voxel_size = voxel_size;
    row.truncation_distance = truncation_distance;
    row.surface_support_radius = surface_support_radius;
    row.mesh_mode = mesh_mode;
    row.tsdf_mode = object_tsdf_mode;
    row.min_tsdf_support = min_tsdf_support;
    const voxblox::Color visual_color = colorForObject(object);
    row.color_r = static_cast<int>(visual_color.r);
    row.color_g = static_cast<int>(visual_color.g);
    row.color_b = static_cast<int>(visual_color.b);

    if (object.points <= 0 || object.points_file.empty()) {
      row.decision = "skip_empty_object";
      skipped++;
      audit.push_back(row);
      continue;
    }

    LoadedCloud cloud = loadPlyXYZ(joinPath(map_dir, object.points_file));
    if (tsdf_mode == "room_plane_canonicalized_anisotropic" &&
        isRoomStructuralPlane(object)) {
      cloud = canonicalizeRoomStructuralPlaneCloud(cloud);
    }
    row.loaded_points = static_cast<int>(cloud.samples.size());
    if (cloud.samples.empty()) {
      row.decision = "skip_empty_cloud";
      skipped++;
      audit.push_back(row);
      continue;
    }
    const float normal_radius =
        std::max(2.0f * surface_support_radius, truncation_distance);
    row.normal_points =
        static_cast<int>(estimateLocalPlaneNormals(&cloud, normal_radius));

    Submap::Config cfg;
    cfg.voxel_size = voxel_size;
    cfg.truncation_distance = truncation_distance;
    cfg.voxels_per_side = 16;

    Submap* submap = output.createSubmap(cfg);
    submap->setName(object.name);
    submap->setInstanceID(object.label_id);
    submap->setClassID(object.class_id);
    const PanopticLabel panoptic_label = panopticLabelFor(object);
    submap->setLabel(panoptic_label);
    submap->setChangeState(changeStateForSessionState(object.session_state));
    submap->setIsActive(true);
    submap->setWasTracked(true);
    submap->setFrameName(object.name + "_base1");
    panoptic_mapping::Transformation identity;
    identity.setIdentity();
    submap->setT_M_S(identity);

    integrateSurfaceCloud(
        cloud,
        submap,
        surface_support_radius,
        min_tsdf_support,
        object_tsdf_mode,
        visual_color);
    submap->updateEverything(false);
    if (mesh_mode == "surface_samples") {
      replaceMeshWithSurfaceSamples(cloud, submap, voxel_size, visual_color);
    } else if (mesh_mode == "projected_marching_cubes") {
      projectMeshVerticesToSurfaceSamples(
          cloud,
          submap,
          std::max(truncation_distance, surface_support_radius));
    } else if (mesh_mode != "marching_cubes") {
      throw std::runtime_error("unsupported mesh mode: " + mesh_mode);
    }

    row.submap_id = submap->getID();
    row.panoptic_label = labelToString(panoptic_label);
    row.allocated_blocks =
        static_cast<int>(submap->getTsdfLayer().getNumberOfAllocatedBlocks());
    row.mesh_vertices = countMeshVertices(*submap);
    row.decision = "materialized_base1_object_to_panmap_submap";
    created++;
    audit.push_back(row);
  }

  output.updateInstanceToSubmapIDTable();
  if (!output.saveToFile(out_map_path)) {
    std::cerr << "failed to save " << out_map_path << "\n";
    return 4;
  }

  {
    std::ofstream out(diag_dir + "/panmap_materialization_audit.csv");
    out << "label_id,name,class_id,panoptic_id,input_points,loaded_points,"
           "normal_points,submap_id,panoptic_label,voxel_size,truncation_distance,"
           "surface_support_radius,min_tsdf_support,mesh_mode,tsdf_mode,"
           "color_r,color_g,color_b,allocated_blocks,mesh_vertices,decision\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& row : audit) {
      out << row.label_id << ','
          << csvEscape(row.name) << ','
          << row.class_id << ','
          << row.panoptic_id << ','
          << row.input_points << ','
          << row.loaded_points << ','
          << row.normal_points << ','
          << row.submap_id << ','
          << row.panoptic_label << ','
          << row.voxel_size << ','
          << row.truncation_distance << ','
          << row.surface_support_radius << ','
          << row.min_tsdf_support << ','
          << row.mesh_mode << ','
          << row.tsdf_mode << ','
          << row.color_r << ','
          << row.color_g << ','
          << row.color_b << ','
          << row.allocated_blocks << ','
          << row.mesh_vertices << ','
          << row.decision << '\n';
    }
  }
  {
    std::ofstream out(diag_dir + "/panmap_materialization_summary.csv");
    out << "input_objects,created_submaps,skipped_objects,output_panmap,"
           "voxel_size,truncation_distance,surface_support_radius,min_tsdf_support,"
           "mesh_mode,tsdf_mode,method_role\n";
    out << objects.size() << ',' << created << ',' << skipped << ','
        << out_map_path << ',' << fallback_voxel_size << ','
        << fallback_truncation_distance << ','
        << fallback_surface_support_radius
        << ',' << min_tsdf_support
        << ',' << mesh_mode
        << ',' << tsdf_mode
        << ",format_backend_only\n";
  }

  std::cerr << "panmap_materializer created_submaps=" << created
            << " skipped=" << skipped << " out=" << out_map_path << "\n";
  return 0;
}
