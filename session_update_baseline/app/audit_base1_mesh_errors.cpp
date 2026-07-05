#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <hydra/utils/nearest_neighbor_utilities.h>
#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace fs = std::filesystem;

namespace {

struct Args {
  std::string map_file;
  std::string gt_background_ply;
  std::string output_dir;
  std::string map_time = "latest";
};

struct ObjectBox {
  khronos::NodeId id = 0;
  int label = -1;
  khronos::BoundingBox box;
  std::size_t mesh_vertices = 0;
};

struct Counts {
  std::uint64_t inliers = 0;
  std::uint64_t outliers = 0;
  std::uint64_t outliers_inside_object_bbox = 0;
  std::uint64_t outliers_outside_object_bbox = 0;
};

bool insideBox(const khronos::BoundingBox& box, const khronos::Point& point) {
  const auto min_corner = box.world_P_center - box.dimensions * 0.5f;
  const auto max_corner = box.world_P_center + box.dimensions * 0.5f;
  return point.x() >= min_corner.x() && point.x() <= max_corner.x() &&
         point.y() >= min_corner.y() && point.y() <= max_corner.y() &&
         point.z() >= min_corner.z() && point.z() <= max_corner.z();
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
    } else if (key == "--gt_background_ply") {
      args.gt_background_ply = value();
    } else if (key == "--output_dir") {
      args.output_dir = value();
    } else if (key == "--map_time") {
      args.map_time = value();
    } else {
      throw std::runtime_error("Unknown argument: " + key);
    }
  }
  if (args.map_file.empty() || args.gt_background_ply.empty() || args.output_dir.empty()) {
    throw std::runtime_error("--map_file, --gt_background_ply, and --output_dir are required");
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

std::vector<khronos::Point> loadPlyVertices(const std::string& path) {
  pcl::PLYReader reader;
  pcl::PolygonMesh mesh;
  if (reader.read(path, mesh) != 0) {
    throw std::runtime_error("Failed to read PLY: " + path);
  }
  pcl::PointCloud<pcl::PointXYZ> cloud;
  pcl::fromPCLPointCloud2(mesh.cloud, cloud);
  std::vector<khronos::Point> points;
  points.reserve(cloud.size());
  for (const auto& point : cloud) {
    points.emplace_back(point.x, point.y, point.z);
  }
  return points;
}

std::vector<ObjectBox> collectObjectBoxes(const khronos::DynamicSceneGraph& dsg) {
  std::vector<ObjectBox> boxes;
  if (!dsg.hasLayer(khronos::DsgLayers::OBJECTS)) {
    return boxes;
  }
  for (const auto& [id, node] : dsg.getLayer(khronos::DsgLayers::OBJECTS).nodes()) {
    const auto& attrs = node->attributes<khronos::KhronosObjectAttributes>();
    boxes.push_back(ObjectBox{id, attrs.semantic_label, attrs.bounding_box, attrs.mesh.numVertices()});
  }
  return boxes;
}

std::size_t containingObject(const std::vector<ObjectBox>& boxes, const khronos::Point& point) {
  for (std::size_t i = 0; i < boxes.size(); ++i) {
    if (insideBox(boxes[i].box, point)) {
      return i;
    }
  }
  return std::numeric_limits<std::size_t>::max();
}

void writeSummary(const fs::path& path,
                  const std::vector<double>& thresholds,
                  const std::vector<Counts>& gt_counts,
                  const std::vector<Counts>& dsg_counts) {
  std::ofstream out(path);
  out << "threshold,precision,recall,gt_inliers,gt_outliers,dsg_inliers,dsg_outliers,"
         "gt_outliers_inside_object_bbox,gt_outliers_outside_object_bbox,"
         "dsg_outliers_inside_object_bbox,dsg_outliers_outside_object_bbox\n";
  for (std::size_t i = 0; i < thresholds.size(); ++i) {
    const double precision =
        static_cast<double>(dsg_counts[i].inliers) /
        static_cast<double>(dsg_counts[i].inliers + dsg_counts[i].outliers);
    const double recall =
        static_cast<double>(gt_counts[i].inliers) /
        static_cast<double>(gt_counts[i].inliers + gt_counts[i].outliers);
    out << thresholds[i] << "," << precision << "," << recall << ","
        << gt_counts[i].inliers << "," << gt_counts[i].outliers << ","
        << dsg_counts[i].inliers << "," << dsg_counts[i].outliers << ","
        << gt_counts[i].outliers_inside_object_bbox << ","
        << gt_counts[i].outliers_outside_object_bbox << ","
        << dsg_counts[i].outliers_inside_object_bbox << ","
        << dsg_counts[i].outliers_outside_object_bbox << "\n";
  }
}

void writeObjectBuckets(
    const fs::path& path,
    const std::vector<double>& thresholds,
    const std::vector<ObjectBox>& boxes,
    const std::vector<std::unordered_map<khronos::NodeId, std::uint64_t>>& gt_by_object,
    const std::vector<std::unordered_map<khronos::NodeId, std::uint64_t>>& dsg_by_object) {
  std::ofstream out(path);
  out << "threshold,side,object_id,label,count,object_mesh_vertices,bbox_cx,bbox_cy,bbox_cz,"
         "bbox_dx,bbox_dy,bbox_dz\n";
  std::unordered_map<khronos::NodeId, const ObjectBox*> by_id;
  for (const auto& box : boxes) {
    by_id[box.id] = &box;
  }
  auto write = [&](double threshold,
                   const std::string& side,
                   const std::unordered_map<khronos::NodeId, std::uint64_t>& counts) {
    for (const auto& [id, count] : counts) {
      const auto it = by_id.find(id);
      if (it == by_id.end()) {
        continue;
      }
      const auto& box = *it->second;
      out << threshold << "," << side << "," << id << "," << box.label << "," << count << ","
          << box.mesh_vertices << "," << box.box.world_P_center.x() << ","
          << box.box.world_P_center.y() << "," << box.box.world_P_center.z() << ","
          << box.box.dimensions.x() << "," << box.box.dimensions.y() << ","
          << box.box.dimensions.z() << "\n";
    }
  };
  for (std::size_t i = 0; i < thresholds.size(); ++i) {
    write(thresholds[i], "gt_fn", gt_by_object[i]);
    write(thresholds[i], "dsg_fp", dsg_by_object[i]);
  }
}

void writeAxisBins(const fs::path& path,
                   const std::vector<khronos::Point>& points,
                   double resolution_m) {
  std::array<std::map<int, std::uint64_t>, 3> bins;
  for (const auto& point : points) {
    bins[0][static_cast<int>(std::llround(point.x() / resolution_m))]++;
    bins[1][static_cast<int>(std::llround(point.y() / resolution_m))]++;
    bins[2][static_cast<int>(std::llround(point.z() / resolution_m))]++;
  }
  std::ofstream out(path);
  out << "axis,bin,coord,count\n";
  const std::array<std::string, 3> names{"x", "y", "z"};
  for (std::size_t axis = 0; axis < bins.size(); ++axis) {
    std::vector<std::pair<int, std::uint64_t>> sorted(bins[axis].begin(), bins[axis].end());
    std::sort(sorted.begin(), sorted.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.second > rhs.second;
    });
    const std::size_t limit = std::min<std::size_t>(sorted.size(), 40);
    for (std::size_t i = 0; i < limit; ++i) {
      out << names[axis] << "," << sorted[i].first << ","
          << sorted[i].first * resolution_m << "," << sorted[i].second << "\n";
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parseArgs(argc, argv);
    fs::create_directories(args.output_dir);

    auto map = khronos::SpatioTemporalMap::load(args.map_file);
    if (!map) {
      throw std::runtime_error("Failed to load map: " + args.map_file);
    }
    const auto stamp = selectMapTime(*map, args.map_time);
    auto dsg = map->getDsgPtr(stamp);
    if (!dsg || !dsg->hasMesh()) {
      throw std::runtime_error("Selected DSG has no mesh");
    }

    std::vector<khronos::Point> dsg_points;
    dsg_points.reserve(dsg->mesh()->numVertices());
    for (std::size_t i = 0; i < dsg->mesh()->numVertices(); ++i) {
      dsg_points.push_back(dsg->mesh()->pos(i));
    }
    const auto gt_points = loadPlyVertices(args.gt_background_ply);
    const auto boxes = collectObjectBoxes(*dsg);

    hydra::PointNeighborSearch dsg_search(dsg_points);
    hydra::PointNeighborSearch gt_search(gt_points);
    const std::vector<double> thresholds{0.05, 0.10, 0.20};
    std::vector<double> threshold_sq;
    for (const auto threshold : thresholds) {
      threshold_sq.push_back(threshold * threshold);
    }

    std::vector<Counts> gt_counts(thresholds.size());
    std::vector<Counts> dsg_counts(thresholds.size());
    std::vector<std::unordered_map<khronos::NodeId, std::uint64_t>> gt_by_object(thresholds.size());
    std::vector<std::unordered_map<khronos::NodeId, std::uint64_t>> dsg_by_object(thresholds.size());

    for (const auto& point : gt_points) {
      float distance_sq = std::numeric_limits<float>::max();
      std::size_t nearest = 0;
      dsg_search.search(point, distance_sq, nearest);
      const auto box_idx = containingObject(boxes, point);
      for (std::size_t i = 0; i < thresholds.size(); ++i) {
        if (distance_sq <= threshold_sq[i]) {
          ++gt_counts[i].inliers;
        } else {
          ++gt_counts[i].outliers;
          if (box_idx == std::numeric_limits<std::size_t>::max()) {
            ++gt_counts[i].outliers_outside_object_bbox;
          } else {
            ++gt_counts[i].outliers_inside_object_bbox;
            gt_by_object[i][boxes[box_idx].id]++;
          }
        }
      }
    }

    for (const auto& point : dsg_points) {
      float distance_sq = std::numeric_limits<float>::max();
      std::size_t nearest = 0;
      gt_search.search(point, distance_sq, nearest);
      const auto box_idx = containingObject(boxes, point);
      for (std::size_t i = 0; i < thresholds.size(); ++i) {
        if (distance_sq <= threshold_sq[i]) {
          ++dsg_counts[i].inliers;
        } else {
          ++dsg_counts[i].outliers;
          if (box_idx == std::numeric_limits<std::size_t>::max()) {
            ++dsg_counts[i].outliers_outside_object_bbox;
          } else {
            ++dsg_counts[i].outliers_inside_object_bbox;
            dsg_by_object[i][boxes[box_idx].id]++;
          }
        }
      }
    }

    writeSummary(fs::path(args.output_dir) / "mesh_error_summary.csv",
                 thresholds,
                 gt_counts,
                 dsg_counts);
    writeObjectBuckets(fs::path(args.output_dir) / "mesh_error_object_buckets.csv",
                       thresholds,
                       boxes,
                       gt_by_object,
                       dsg_by_object);
    writeAxisBins(fs::path(args.output_dir) / "dsg_axis_bins_8cm.csv", dsg_points, 0.08);
    writeAxisBins(fs::path(args.output_dir) / "gt_axis_bins_8cm.csv", gt_points, 0.08);
    std::cout << "AUDIT_DONE\n";
    std::cout << "dsg_vertices=" << dsg_points.size() << "\n";
    std::cout << "gt_vertices=" << gt_points.size() << "\n";
    std::cout << "objects=" << boxes.size() << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
