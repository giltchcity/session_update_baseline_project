#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <khronos/spatio_temporal_map/spatio_temporal_map.h>

#include "session_update_baseline/base1/object_guided_map_reconciler.h"

namespace fs = std::filesystem;

namespace {

struct Args {
  std::string map_file;
  std::string output_dir;
  std::string mode = "cleanup";
  std::string dynamic_mode = "within_session";
  std::string map_time = "latest";
  std::string prior_map;
  std::string prior_object_memory;
  std::string object_changes_csv;
  std::string object_move_decision = "hard";
  std::string synthetic_change_file;
  double object_distance_m = 0.05;
  double bbox_margin_m = 0.05;
  double injection_min_separation_m = 0.0;
  bool temporal_background_repair = false;
  double temporal_background_min_separation_m = 0.08;
  bool temporal_object_repair = false;
  double temporal_object_min_separation_m = 0.08;
  bool horizontal_plane_fill = false;
  std::string horizontal_plane_fill_mode = "footprint";
  double horizontal_plane_grid_resolution_m = 0.08;
  std::size_t horizontal_plane_min_support_vertices = 5000;
  double object_move_skip_probability = 1.0;
  double object_move_time_scale_s = 30.0;
  double prior_match_distance_m = 0.75;
  std::size_t min_object_mesh_vertices = 20;
  double repair_global_vertex_ratio_threshold = 0.05;
  bool require_same_label = false;
  bool require_bbox_containment = true;
  bool dry_run = false;
};

bool parseBool(const std::string& value) {
  return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "on";
}

void printUsage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0 << " --map_file PATH --output_dir DIR [options]\n\n"
      << "Options:\n"
      << "  --mode no_op|audit|cleanup|injection|full default: cleanup\n"
      << "  --dynamic_mode within_session|cross_session  default: within_session\n"
      << "  --map_time latest|earliest|index:N|timestamp:NS default: latest\n"
      << "  --prior_map PATH                       optional, for cross-session mode\n"
      << "  --prior_object_memory PATH             optional, for cross-session mode\n"
      << "  --object_changes_csv PATH              optional Khronos object_changes.csv\n"
      << "  --object_move_decision hard|probability|expected_utility default: hard\n"
      << "  --synthetic_change_file PATH           optional forced-change JSON for tests\n"
      << "  --object_distance_m FLOAT               default: 0.05\n"
      << "  --bbox_margin_m FLOAT                   default: 0.05\n"
      << "  --injection_min_separation_m FLOAT      default: 0.0; <=0 uses full object append\n"
      << "  --injection_hole_radius_m FLOAT         alias for --injection_min_separation_m\n"
      << "  --temporal_background_repair true|false default: false\n"
      << "  --temporal_background_min_separation_m FLOAT default: 0.08\n"
      << "  --temporal_object_repair true|false     default: false\n"
      << "  --temporal_object_min_separation_m FLOAT default: 0.08\n"
      << "  --horizontal_plane_fill true|false      default: false\n"
      << "  --horizontal_plane_fill_mode footprint|graph_cut default: footprint\n"
      << "  --horizontal_plane_grid_resolution_m FLOAT default: 0.08\n"
      << "  --horizontal_plane_min_support_vertices INT default: 5000\n"
      << "  --object_move_skip_probability FLOAT    default: 1.0; <1 uses probabilistic object move gate\n"
      << "  --object_move_time_scale_s FLOAT        default: 30.0\n"
      << "  --prior_match_distance_m FLOAT          default: 0.75\n"
      << "  --min_object_mesh_vertices INT          default: 20\n"
      << "  --repair_global_vertex_ratio_threshold FLOAT default: 0.05\n"
      << "  --require_same_label true|false         default: false\n"
      << "  --require_bbox_containment true|false   default: true\n"
      << "  --dry_run true|false                    default: false\n";
}

Args parseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto needValue = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + name);
      }
      return argv[++i];
    };

    if (key == "--map_file") {
      args.map_file = needValue(key);
    } else if (key == "--output_dir") {
      args.output_dir = needValue(key);
    } else if (key == "--mode") {
      args.mode = needValue(key);
    } else if (key == "--dynamic_mode") {
      args.dynamic_mode = needValue(key);
    } else if (key == "--map_time") {
      args.map_time = needValue(key);
    } else if (key == "--prior_map") {
      args.prior_map = needValue(key);
    } else if (key == "--prior_object_memory") {
      args.prior_object_memory = needValue(key);
    } else if (key == "--object_changes_csv") {
      args.object_changes_csv = needValue(key);
    } else if (key == "--object_move_decision") {
      args.object_move_decision = needValue(key);
    } else if (key == "--synthetic_change_file") {
      args.synthetic_change_file = needValue(key);
    } else if (key == "--object_distance_m") {
      args.object_distance_m = std::stod(needValue(key));
    } else if (key == "--bbox_margin_m") {
      args.bbox_margin_m = std::stod(needValue(key));
    } else if (key == "--injection_min_separation_m" || key == "--injection_hole_radius_m") {
      args.injection_min_separation_m = std::stod(needValue(key));
    } else if (key == "--temporal_background_repair") {
      args.temporal_background_repair = parseBool(needValue(key));
    } else if (key == "--temporal_background_min_separation_m") {
      args.temporal_background_min_separation_m = std::stod(needValue(key));
    } else if (key == "--temporal_object_repair") {
      args.temporal_object_repair = parseBool(needValue(key));
    } else if (key == "--temporal_object_min_separation_m") {
      args.temporal_object_min_separation_m = std::stod(needValue(key));
    } else if (key == "--horizontal_plane_fill") {
      args.horizontal_plane_fill = parseBool(needValue(key));
    } else if (key == "--horizontal_plane_fill_mode") {
      args.horizontal_plane_fill_mode = needValue(key);
    } else if (key == "--horizontal_plane_grid_resolution_m") {
      args.horizontal_plane_grid_resolution_m = std::stod(needValue(key));
    } else if (key == "--horizontal_plane_min_support_vertices") {
      args.horizontal_plane_min_support_vertices =
          static_cast<std::size_t>(std::stoul(needValue(key)));
    } else if (key == "--object_move_skip_probability") {
      args.object_move_skip_probability = std::stod(needValue(key));
    } else if (key == "--object_move_time_scale_s") {
      args.object_move_time_scale_s = std::stod(needValue(key));
    } else if (key == "--prior_match_distance_m") {
      args.prior_match_distance_m = std::stod(needValue(key));
    } else if (key == "--min_object_mesh_vertices") {
      args.min_object_mesh_vertices =
          static_cast<std::size_t>(std::stoul(needValue(key)));
    } else if (key == "--repair_global_vertex_ratio_threshold") {
      args.repair_global_vertex_ratio_threshold = std::stod(needValue(key));
    } else if (key == "--require_same_label") {
      args.require_same_label = parseBool(needValue(key));
    } else if (key == "--require_bbox_containment") {
      args.require_bbox_containment = parseBool(needValue(key));
    } else if (key == "--dry_run") {
      args.dry_run = parseBool(needValue(key));
    } else if (key == "--help" || key == "-h") {
      printUsage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + key);
    }
  }

  if (args.map_file.empty()) {
    throw std::runtime_error("--map_file is required");
  }
  if (args.output_dir.empty()) {
    throw std::runtime_error("--output_dir is required");
  }
  if (args.mode != "no_op" && args.mode != "audit" && args.mode != "cleanup" &&
      args.mode != "injection" && args.mode != "full") {
    throw std::runtime_error("--mode must be no_op, audit, cleanup, injection, or full");
  }
  if (args.dynamic_mode != "within_session" && args.dynamic_mode != "cross_session") {
    throw std::runtime_error("--dynamic_mode must be within_session or cross_session");
  }
  if (args.object_move_decision != "hard" && args.object_move_decision != "probability" &&
      args.object_move_decision != "expected_utility") {
    throw std::runtime_error(
        "--object_move_decision must be hard, probability, or expected_utility");
  }
  if (args.horizontal_plane_fill_mode != "footprint" &&
      args.horizontal_plane_fill_mode != "graph_cut") {
    throw std::runtime_error("--horizontal_plane_fill_mode must be footprint or graph_cut");
  }
  return args;
}

khronos::TimeStamp selectMapTime(const khronos::SpatioTemporalMap& map,
                                 const std::string& selector) {
  const auto& stamps = map.stamps();
  if (stamps.empty()) {
    throw std::runtime_error("Input map has no timestamps");
  }
  if (selector == "latest") {
    return map.latest();
  }
  if (selector == "earliest") {
    return map.earliest();
  }
  const std::string index_prefix = "index:";
  if (selector.rfind(index_prefix, 0) == 0) {
    const auto index = static_cast<std::size_t>(
        std::stoul(selector.substr(index_prefix.size())));
    if (index >= stamps.size()) {
      throw std::runtime_error("--map_time index out of range");
    }
    return stamps[index];
  }
  const std::string timestamp_prefix = "timestamp:";
  if (selector.rfind(timestamp_prefix, 0) == 0) {
    return static_cast<khronos::TimeStamp>(
        std::stoull(selector.substr(timestamp_prefix.size())));
  }
  throw std::runtime_error("Unsupported --map_time selector: " + selector);
}

void writeCommand(const fs::path& output_dir, int argc, char** argv) {
  std::ofstream out(output_dir / "command.txt");
  for (int i = 0; i < argc; ++i) {
    if (i > 0) {
      out << " ";
    }
    out << argv[i];
  }
  out << "\n";
}

void writeConfig(const fs::path& output_dir, const Args& args) {
  std::ofstream out(output_dir / "config.yaml");
  out << "map_file: \"" << args.map_file << "\"\n";
  out << "output_dir: \"" << args.output_dir << "\"\n";
  out << "mode: \"" << args.mode << "\"\n";
  out << "dynamic_mode: \"" << args.dynamic_mode << "\"\n";
  out << "map_time: \"" << args.map_time << "\"\n";
  out << "prior_map: \"" << args.prior_map << "\"\n";
  out << "prior_object_memory: \"" << args.prior_object_memory << "\"\n";
  out << "object_changes_csv: \"" << args.object_changes_csv << "\"\n";
  out << "object_move_decision: \"" << args.object_move_decision << "\"\n";
  out << "synthetic_change_file: \"" << args.synthetic_change_file << "\"\n";
  out << "object_distance_m: " << args.object_distance_m << "\n";
  out << "bbox_margin_m: " << args.bbox_margin_m << "\n";
  out << "injection_min_separation_m: " << args.injection_min_separation_m << "\n";
  out << "injection_hole_radius_m: " << args.injection_min_separation_m << "\n";
  out << "temporal_background_repair: "
      << (args.temporal_background_repair ? "true" : "false") << "\n";
  out << "temporal_background_min_separation_m: "
      << args.temporal_background_min_separation_m << "\n";
  out << "temporal_object_repair: "
      << (args.temporal_object_repair ? "true" : "false") << "\n";
  out << "temporal_object_min_separation_m: "
      << args.temporal_object_min_separation_m << "\n";
  out << "horizontal_plane_fill: "
      << (args.horizontal_plane_fill ? "true" : "false") << "\n";
  out << "horizontal_plane_fill_mode: \"" << args.horizontal_plane_fill_mode << "\"\n";
  out << "horizontal_plane_grid_resolution_m: "
      << args.horizontal_plane_grid_resolution_m << "\n";
  out << "horizontal_plane_min_support_vertices: "
      << args.horizontal_plane_min_support_vertices << "\n";
  out << "object_move_skip_probability: " << args.object_move_skip_probability << "\n";
  out << "object_move_time_scale_s: " << args.object_move_time_scale_s << "\n";
  out << "prior_match_distance_m: " << args.prior_match_distance_m << "\n";
  out << "min_object_mesh_vertices: " << args.min_object_mesh_vertices << "\n";
  out << "repair_global_vertex_ratio_threshold: " << args.repair_global_vertex_ratio_threshold
      << "\n";
  out << "require_same_label: " << (args.require_same_label ? "true" : "false") << "\n";
  out << "require_bbox_containment: " << (args.require_bbox_containment ? "true" : "false")
      << "\n";
  out << "dry_run: " << (args.dry_run ? "true" : "false") << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  try {
    const Args args = parseArgs(argc, argv);
    const fs::path output_dir(args.output_dir);
    fs::create_directories(output_dir);
    writeCommand(output_dir, argc, argv);
    writeConfig(output_dir, args);

    auto map = khronos::SpatioTemporalMap::load(args.map_file);
    if (!map) {
      std::cerr << "Failed to load map: " << args.map_file << "\n";
      return 2;
    }

    const auto selected_stamp = selectMapTime(*map, args.map_time);

    bool prior_map_loaded = false;
    std::size_t prior_memory_objects = 0;
    if (!args.prior_map.empty()) {
      auto prior_map = khronos::SpatioTemporalMap::load(args.prior_map);
      if (!prior_map) {
        std::cerr << "Failed to load prior map: " << args.prior_map << "\n";
        return 6;
      }
      prior_map_loaded = true;
    }
    prior_memory_objects =
        session_update::base1::countPriorMemoryObjects(args.prior_object_memory);

    auto dsg = map->getDsgPtr(selected_stamp);
    if (!dsg) {
      std::cerr << "Failed to extract latest DSG from map: " << args.map_file << "\n";
      return 3;
    }

    const fs::path original_map = output_dir / "original_final.4dmap";
    if (!session_update::base1::saveMapWithSingleDsg(dsg, selected_stamp, original_map.string())) {
      std::cerr << "Failed to save original map copy: " << original_map << "\n";
      return 4;
    }

    session_update::base1::ReconcilerConfig config;
    config.mode = args.mode;
    config.dynamic_mode = args.dynamic_mode;
    config.prior_map = args.prior_map;
    config.prior_object_memory = args.prior_object_memory;
    config.object_changes_csv = args.object_changes_csv;
    config.object_move_decision = args.object_move_decision;
    config.synthetic_change_file = args.synthetic_change_file;
    config.object_distance_m = args.object_distance_m;
    config.bbox_margin_m = args.bbox_margin_m;
    config.injection_min_separation_m = args.injection_min_separation_m;
    config.temporal_background_repair = args.temporal_background_repair;
    config.temporal_background_min_separation_m =
        args.temporal_background_min_separation_m;
    config.temporal_object_repair = args.temporal_object_repair;
    config.temporal_object_min_separation_m = args.temporal_object_min_separation_m;
    config.horizontal_plane_fill = args.horizontal_plane_fill;
    config.horizontal_plane_fill_mode = args.horizontal_plane_fill_mode;
    config.horizontal_plane_grid_resolution_m = args.horizontal_plane_grid_resolution_m;
    config.horizontal_plane_min_support_vertices =
        args.horizontal_plane_min_support_vertices;
    config.object_move_skip_probability = args.object_move_skip_probability;
    config.object_move_time_scale_s = args.object_move_time_scale_s;
    config.prior_match_distance_m = args.prior_match_distance_m;
    config.min_object_mesh_vertices = args.min_object_mesh_vertices;
    config.repair_global_vertex_ratio_threshold = args.repair_global_vertex_ratio_threshold;
    config.require_same_label = args.require_same_label;
    config.require_bbox_containment = args.require_bbox_containment;
    config.dry_run = args.dry_run;
    if (args.temporal_background_repair || args.temporal_object_repair) {
      for (const auto stamp : map->stamps()) {
        if (stamp >= selected_stamp) {
          continue;
        }
        auto prior_dsg = map->getDsgPtr(stamp);
        if (prior_dsg) {
          config.temporal_background_dsgs.push_back(prior_dsg);
        }
      }
    }

    session_update::base1::ObjectGuidedMapReconciler reconciler(config);
    auto result = reconciler.reconcile(*dsg);

    const fs::path improved_map = output_dir / "improved_final.4dmap";
    if (!session_update::base1::saveMapWithSingleDsg(dsg, selected_stamp, improved_map.string())) {
      std::cerr << "Failed to save improved map: " << improved_map << "\n";
      return 5;
    }

    session_update::base1::writeObjectAuditCsv((output_dir / "object_audit.csv").string(),
                                               result.object_rows);
    session_update::base1::writeMeshUpdateCsv((output_dir / "mesh_update_summary.csv").string(),
                                              result.mesh_summary);
    session_update::base1::writeObjectUpdateCsv((output_dir / "object_update_summary.csv").string(),
                                                result.object_rows);
    session_update::base1::writeMeshVertexUpdateCsv(
        (output_dir / "mesh_vertex_update_summary.csv").string(), result.vertex_update_rows);
    session_update::base1::writeEvidenceSummaryJson(
        (output_dir / "evidence_summary.json").string(),
        config,
        result.mesh_summary,
        prior_map_loaded,
        prior_memory_objects);
    session_update::base1::writeObjectMemoryJson((output_dir / "object_memory.json").string(),
                                                 result.object_rows);

    std::cout << "BASE1_DONE\n";
    std::cout << "original_final=" << original_map << "\n";
    std::cout << "improved_final=" << improved_map << "\n";
    std::cout << "mode=" << args.mode << "\n";
    std::cout << "dynamic_mode=" << args.dynamic_mode << "\n";
    std::cout << "map_time=" << args.map_time << "\n";
    std::cout << "selected_stamp=" << selected_stamp << "\n";
    std::cout << "prior_map_loaded=" << (prior_map_loaded ? "true" : "false") << "\n";
    std::cout << "prior_memory_objects=" << prior_memory_objects << "\n";
    std::cout << "synthetic_change_file=" << args.synthetic_change_file << "\n";
    std::cout << "initial_vertices=" << result.mesh_summary.initial_vertices << "\n";
    std::cout << "final_vertices=" << result.mesh_summary.final_vertices << "\n";
    std::cout << "removed_vertices=" << result.mesh_summary.removed_vertices << "\n";
    std::cout << "objects_total=" << result.mesh_summary.objects_total << "\n";
    std::cout << "objects_used_for_cleanup=" << result.mesh_summary.objects_used_for_cleanup
              << "\n";
    std::cout << "cleanup_source_points=" << result.mesh_summary.cleanup_source_points << "\n";
    std::cout << "injected_objects=" << result.mesh_summary.injected_objects << "\n";
    std::cout << "injected_vertices=" << result.mesh_summary.injected_vertices << "\n";
    std::cout << "injected_faces=" << result.mesh_summary.injected_faces << "\n";
    std::cout << "temporal_background_sources="
              << result.mesh_summary.temporal_background_sources << "\n";
    std::cout << "temporal_background_injected_vertices="
              << result.mesh_summary.temporal_background_injected_vertices << "\n";
    std::cout << "temporal_background_injected_faces="
              << result.mesh_summary.temporal_background_injected_faces << "\n";
    std::cout << "temporal_object_sources="
              << result.mesh_summary.temporal_object_sources << "\n";
    std::cout << "temporal_object_injected_vertices="
              << result.mesh_summary.temporal_object_injected_vertices << "\n";
    std::cout << "temporal_object_injected_faces="
              << result.mesh_summary.temporal_object_injected_faces << "\n";
    std::cout << "horizontal_planes_filled="
              << result.mesh_summary.horizontal_planes_filled << "\n";
    std::cout << "horizontal_plane_vertices="
              << result.mesh_summary.horizontal_plane_vertices << "\n";
    std::cout << "horizontal_plane_faces="
              << result.mesh_summary.horizontal_plane_faces << "\n";
    std::cout << "horizontal_plane_graph_cut_cells="
              << result.mesh_summary.horizontal_plane_graph_cut_cells << "\n";
    std::cout << "horizontal_plane_graph_cut_fill_cells="
              << result.mesh_summary.horizontal_plane_graph_cut_fill_cells << "\n";
    std::cout << "prior_memory_objects_used_for_matching="
              << result.mesh_summary.prior_memory_objects << "\n";
    std::cout << "prior_matched_objects=" << result.mesh_summary.prior_matched_objects << "\n";
    std::cout << "prior_unmatched_current_objects="
              << result.mesh_summary.prior_unmatched_current_objects << "\n";
    std::cout << "forced_absent_prior_objects="
              << result.mesh_summary.forced_absent_prior_objects << "\n";
    std::cout << "forced_absent_vertices_removed="
              << result.mesh_summary.forced_absent_vertices_removed << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    printUsage(argv[0]);
    return 1;
  }
}
