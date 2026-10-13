#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <unordered_set>
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
  std::string object_injection_policy = "all";
  double object_surface_support_distance_m = 0.10;
  std::string object_alignment_policy = "none";
  double object_alignment_support_distance_m = 0.20;
  double object_alignment_max_translation_m = 0.08;
  std::size_t object_alignment_min_support_vertices = 50;
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
  std::string horizontal_plane_candidate_policy = "extreme_pair";
  int horizontal_plane_support_band_cells = -1;
  bool axis_aligned_plane_fill = false;
  int axis_aligned_plane_support_band_cells = -1;
  std::string axis_aligned_plane_candidate_policy = "boundary";
  bool structural_plane_visibility_filter = false;
  std::string structural_plane_visibility_scope = "all";
  bool structural_plane_output_supersample = false;
  std::string structural_plane_output_supersample_scope = "all";
  double horizontal_plane_grid_resolution_m = 0.08;
  std::size_t horizontal_plane_min_support_vertices = 5000;
  bool free_space_culling = false;
  std::string free_space_culling_scope = "added";
  std::string free_space_culling_decision = "no_present";
  double free_space_culling_block_size_m = 0.5;
  double free_space_culling_radial_tolerance_m = 0.08;
  double free_space_culling_depth_tolerance_m = 0.30;
  double free_space_culling_active_window_duration_s = 3.0;
  std::size_t free_space_culling_min_absent = 1;
  std::size_t free_space_culling_max_present = 0;
  bool dynamic_residue_cleanup = false;
  double dynamic_residue_support_distance_m = 0.10;
  std::size_t dynamic_residue_min_track_frames = 15;
  bool dynamic_residue_require_time_overlap = true;
  std::string dynamic_residue_decision = "no_present";
  std::string dynamic_residue_unobserved_policy = "keep";
  std::size_t dynamic_residue_min_absent = 1;
  std::size_t dynamic_residue_max_present = 0;
  std::string dynamic_residue_protected_labels = "";
  bool dynamic_residue_protect_horizontal_surfaces = true;
  double dynamic_residue_horizontal_normal_z_min = 0.85;
  bool volume_graph_cut_fill = false;
  std::string volume_graph_cut_surface_policy = "all_boundaries";
  double volume_graph_cut_resolution_m = 0.16;
  std::size_t volume_graph_cut_max_cells = 350000;
  bool cross_session_remove_absent_prior = true;
  double cross_session_mesh_merge_distance_m = 0.08;
  bool object_cleanup_reobservation_gate = false;
  double object_cleanup_support_distance_m = 0.08;
  double object_move_skip_probability = 1.0;
  double object_move_time_scale_s = 30.0;
  double prior_match_distance_m = 0.75;
  std::size_t min_object_mesh_vertices = 20;
  double repair_global_vertex_ratio_threshold = 0.05;
  bool require_same_label = false;
  bool require_bbox_containment = true;
  bool save_original_copy = true;
  bool dry_run = false;
};

bool parseBool(const std::string& value) {
  return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "on";
}

std::unordered_set<int> parseIntSet(const std::string& value) {
  std::unordered_set<int> result;
  std::stringstream stream(value);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (!token.empty()) {
      result.insert(std::stoi(token));
    }
  }
  return result;
}

void printUsage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0 << " --map_file PATH --output_dir DIR [options]\n\n"
      << "Options:\n"
      << "  --mode no_op|audit|cleanup|dynamic_cleanup|injection|full default: cleanup\n"
      << "  --dynamic_mode within_session|cross_session  default: within_session\n"
      << "  --map_time latest|earliest|index:N|timestamp:NS default: latest\n"
      << "  --prior_map PATH                       optional prior map for diagnostics\n"
      << "  --prior_object_memory PATH             optional rolling Base1 object memory\n"
      << "  --object_changes_csv PATH              optional Khronos object_changes.csv\n"
      << "  --object_move_decision hard|probability|expected_utility default: hard\n"
      << "  --object_injection_policy all|repair_candidates|repair_or_supported default: all\n"
      << "  --object_surface_support_distance_m FLOAT default: 0.10\n"
      << "  --object_alignment_policy none|translation default: none\n"
      << "  --object_alignment_support_distance_m FLOAT default: 0.20\n"
      << "  --object_alignment_max_translation_m FLOAT default: 0.08\n"
      << "  --object_alignment_min_support_vertices INT default: 50\n"
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
      << "  --horizontal_plane_candidate_policy extreme_pair|dominant_all|upper_band default: extreme_pair\n"
      << "  --horizontal_plane_support_band_cells INT default: -1; <0 disables horizontal support band\n"
      << "  --axis_aligned_plane_fill true|false    default: false; graph-cut outer walls\n"
      << "  --axis_aligned_plane_support_band_cells INT default: -1; <0 disables wall band\n"
      << "  --axis_aligned_plane_candidate_policy boundary|strong_all default: boundary\n"
      << "  --structural_plane_visibility_filter true|false default: false; add ray/object unary to plane fill\n"
      << "  --structural_plane_visibility_scope all|vertical default: all\n"
      << "  --structural_plane_output_supersample true|false default: false; add 4 sub-cell plane samples\n"
      << "  --structural_plane_output_supersample_scope all|observed|support default: all\n"
      << "  --horizontal_plane_grid_resolution_m FLOAT default: 0.08\n"
      << "  --horizontal_plane_min_support_vertices INT default: 5000\n"
      << "  --free_space_culling true|false       default: false; cull added points contradicted by rays\n"
      << "  --free_space_culling_scope added|objects default: added\n"
      << "  --free_space_culling_decision no_present|absence_majority default: no_present\n"
      << "  --free_space_culling_block_size_m FLOAT default: 0.5\n"
      << "  --free_space_culling_radial_tolerance_m FLOAT default: 0.08\n"
      << "  --free_space_culling_depth_tolerance_m FLOAT default: 0.30\n"
      << "  --free_space_culling_active_window_duration_s FLOAT default: 3.0\n"
      << "  --free_space_culling_min_absent INT   default: 1\n"
      << "  --free_space_culling_max_present INT  default: 0\n"
      << "  --dynamic_residue_cleanup true|false default: false\n"
      << "  --dynamic_residue_support_distance_m FLOAT default: 0.10\n"
      << "  --dynamic_residue_min_track_frames INT default: 15\n"
      << "  --dynamic_residue_require_time_overlap true|false default: true\n"
      << "  --dynamic_residue_decision no_present|absence_majority default: no_present\n"
      << "  --dynamic_residue_unobserved_policy keep|quarantine default: keep\n"
      << "  --dynamic_residue_min_absent INT default: 1\n"
      << "  --dynamic_residue_max_present INT default: 0\n"
      << "  --dynamic_residue_protected_labels CSV default: empty\n"
      << "  --dynamic_residue_protect_horizontal_surfaces true|false default: true\n"
      << "  --dynamic_residue_horizontal_normal_z_min FLOAT default: 0.85\n"
      << "  --volume_graph_cut_fill true|false    default: false; 3D free/full voxel graph cut\n"
      << "  --volume_graph_cut_surface_policy all_boundaries|structural_boundaries|structural_plane_snap_boundaries|marching_cubes|structural_marching_cubes|structural_binary_marching_cubes default: all_boundaries\n"
      << "  --volume_graph_cut_resolution_m FLOAT default: 0.16\n"
      << "  --volume_graph_cut_max_cells INT      default: 350000\n"
      << "  --cross_session_remove_absent_prior true|false default: true\n"
      << "  --cross_session_mesh_merge_distance_m FLOAT default: 0.08\n"
      << "  --object_cleanup_reobservation_gate true|false default: false\n"
      << "  --object_cleanup_support_distance_m FLOAT default: 0.08\n"
      << "  --object_move_skip_probability FLOAT    default: 1.0; <1 uses probabilistic object move gate\n"
      << "  --object_move_time_scale_s FLOAT        default: 30.0\n"
      << "  --prior_match_distance_m FLOAT          default: 0.75\n"
      << "  --min_object_mesh_vertices INT          default: 20\n"
      << "  --repair_global_vertex_ratio_threshold FLOAT default: 0.05\n"
      << "  --require_same_label true|false         default: false\n"
      << "  --require_bbox_containment true|false   default: true\n"
      << "  --save_original_copy true|false         default: true\n"
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
    } else if (key == "--object_injection_policy") {
      args.object_injection_policy = needValue(key);
    } else if (key == "--object_surface_support_distance_m") {
      args.object_surface_support_distance_m = std::stod(needValue(key));
    } else if (key == "--object_alignment_policy") {
      args.object_alignment_policy = needValue(key);
    } else if (key == "--object_alignment_support_distance_m") {
      args.object_alignment_support_distance_m = std::stod(needValue(key));
    } else if (key == "--object_alignment_max_translation_m") {
      args.object_alignment_max_translation_m = std::stod(needValue(key));
    } else if (key == "--object_alignment_min_support_vertices") {
      args.object_alignment_min_support_vertices =
          static_cast<std::size_t>(std::stoul(needValue(key)));
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
    } else if (key == "--horizontal_plane_candidate_policy") {
      args.horizontal_plane_candidate_policy = needValue(key);
    } else if (key == "--horizontal_plane_support_band_cells") {
      args.horizontal_plane_support_band_cells = std::stoi(needValue(key));
    } else if (key == "--axis_aligned_plane_fill") {
      args.axis_aligned_plane_fill = parseBool(needValue(key));
    } else if (key == "--axis_aligned_plane_support_band_cells") {
      args.axis_aligned_plane_support_band_cells = std::stoi(needValue(key));
    } else if (key == "--axis_aligned_plane_candidate_policy") {
      args.axis_aligned_plane_candidate_policy = needValue(key);
    } else if (key == "--structural_plane_visibility_filter") {
      args.structural_plane_visibility_filter = parseBool(needValue(key));
    } else if (key == "--structural_plane_visibility_scope") {
      args.structural_plane_visibility_scope = needValue(key);
    } else if (key == "--structural_plane_output_supersample") {
      args.structural_plane_output_supersample = parseBool(needValue(key));
    } else if (key == "--structural_plane_output_supersample_scope") {
      args.structural_plane_output_supersample_scope = needValue(key);
    } else if (key == "--horizontal_plane_grid_resolution_m") {
      args.horizontal_plane_grid_resolution_m = std::stod(needValue(key));
    } else if (key == "--horizontal_plane_min_support_vertices") {
      args.horizontal_plane_min_support_vertices =
          static_cast<std::size_t>(std::stoul(needValue(key)));
    } else if (key == "--free_space_culling") {
      args.free_space_culling = parseBool(needValue(key));
    } else if (key == "--free_space_culling_scope") {
      args.free_space_culling_scope = needValue(key);
    } else if (key == "--free_space_culling_decision") {
      args.free_space_culling_decision = needValue(key);
    } else if (key == "--free_space_culling_block_size_m") {
      args.free_space_culling_block_size_m = std::stod(needValue(key));
    } else if (key == "--free_space_culling_radial_tolerance_m") {
      args.free_space_culling_radial_tolerance_m = std::stod(needValue(key));
    } else if (key == "--free_space_culling_depth_tolerance_m") {
      args.free_space_culling_depth_tolerance_m = std::stod(needValue(key));
    } else if (key == "--free_space_culling_active_window_duration_s") {
      args.free_space_culling_active_window_duration_s = std::stod(needValue(key));
    } else if (key == "--free_space_culling_min_absent") {
      args.free_space_culling_min_absent =
          static_cast<std::size_t>(std::stoul(needValue(key)));
    } else if (key == "--free_space_culling_max_present") {
      args.free_space_culling_max_present =
          static_cast<std::size_t>(std::stoul(needValue(key)));
    } else if (key == "--dynamic_residue_cleanup") {
      args.dynamic_residue_cleanup = parseBool(needValue(key));
    } else if (key == "--dynamic_residue_support_distance_m") {
      args.dynamic_residue_support_distance_m = std::stod(needValue(key));
    } else if (key == "--dynamic_residue_min_track_frames") {
      args.dynamic_residue_min_track_frames =
          static_cast<std::size_t>(std::stoul(needValue(key)));
    } else if (key == "--dynamic_residue_require_time_overlap") {
      args.dynamic_residue_require_time_overlap = parseBool(needValue(key));
    } else if (key == "--dynamic_residue_decision") {
      args.dynamic_residue_decision = needValue(key);
    } else if (key == "--dynamic_residue_unobserved_policy") {
      args.dynamic_residue_unobserved_policy = needValue(key);
    } else if (key == "--dynamic_residue_min_absent") {
      args.dynamic_residue_min_absent =
          static_cast<std::size_t>(std::stoul(needValue(key)));
    } else if (key == "--dynamic_residue_max_present") {
      args.dynamic_residue_max_present =
          static_cast<std::size_t>(std::stoul(needValue(key)));
    } else if (key == "--dynamic_residue_protected_labels") {
      args.dynamic_residue_protected_labels = needValue(key);
    } else if (key == "--dynamic_residue_protect_horizontal_surfaces") {
      args.dynamic_residue_protect_horizontal_surfaces =
          parseBool(needValue(key));
    } else if (key == "--dynamic_residue_horizontal_normal_z_min") {
      args.dynamic_residue_horizontal_normal_z_min =
          std::stod(needValue(key));
    } else if (key == "--volume_graph_cut_fill") {
      args.volume_graph_cut_fill = parseBool(needValue(key));
    } else if (key == "--volume_graph_cut_surface_policy") {
      args.volume_graph_cut_surface_policy = needValue(key);
    } else if (key == "--volume_graph_cut_resolution_m") {
      args.volume_graph_cut_resolution_m = std::stod(needValue(key));
    } else if (key == "--volume_graph_cut_max_cells") {
      args.volume_graph_cut_max_cells =
          static_cast<std::size_t>(std::stoul(needValue(key)));
    } else if (key == "--cross_session_remove_absent_prior") {
      args.cross_session_remove_absent_prior = parseBool(needValue(key));
    } else if (key == "--cross_session_mesh_merge_distance_m") {
      args.cross_session_mesh_merge_distance_m = std::stod(needValue(key));
    } else if (key == "--object_cleanup_reobservation_gate") {
      args.object_cleanup_reobservation_gate = parseBool(needValue(key));
    } else if (key == "--object_cleanup_support_distance_m") {
      args.object_cleanup_support_distance_m = std::stod(needValue(key));
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
    } else if (key == "--save_original_copy") {
      args.save_original_copy = parseBool(needValue(key));
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
      args.mode != "dynamic_cleanup" && args.mode != "injection" && args.mode != "full") {
    throw std::runtime_error(
        "--mode must be no_op, audit, cleanup, dynamic_cleanup, injection, or full");
  }
  if (args.dynamic_mode != "within_session" && args.dynamic_mode != "cross_session") {
    throw std::runtime_error("--dynamic_mode must be within_session or cross_session");
  }
  if (args.object_move_decision != "hard" && args.object_move_decision != "probability" &&
      args.object_move_decision != "expected_utility") {
    throw std::runtime_error(
        "--object_move_decision must be hard, probability, or expected_utility");
  }
  if (args.object_injection_policy != "all" &&
      args.object_injection_policy != "repair_candidates" &&
      args.object_injection_policy != "repair_or_supported") {
    throw std::runtime_error(
        "--object_injection_policy must be all, repair_candidates, or repair_or_supported");
  }
  if (args.object_alignment_policy != "none" &&
      args.object_alignment_policy != "translation") {
    throw std::runtime_error("--object_alignment_policy must be none or translation");
  }
  if (args.horizontal_plane_fill_mode != "footprint" &&
      args.horizontal_plane_fill_mode != "graph_cut") {
    throw std::runtime_error("--horizontal_plane_fill_mode must be footprint or graph_cut");
  }
  if (args.horizontal_plane_candidate_policy != "extreme_pair" &&
      args.horizontal_plane_candidate_policy != "dominant_all" &&
      args.horizontal_plane_candidate_policy != "upper_band") {
    throw std::runtime_error(
        "--horizontal_plane_candidate_policy must be extreme_pair, dominant_all, or upper_band");
  }
  if (args.free_space_culling_scope != "added" &&
      args.free_space_culling_scope != "objects") {
    throw std::runtime_error("--free_space_culling_scope must be added or objects");
  }
  if (args.free_space_culling_decision != "no_present" &&
      args.free_space_culling_decision != "absence_majority") {
    throw std::runtime_error(
        "--free_space_culling_decision must be no_present or absence_majority");
  }
  if (args.dynamic_residue_decision != "no_present" &&
      args.dynamic_residue_decision != "absence_majority") {
    throw std::runtime_error(
        "--dynamic_residue_decision must be no_present or absence_majority");
  }
  if (args.dynamic_residue_unobserved_policy != "keep" &&
      args.dynamic_residue_unobserved_policy != "quarantine") {
    throw std::runtime_error(
        "--dynamic_residue_unobserved_policy must be keep or quarantine");
  }
  if (args.dynamic_residue_support_distance_m <= 0.0) {
    throw std::runtime_error("--dynamic_residue_support_distance_m must be > 0");
  }
  if (args.dynamic_residue_horizontal_normal_z_min < 0.0 ||
      args.dynamic_residue_horizontal_normal_z_min > 1.0) {
    throw std::runtime_error(
        "--dynamic_residue_horizontal_normal_z_min must be in [0, 1]");
  }
  if (args.cross_session_mesh_merge_distance_m < 0.0) {
    throw std::runtime_error("--cross_session_mesh_merge_distance_m must be >= 0");
  }
  if (args.object_cleanup_support_distance_m <= 0.0) {
    throw std::runtime_error("--object_cleanup_support_distance_m must be > 0");
  }
  if (args.structural_plane_visibility_scope != "all" &&
      args.structural_plane_visibility_scope != "vertical") {
    throw std::runtime_error("--structural_plane_visibility_scope must be all or vertical");
  }
  if (args.structural_plane_output_supersample_scope != "all" &&
      args.structural_plane_output_supersample_scope != "observed" &&
      args.structural_plane_output_supersample_scope != "support") {
    throw std::runtime_error(
        "--structural_plane_output_supersample_scope must be all, observed, or support");
  }
  if (args.axis_aligned_plane_candidate_policy != "boundary" &&
      args.axis_aligned_plane_candidate_policy != "strong_all") {
    throw std::runtime_error(
        "--axis_aligned_plane_candidate_policy must be boundary or strong_all");
  }
  if (args.volume_graph_cut_surface_policy != "all_boundaries" &&
      args.volume_graph_cut_surface_policy != "structural_boundaries" &&
      args.volume_graph_cut_surface_policy != "structural_plane_snap_boundaries" &&
      args.volume_graph_cut_surface_policy != "marching_cubes" &&
      args.volume_graph_cut_surface_policy != "structural_marching_cubes" &&
      args.volume_graph_cut_surface_policy != "structural_binary_marching_cubes") {
    throw std::runtime_error(
        "--volume_graph_cut_surface_policy must be all_boundaries, structural_boundaries, "
        "structural_plane_snap_boundaries, marching_cubes, structural_marching_cubes, "
        "or structural_binary_marching_cubes");
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
  out << "object_injection_policy: \"" << args.object_injection_policy << "\"\n";
  out << "object_surface_support_distance_m: "
      << args.object_surface_support_distance_m << "\n";
  out << "object_alignment_policy: \"" << args.object_alignment_policy << "\"\n";
  out << "object_alignment_support_distance_m: "
      << args.object_alignment_support_distance_m << "\n";
  out << "object_alignment_max_translation_m: "
      << args.object_alignment_max_translation_m << "\n";
  out << "object_alignment_min_support_vertices: "
      << args.object_alignment_min_support_vertices << "\n";
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
  out << "horizontal_plane_candidate_policy: \""
      << args.horizontal_plane_candidate_policy << "\"\n";
  out << "horizontal_plane_support_band_cells: "
      << args.horizontal_plane_support_band_cells << "\n";
  out << "axis_aligned_plane_fill: "
      << (args.axis_aligned_plane_fill ? "true" : "false") << "\n";
  out << "axis_aligned_plane_support_band_cells: "
      << args.axis_aligned_plane_support_band_cells << "\n";
  out << "axis_aligned_plane_candidate_policy: \""
      << args.axis_aligned_plane_candidate_policy << "\"\n";
  out << "structural_plane_visibility_filter: "
      << (args.structural_plane_visibility_filter ? "true" : "false") << "\n";
  out << "structural_plane_visibility_scope: \""
      << args.structural_plane_visibility_scope << "\"\n";
  out << "structural_plane_output_supersample: "
      << (args.structural_plane_output_supersample ? "true" : "false") << "\n";
  out << "structural_plane_output_supersample_scope: \""
      << args.structural_plane_output_supersample_scope << "\"\n";
  out << "horizontal_plane_grid_resolution_m: "
      << args.horizontal_plane_grid_resolution_m << "\n";
  out << "horizontal_plane_min_support_vertices: "
      << args.horizontal_plane_min_support_vertices << "\n";
  out << "free_space_culling: " << (args.free_space_culling ? "true" : "false") << "\n";
  out << "free_space_culling_scope: \"" << args.free_space_culling_scope << "\"\n";
  out << "free_space_culling_decision: \"" << args.free_space_culling_decision << "\"\n";
  out << "free_space_culling_block_size_m: "
      << args.free_space_culling_block_size_m << "\n";
  out << "free_space_culling_radial_tolerance_m: "
      << args.free_space_culling_radial_tolerance_m << "\n";
  out << "free_space_culling_depth_tolerance_m: "
      << args.free_space_culling_depth_tolerance_m << "\n";
  out << "free_space_culling_active_window_duration_s: "
      << args.free_space_culling_active_window_duration_s << "\n";
  out << "free_space_culling_min_absent: "
      << args.free_space_culling_min_absent << "\n";
  out << "free_space_culling_max_present: "
      << args.free_space_culling_max_present << "\n";
  out << "dynamic_residue_cleanup: "
      << ((args.dynamic_residue_cleanup || args.mode == "dynamic_cleanup")
              ? "true"
              : "false")
      << "\n";
  out << "dynamic_residue_support_distance_m: "
      << args.dynamic_residue_support_distance_m << "\n";
  out << "dynamic_residue_min_track_frames: "
      << args.dynamic_residue_min_track_frames << "\n";
  out << "dynamic_residue_require_time_overlap: "
      << (args.dynamic_residue_require_time_overlap ? "true" : "false") << "\n";
  out << "dynamic_residue_decision: \"" << args.dynamic_residue_decision << "\"\n";
  out << "dynamic_residue_unobserved_policy: \""
      << args.dynamic_residue_unobserved_policy << "\"\n";
  out << "dynamic_residue_min_absent: " << args.dynamic_residue_min_absent << "\n";
  out << "dynamic_residue_max_present: " << args.dynamic_residue_max_present << "\n";
  out << "dynamic_residue_protected_labels: \""
      << args.dynamic_residue_protected_labels << "\"\n";
  out << "dynamic_residue_protect_horizontal_surfaces: "
      << (args.dynamic_residue_protect_horizontal_surfaces ? "true" : "false")
      << "\n";
  out << "dynamic_residue_horizontal_normal_z_min: "
      << args.dynamic_residue_horizontal_normal_z_min << "\n";
  out << "volume_graph_cut_fill: "
      << (args.volume_graph_cut_fill ? "true" : "false") << "\n";
  out << "volume_graph_cut_surface_policy: \""
      << args.volume_graph_cut_surface_policy << "\"\n";
  out << "volume_graph_cut_resolution_m: "
      << args.volume_graph_cut_resolution_m << "\n";
  out << "volume_graph_cut_max_cells: "
      << args.volume_graph_cut_max_cells << "\n";
  out << "cross_session_remove_absent_prior: "
      << (args.cross_session_remove_absent_prior ? "true" : "false") << "\n";
  out << "cross_session_mesh_merge_distance_m: "
      << args.cross_session_mesh_merge_distance_m << "\n";
  out << "object_cleanup_reobservation_gate: "
      << (args.object_cleanup_reobservation_gate ? "true" : "false") << "\n";
  out << "object_cleanup_support_distance_m: "
      << args.object_cleanup_support_distance_m << "\n";
  out << "object_move_skip_probability: " << args.object_move_skip_probability << "\n";
  out << "object_move_time_scale_s: " << args.object_move_time_scale_s << "\n";
  out << "prior_match_distance_m: " << args.prior_match_distance_m << "\n";
  out << "min_object_mesh_vertices: " << args.min_object_mesh_vertices << "\n";
  out << "repair_global_vertex_ratio_threshold: " << args.repair_global_vertex_ratio_threshold
      << "\n";
  out << "require_same_label: " << (args.require_same_label ? "true" : "false") << "\n";
  out << "require_bbox_containment: " << (args.require_bbox_containment ? "true" : "false")
      << "\n";
  out << "save_original_copy: " << (args.save_original_copy ? "true" : "false") << "\n";
  out << "dry_run: " << (args.dry_run ? "true" : "false") << "\n";
}

session_update::base1::ReconcilerConfig makeReconcilerConfig(const Args& args) {
  session_update::base1::ReconcilerConfig config;
  config.mode = args.mode;
  config.dynamic_mode = args.dynamic_mode;
  config.prior_map = args.prior_map;
  config.prior_object_memory = args.prior_object_memory;
  config.object_changes_csv = args.object_changes_csv;
  config.object_move_decision = args.object_move_decision;
  config.object_injection_policy = args.object_injection_policy;
  config.object_surface_support_distance_m = args.object_surface_support_distance_m;
  config.object_alignment_policy = args.object_alignment_policy;
  config.object_alignment_support_distance_m = args.object_alignment_support_distance_m;
  config.object_alignment_max_translation_m = args.object_alignment_max_translation_m;
  config.object_alignment_min_support_vertices = args.object_alignment_min_support_vertices;
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
  config.horizontal_plane_candidate_policy = args.horizontal_plane_candidate_policy;
  config.horizontal_plane_support_band_cells = args.horizontal_plane_support_band_cells;
  config.axis_aligned_plane_fill = args.axis_aligned_plane_fill;
  config.axis_aligned_plane_support_band_cells =
      args.axis_aligned_plane_support_band_cells;
  config.axis_aligned_plane_candidate_policy = args.axis_aligned_plane_candidate_policy;
  config.structural_plane_visibility_filter = args.structural_plane_visibility_filter;
  config.structural_plane_visibility_scope = args.structural_plane_visibility_scope;
  config.structural_plane_output_supersample = args.structural_plane_output_supersample;
  config.structural_plane_output_supersample_scope =
      args.structural_plane_output_supersample_scope;
  config.horizontal_plane_grid_resolution_m = args.horizontal_plane_grid_resolution_m;
  config.horizontal_plane_min_support_vertices =
      args.horizontal_plane_min_support_vertices;
  config.free_space_culling = args.free_space_culling;
  config.free_space_culling_scope = args.free_space_culling_scope;
  config.free_space_culling_decision = args.free_space_culling_decision;
  config.free_space_culling_block_size_m = args.free_space_culling_block_size_m;
  config.free_space_culling_radial_tolerance_m =
      args.free_space_culling_radial_tolerance_m;
  config.free_space_culling_depth_tolerance_m =
      args.free_space_culling_depth_tolerance_m;
  config.free_space_culling_active_window_duration_s =
      args.free_space_culling_active_window_duration_s;
  config.free_space_culling_min_absent = args.free_space_culling_min_absent;
  config.free_space_culling_max_present = args.free_space_culling_max_present;
  config.dynamic_residue_cleanup =
      args.dynamic_residue_cleanup || args.mode == "dynamic_cleanup";
  config.dynamic_residue_support_distance_m =
      args.dynamic_residue_support_distance_m;
  config.dynamic_residue_min_track_frames = args.dynamic_residue_min_track_frames;
  config.dynamic_residue_require_time_overlap =
      args.dynamic_residue_require_time_overlap;
  config.dynamic_residue_decision = args.dynamic_residue_decision;
  config.dynamic_residue_unobserved_policy =
      args.dynamic_residue_unobserved_policy;
  config.dynamic_residue_min_absent = args.dynamic_residue_min_absent;
  config.dynamic_residue_max_present = args.dynamic_residue_max_present;
  config.dynamic_residue_protected_labels =
      parseIntSet(args.dynamic_residue_protected_labels);
  config.dynamic_residue_protect_horizontal_surfaces =
      args.dynamic_residue_protect_horizontal_surfaces;
  config.dynamic_residue_horizontal_normal_z_min =
      args.dynamic_residue_horizontal_normal_z_min;
  config.volume_graph_cut_fill = args.volume_graph_cut_fill;
  config.volume_graph_cut_surface_policy = args.volume_graph_cut_surface_policy;
  config.volume_graph_cut_resolution_m = args.volume_graph_cut_resolution_m;
  config.volume_graph_cut_max_cells = args.volume_graph_cut_max_cells;
  config.cross_session_remove_absent_prior =
      args.cross_session_remove_absent_prior;
  config.cross_session_mesh_merge_distance_m =
      args.cross_session_mesh_merge_distance_m;
  config.object_cleanup_reobservation_gate =
      args.object_cleanup_reobservation_gate;
  config.object_cleanup_support_distance_m =
      args.object_cleanup_support_distance_m;
  config.object_move_skip_probability = args.object_move_skip_probability;
  config.object_move_time_scale_s = args.object_move_time_scale_s;
  config.prior_match_distance_m = args.prior_match_distance_m;
  config.min_object_mesh_vertices = args.min_object_mesh_vertices;
  config.repair_global_vertex_ratio_threshold =
      args.repair_global_vertex_ratio_threshold;
  config.require_same_label = args.require_same_label;
  config.require_bbox_containment = args.require_bbox_containment;
  config.dry_run = args.dry_run;
  return config;
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
    khronos::DynamicSceneGraph::Ptr prior_session_dsg;
    if (!args.prior_map.empty()) {
      auto prior_map = khronos::SpatioTemporalMap::load(args.prior_map);
      if (!prior_map) {
        std::cerr << "Failed to load prior map: " << args.prior_map << "\n";
        return 6;
      }
      const auto prior_stamp = selectMapTime(*prior_map, "latest");
      prior_session_dsg = prior_map->getDsgPtr(prior_stamp);
      if (!prior_session_dsg) {
        std::cerr << "Failed to extract latest DSG from prior map: " << args.prior_map << "\n";
        return 7;
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
    if (args.save_original_copy &&
        !session_update::base1::saveMapWithSingleDsg(
            dsg, selected_stamp, original_map.string())) {
        std::cerr << "Failed to save original map copy: " << original_map << "\n";
        return 4;
    }

    auto config = makeReconcilerConfig(args);
    config.prior_session_dsg = prior_session_dsg;
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
    std::cout << "original_final="
              << (args.save_original_copy ? original_map.string() : args.map_file) << "\n";
    std::cout << "improved_final=" << improved_map << "\n";
    std::cout << "mode=" << args.mode << "\n";
    std::cout << "dynamic_mode=" << args.dynamic_mode << "\n";
    std::cout << "map_time=" << args.map_time << "\n";
    std::cout << "selected_stamp=" << selected_stamp << "\n";
    std::cout << "prior_map_loaded=" << (prior_map_loaded ? "true" : "false") << "\n";
    std::cout << "prior_memory_objects=" << prior_memory_objects << "\n";
    std::cout << "cross_session_prior_vertices="
              << result.mesh_summary.cross_session_prior_vertices << "\n";
    std::cout << "cross_session_prior_absent_vertices="
              << result.mesh_summary.cross_session_prior_absent_vertices << "\n";
    std::cout << "cross_session_prior_persistent_vertices="
              << result.mesh_summary.cross_session_prior_persistent_vertices << "\n";
    std::cout << "cross_session_prior_unobserved_vertices="
              << result.mesh_summary.cross_session_prior_unobserved_vertices << "\n";
    std::cout << "cross_session_current_injected_vertices="
              << result.mesh_summary.cross_session_current_injected_vertices << "\n";
    std::cout << "synthetic_change_file=" << args.synthetic_change_file << "\n";
    std::cout << "object_injection_policy=" << args.object_injection_policy << "\n";
    std::cout << "object_surface_support_distance_m="
              << args.object_surface_support_distance_m << "\n";
    std::cout << "object_alignment_policy=" << args.object_alignment_policy << "\n";
    std::cout << "object_alignment_candidates="
              << result.mesh_summary.object_alignment_candidates << "\n";
    std::cout << "object_alignment_applied="
              << result.mesh_summary.object_alignment_applied << "\n";
    std::cout << "object_alignment_support_vertices="
              << result.mesh_summary.object_alignment_support_vertices << "\n";
    std::cout << "initial_vertices=" << result.mesh_summary.initial_vertices << "\n";
    std::cout << "final_vertices=" << result.mesh_summary.final_vertices << "\n";
    std::cout << "removed_vertices=" << result.mesh_summary.removed_vertices << "\n";
    std::cout << "objects_total=" << result.mesh_summary.objects_total << "\n";
    std::cout << "objects_used_for_cleanup=" << result.mesh_summary.objects_used_for_cleanup
              << "\n";
    std::cout << "cleanup_source_points=" << result.mesh_summary.cleanup_source_points << "\n";
    std::cout << "object_cleanup_present_protected_vertices="
              << result.mesh_summary.object_cleanup_present_protected_vertices << "\n";
    std::cout << "object_cleanup_unobserved_protected_vertices="
              << result.mesh_summary.object_cleanup_unobserved_protected_vertices << "\n";
    std::cout << "object_cleanup_absent_confirmed_vertices="
              << result.mesh_summary.object_cleanup_absent_confirmed_vertices << "\n";
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
    std::cout << "horizontal_plane_candidate_policy="
              << args.horizontal_plane_candidate_policy << "\n";
    std::cout << "horizontal_plane_support_band_cells="
              << args.horizontal_plane_support_band_cells << "\n";
    std::cout << "axis_aligned_planes_filled="
              << result.mesh_summary.axis_aligned_planes_filled << "\n";
    std::cout << "axis_aligned_plane_support_band_cells="
              << args.axis_aligned_plane_support_band_cells << "\n";
    std::cout << "axis_aligned_plane_candidate_policy="
              << args.axis_aligned_plane_candidate_policy << "\n";
    std::cout << "structural_plane_visibility_filter="
              << (args.structural_plane_visibility_filter ? "true" : "false") << "\n";
    std::cout << "structural_plane_visibility_scope="
              << args.structural_plane_visibility_scope << "\n";
    std::cout << "structural_plane_output_supersample="
              << (args.structural_plane_output_supersample ? "true" : "false") << "\n";
    std::cout << "structural_plane_output_supersample_scope="
              << args.structural_plane_output_supersample_scope << "\n";
    std::cout << "axis_aligned_plane_vertices="
              << result.mesh_summary.axis_aligned_plane_vertices << "\n";
    std::cout << "axis_aligned_plane_faces="
              << result.mesh_summary.axis_aligned_plane_faces << "\n";
    std::cout << "axis_aligned_plane_graph_cut_cells="
              << result.mesh_summary.axis_aligned_plane_graph_cut_cells << "\n";
    std::cout << "axis_aligned_plane_graph_cut_fill_cells="
              << result.mesh_summary.axis_aligned_plane_graph_cut_fill_cells << "\n";
    std::cout << "free_space_culling="
              << (args.free_space_culling ? "true" : "false") << "\n";
    std::cout << "free_space_culling_scope=" << args.free_space_culling_scope << "\n";
    std::cout << "free_space_culling_decision=" << args.free_space_culling_decision << "\n";
    std::cout << "free_space_checked_vertices="
              << result.mesh_summary.free_space_checked_vertices << "\n";
    std::cout << "free_space_absent_vertices="
              << result.mesh_summary.free_space_absent_vertices << "\n";
    std::cout << "free_space_present_vertices="
              << result.mesh_summary.free_space_present_vertices << "\n";
    std::cout << "free_space_removed_vertices="
              << result.mesh_summary.free_space_removed_vertices << "\n";
    std::cout << "dynamic_residue_cleanup="
              << ((args.dynamic_residue_cleanup || args.mode == "dynamic_cleanup")
                      ? "true"
                      : "false")
              << "\n";
    std::cout << "dynamic_residue_tracks="
              << result.mesh_summary.dynamic_residue_tracks << "\n";
    std::cout << "dynamic_residue_support_points="
              << result.mesh_summary.dynamic_residue_support_points << "\n";
    std::cout << "dynamic_residue_candidates="
              << result.mesh_summary.dynamic_residue_candidates << "\n";
    std::cout << "dynamic_residue_protected_vertices="
              << result.mesh_summary.dynamic_residue_protected_vertices << "\n";
    std::cout << "dynamic_residue_absent_confirmed_vertices="
              << result.mesh_summary.dynamic_residue_absent_confirmed_vertices << "\n";
    std::cout << "dynamic_residue_present_rejected_vertices="
              << result.mesh_summary.dynamic_residue_present_rejected_vertices << "\n";
    std::cout << "dynamic_residue_unobserved_vertices="
              << result.mesh_summary.dynamic_residue_unobserved_vertices << "\n";
    std::cout << "dynamic_residue_quarantined_vertices="
              << result.mesh_summary.dynamic_residue_quarantined_vertices << "\n";
    std::cout << "dynamic_residue_removed_vertices="
              << result.mesh_summary.dynamic_residue_removed_vertices << "\n";
    std::cout << "volume_graph_cut_fill="
              << (args.volume_graph_cut_fill ? "true" : "false") << "\n";
    std::cout << "volume_graph_cut_surface_policy="
              << args.volume_graph_cut_surface_policy << "\n";
    std::cout << "volume_graph_cut_cells="
              << result.mesh_summary.volume_graph_cut_cells << "\n";
    std::cout << "volume_graph_cut_free_evidence_cells="
              << result.mesh_summary.volume_graph_cut_free_evidence_cells << "\n";
    std::cout << "volume_graph_cut_full_evidence_cells="
              << result.mesh_summary.volume_graph_cut_full_evidence_cells << "\n";
    std::cout << "volume_graph_cut_structural_cells="
              << result.mesh_summary.volume_graph_cut_structural_cells << "\n";
    std::cout << "volume_graph_cut_full_cells="
              << result.mesh_summary.volume_graph_cut_full_cells << "\n";
    std::cout << "volume_graph_cut_vertices="
              << result.mesh_summary.volume_graph_cut_vertices << "\n";
    std::cout << "volume_graph_cut_faces="
              << result.mesh_summary.volume_graph_cut_faces << "\n";
    std::cout << "prior_memory_objects_used_for_matching="
              << result.mesh_summary.prior_memory_objects << "\n";
    std::cout << "prior_matched_objects=" << result.mesh_summary.prior_matched_objects << "\n";
    std::cout << "prior_unmatched_current_objects="
              << result.mesh_summary.prior_unmatched_current_objects << "\n";
    std::cout << "prior_restored_objects="
              << result.mesh_summary.prior_restored_objects << "\n";
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
