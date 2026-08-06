#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <khronos/common/common_types.h>
#include <spark_dsg/mesh.h>

namespace session_update::base1 {

struct ReconcilerConfig {
  std::string mode = "cleanup";
  std::string dynamic_mode = "within_session";
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
  std::unordered_set<int> dynamic_residue_protected_labels;
  bool dynamic_residue_protect_horizontal_surfaces = true;
  double dynamic_residue_horizontal_normal_z_min = 0.85;
  bool volume_graph_cut_fill = false;
  std::string volume_graph_cut_surface_policy = "all_boundaries";
  double volume_graph_cut_resolution_m = 0.16;
  std::size_t volume_graph_cut_max_cells = 350000;
  std::vector<khronos::DynamicSceneGraph::Ptr> temporal_background_dsgs;
  khronos::DynamicSceneGraph::Ptr prior_session_dsg;
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
  bool dry_run = false;
};

struct ObjectAuditRow {
  khronos::NodeId node_id = 0;
  int semantic_label = -1;
  uint64_t first_observed_ns = 0;
  uint64_t last_observed_ns = 0;
  float bbox_cx = 0.0f;
  float bbox_cy = 0.0f;
  float bbox_cz = 0.0f;
  float bbox_dx = 0.0f;
  float bbox_dy = 0.0f;
  float bbox_dz = 0.0f;
  float bbox_volume = 0.0f;
  std::size_t object_mesh_vertices = 0;
  std::size_t object_mesh_faces = 0;
  std::size_t dynamic_trajectory_points = 0;
  std::size_t dynamic_point_frames = 0;
  std::size_t global_vertices_in_bbox = 0;
  std::size_t global_vertices_in_bbox_same_label = 0;
  std::size_t global_vertices_in_dynamic_swept_bbox = 0;
  std::size_t global_vertices_in_dynamic_swept_bbox_same_label = 0;
  std::size_t global_vertices_in_dynamic_swept_bbox_time_overlap = 0;
  std::size_t global_vertices_near_dynamic_points = 0;
  std::size_t global_vertices_near_dynamic_points_same_label = 0;
  std::size_t global_vertices_near_dynamic_points_time_overlap = 0;
  std::size_t dynamic_residue_candidates = 0;
  std::size_t dynamic_residue_protected = 0;
  std::size_t dynamic_residue_absent_confirmed = 0;
  std::size_t dynamic_residue_present_rejected = 0;
  std::size_t dynamic_residue_unobserved = 0;
  std::size_t dynamic_residue_quarantined = 0;
  std::size_t dynamic_residue_removed = 0;
  std::size_t vertices_candidate = 0;
  std::size_t vertices_removed = 0;
  bool repair_candidate = false;
  std::size_t repair_candidate_vertices = 0;
  uint64_t change_first_absent_ns = 0;
  uint64_t change_last_absent_ns = 0;
  uint64_t change_first_persistent_ns = 0;
  uint64_t change_last_persistent_ns = 0;
  double object_move_probability = -1.0;
  bool skipped_by_object_move = false;
  bool object_alignment_applied = false;
  std::size_t object_alignment_support_vertices = 0;
  double object_alignment_translation_norm_m = 0.0;
  double object_alignment_before_median_m = 0.0;
  double object_alignment_after_median_m = 0.0;
  bool prior_matched = false;
  khronos::NodeId prior_match_object_id = 0;
  double prior_match_distance_m = std::numeric_limits<double>::infinity();
  std::size_t prior_absent_evidence = 0;
  std::size_t prior_present_evidence = 0;
  double prior_stationarity_alpha = 0.0;
  double prior_stationarity_beta = 0.0;
  double prior_stationarity_mean = 0.0;
  std::string session_state = "unknown";
};

struct MeshUpdateSummary {
  std::size_t initial_vertices = 0;
  std::size_t initial_faces = 0;
  std::size_t final_vertices = 0;
  std::size_t final_faces = 0;
  std::size_t candidate_vertices = 0;
  std::size_t removed_vertices = 0;
  std::size_t objects_total = 0;
  std::size_t objects_with_private_mesh = 0;
  std::size_t objects_used_for_cleanup = 0;
  std::size_t cleanup_source_points = 0;
  std::size_t object_cleanup_present_protected_vertices = 0;
  std::size_t object_cleanup_unobserved_protected_vertices = 0;
  std::size_t object_cleanup_absent_confirmed_vertices = 0;
  std::size_t repair_candidate_objects = 0;
  std::size_t repair_candidate_vertices = 0;
  std::size_t injected_objects = 0;
  std::size_t injected_vertices = 0;
  std::size_t injected_faces = 0;
  std::size_t object_alignment_candidates = 0;
  std::size_t object_alignment_applied = 0;
  std::size_t object_alignment_support_vertices = 0;
  std::size_t temporal_background_sources = 0;
  std::size_t temporal_background_injected_vertices = 0;
  std::size_t temporal_background_injected_faces = 0;
  std::size_t temporal_object_sources = 0;
  std::size_t temporal_object_injected_vertices = 0;
  std::size_t temporal_object_injected_faces = 0;
  std::size_t horizontal_planes_filled = 0;
  std::size_t horizontal_plane_vertices = 0;
  std::size_t horizontal_plane_faces = 0;
  std::size_t horizontal_plane_graph_cut_cells = 0;
  std::size_t horizontal_plane_graph_cut_fill_cells = 0;
  std::size_t axis_aligned_planes_filled = 0;
  std::size_t axis_aligned_plane_vertices = 0;
  std::size_t axis_aligned_plane_faces = 0;
  std::size_t axis_aligned_plane_graph_cut_cells = 0;
  std::size_t axis_aligned_plane_graph_cut_fill_cells = 0;
  std::size_t free_space_checked_vertices = 0;
  std::size_t free_space_absent_vertices = 0;
  std::size_t free_space_present_vertices = 0;
  std::size_t free_space_removed_vertices = 0;
  std::size_t dynamic_residue_tracks = 0;
  std::size_t dynamic_residue_support_points = 0;
  std::size_t dynamic_residue_candidates = 0;
  std::size_t dynamic_residue_protected_vertices = 0;
  std::size_t dynamic_residue_absent_confirmed_vertices = 0;
  std::size_t dynamic_residue_present_rejected_vertices = 0;
  std::size_t dynamic_residue_unobserved_vertices = 0;
  std::size_t dynamic_residue_quarantined_vertices = 0;
  std::size_t dynamic_residue_removed_vertices = 0;
  std::size_t volume_graph_cut_cells = 0;
  std::size_t volume_graph_cut_free_evidence_cells = 0;
  std::size_t volume_graph_cut_full_evidence_cells = 0;
  std::size_t volume_graph_cut_structural_cells = 0;
  std::size_t volume_graph_cut_full_cells = 0;
  std::size_t volume_graph_cut_vertices = 0;
  std::size_t volume_graph_cut_faces = 0;
  std::size_t prior_memory_objects = 0;
  std::size_t prior_matched_objects = 0;
  std::size_t prior_absent_objects = 0;
  std::size_t prior_unobserved_objects = 0;
  std::size_t prior_unmatched_current_objects = 0;
  std::size_t prior_restored_objects = 0;
  std::size_t prior_absent_vertices_removed = 0;
  std::size_t forced_absent_prior_objects = 0;
  std::size_t forced_absent_vertices_removed = 0;
  std::size_t cross_session_prior_vertices = 0;
  std::size_t cross_session_prior_faces = 0;
  std::size_t cross_session_current_vertices = 0;
  std::size_t cross_session_current_faces = 0;
  std::size_t cross_session_prior_checked_vertices = 0;
  std::size_t cross_session_prior_absent_vertices = 0;
  std::size_t cross_session_prior_persistent_vertices = 0;
  std::size_t cross_session_prior_unobserved_vertices = 0;
  std::size_t cross_session_current_injected_vertices = 0;
  std::size_t cross_session_current_injected_faces = 0;
};

struct ReconcileResult {
  MeshUpdateSummary mesh_summary;
  std::vector<ObjectAuditRow> object_rows;
  struct VertexUpdateRow {
    std::size_t vertex_index = 0;
    khronos::NodeId object_id = 0;
    int vertex_label = -1;
    int object_label = -1;
    double distance_m = 0.0;
    std::string decision;
  };
  std::vector<VertexUpdateRow> vertex_update_rows;
};

class ObjectGuidedMapReconciler {
 public:
  explicit ObjectGuidedMapReconciler(ReconcilerConfig config);

  ReconcileResult reconcile(khronos::DynamicSceneGraph& dsg) const;

 private:
  struct CleanupSource {
    khronos::NodeId node_id = 0;
    int semantic_label = -1;
    khronos::BoundingBox bounding_box;
    std::size_t first_point_index = 0;
    std::size_t num_points = 0;
  };

  struct PriorObject {
    khronos::NodeId object_id = 0;
    int semantic_label = -1;
    float bbox_cx = 0.0f;
    float bbox_cy = 0.0f;
    float bbox_cz = 0.0f;
    float bbox_dx = 0.0f;
    float bbox_dy = 0.0f;
    float bbox_dz = 0.0f;
    std::size_t object_mesh_vertices = 0;
    double stationarity_alpha = 2.0;
    double stationarity_beta = 1.0;
    std::string session_state;
  };

  struct SyntheticChangeSpec {
    std::unordered_set<khronos::NodeId> force_absent_prior_object_ids;
    bool delete_global_vertices_in_forced_absent_bbox = true;
  };

  struct ObjectChangeEvidence {
    uint64_t first_absent_ns = 0;
    uint64_t last_absent_ns = 0;
    uint64_t first_persistent_ns = 0;
    uint64_t last_persistent_ns = 0;
    double move_probability = 0.0;
    bool skip_injection = false;
  };

  static bool isInsideExpandedBox(const khronos::BoundingBox& box,
                                  const khronos::Point& point,
                                  double margin_m);

  std::vector<ObjectAuditRow> auditObjects(const khronos::DynamicSceneGraph& dsg) const;

  std::vector<CleanupSource> collectCleanupSources(
      const khronos::DynamicSceneGraph& dsg,
      std::vector<khronos::Point>* source_points,
      std::vector<std::size_t>* point_to_source,
      std::vector<ObjectAuditRow>* object_rows) const;

  std::vector<PriorObject> loadPriorObjects() const;

  std::unordered_map<khronos::NodeId, ObjectChangeEvidence> loadObjectChangeEvidence() const;

  SyntheticChangeSpec loadSyntheticChangeSpec() const;

  static khronos::BoundingBox makeBox(const PriorObject& object);

  void applyForcedAbsentPriorObjects(const std::vector<PriorObject>& prior_objects,
                                     const SyntheticChangeSpec& synthetic_change,
                                     spark_dsg::Mesh& mesh,
                                     std::unordered_set<std::size_t>* vertices_to_delete,
                                     std::vector<ReconcileResult::VertexUpdateRow>* update_rows,
                                     std::vector<ObjectAuditRow>* object_rows,
                                     MeshUpdateSummary* summary) const;

  void applyAbsentPriorMemoryObjects(spark_dsg::Mesh& mesh,
                                     std::unordered_set<std::size_t>* vertices_to_delete,
                                     std::vector<ReconcileResult::VertexUpdateRow>* update_rows,
                                     std::vector<ObjectAuditRow>* object_rows,
                                     MeshUpdateSummary* summary) const;

  void applyPriorObjectMemory(const khronos::DynamicSceneGraph& dsg,
                              std::vector<ObjectAuditRow>* object_rows,
                              MeshUpdateSummary* summary) const;

  void restorePriorObjectNodes(khronos::DynamicSceneGraph& dsg,
                               std::vector<ObjectAuditRow>* object_rows,
                               MeshUpdateSummary* summary) const;

  ReconcilerConfig config_;
};

bool saveMapWithSingleDsg(const khronos::DynamicSceneGraph::Ptr& dsg,
                          khronos::TimeStamp stamp,
                          const std::string& path);

void writeObjectAuditCsv(const std::string& path, const std::vector<ObjectAuditRow>& rows);
void writeMeshUpdateCsv(const std::string& path, const MeshUpdateSummary& summary);
void writeObjectUpdateCsv(const std::string& path, const std::vector<ObjectAuditRow>& rows);
void writeMeshVertexUpdateCsv(const std::string& path,
                              const std::vector<ReconcileResult::VertexUpdateRow>& rows);
void writeEvidenceSummaryJson(const std::string& path,
                              const ReconcilerConfig& config,
                              const MeshUpdateSummary& summary,
                              bool prior_map_loaded,
                              std::size_t prior_memory_objects);
void writeObjectMemoryJson(const std::string& path, const std::vector<ObjectAuditRow>& rows);
std::size_t countPriorMemoryObjects(const std::string& path);

}  // namespace session_update::base1
