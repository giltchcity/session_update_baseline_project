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
  std::vector<khronos::DynamicSceneGraph::Ptr> temporal_background_dsgs;
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
  bool prior_matched = false;
  khronos::NodeId prior_match_object_id = 0;
  double prior_match_distance_m = std::numeric_limits<double>::infinity();
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
  std::size_t repair_candidate_objects = 0;
  std::size_t repair_candidate_vertices = 0;
  std::size_t injected_objects = 0;
  std::size_t injected_vertices = 0;
  std::size_t injected_faces = 0;
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
  std::size_t prior_memory_objects = 0;
  std::size_t prior_matched_objects = 0;
  std::size_t prior_unmatched_current_objects = 0;
  std::size_t forced_absent_prior_objects = 0;
  std::size_t forced_absent_vertices_removed = 0;
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

  void applyPriorObjectMemory(std::vector<ObjectAuditRow>* object_rows,
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
