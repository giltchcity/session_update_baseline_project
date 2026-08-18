/** -----------------------------------------------------------------------------
 * Copyright (c) 2024 Massachusetts Institute of Technology.
 * All Rights Reserved.
 *
 * AUTHORS:      Lukas Schmid <lschmid@mit.edu>, Marcus Abate <mabate@mit.edu>,
 *               Yun Chang <yunchang@mit.edu>, Luca Carlone <lcarlone@mit.edu>
 * AFFILIATION:  MIT SPARK Lab, Massachusetts Institute of Technology
 * YEAR:         2024
 * SOURCE:       https://github.com/MIT-SPARK/Khronos
 * LICENSE:      BSD 3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * -------------------------------------------------------------------------- */

#pragma once

#include <limits>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <config_utilities/config_utilities.h>
#include <hydra/common/global_info.h>
#include <hydra/common/robot_prefix_config.h>
#include <spatial_hash/grid.h>

#include "khronos/backend/change_detection/physical_evidence_store.h"
#include "khronos/common/common_types.h"

namespace khronos {

/**
 * @brief Utility class that checks if a point has been observed to be absent by comparing it to a
 * set of deformable rays stored in the DSG.
 */
class RayVerificator {
 public:
  // Types.
  using Ptr = std::shared_ptr<RayVerificator>;
  using ConstPtr = std::shared_ptr<const RayVerificator>;
  using IndexSet = std::unordered_set<size_t>;
  using SpatialHash = BlockIndexMap<IndexSet>;
  using PhysicalEvidenceSnapshot =
      std::optional<PhysicalEvidenceStore::Snapshot>;

  enum class UpdateMode { kFullReset, kIncremental };

  /**
   * @brief Observable accounting for the full-versus-incremental contract.
   *
   * These counters are intentionally part of the API: callers and regression
   * tests must be able to prove that a growing, fixed DSG did not silently take
   * the expensive full-rebuild path. Times are wall-clock microseconds.
   */
  struct Statistics {
    size_t full_resets = 0;
    size_t incremental_rebinds = 0;
    size_t incremental_updates = 0;
    size_t rejected_prefixes = 0;
    size_t indexed_poses = 0;
    size_t indexed_vertices = 0;
    size_t indexed_objects = 0;
    size_t rays = 0;
    size_t last_new_poses = 0;
    size_t last_new_vertices = 0;
    size_t last_new_rays = 0;
    size_t last_reobserved_vertices = 0;
    size_t last_reobserved_objects = 0;
    uint64_t last_prefix_check_us = 0;
    uint64_t last_update_us = 0;
    UpdateMode last_mode = UpdateMode::kFullReset;
  };

  // Config.
  struct Config {
    int verbosity = hydra::GlobalInfo::instance().getConfig().default_verbosity;

    // Size of the blocks used for background hashing in meters. This is independent of the mesh
    // block size.
    float block_size = 1.f;

    // Maximum distance to the ray to cound as occlusion in meters.
    float radial_tolerance = 0.1f;

    // Maximum depth difference within which points are considered to be the same in meters.
    float depth_tolerance = 0.1f;

    // Time stamps to raycast for verification.
    // NOTE(lschmid): Could add uniform, random, all (that'd be expensive though).
    enum class RayPolicy {
      kFirst,
      kLast,
      kFirstAndLast,
      kMiddle,
      kAll,
      kRandom,
      kRandom3,
    } ray_policy = RayPolicy::kMiddle;

    // Time to subtract from the latest time stamp to compensat the Active Window in s.
    float active_window_duration = 0.f;

    // Robot prefix to use for the ray verificator. TODO(lschmid): This should probably be moved to
    // the query.
    hydra::RobotPrefixConfig prefix;
  } const config;

  // Construction.
  explicit RayVerificator(const Config& config);
  virtual ~RayVerificator() = default;

  // Processing.
  struct CoverageReasonCounts {
    size_t same_id = 0;
    size_t different_id = 0;
    size_t unidentified_object = 0;
    size_t background_replacement = 0;
    size_t free_space = 0;
    size_t geometric_occlusion = 0;
    size_t unavailable = 0;
    size_t invalid = 0;
    size_t no_overlap = 0;

    void merge(const CoverageReasonCounts& other) {
      same_id += other.same_id;
      different_id += other.different_id;
      unidentified_object += other.unidentified_object;
      background_replacement += other.background_replacement;
      free_space += other.free_space;
      geometric_occlusion += other.geometric_occlusion;
      unavailable += other.unavailable;
      invalid += other.invalid;
      no_overlap += other.no_overlap;
    }
  };

  struct CheckResult {
    // Timestamps of all rays that marked the point as absent or present.
    std::vector<uint64_t> absent;
    std::vector<uint64_t> present;

    // Timestamps of geometrically covering rays whose typed endpoint cannot
    // decide the state of the queried physical object.
    std::vector<uint64_t> inconclusive;
    CoverageReasonCounts reasons;

    // Merge other check results into this one.
    void merge(const CheckResult& other) {
      absent.insert(absent.end(), other.absent.begin(), other.absent.end());
      present.insert(present.end(), other.present.begin(), other.present.end());
      inconclusive.insert(inconclusive.end(),
                          other.inconclusive.begin(),
                          other.inconclusive.end());
      reasons.merge(other.reasons);
    }
  };

  // Extra details about the check that can be optionally requested for visualization.
  struct CheckDetails {
    Points start;
    Points end;
    std::vector<float> range;
    enum class Result { kNoOverlap, kOccludded, kAbsent, kMatch };
    std::vector<Result> result;
  };

  /**
   * @brief Checks if a point is observed to be present or absent in the given time window.
   * @param point The 3D position in world frame to check.
   * @param earliest Only consder measurments that are at least this old, as nanosecond stamp.
   * @param latest Only consder measurments that are at most this old, as nanosecond stamp.
   * @param details Optional pointer to a CheckDetails struct to fill with details about the check.
   * @return The result of the check.
   */
  CheckResult check(const Point& point,
                    const uint64_t earliest = 0ul,
                    const uint64_t latest = std::numeric_limits<uint64_t>::max(),
                    CheckDetails* details = nullptr) const;

  /**
   * @brief Check a surface belonging to one stable physical identity.
   *
   * Unlike check(), an endpoint from another object or an occluding surface is
   * coverage but not evidence that the queried object is absent. The overload
   * without a snapshot is intended for isolated queries and captures the
   * store's current immutable snapshot for that call.
   */
  CheckResult checkPhysical(
      const Point& point,
      size_t physical_id,
      const uint64_t earliest = 0ul,
      const uint64_t latest = std::numeric_limits<uint64_t>::max(),
      CheckDetails* details = nullptr) const;

  /**
   * @brief Check against a caller-frozen evidence snapshot.
   *
   * Batch callers must use one snapshot for their complete update so an
   * asynchronous frame ingest cannot split one change decision across store
   * versions.
   */
  CheckResult checkPhysical(
      const Point& point,
      size_t physical_id,
      const PhysicalEvidenceSnapshot& evidence_snapshot,
      const uint64_t earliest = 0ul,
      const uint64_t latest = std::numeric_limits<uint64_t>::max(),
      CheckDetails* details = nullptr) const;

  /**
   * @brief Check the actual triangle surface of one physical object, not its
   * sparse vertex set.
   *
   * A reconstructed object mesh is a sampling of a continuous surface. Querying
   * only vertices lets rays pass through the holes between samples and turn
   * "this sparse view is not the whole object" into false absence evidence.
   * This overload intersects every mesh triangle against the indexed ray
   * segments and classifies each intersection with the endpoint identity at
   * the exact timestamp, so a ray only contradicts a surface it actually
   * crossed.
   *
   * If the mesh has no faces the vertex-based check is used as the only
   * available fallback.
   */
  CheckResult checkPhysicalSurface(
      size_t physical_id,
      const spark_dsg::Mesh& mesh,
      const BoundingBox& bbox,
      const PhysicalEvidenceSnapshot& evidence_snapshot,
      const uint64_t earliest = 0ul,
      const uint64_t latest = std::numeric_limits<uint64_t>::max(),
      CheckDetails* details = nullptr) const;

  /**
   * @brief Unique-ray surface evidence for one physical object mesh.
   *
   * Each sensor ray is one measurement; if one ray passes several surface
   * samples it must not be counted several times. The returned counts are
   * therefore unique ray indices, and `surface_samples` is the number of mesh
   * triangle centroids (or vertices for topology-free meshes) that were
   * queried. Ratios of these counts are scale-free.
   */
  struct SurfaceEvidenceCounts {
    size_t support_rays = 0;
    size_t contradiction_rays = 0;
    size_t surface_samples = 0;
    std::unordered_set<size_t> support_indices;
    std::unordered_set<size_t> contradiction_indices;

    // Per-(sample, ray) six-class evidence votes for the verification ledger.
    // These are NOT unique-ray counts: one ray passing several samples votes
    // several times, so ratios between vote classes are comparable, but the
    // sums differ from support_rays/contradiction_rays by design.
    size_t supported_votes = 0;
    size_t free_space_votes = 0;
    size_t replaced_by_other_votes = 0;
    size_t replaced_by_background_votes = 0;
    size_t occluded_votes = 0;
    size_t unobserved_samples = 0;
  };

  SurfaceEvidenceCounts countPhysicalSurface(
      size_t physical_id,
      const spark_dsg::Mesh& mesh,
      const BoundingBox& bbox,
      const PhysicalEvidenceSnapshot& evidence_snapshot,
      const uint64_t earliest = 0ul,
      const uint64_t latest = std::numeric_limits<uint64_t>::max()) const;

  void setPhysicalEvidenceStore(PhysicalEvidenceStore::Ptr store);
  PhysicalEvidenceSnapshot physicalEvidenceSnapshot() const;

  /**
   * @brief Bind a DSG snapshot for absence checks. A verified append-only
   * snapshot preserves the existing ray index; any prefix mismatch rebuilds
   * all state and returns kFullReset.
   * @param dsg The DynamicSceneGraph to use for ray verification.
   */
  UpdateMode setDsg(std::shared_ptr<const DynamicSceneGraph> dsg);

  /**
   * @brief Incrementally add new measurements of a single growing dynamic scene graph to the ray
   * verificator. The indexed pose/mesh prefix is verified byte-for-byte before
   * reuse. Any removal, timestamp change, deformation, or out-of-order pose
   * causes a conservative full reset; only a stable prefix plus a new suffix is
   * processed incrementally.
   */
  UpdateMode updateDsg();

  /**
   * @brief Discard every cached measurement and rebuild from the bound DSG.
   *
   * This is the required path after a loop closure or any deformation/removal
   * that violates the verified append-only prefix contract.
   */
  void forceReset();

  /**
   * @brief Recompute the spatial hashing function used to query points. This may not be required
   * for small deformations of the DSG.
   */
  void recomputeHash();

  // Accessors.
  const Config& getConfig() const { return config; }

  /**
   * @brief Get the set of vertices that were re-observed during the last update.
   */
  const IndexSet& getReobservedVertices() const { return reobserved_vertices_; }

  /**
   * @brief Get the set of objects that were re-observed during the last update.
   */
  const IndexSet& getReobservedObjects() const { return reobserved_objects_; }

  const Statistics& getStatistics() const { return statistics_; }

 private:
  struct Ray {
    Ray() = default;
    Ray(const uint64_t timestamp, const NodeId source_node, const size_t target_index)
        : timestamp(timestamp), source_node(source_node), target_index(target_index) {}

    // Time stamp of the measurement.
    uint64_t timestamp;

    // Node ID of the agent node associated with this measurement (ray source point).
    NodeId source_node;

    // Index of the vertex associated with this measurement (ray target points).
    size_t target_index;
  };

  // Data to lookup ray queries.
  std::vector<Ray> rays_;

  // Time stamps and pointers to all source points. These are sorted by timestamp.
  std::vector<uint64_t> timestamps_;
  // Node IDs for each timestamp
  std::vector<NodeId> node_ids_;

  // Spatial Hash.
  spatial_hash::Grid<BlockIndex> grid_;
  SpatialHash block_seen_by_rays_;  // Map which blocks were seen by which rays. Stores the index to
                                    // the ray.
  SpatialHash vertices_in_block_;   // Map which vertices are contained in which blocks.
  SpatialHash objects_in_block_;    // Map which objects are contained in which blocks.

  // Cached data.
  std::shared_ptr<const DynamicSceneGraph> dsg_;
  PhysicalEvidenceStore::Ptr physical_evidence_store_;
  unsigned int seed_;

  // Variables.
  // The index of the first not-yet-indexed mesh vertex.
  size_t previous_vertex_index_ = 0;

  struct PoseSnapshot {
    uint64_t timestamp = 0;
    Point position = Point::Zero();
    Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
  };

  struct ObjectSnapshot {
    Points points;
    Point center = Point::Zero();
    std::optional<size_t> instance_id;
    std::vector<uint64_t> first_seen;
    std::vector<uint64_t> last_seen;
    std::vector<uint64_t> first_observed;
    std::vector<uint64_t> last_observed;
    uint64_t observation_first = 0;
    uint64_t observation_last = 0;
  };

  std::unordered_map<NodeId, PoseSnapshot> indexed_poses_;
  std::unordered_map<NodeId, ObjectSnapshot> indexed_objects_;
  std::unordered_map<NodeId, BlockIndexSet> object_blocks_;
  Points indexed_vertex_positions_;
  std::vector<uint64_t> indexed_vertex_first_seen_;
  std::vector<uint64_t> indexed_vertex_last_seen_;
  Statistics statistics_;

  // Newly (during last update) re-observed vertices and objects for change detection. These are set
  // during 'updateDsg()'.
  IndexSet reobserved_vertices_;
  IndexSet reobserved_objects_;

  // Helper functions.
  bool hasStablePrefix(const DynamicSceneGraph& dsg) const;
  bool hasStablePosePrefix(const DynamicSceneGraph& dsg) const;
  bool hasStableMeshPrefix(const DynamicSceneGraph& dsg) const;
  bool hasStableObjectSet(const DynamicSceneGraph& dsg) const;
  void resetState(std::shared_ptr<const DynamicSceneGraph> dsg, bool rejected_prefix);

  // Emit the rays for one vertex from an already-resolved source set.
  BlockIndexSet emitVertexRays(size_t vertex_index,
                               const std::unordered_set<size_t>& source_indices);

  // Allocate all new nodes as potential sources for rays.
  size_t addPoseNodes();

  // Add all new vertices as rays. Requires that addPoseNodes is called first. Returns the observed
  // blocks.
  BlockIndexSet addVertices();

  // Compute for a single vertex which sources it is associated with.
  std::unordered_set<size_t> computeVertexSources(const size_t first_seen, const size_t last_seen);

  // Add a single ray to the hash. Returns the observed blocks.
  BlockIndexSet addRayToHash(const size_t ray_index);

  void updateObjectsInHash();
  void rebuildObjectsInHash();
  ObjectSnapshot makeObjectSnapshot(const KhronosObjectAttributes& attrs) const;
  BlockIndexSet computeObjectBlocks(const KhronosObjectAttributes& attrs) const;
  static bool objectSnapshotsEqual(const ObjectSnapshot& lhs,
                                   const ObjectSnapshot& rhs);

  // Struct to lookup rays for a given dsg somewhat efficiently.
  struct RayLookup {
    RayLookup(const DynamicSceneGraph& dsg, const Config& config)
        : vertices_(dsg.mesh()->points),
          nodes_(
              dsg.getLayer(dsg.getLayerKey(DsgLayers::AGENTS)->layer, config.prefix.key).nodes()) {}

    Point getSource(const Ray& ray) const {
      return nodes_.at(ray.source_node)
          ->attributes<spark_dsg::NodeAttributes>()
          .position.cast<float>();
    }

    Point getTarget(const Ray& ray) const { return vertices_.at(ray.target_index); }

   private:
    const std::vector<spark_dsg::Mesh::Pos>& vertices_;
    const spark_dsg::SceneGraphLayer::Nodes& nodes_;
  };
};

void declare_config(RayVerificator::Config& config);

}  // namespace khronos
