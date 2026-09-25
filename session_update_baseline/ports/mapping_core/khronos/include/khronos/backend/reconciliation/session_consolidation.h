#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <hydra/utils/nearest_neighbor_utilities.h>

#include "khronos/backend/change_detection/physical_evidence_store.h"
#include "khronos/common/common_types.h"

namespace khronos {

/**
 * @brief Session-end consolidation of the present map with this session's own
 * depth evidence.
 *
 * The map a session ends with holds two kinds of surface: memory inherited from
 * earlier sessions and surface built in this session. Each surface element
 * (mesh vertex) is tested against every depth frame this session stored, at the
 * element's own pixel and over its localization ball: the element is only known
 * to within tau = max(v/2, sigma(z)) of the layer that produced it (v: TSDF
 * voxel of that layer, sigma(z): this session's depth noise at range z,
 * measured on its own surfaces), so a frame either hits the element (a sample of
 * the ball measures a surface within tau), sees through it (every sample of the
 * ball is observed beyond it by more than tau), or is blocked in front of it.
 *
 * Rules (majority over the session's frames; no other constants):
 *  - surface built this session: retired when the frames that see through it
 *    within the truncation band outnumber the frames that hit it on its own
 *    ray (repeated visits of one surface that left a stale sheet);
 *  - memory: retired when the frames that see through it outnumber the frames
 *    that hit it (the present contradicts it), or when it was never hit or
 *    seen through and most of its blocked views are blocked within the
 *    truncation band in front of it (a TSDF cannot hold a separate surface
 *    that close behind the observed one);
 *  - memory this session never observed stays;
 *  - memory a previous session's consolidation retired is retired again (the
 *    next session reasons on the unconsolidated state, see setChain).
 *
 * Only geometry is edited: object identities, states, presence and bounding
 * boxes are untouched, and an object's surface is never removed entirely
 * (whether an object exists is decided by change detection). The consolidation
 * runs after the terminal change-detection pass on the final snapshot only, so
 * every earlier snapshot and every change-detection input is unchanged.
 */
class SessionConsolidation {
 public:
  struct Config {
    // Worker threads for the per-frame evidence pass.
    int num_threads = 4;
    // An element of the current map within this distance of an inherited
    // surface point is memory (the inherited map is carried over unchanged, so
    // this only absorbs float round-off of the stored positions).
    float memory_match_distance = 0.003f;
    // Range bin of the depth noise model and the samples a bin needs to carry
    // its own estimate (estimator settings, not decision thresholds).
    float range_bin = 0.5f;
    size_t min_bin_samples = 1000;
  };

  struct Scales {
    float background_voxel = 0.f;
    float background_truncation = 0.f;
    // > 0: absolute object voxel; < 0: fraction of the object's largest extent
    // (the object extractor's convention), bounded below by object_min_voxel.
    float object_voxel = 0.f;
    float object_min_voxel = 0.f;
  };

  struct Result {
    size_t frames = 0;
    size_t elements = 0;
    size_t memory_elements = 0;
    size_t retired_own = 0;
    size_t retired_memory_seen_through = 0;
    size_t retired_memory_hidden = 0;
    size_t retired_chain = 0;
    size_t objects_kept_whole = 0;
    size_t background_vertices_erased = 0;
    size_t object_vertices_erased = 0;
    std::vector<float> sigma_background;  // per range bin [m]
    std::vector<float> sigma_objects;
    // World positions of every retired element (for the next session's chain).
    std::vector<Eigen::Vector3f> retired_positions;
    std::string summary() const;
  };

  explicit SessionConsolidation(const Config& config) : config(config) {}

  void setScales(const Scales& scales) { scales_ = scales; }

  /** Inherited surface positions (world frame) of the state this session started from. */
  void setMemory(std::vector<Eigen::Vector3f> points);

  /**
   * Positions the previous sessions' consolidations retired. The next session
   * reasons on the unconsolidated state (so object and change reasoning is
   * exactly that of the plain map); its consolidation retires these again
   * where they are still memory.
   */
  void setChain(std::vector<Eigen::Vector3f> points);

  /** Edit `dsg` (the final snapshot) in place. */
  Result apply(DynamicSceneGraph& dsg, const PhysicalEvidenceStore::Snapshot& evidence) const;

  const Config config;

 private:
  Scales scales_;
  std::vector<Eigen::Vector3f> memory_points_;
  std::unique_ptr<hydra::PointNeighborSearch> memory_search_;
  std::vector<Eigen::Vector3f> chain_points_;
  std::unique_ptr<hydra::PointNeighborSearch> chain_search_;
};

}  // namespace khronos
