#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include <Eigen/Core>
#include <spark_dsg/mesh.h>
#include <spatial_hash/types.h>

namespace khronos {

/**
 * @brief Scene memory handed from the backend to the active window as a
 * re-integrable prior, and the ledger of what the active window did with it.
 *
 * The previous session's current geometry is published once, after the prior
 * state is loaded. When the active window allocates a TSDF block for the first
 * time in this session, it seeds the block's voxels from this surface with
 * `prior_weight`, so this session's measurements refine the inherited estimate
 * instead of building a second copy beside it. When such a block is archived,
 * its mesh replaces the inherited vertices inside the block.
 */
struct InheritedGeometry {
  using Ptr = std::shared_ptr<InheritedGeometry>;

  // Published by the backend. Read-only afterwards.
  spark_dsg::Mesh mesh;
  uint64_t horizon_ns = 0;
  bool ready = false;

  // Written by the active window, consumed by the backend.
  std::mutex mutex;
  float block_size = 0.f;
  std::vector<spatial_hash::BlockIndex> seeded_pending;   // seeded, still in the window
  std::vector<spatial_hash::BlockIndex> archived_seeded;  // seeded and archived, unconsumed
  size_t seeded_blocks = 0;
  size_t seeded_voxels = 0;
  size_t unmeasured_archived = 0;  // seeded blocks archived without any measurement (kept)

  // Inherited vertices this session's own fusion carved away, recorded at the
  // moment a block is archived, which is the last moment its TSDF exists.
  //
  // A block leaving the window has its mesh extracted and its voxels dropped, so
  // by the time the backend reconciles, the only surviving evidence is the mesh.
  // That is why memory could previously only be retired where the session
  // REBUILT a surface: where the session instead measured empty space and built
  // nothing, there was no longer anything to compare against, and the inherited
  // copy of a since-removed object stayed in the map. Reading the fused signed
  // distance here uses the same measurements marching cubes used, one step
  // before they are discarded. No new threshold: a voxel counts as carved when
  // this session gave it weight and its distance is beyond the truncation band,
  // which is the map's own definition of observed free space.
  // Positions, not vertex indices: the backend erases vertices from the live
  // mesh as it reconciles, which reindexes it, so an index published here would
  // not survive the trip.
  std::vector<Eigen::Vector3f> carved_points;
  size_t carved_checked = 0;
};

}  // namespace khronos
