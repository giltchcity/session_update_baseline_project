#pragma once

#include <memory>
#include <mutex>
#include <vector>

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
};

}  // namespace khronos
