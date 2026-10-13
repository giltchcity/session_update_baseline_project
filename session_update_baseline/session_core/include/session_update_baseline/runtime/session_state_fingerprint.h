#pragma once

#include <cstddef>
#include <cstdint>

#include <spark_dsg/dynamic_scene_graph.h>

namespace session_update::runtime {

struct CanonicalSceneFingerprint {
  std::uint64_t fnv1a64 = 0;
  std::size_t encoded_bytes = 0;
  std::size_t object_records = 0;
};

/**
 * Hash the authoritative current scene independently of DSG container order.
 *
 * The canonical stream contains the complete global mesh (all enabled fields
 * and faces) followed by Khronos object records sorted by physical instance,
 * semantic class, and complete record bytes. Object records contain current
 * private geometry, semantic/pose attributes, presence intervals, and D1
 * trajectory history. DSG node IDs and unordered container iteration order are
 * deliberately excluded because serialize/load/reseed may change their order
 * without changing the physical scene.
 */
CanonicalSceneFingerprint canonicalCurrentSceneFingerprint(
    const spark_dsg::DynamicSceneGraph& dsg);

}  // namespace session_update::runtime
