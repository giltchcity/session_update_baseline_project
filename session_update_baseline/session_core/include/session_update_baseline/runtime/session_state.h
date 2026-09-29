#pragma once

#include <stdexcept>
#include <utility>

#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <spark_dsg/node_symbol.h>

namespace session_update::runtime {

struct SessionSeed {
  khronos::TimeStamp stamp;
  spark_dsg::DynamicSceneGraph::Ptr dsg;
};

inline SessionSeed latestSessionSeed(khronos::SpatioTemporalMap& prior) {
  if (prior.numTimeSteps() == 0) {
    throw std::invalid_argument("Cannot seed a session from an empty map");
  }
  const auto stamp = prior.latest();
  auto dsg = prior.getDsgPtr(stamp);
  if (!dsg) {
    throw std::runtime_error("Prior map has no readable latest state");
  }
  return {stamp, std::move(dsg)};
}

inline void initializeSessionTimeline(khronos::SpatioTemporalMap& output,
                                      const SessionSeed& seed) {
  if (output.numTimeSteps() != 0) {
    throw std::invalid_argument("Session timeline must be empty before seeding");
  }
  if (!seed.dsg) {
    throw std::invalid_argument("Session seed has no DSG");
  }
  // Keep exactly one initial snapshot. The complete ancestor timeline remains
  // in its own artifact and must never be recursively copied into this output.
  output.update(seed.dsg->clone(), seed.stamp);
}

/**
 * @brief Materialize the previous session's latest current state as the working
 * input to the ordinary hidden-change pipeline.
 *
 * This is a serialization adapter, not a D3-specific reconciliation path. It
 * intentionally imports the same two kinds of state that survive a D2 hidden
 * interval: current background geometry and current physical objects. New
 * observations are subsequently processed by the normal change detector and
 * reconciler.
 *
 * Agent poses are observation provenance, not persistent scene state. They are
 * not copied into a new process (whose pose IDs restart); per-surface sensor
 * bounds remain on mesh/object attributes and new-session rays are built only
 * from new observations.
 */
inline void initializeHiddenChangeWorkingDsg(
    const SessionSeed& seed,
    spark_dsg::DynamicSceneGraph& output,
    char memory_prefix = 'M') {
  if (!seed.dsg || !seed.dsg->hasMesh()) {
    throw std::invalid_argument("Session seed has no current background mesh");
  }

  output.setMesh(seed.dsg->mesh()->clone());
  if (!seed.dsg->hasLayer(khronos::DsgLayers::OBJECTS)) {
    return;
  }

  const auto& objects = seed.dsg->getLayer(khronos::DsgLayers::OBJECTS);
  for (const auto& [node_id, node] : objects.nodes()) {
    spark_dsg::NodeSymbol source(node_id);
    spark_dsg::NodeSymbol memory_id(memory_prefix, source.categoryId());
    while (output.hasNode(memory_id)) {
      ++memory_id;
    }
    output.emplaceNode(khronos::DsgLayers::OBJECTS,
                       memory_id,
                       node->attributes().clone());
  }
}

/**
 * @brief Seed the two backend working graphs while preserving Hydra's shared-mesh
 * invariant.
 *
 * Hydra's backend applies each MeshDelta only to the private graph's mesh, while
 * change detection snapshots the unmerged graph. The backend constructor makes
 * those graphs share one mesh pointer for exactly this reason. Seeding both
 * graphs independently would clone the prior mesh twice and silently freeze the
 * change-detection graph at the previous session's final geometry.
 */
inline void initializeHiddenChangeWorkingDsgPair(
    const SessionSeed& seed,
    spark_dsg::DynamicSceneGraph& private_graph,
    spark_dsg::DynamicSceneGraph& unmerged_graph,
    char memory_prefix = 'M') {
  initializeHiddenChangeWorkingDsg(seed, private_graph, memory_prefix);
  initializeHiddenChangeWorkingDsg(seed, unmerged_graph, memory_prefix);
  unmerged_graph.setMesh(private_graph.mesh());
}

}  // namespace session_update::runtime
