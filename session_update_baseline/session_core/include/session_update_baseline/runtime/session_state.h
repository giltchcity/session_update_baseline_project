#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <khronos/utils/khronos_attribute_utils.h>
#include <spark_dsg/node_symbol.h>

#include "session_core/runtime/session_bundle.h"

namespace session_update::runtime {

// New-protocol output directories are immutable after publication. Validate the
// complete bundle before opening its requested map; genuine legacy maps remain readable.
inline std::unique_ptr<khronos::SpatioTemporalMap> loadSessionMap(const std::string& path) {
  uint64_t stamp=0;
  const bool bound=khronos::session_io::verifyBundle(path,nullptr,&stamp);
  auto map=khronos::SpatioTemporalMap::load(path);
  if (bound && map && (map->numTimeSteps()==0 || map->stamps().back()!=stamp ||
                       map->latest()!=stamp)) {
    throw std::runtime_error("Requested map timestamp does not match session bundle");
  }
  return map;
}

struct SessionSeed {
  khronos::TimeStamp stamp;
  spark_dsg::DynamicSceneGraph::Ptr dsg;
};

// README (16): one causal snapshot projection for seed, viewer, export and audit.
inline spark_dsg::DynamicSceneGraph::Ptr sessionSceneAt(
    const khronos::SpatioTemporalMap& map, khronos::TimeStamp stamp,
    khronos::TimeStamp* selected_stamp = nullptr) {
  const auto& stamps = map.stamps();
  const auto end = std::upper_bound(stamps.begin(),stamps.end(),stamp);
  if (end == stamps.begin()) return nullptr;
  const size_t index = static_cast<size_t>(end-stamps.begin()-1);
  if (selected_stamp) *selected_stamp = stamps[index];
  const auto scene = map.rawDsg(index);
  if (!scene) throw std::runtime_error("Unreadable session snapshot");
  return scene->clone();
}

// README (5), (16): read the state of the selected causal snapshot. A legacy
// snapshot carries this same presence state in native observation intervals.
inline bool hasSessionCurrentState(const khronos::KhronosObjectAttributes& attrs,
                                   khronos::TimeStamp snapshot_stamp) {
  const auto state = attrs.details.find("session_current_exists");
  return state != attrs.details.end() && state->second.size() == 1
      ? state->second.front() != 0 : khronos::isPresent(attrs, snapshot_stamp);
}

// Display is a projection of the stored state. Keep the source's closed nodes
// and attributes intact; reuse native mesh composition on a temporary view.
inline spark_dsg::Mesh::Ptr composeSessionCurrentMesh(
    const khronos::DynamicSceneGraph& dsg, khronos::TimeStamp snapshot_stamp) {
  auto visible = dsg.clone();
  std::vector<spark_dsg::NodeId> closed;
  if (visible->hasLayer(khronos::DsgLayers::OBJECTS)) {
    for (const auto& [id, node] : visible->getLayer(khronos::DsgLayers::OBJECTS).nodes()) {
      const auto* attrs = node->tryAttributes<khronos::KhronosObjectAttributes>();
      if (attrs && !hasSessionCurrentState(*attrs, snapshot_stamp)) closed.push_back(id);
    }
  }
  for (const auto id : closed) visible->removeNode(id);
  return khronos::composeCurrentSceneMesh(*visible);
}

inline SessionSeed latestSessionSeed(khronos::SpatioTemporalMap& prior) {
  if (prior.numTimeSteps() == 0) {
    throw std::invalid_argument("Cannot seed a session from an empty map");
  }
  const auto stamp = prior.latest();
  auto dsg = sessionSceneAt(prior,stamp);
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
