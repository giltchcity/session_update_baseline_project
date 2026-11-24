#pragma once
// Physical objects in the scene graph (session layer): the identity a node carries, the merge of
// the nodes of one identity, and the reduction of a snapshot to one node per identity.
#include <optional>
#include <vector>

#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/node_attributes.h>

#include "khronos/common/common_types.h"

namespace khronos {
class PersistentObjectState;

namespace session {

// The physical instance id a node carries (details "instance_id"), if any.
std::optional<size_t> physicalInstanceId(const KhronosObjectAttributes& attrs);

// The attributes of the merge of `nodes`, all of one physical identity (the scene-graph
// updater's merge of object nodes; for nodes without a physical id, nullptr).
spark_dsg::NodeAttributes::Ptr mergePhysicalAttributes(const DynamicSceneGraph& graph,
                                                       const std::vector<NodeId>& nodes);

/**
 * Consolidate duplicate segments carrying one physical instance ID.
 *
 * Change detection operates on an unmerged snapshot, whereas ordinary DSG update merges are
 * materialized only in the private graph. This explicit canonicalization therefore runs on the
 * snapshot that is reconciled and saved, guaranteeing one current node per physical object in
 * both D2 and D3.
 *
 * @param registry Backend-domain persistent physical-object registry. When non-null, the merged
 * geometry-authoritative fields (mesh, bounding box, position, geometry-support details) for each
 * physical ID are taken from that ID's persistent canonical geometry rather than solely from the
 * segments visible in this one call: track segments become observations of one accumulating
 * physical object rather than each canonicalization round re-electing a winner from scratch.
 * When null, a temporary registry local to this call is used, which reproduces the previous
 * fresh/single-round reduction.
 */
size_t canonicalizePhysicalObjects(DynamicSceneGraph& graph, PersistentObjectState* registry = nullptr);

}  // namespace session
}  // namespace khronos
