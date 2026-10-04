#include "session_core/adapters/physical_object_access.h"
#include "khronos/backend/update_khronos_objects_functor.h"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <config_utilities/config.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <hydra/backend/backend_utilities.h>
#include <hydra/common/global_info.h>
#include <hydra/utils/mesh_utilities.h>
#include <kimera_pgmo/deformation_graph.h>
#include <kimera_pgmo/utils/common_functions.h>

#include "khronos/utils/khronos_attribute_utils.h"

namespace khronos {
using session_detail::getPhysicalInstanceId;
using session_detail::firstObservation;
std::optional<size_t> UpdateKhronosObjectsFunctor::physicalInstanceId(
    const KhronosObjectAttributes& attrs) {
  return getPhysicalInstanceId(attrs);
}

spark_dsg::NodeAttributes::Ptr UpdateKhronosObjectsFunctor::mergeObjectAttributes(
    const DynamicSceneGraph& graph, const std::vector<NodeId>& nodes) {
  if (nodes.empty()) {
    return nullptr;
  }

  struct Segment {
    NodeId node_id;
    const KhronosObjectAttributes* attrs;
  };
  std::vector<Segment> segments;
  segments.reserve(nodes.size());
  for (const auto node_id : nodes) {
    if (!graph.hasNode(node_id)) {
      continue;
    }
    const auto* candidate =
        graph.getNode(node_id).tryAttributes<KhronosObjectAttributes>();
    if (candidate) {
      segments.push_back({node_id, candidate});
    }
  }
  if (segments.empty()) {
    return graph.getNode(nodes.front()).attributes().clone();
  }

  const auto shared_instance = getPhysicalInstanceId(*segments.front().attrs);
  const bool one_physical_object =
      shared_instance &&
      std::all_of(segments.begin(), segments.end(), [shared_instance](const auto& item) {
        return getPhysicalInstanceId(*item.attrs) == shared_instance;
      });
  if (!one_physical_object) {
    return segments.front().attrs->clone();
  }

  for (const auto& segment : segments) {
    if (segment.attrs->first_observed_ns.size() !=
            segment.attrs->last_observed_ns.size() ||
        segment.attrs->first_observed_ns.empty()) {
      throw std::invalid_argument(
          "Cannot canonicalize a physical segment with invalid presence vectors");
    }
    for (size_t i = 0; i < segment.attrs->first_observed_ns.size(); ++i) {
      if (segment.attrs->first_observed_ns[i] >
          segment.attrs->last_observed_ns[i]) {
        throw std::invalid_argument(
            "Cannot canonicalize an inverted physical presence interval");
      }
    }
  }

  // Reduce visibility segments in real direct-observation order. Stable node
  // IDs are only the deterministic final tie breaker.
  std::sort(segments.begin(), segments.end(), [](const auto& lhs, const auto& rhs) {
    return std::make_tuple(observationFirstStamp(*lhs.attrs),
                           observationLastStamp(*lhs.attrs),
                           lhs.node_id) <
           std::make_tuple(observationFirstStamp(*rhs.attrs),
                           observationLastStamp(*rhs.attrs),
                           rhs.node_id);
  });

  // Geometry ownership (Invariants 1/2): the newest direct segment defines the
  // presence right boundary, but it does not unconditionally own the merged
  // static geometry. An established high-confidence current mesh must not be
  // regressed by a weaker later observation:
  //  - trajectory-only segments (empty mesh) never own static geometry;
  //  - a segment carrying D1 motion evidence (tracker-measured displacement
  //    >= min_dynamic_displacement) takes over: its pose is the new current;
  //  - otherwise a stationary re-observation takes over only with at least as
  //    many reconstruction frames as the current holder (support gate). With
  //    less support the established holder keeps its mesh/bbox/position and
  //    the new segment only extends presence and trajectory history.
  constexpr size_t kLegacySeedSupport = 1000;  // pre-fix attrs without the
                                               // detail are treated as
                                               // established (protected)
  const auto detailValue = [](const KhronosObjectAttributes& attrs,
                              const char* key) -> size_t {
    const auto iter = attrs.details.find(key);
    return (iter == attrs.details.end() || iter->second.empty()) ? 0
                                                                 : iter->second.front();
  };
  const auto reconstructionSupport =
      [&](const KhronosObjectAttributes& attrs) -> size_t {
    const size_t support = detailValue(attrs, kReconstructionFramesDetail);
    return support == 0 ? kLegacySeedSupport : support;
  };
  const auto hasMotionEvidence = [&](const KhronosObjectAttributes& attrs) {
    return detailValue(attrs, kHasDynamicHistoryDetail) != 0;
  };

  const KhronosObjectAttributes* geometry_holder = nullptr;
  for (const auto& segment : segments) {
    const auto* attrs = segment.attrs;
    if (attrs->mesh.points.empty()) {
      continue;  // trajectory-only segment: never owns static geometry
    }
    if (!geometry_holder) {
      geometry_holder = attrs;
      continue;
    }
    if (hasMotionEvidence(*attrs)) {
      geometry_holder = attrs;  // D1: tracker confirmed a real move
      continue;
    }
    if (reconstructionSupport(*attrs) >= reconstructionSupport(*geometry_holder)) {
      geometry_holder = attrs;  // equal-or-better supported re-observation
    }
    // else: weaker stationary re-observation -> established geometry is kept.
  }

  // The newest direct segment exclusively owns the right/presence boundary. A
  // trajectory-only newest segment intentionally keeps an empty static mesh;
  // borrowing an older settled mesh would put D1 motion geometry back into the
  // static map.
  const auto newest_iter = std::max_element(
      segments.begin(), segments.end(), [](const auto& lhs, const auto& rhs) {
        return std::make_tuple(observationLastStamp(*lhs.attrs),
                               observationFirstStamp(*lhs.attrs),
                               lhs.node_id) <
               std::make_tuple(observationLastStamp(*rhs.attrs),
                               observationFirstStamp(*rhs.attrs),
                               rhs.node_id);
      });
  const auto* newest = newest_iter->attrs;
  auto result = newest->clone();
  auto& merged = *CHECK_NOTNULL(
      dynamic_cast<KhronosObjectAttributes*>(result.get()));
  // Gate the geometric takeover: current geometry comes from the support gate's
  // holder, not unconditionally from the newest segment. A trajectory-only
  // newest segment keeps its empty mesh (D1 intent, see comment above).
  if (geometry_holder && geometry_holder != newest &&
      !newest->mesh.points.empty()) {
    // Compatibility path (P54): with a registry the materialized state geometry replaces this
    // choice; logged so that production runs show whether it is ever taken.
    LOG(INFO) << "LEGACY_GEOMETRY_HOLDER inst=" << *shared_instance;
    merged.mesh = geometry_holder->mesh;
    merged.bounding_box = geometry_holder->bounding_box;
    merged.position = geometry_holder->position;
    merged.details[kReconstructionFramesDetail] = {
        detailValue(*geometry_holder, kReconstructionFramesDetail)};
    merged.details[kHasDynamicHistoryDetail] = {
        detailValue(*geometry_holder, kHasDynamicHistoryDetail)};
  }
  const auto authoritative_right = merged.last_observed_ns.empty()
                                       ? observationLastStamp(*newest)
                                       : merged.last_observed_ns.back();

  // Each historical segment may extend only to the next direct segment. This
  // pairwise cap preserves real finite gaps in a three-or-more-segment history:
  // clipping every old segment straight to newest.first would let S1's open
  // interval jump across a finite S2->S3 absence gap.
  using PresenceInterval = std::pair<TimeStamp, TimeStamp>;
  std::vector<PresenceInterval> candidates;
  for (size_t segment_index = 0; segment_index < segments.size();
       ++segment_index) {
    const auto& segment = segments[segment_index];
    const auto* source = segment.attrs;
    const auto next_direct_first = segment_index + 1 < segments.size()
                                       ? observationFirstStamp(
                                             *segments[segment_index + 1].attrs)
                                       : 0;
    for (size_t i = 0; i < source->first_observed_ns.size(); ++i) {
      auto clipped_last = source->last_observed_ns[i];
      if (source != newest && next_direct_first > 0 &&
          clipped_last > next_direct_first) {
        clipped_last = next_direct_first;
      }
      if (source != newest && clipped_last > authoritative_right) {
        clipped_last = authoritative_right;
      }
      if (source->first_observed_ns[i] <= clipped_last) {
        candidates.emplace_back(source->first_observed_ns[i], clipped_last);
      }
    }
  }

  // Canonicalization first clips every segment, then performs one batch union.
  // Do not mutate the result interval-by-interval: later segment boundaries
  // must be evaluated against the original history so finite gaps remain
  // deterministic.
  std::sort(candidates.begin(), candidates.end());
  std::vector<PresenceInterval> reduced;
  for (const auto& candidate : candidates) {
    if (reduced.empty() || candidate.first > reduced.back().second) {
      reduced.push_back(candidate);
    } else {
      reduced.back().second = std::max(reduced.back().second, candidate.second);
    }
  }

  // The newest segment is the only source allowed to define the final right
  // edge. This assertion is materialized rather than inferred from union order
  // so an optimistic old interval can never resurrect a terminally absent ID.
  while (!reduced.empty() && reduced.back().first > authoritative_right) {
    reduced.pop_back();
  }
  if (!reduced.empty()) {
    reduced.back().second = authoritative_right;
  }
  merged.first_observed_ns.clear();
  merged.last_observed_ns.clear();
  for (const auto& interval : reduced) {
    if (interval.first <= interval.second) {
      merged.first_observed_ns.push_back(interval.first);
      merged.last_observed_ns.push_back(interval.second);
    }
  }
  std::vector<const KhronosObjectAttributes*> attrs;
  attrs.reserve(segments.size());
  for (const auto& segment : segments) {
    attrs.push_back(segment.attrs);
  }
  mergeTrajectoryHistory(attrs, merged);
  setObservationBounds(
      merged, observationFirstStamp(*newest), observationLastStamp(*newest));
  return result;
}

size_t UpdateKhronosObjectsFunctor::canonicalizePhysicalObjects(
    DynamicSceneGraph& graph, PersistentObjectState* registry) {
  if (!graph.hasLayer(DsgLayers::OBJECTS)) {
    return 0;
  }

  // A null registry reproduces the previous fresh/single-round reduction for
  // existing call sites that never threaded a persistent registry through:
  // `mergeObjectAttributes`' winner-takes-all geometry stays authoritative and
  // this local instance never survives past this one call. The persistent
  // fragment materialization (applyPhysicalGeometry) only runs when a real
  // registry owns the physical-ID history across rounds.
  PersistentObjectState local_registry;
  PersistentObjectState& effective_registry = registry ? *registry : local_registry;

  std::map<size_t, std::vector<NodeId>> groups;
  const auto& objects = graph.getLayer(DsgLayers::OBJECTS);
  for (const auto& [node_id, node] : objects.nodes()) {
    const auto attrs = node->tryAttributes<KhronosObjectAttributes>();
    if (!attrs) {
      continue;
    }
    if (const auto instance_id = getPhysicalInstanceId(*attrs)) {
      groups[*instance_id].push_back(node_id);
    }
  }

  size_t merged_count = 0;
  for (auto& [instance_id, nodes] : groups) {
    (void)instance_id;
    // Singleton groups still need the registry write-back: a physical object
    // whose state was closed by surface evidence (for example the A-only I20
    // laptop in Session B) has no second node to trigger a merge, but its
    // CURRENT mesh still has to be materialized as empty on the surviving
    // node. Skipping singletons is what left closed objects visible as ghosts.
    if (nodes.size() < 2) {
      const auto target = nodes.front();
      const auto* source_attrs =
          graph.getNode(target).tryAttributes<KhronosObjectAttributes>();
      if (!source_attrs) {
        continue;
      }
      auto merged_attrs = source_attrs->clone();
      if (registry && dynamic_cast<KhronosObjectAttributes*>(merged_attrs.get())) {
        effective_registry.applyPhysicalGeometry(graph, nodes,
            *static_cast<KhronosObjectAttributes*>(merged_attrs.get()));
      }
      if (!graph.setNodeAttributes(target, std::move(merged_attrs))) {
        throw std::runtime_error(
            "Failed to materialize singleton physical object attributes");
      }
      continue;
    }
    std::sort(nodes.begin(), nodes.end(), [&graph](NodeId lhs, NodeId rhs) {
      const auto& lhs_attrs =
          graph.getNode(lhs).attributes<KhronosObjectAttributes>();
      const auto& rhs_attrs =
          graph.getNode(rhs).attributes<KhronosObjectAttributes>();
      const auto lhs_first = firstObservation(lhs_attrs);
      const auto rhs_first = firstObservation(rhs_attrs);
      return lhs_first == rhs_first ? lhs < rhs : lhs_first < rhs_first;
    });

    const auto target = nodes.front();
    auto merged_attrs = mergeObjectAttributes(graph, nodes);
    if (registry && dynamic_cast<KhronosObjectAttributes*>(merged_attrs.get())) {
      effective_registry.applyPhysicalGeometry(graph, nodes,
          *static_cast<KhronosObjectAttributes*>(merged_attrs.get()));
    }
    for (size_t i = 1; i < nodes.size(); ++i) {
      if (graph.mergeNodes(nodes[i], target)) {
        ++merged_count;
      }
    }
    if (!graph.setNodeAttributes(target, std::move(merged_attrs))) {
      throw std::runtime_error(
          "Failed to materialize canonical physical object attributes");
    }
  }
  return merged_count;
}


}  // namespace khronos
