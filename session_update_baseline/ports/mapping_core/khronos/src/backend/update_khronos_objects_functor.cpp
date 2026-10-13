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

void declare_config(UpdateKhronosObjectsFunctor::Config& config) {
  using namespace config;
  name("UpdateKhronosObjectsFunctor::Config");
  field(config.deformation_interpolator, "deformation_interpolator");
  field(config.merge_proposer, "merge_proposer");
  field(config.merge_require_same_label, "merge_require_same_label");
  field(config.merge_require_no_co_visibility, "merge_require_no_co_visibility");
  field(config.merge_min_iou, "merge_min_iou");
}

UpdateKhronosObjectsFunctor::UpdateKhronosObjectsFunctor(const Config& config)
    : config(config::checkValid(config)),
      merge_proposer(config.merge_proposer),
      deformation_interpolator(config.deformation_interpolator) {}

hydra::UpdateFunctor::Hooks UpdateKhronosObjectsFunctor::hooks() const {
  auto my_hooks = UpdateFunctor::hooks();
  my_hooks.find_merges = [this](const auto& graph, const auto& info) {
    return findMerges(graph, info);
  };
  my_hooks.merge = &UpdateKhronosObjectsFunctor::mergeObjectAttributes;

  return my_hooks;
}

void UpdateKhronosObjectsFunctor::call(const DynamicSceneGraph& unmerged,
                                       SharedDsgInfo& dsg,
                                       const UpdateInfo::ConstPtr& info) const {
  Timer spin_timer("backend/update_khronos_objects", info->timestamp_ns);
  if (!unmerged.hasLayer(DsgLayers::OBJECTS)) {
    VLOG(5) << "Skipping khronos object update due to missing layer";
    return;
  }

  // we want to use the unmerged graph for most things
  const auto& objects = unmerged.getLayer(DsgLayers::OBJECTS);
  // we want to iterate over the unmerged graph
  const auto new_loopclosure = info->loop_closure_detected;
  active_tracker.clear();  // reset from previous pass
  LayerView view = new_loopclosure ? LayerView(objects) : active_tracker.view(objects);

  // interpolate to update
  deformation_interpolator.interpolateNodePositions(unmerged, *dsg.graph, info, view);
}

MergeList UpdateKhronosObjectsFunctor::findMerges(const DynamicSceneGraph& graph,
                                                  const UpdateInfo::ConstPtr& info) const {
  if (!graph.hasLayer(DsgLayers::OBJECTS)) {
    return {};
  }

  const auto new_lcd = info->loop_closure_detected;
  const auto& objects = graph.getLayer(DsgLayers::OBJECTS);
  // freeze layer view to avoid messing with tracker
  LayerView view = new_lcd ? LayerView(objects) : active_tracker.view(objects, true);

  MergeList proposals;

  // A physical instance ID is an identity key, not a semantic category.  It must survive a
  // tracking-window break and a serialized session boundary, even when the object moved far enough
  // that its old and new boxes no longer overlap.  Consolidate every currently touched physical-ID
  // group into its oldest deterministic node.  Nodes without a physical ID continue to use the
  // upstream semantic/geometric association below.
  std::set<NodeId> touched_nodes;
  for (const auto& node : view) {
    touched_nodes.insert(node.id);
  }

  std::map<size_t, std::vector<const SceneGraphNode*>> physical_groups;
  for (const auto& [id, node] : objects.nodes()) {
    const auto attrs = node->tryAttributes<KhronosObjectAttributes>();
    if (!attrs) {
      continue;
    }
    const auto instance_id = getPhysicalInstanceId(*attrs);
    if (instance_id) {
      physical_groups[*instance_id].push_back(node.get());
    }
  }

  for (const auto& [instance_id, nodes] : physical_groups) {
    (void)instance_id;
    if (nodes.size() < 2 ||
        std::none_of(nodes.begin(), nodes.end(), [&touched_nodes](const auto* node) {
          return touched_nodes.count(node->id) > 0;
        })) {
      continue;
    }

    const auto canonical = *std::min_element(
        nodes.begin(), nodes.end(), [](const SceneGraphNode* lhs, const SceneGraphNode* rhs) {
          const auto& lhs_attrs = lhs->template attributes<KhronosObjectAttributes>();
          const auto& rhs_attrs = rhs->template attributes<KhronosObjectAttributes>();
          const auto lhs_time = firstObservation(lhs_attrs);
          const auto rhs_time = firstObservation(rhs_attrs);
          return lhs_time == rhs_time ? lhs->id < rhs->id : lhs_time < rhs_time;
        });
    for (const auto* node : nodes) {
      if (node->id != canonical->id) {
        proposals.push_back({node->id, canonical->id});
      }
    }
  }

  merge_proposer.findMerges(
      objects,
      view,
      [this](const SceneGraphNode& lhs, const SceneGraphNode& rhs) {
        const auto lhs_attrs = lhs.tryAttributes<spark_dsg::KhronosObjectAttributes>();
        const auto rhs_attrs = rhs.tryAttributes<spark_dsg::KhronosObjectAttributes>();

        if (!lhs_attrs || !rhs_attrs) {
          return false;
        }

        // Explicit physical identities are handled above.  Never let a semantic/bbox heuristic
        // merge a physical object with a different ID (for example the three S74 computers).
        if (getPhysicalInstanceId(*lhs_attrs) || getPhysicalInstanceId(*rhs_attrs)) {
          return false;
        }

        if (config.merge_require_same_label) {
          if (lhs_attrs->semantic_label != rhs_attrs->semantic_label) {
            return false;
          }
        }

        if (config.merge_require_no_co_visibility) {
          if (lhs_attrs->first_observed_ns.front() < rhs_attrs->first_observed_ns.front() &&
              lhs_attrs->last_observed_ns.front() > rhs_attrs->first_observed_ns.front()) {
            return false;
          }

          if (lhs_attrs->last_observed_ns.front() > rhs_attrs->first_observed_ns.front() &&
              lhs_attrs->first_observed_ns.front() < rhs_attrs->last_observed_ns.front()) {
            return false;
          }
        }

        if (!lhs_attrs->bounding_box.intersects(rhs_attrs->bounding_box)) {
          return false;
        }

        return config.merge_min_iou == 0 ||
               lhs_attrs->bounding_box.computeIoU(rhs_attrs->bounding_box) >= config.merge_min_iou;
      },
      proposals);
  return proposals;
}

}  // namespace khronos
