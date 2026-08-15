#pragma once
#include <optional>

#include <config_utilities/factory.h>
#include <hydra/backend/association_strategies.h>
#include <hydra/backend/deformation_interpolator.h>
#include <hydra/backend/merge_tracker.h>
#include <hydra/backend/update_functions.h>
#include <hydra/utils/active_window_tracker.h>

#include "khronos/backend/reconciliation/persistent_object_state.h"
#include "khronos/common/common_types.h"
namespace khronos {
using hydra::MergeList;
using hydra::SharedDsgInfo;
using hydra::UpdateInfo;
using spark_dsg::LayerView;
struct UpdateKhronosObjectsFunctor : public hydra::UpdateFunctor {
  struct Config {
    //! Interpolator for object nodes using deformation graph
    hydra::DeformationInterpolator::Config deformation_interpolator;
    //! Require merges to have same semantic label
    bool merge_require_same_label = true;
    //! Require merges to not be co-visible
    bool merge_require_no_co_visibility = false;
    //! Min IOU to be considered a merge
    double merge_min_iou = 0.5;
    //! Association strategy for finding matches to active nodes
    hydra::MergeProposer::Config merge_proposer = {
        config::VirtualConfig<hydra::AssociationStrategy>{
            hydra::association::SemanticNearestNode::Config{}}};
  } const config;

  explicit UpdateKhronosObjectsFunctor(const Config& config);
  Hooks hooks() const override;
  void call(const DynamicSceneGraph& unmerged,
            SharedDsgInfo& dsg,
            const UpdateInfo::ConstPtr& info) const override;

  MergeList findMerges(const DynamicSceneGraph& graph, const UpdateInfo::ConstPtr& info) const;

  static std::optional<size_t> physicalInstanceId(const KhronosObjectAttributes& attrs);
  static spark_dsg::NodeAttributes::Ptr mergeObjectAttributes(
      const DynamicSceneGraph& graph, const std::vector<NodeId>& nodes);

  /**
   * @brief Consolidate duplicate segments carrying one physical instance ID.
   *
   * Change detection operates on an unmerged snapshot, whereas ordinary DSG
   * update merges are materialized only in the private graph. This explicit
   * canonicalization therefore runs on the snapshot that is reconciled and
   * saved, guaranteeing one current node per physical object in both D2 and D3.
   *
   * @param registry Backend-domain persistent physical-object registry. When
   * non-null, the merged geometry-authoritative fields (mesh, bounding box,
   * position, geometry-support details) for each physical ID are taken from
   * that ID's persistent canonical geometry rather than solely from the
   * segments visible in this one call: track segments become observations
   * of one accumulating physical object rather than each canonicalization
   * round re-electing a winner from scratch. When null, a temporary
   * registry local to this call is used, which reproduces the previous
   * fresh/single-round reduction (existing single-argument call sites keep
   * their prior behavior unchanged).
   */
  static size_t canonicalizePhysicalObjects(DynamicSceneGraph& graph,
                                            PersistentObjectState* registry = nullptr);

  const hydra::MergeProposer merge_proposer;
  const hydra::DeformationInterpolator deformation_interpolator;

  mutable hydra::ActiveWindowTracker active_tracker;
  mutable std::unordered_map<NodeId, Eigen::Vector3d> cached_pos_;

 private:
  inline static const auto registration_ =
      config::RegistrationWithConfig<UpdateFunctor, UpdateKhronosObjectsFunctor, Config>(
          "UpdateKhronosObjectsFunctor");
};

void declare_config(UpdateKhronosObjectsFunctor::Config& config);

}  // namespace khronos
