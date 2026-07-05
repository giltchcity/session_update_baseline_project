#pragma once

#include "session_update_baseline/panoptic_core/types.hpp"

namespace session_update::panoptic_core {

// Portable version of Panoptic Mapping's TsdfRegistrator::submapsConflict().
//
// Source:
//   vendor/panoptic_mapping/src/map_management/tsdf_registrator.cpp
//
// Key retained predicates:
//   - Free-space active submap conflicts when distance >= rejection_distance.
//   - Non-free-space active submap conflicts when distance <= -rejection_distance.
//   - Non-free-space active submap matches when distance <= rejection_distance.
//   - Weighted rejection/acceptance thresholds follow Panoptic's original
//     normalize_by_voxel_weight branch.
class TsdfConflictChecker {
 public:
  struct Config {
    double min_voxel_weight = 1e-6;
    double error_threshold = -1.0;
    int match_rejection_points = 50;
    double match_rejection_percentage = 0.1;
    int match_acceptance_points = 50;
    double match_acceptance_percentage = 0.1;
    bool normalize_by_voxel_weight = true;
    double normalization_max_weight = 5000.0;
  };

  TsdfConflictChecker();
  explicit TsdfConflictChecker(const Config& config);

  PairEvaluation evaluatePair(const SubmapSurface& reference,
                              const SubmapSurface& query_submap,
                              const DistanceQuery& query) const;

  ChangeState updateInactiveState(const SubmapSurface& inactive_reference,
                                  ChangeState current_state,
                                  const SubmapSurface& active_other,
                                  const DistanceQuery& active_query) const;

 private:
  double combinedWeight(double w1, double w2) const;

  Config config_;
};

}  // namespace session_update::panoptic_core
