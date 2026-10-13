#include "session_update_baseline/panoptic_core/tsdf_conflict_checker.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace session_update::panoptic_core {

TsdfConflictChecker::TsdfConflictChecker()
    : TsdfConflictChecker(Config{}) {}

TsdfConflictChecker::TsdfConflictChecker(const Config& config)
    : config_(config) {}

PairEvaluation TsdfConflictChecker::evaluatePair(
    const SubmapSurface& reference,
    const SubmapSurface& query_submap,
    const DistanceQuery& query) const {
  PairEvaluation result;

  const Eigen::Isometry3d T_query_reference =
      query_submap.T_map_submap.inverse() * reference.T_map_submap;

  const double rejection_count =
      config_.normalize_by_voxel_weight
          ? std::numeric_limits<double>::max()
          : std::max(static_cast<double>(config_.match_rejection_points),
                     config_.match_rejection_percentage *
                         static_cast<double>(
                             reference.iso_surface_points.size()));

  const double rejection_distance =
      config_.error_threshold > 0.0
          ? config_.error_threshold
          : -config_.error_threshold * query_submap.voxel_size;

  for (const auto& point : reference.iso_surface_points) {
    if (point.weight < config_.min_voxel_weight) {
      continue;
    }

    const Eigen::Vector3d position_in_query =
        T_query_reference * point.position;
    const auto sample = query.query(position_in_query);
    if (!sample || sample->weight < config_.min_voxel_weight) {
      continue;
    }

    ++result.queried_points;

    double weight = 1.0;
    if (config_.normalize_by_voxel_weight) {
      weight = combinedWeight(sample->weight, point.weight);
      result.total_weight += weight;
    }

    double distance = sample->distance;

    if (query_submap.label == PanopticLabel::kFreeSpace) {
      // Original Panoptic predicate:
      //   if (distance >= rejection_distance) conflicting_points += weight;
      if (distance >= rejection_distance) {
        result.conflicting_weight += weight;
      }
    } else {
      // Original Panoptic class-layer guard:
      //   if class voxel exists and !belongsToSubmap(), set distance to
      //   truncation_distance. We expose this as sample.belongs_to_submap.
      if (!sample->belongs_to_submap) {
        distance = query_submap.truncation_distance;
      }

      // Original Panoptic predicates:
      //   distance <= -rejection_distance -> conflict
      //   distance <=  rejection_distance -> match
      if (distance <= -rejection_distance) {
        result.conflicting_weight += weight;
      } else if (distance <= rejection_distance) {
        result.matched_weight += weight;
      }
    }

    if (result.conflicting_weight > rejection_count) {
      result.conflicts = true;
      result.matches = false;
      return result;
    }
  }

  if (reference.iso_surface_points.empty()) {
    return result;
  }

  if (config_.normalize_by_voxel_weight) {
    const double rejection_weight =
        std::max(static_cast<double>(config_.match_rejection_points) /
                     static_cast<double>(reference.iso_surface_points.size()),
                 config_.match_rejection_percentage) *
        result.total_weight;

    if (result.conflicting_weight > rejection_weight) {
      result.conflicts = true;
      result.matches = false;
      return result;
    }

    const double acceptance_weight =
        std::max(static_cast<double>(config_.match_acceptance_points) /
                     static_cast<double>(reference.iso_surface_points.size()),
                 config_.match_acceptance_percentage) *
        result.total_weight;
    result.matches = result.matched_weight > acceptance_weight;
  } else {
    const double acceptance_count =
        std::max(static_cast<double>(config_.match_acceptance_points),
                 config_.match_acceptance_percentage *
                     static_cast<double>(
                         reference.iso_surface_points.size()));
    result.matches = result.matched_weight > acceptance_count;
  }

  result.conflicts = false;
  return result;
}

ChangeState TsdfConflictChecker::updateInactiveState(
    const SubmapSurface& inactive_reference,
    ChangeState current_state,
    const SubmapSurface& active_other,
    const DistanceQuery& active_query) const {
  const PairEvaluation pair =
      evaluatePair(inactive_reference, active_other, active_query);

  // Adapted from TsdfRegistrator::checkSubmapForChange():
  //
  //   if (submapsConflict(*submap, other, &submaps_match)) {
  //     submap->setChangeState(ChangeState::kAbsent);
  //   } else if (submap->getClassID() == other.getClassID() &&
  //              submaps_match) {
  //     submap->setChangeState(ChangeState::kPersistent);
  //   }
  if (pair.conflicts) {
    return ChangeState::kAbsent;
  }

  if (inactive_reference.class_id == active_other.class_id && pair.matches) {
    return ChangeState::kPersistent;
  }

  return current_state;
}

double TsdfConflictChecker::combinedWeight(double w1, double w2) const {
  // Adapted from TsdfRegistrator::computeCombinedWeight().
  if (w1 <= 0.0 || w2 <= 0.0) {
    return 0.0;
  }

  if (w1 >= config_.normalization_max_weight &&
      w2 >= config_.normalization_max_weight) {
    return 1.0;
  }

  return std::sqrt(
      std::min(w1 / config_.normalization_max_weight, 1.0) *
      std::min(w2 / config_.normalization_max_weight, 1.0));
}

}  // namespace session_update::panoptic_core
