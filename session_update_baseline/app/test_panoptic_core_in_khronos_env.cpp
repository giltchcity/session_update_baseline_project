#include <cstdlib>
#include <iostream>
#include <optional>

#include "session_update_baseline/panoptic_core/tsdf_conflict_checker.hpp"

namespace panoptic = session_update::panoptic_core;

class ConstantDistanceQuery final : public panoptic::DistanceQuery {
 public:
  explicit ConstantDistanceQuery(double distance) : distance_(distance) {}

  std::optional<panoptic::DistanceSample> query(
      const Eigen::Vector3d&) const override {
    return panoptic::DistanceSample{distance_, 1.0, true};
  }

 private:
  double distance_;
};

int main() {
  panoptic::TsdfConflictChecker::Config config;
  config.normalize_by_voxel_weight = false;
  config.match_rejection_points = 0;
  config.match_rejection_percentage = 0.0;
  config.match_acceptance_points = 0;
  config.match_acceptance_percentage = 0.0;

  panoptic::SubmapSurface old_surface;
  old_surface.class_id = 7;
  old_surface.label = panoptic::PanopticLabel::kInstance;
  old_surface.voxel_size = 0.05;
  old_surface.iso_surface_points.push_back(
      panoptic::IsoSurfacePoint{Eigen::Vector3d::Zero(), 1.0});

  panoptic::SubmapSurface current_surface = old_surface;
  panoptic::TsdfConflictChecker checker(config);

  ConstantDistanceQuery matching_query(0.0);
  const auto persistent = checker.updateInactiveState(
      old_surface,
      panoptic::ChangeState::kUnobserved,
      current_surface,
      matching_query);
  if (persistent != panoptic::ChangeState::kPersistent) {
    std::cerr << "Expected persistent state\n";
    return EXIT_FAILURE;
  }

  ConstantDistanceQuery conflicting_query(-0.20);
  const auto absent = checker.updateInactiveState(
      old_surface,
      panoptic::ChangeState::kUnobserved,
      current_surface,
      conflicting_query);
  if (absent != panoptic::ChangeState::kAbsent) {
    std::cerr << "Expected absent state\n";
    return EXIT_FAILURE;
  }

  std::cout << "PANOPTIC_CORE_PORT_OK persistent_and_absent\n";
  return EXIT_SUCCESS;
}
