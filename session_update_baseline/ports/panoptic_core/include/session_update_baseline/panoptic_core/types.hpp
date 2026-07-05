#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Geometry>

namespace session_update::panoptic_core {

// Adapted from panoptic_mapping/common/common.h.
enum class PanopticLabel { kUnknown = 0, kInstance, kBackground, kFreeSpace };

inline std::string toString(PanopticLabel label) {
  switch (label) {
    case PanopticLabel::kUnknown:
      return "Unknown";
    case PanopticLabel::kInstance:
      return "Instance";
    case PanopticLabel::kBackground:
      return "Background";
    case PanopticLabel::kFreeSpace:
      return "FreeSpace";
  }
  return "UnknownPanopticLabel";
}

// Adapted from panoptic_mapping/common/common.h.
enum class ChangeState {
  kNew = 0,
  kMatched,
  kUnobserved,
  kAbsent,
  kPersistent,
};

inline std::string toString(ChangeState state) {
  switch (state) {
    case ChangeState::kNew:
      return "New";
    case ChangeState::kMatched:
      return "Matched";
    case ChangeState::kUnobserved:
      return "Unobserved";
    case ChangeState::kAbsent:
      return "Absent";
    case ChangeState::kPersistent:
      return "Persistent";
  }
  return "UnknownChangeState";
}

// Adapted from panoptic_mapping::IsoSurfacePoint.
struct IsoSurfacePoint {
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  double weight = 0.0;
};

// Lightweight replacement for the parts of panoptic_mapping::Submap used by
// TsdfRegistrator::submapsConflict().
struct SubmapSurface {
  int id = -1;
  int class_id = -1;
  std::string name;
  PanopticLabel label = PanopticLabel::kUnknown;
  double voxel_size = 0.05;
  double truncation_distance = 0.15;
  Eigen::Isometry3d T_map_submap = Eigen::Isometry3d::Identity();
  std::vector<IsoSurfacePoint> iso_surface_points;
};

struct DistanceSample {
  double distance = 0.0;
  double weight = 0.0;
  bool belongs_to_submap = true;
};

// Replacement for voxblox::Interpolator<TsdfVoxel>. This is the seam where a
// Khronos mesh, a TSDF layer, or an exported point cloud can provide the same
// distance/weight query used by Panoptic's original conflict rule.
class DistanceQuery {
 public:
  virtual ~DistanceQuery() = default;

  virtual std::optional<DistanceSample> query(
      const Eigen::Vector3d& position_in_query_map) const = 0;
};

struct PairEvaluation {
  bool conflicts = false;
  bool matches = false;
  double conflicting_weight = 0.0;
  double matched_weight = 0.0;
  double total_weight = 0.0;
  std::size_t queried_points = 0;
};

}  // namespace session_update::panoptic_core

