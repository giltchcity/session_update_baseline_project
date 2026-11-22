#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "khronos/active_window/online_front.h"

namespace khronos {

// Local identity evidence from a completed visit, expressed in world coordinates.
// An unexplained or independently supported measurement at another location keeps
// its proposed identity. A handoff never merges two physical IDs.
class PredictedSiteHandoff {
 public:
  explicit PredictedSiteHandoff(const std::string& path);

  uint64_t sourceStamp() const { return source_stamp_; }
  size_t size() const { return sites_.size(); }
  void validateSeed(uint64_t stamp, const std::map<int, int>& classes) const;

  // The original map and its historical evidence remain owned by the caller.
  bool keepInheritedVertex(int id, int semantic, const Eigen::Vector3d& point) const;
  void applyFrame(const cv::Mat& depth, const cv::Mat& raw_semantics, cv::Mat& ids,
                  cv::Mat& semantics, const OnlineFront::Intrinsics& camera,
                  const Eigen::Isometry3d& pose, int64_t stamp);

 private:
  using Cell = std::array<int64_t, 3>;
  enum class Role { Unresolved, Shared, Independent };
  struct Sample {
    Eigen::Vector3d point;
    Role role = Role::Unresolved;
    int64_t balance = 0;
  };
  struct Site {
    int source, semantic, host, host_semantic;
    std::vector<Sample> samples;
    std::map<Cell, size_t> cells;
  };

  Cell cell(const Eigen::Vector3d& point) const;
  const Site* explanatorySite(int id, int semantic, const Eigen::Vector3d& point) const;
  void observe(const cv::Mat& depth, const cv::Mat& raw_semantics, const cv::Mat& ids,
               const cv::Mat& semantics, const OnlineFront::Intrinsics& camera,
               const Eigen::Isometry3d& pose);

  double resolution_;
  uint64_t source_stamp_;
  int64_t last_stamp_ = -1;
  std::vector<Site> sites_;
  std::map<int, std::vector<size_t>> by_source_;
};

}  // namespace khronos
