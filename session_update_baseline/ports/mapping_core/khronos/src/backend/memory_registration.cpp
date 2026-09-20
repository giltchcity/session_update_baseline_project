#include "khronos/backend/memory_registration.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/SVD>
#include <glog/logging.h>
#include <hydra/utils/nearest_neighbor_utilities.h>

namespace khronos {

namespace {

// Horn's closed-form rigid fit: the transform T with source*T ~ target.
Eigen::Isometry3f fitRigid(const Points& source, const Points& target) {
  Eigen::Vector3f source_mean = Eigen::Vector3f::Zero();
  Eigen::Vector3f target_mean = Eigen::Vector3f::Zero();
  for (size_t i = 0; i < source.size(); ++i) {
    source_mean += source[i];
    target_mean += target[i];
  }
  source_mean /= static_cast<float>(source.size());
  target_mean /= static_cast<float>(target.size());

  Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
  for (size_t i = 0; i < source.size(); ++i) {
    covariance += (source[i] - source_mean) * (target[i] - target_mean).transpose();
  }
  const Eigen::JacobiSVD<Eigen::Matrix3f> svd(covariance,
                                              Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3f rotation = svd.matrixV() * svd.matrixU().transpose();
  if (rotation.determinant() < 0.f) {
    Eigen::Matrix3f flip = Eigen::Matrix3f::Identity();
    flip(2, 2) = -1.f;
    rotation = svd.matrixV() * flip * svd.matrixU().transpose();
  }
  Eigen::Isometry3f transform = Eigen::Isometry3f::Identity();
  transform.linear() = rotation;
  transform.translation() = target_mean - rotation * source_mean;
  return transform;
}

float median(std::vector<float>& values) {
  if (values.empty()) return 0.f;
  const size_t middle = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + middle, values.end());
  return values[middle];
}

}  // namespace

MemoryRegistration::Result MemoryRegistration::estimate(const Points& memory,
                                                        const Points& session,
                                                        const Config& config) {
  Result result;
  if (memory.size() < config.min_pairs || session.size() < config.min_pairs ||
      config.association_tolerance <= 0.f) {
    return result;
  }

  const hydra::PointNeighborSearch search(session);
  const float association_squared = config.association_tolerance * config.association_tolerance;
  Eigen::Isometry3f estimate = Eigen::Isometry3f::Identity();

  for (size_t iteration = 0; iteration < config.iterations; ++iteration) {
    // Correspondences under the current estimate.
    Points source;
    Points target;
    std::vector<float> distances;
    source.reserve(memory.size());
    target.reserve(memory.size());
    distances.reserve(memory.size());
    for (const auto& point : memory) {
      const Point moved = estimate * point;
      float distance_squared = std::numeric_limits<float>::max();
      size_t index = 0;
      if (!search.search(moved, distance_squared, index) ||
          distance_squared > association_squared) {
        continue;
      }
      source.push_back(point);
      target.push_back(session[index]);
      distances.push_back(std::sqrt(distance_squared));
    }
    if (source.size() < config.min_pairs) {
      return result;
    }
    if (iteration == 0) {
      std::vector<float> initial = distances;
      result.median_before_m = median(initial);
    }

    // Robust trim: keep pairs within the median plus a median-absolute-deviation
    // band, so a minority of pairs that associate across different surfaces
    // cannot drag the fit.
    std::vector<float> sorted = distances;
    const float centre = median(sorted);
    std::vector<float> deviations;
    deviations.reserve(distances.size());
    for (const float distance : distances) {
      deviations.push_back(std::abs(distance - centre));
    }
    const float spread = median(deviations);
    const float limit = centre + 3.f * spread;
    Points inlier_source;
    Points inlier_target;
    for (size_t i = 0; i < source.size(); ++i) {
      if (distances[i] <= limit) {
        inlier_source.push_back(source[i]);
        inlier_target.push_back(target[i]);
      }
    }
    if (inlier_source.size() < config.min_pairs) {
      return result;
    }
    estimate = fitRigid(inlier_source, inlier_target);
    result.pairs = inlier_source.size();
  }

  // Residual after the fit, on the same association rule.
  std::vector<float> after;
  after.reserve(memory.size());
  for (const auto& point : memory) {
    const Point moved = estimate * point;
    float distance_squared = std::numeric_limits<float>::max();
    size_t index = 0;
    if (search.search(moved, distance_squared, index) &&
        distance_squared <= association_squared) {
      after.push_back(std::sqrt(distance_squared));
    }
  }
  result.median_after_m = median(after);

  const Eigen::AngleAxisf angle(estimate.linear());
  result.rotation_deg = std::abs(angle.angle()) * 180.f / static_cast<float>(M_PI);
  result.translation_m = estimate.translation().norm();
  if (result.rotation_deg > config.max_rotation_deg ||
      result.translation_m > config.max_translation_m) {
    VLOG(1) << "[MemoryRegistration] Rejected implausible correction: "
            << result.rotation_deg << " deg, " << result.translation_m << " m.";
    return result;
  }
  // A correction below the reconstruction agreement scale, or one that does not
  // reduce the residual, is meshing noise rather than a frame error.
  if (result.translation_m < config.agreement && result.rotation_deg < 0.05f) {
    return result;
  }
  if (result.median_after_m >= result.median_before_m) {
    VLOG(1) << "[MemoryRegistration] Rejected correction that does not reduce the residual.";
    return result;
  }

  result.memory_T_session = estimate;
  result.valid = true;
  return result;
}

}  // namespace khronos
