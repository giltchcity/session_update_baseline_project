#pragma once

#include <Eigen/Geometry>

#include "khronos/common/common_types.h"

namespace khronos {

/**
 * @brief Rigid correction that brings scene memory into this session's frame.
 *
 * A session inherits surface geometry expressed in the frame of the session
 * that built it. The transform between two sessions' frames is an input
 * estimate, and its residual error (a fraction of a degree to a few degrees,
 * plus per-session trajectory drift) displaces inherited surfaces by several
 * centimetres. Change detection then compares a displaced memory against fresh
 * measurements, and a global mesh ends up holding two displaced estimates of
 * one surface.
 *
 * The correction is estimated from the only correspondences available online:
 * inherited surface points and the surface this session reconstructed at the
 * same place. It is a point-to-point rigid fit (Horn/Umeyama) on
 * nearest-neighbour pairs inside the change detector's association tolerance,
 * with the pair set trimmed by a robust median/MAD criterion, iterated a few
 * times. Nothing here is tuned per dataset: the association tolerance and the
 * map resolution come from the mapping configuration, and a fit is refused
 * unless it is supported by enough pairs and stays within a sanity bound.
 */
struct MemoryRegistration {
  struct Config {
    // Nearest-neighbour association radius, the change detector's depth
    // tolerance. Pairs farther apart are different surfaces, not the same
    // surface seen twice.
    float association_tolerance = 0.f;
    // Reconstruction agreement scale (half a voxel). A correction smaller than
    // this cannot be distinguished from meshing noise and is not applied.
    float agreement = 0.f;
    // Minimum number of inlier pairs for a fit to be trusted.
    size_t min_pairs = 500;
    // Sanity bounds: a session-to-session frame error is small. A larger
    // estimate means the correspondences are not the same surface.
    float max_rotation_deg = 5.f;
    float max_translation_m = 0.5f;
    size_t iterations = 3;
  };

  struct Result {
    bool valid = false;
    Eigen::Isometry3f memory_T_session = Eigen::Isometry3f::Identity();
    size_t pairs = 0;
    float rotation_deg = 0.f;
    float translation_m = 0.f;
    // Median pair distance before and after applying the estimate.
    float median_before_m = 0.f;
    float median_after_m = 0.f;
  };

  /**
   * @brief Estimate the correction that maps `memory` onto `session`.
   * @param memory Inherited surface points in the current map frame.
   * @param session Surface points this session reconstructed, same frame.
   */
  static Result estimate(const Points& memory, const Points& session, const Config& config);
};

}  // namespace khronos
