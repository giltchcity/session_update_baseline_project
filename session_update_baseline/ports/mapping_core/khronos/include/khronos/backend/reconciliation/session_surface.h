#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "khronos/common/common_types.h"

namespace khronos {

/**
 * @brief The surface a session hands to the next one: every face of its final
 * map (world frame) with the identity of the state the face belongs to and the
 * measurement that placed it.
 *
 * - identity: physical instance id of the object whose current state the face
 *   belongs to, 0 for the static structure (background);
 * - range: q_f [m], the range of the nearest view of the placing session that
 *   reached the face (0: no measurement on record);
 * - scale: s_f, that session's depth scale (every reading scaled by 1 + s_f
 *   makes its frames agree best).
 *
 * (q_f, s_f) give the position error with which the face's surface is known:
 * max(s_f, 0) * q_f along the placing session's line of sight
 * (SessionRefusion). The session-end update produces this state for its final
 * map; the next session's update reads it as its previous surface. It is
 * written next to the session's maps (kFileName) in a versioned binary format.
 */
struct SessionSurface {
  static constexpr const char* kFileName = "session_surface.bin";

  std::vector<Eigen::Vector3f> vertices;
  std::vector<std::array<uint32_t, 3>> faces;
  std::vector<uint32_t> identity;
  std::vector<float> range;
  std::vector<float> scale;

  size_t numFaces() const { return faces.size(); }
  bool empty() const { return faces.empty(); }
  /** Face and record arrays of equal length, face indices within the vertices. */
  bool consistent() const;

  /**
   * The current surfaces of a map snapshot (background mesh and the meshes of
   * current object nodes, with their physical ids) without measurement records:
   * the surface of a map this algorithm did not produce.
   */
  static SessionSurface fromDsg(const DynamicSceneGraph& dsg);

  bool save(const std::string& path) const;
  static bool load(const std::string& path, SessionSurface& surface);
};

}  // namespace khronos
