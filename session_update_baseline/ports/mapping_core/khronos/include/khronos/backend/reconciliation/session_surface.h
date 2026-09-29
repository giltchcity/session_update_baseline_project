#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "khronos/common/common_types.h"

namespace khronos {

/**
 * @brief The surface of a map snapshot: every face of its current surfaces
 * (world frame) with the identity of the entity it belongs to (the physical
 * instance id of its object, 0 for the background) and the position error of
 * its surface.
 *
 * - error: eps_f [m], the position error along the line of sight with which
 *   the session that measured the face placed it, max(s, 0) q (s: that
 *   session's depth scale, q: the range of its nearest view reaching the face;
 *   SessionRefusion, A2). A face keeps its error while it persists.
 *
 * The map is the state a session hands to the next one: the next session-end
 * update reads the surface of the previous final map's latest snapshot as its
 * previous surface (SessionRefusion::Inputs::previous). The map holds geometry
 * and identities; the errors, one per face in the order of fromDsg, are saved
 * next to it (kErrorFileName). A map saved without them records no error
 * (every eps_f = 0).
 */
struct SessionSurface {
  static constexpr const char* kErrorFileName = "surface_error.bin";

  std::vector<Eigen::Vector3f> vertices;
  std::vector<std::array<uint32_t, 3>> faces;
  std::vector<uint32_t> identity;
  std::vector<float> error;

  size_t numFaces() const { return faces.size(); }
  bool empty() const { return faces.empty(); }
  /** One identity and one error per face, face indices within the vertices. */
  bool consistent() const;

  /**
   * The background mesh and the meshes of current object nodes, with their
   * physical ids; errors 0.
   */
  static SessionSurface fromDsg(const DynamicSceneGraph& dsg);

  /** Write the errors (one per face, fromDsg order). */
  static bool saveErrors(const std::string& path, const std::vector<float>& error);
  /** Read the errors saved next to this surface's map; false if unreadable or of another face count. */
  bool loadErrors(const std::string& path);
};

}  // namespace khronos
