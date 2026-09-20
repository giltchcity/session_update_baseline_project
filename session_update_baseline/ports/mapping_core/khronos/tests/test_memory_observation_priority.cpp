/** -----------------------------------------------------------------------------
 * Cross-session memory retirement regression tests.
 *
 * composeObservationPriority() decides, for one physical state observed in an
 * earlier session and again now, which inherited surface survives:
 *
 *  T1 Agreeing memory is kept: an inherited vertex within the agreement scale
 *     of this session's surface is a duplicate below reconstruction
 *     resolution, not a contradiction.
 *  T2 Disagreeing co-observed memory is retired: an inherited vertex in a
 *     voxel this session also built, but farther than the agreement scale
 *     from the session surface, is a stale estimate of a re-measured surface.
 *  T3 Unobserved memory is kept: an inherited vertex far from anything this
 *     session built (a surface this session never looked at) always survives.
 *  T4 The session surface always reaches the composed state, including when
 *     every inherited vertex is retired. Regression test for a gate that
 *     returned early and silently dropped this session's surface.
 *  T5 The agreement scale follows the map resolution: the same geometry
 *     scaled with the voxel size produces the same decisions.
 * -------------------------------------------------------------------------- */

#include <cstdlib>
#include <iostream>
#include <array>
#include <vector>

#include <spark_dsg/bounding_box.h>
#include <spark_dsg/mesh.h>

#include "khronos/backend/reconciliation/persistent_object_state.h"

namespace {

int failures = 0;

void check(bool condition, const std::string& what) {
  if (condition) {
    std::cout << "  ok   " << what << std::endl;
    return;
  }
  std::cout << "  FAIL " << what << std::endl;
  ++failures;
}

// One small triangle per anchor. Composition carries inherited geometry as a
// face union, so surface used in these tests must have faces: isolated vertices
// are not transported into the composed state.
spark_dsg::Mesh makeSurface(const std::vector<Eigen::Vector3f>& anchors) {
  const std::array<Eigen::Vector3f, 3> corners{Eigen::Vector3f(0.f, 0.f, 0.f),
                                               Eigen::Vector3f(0.002f, 0.f, 0.f),
                                               Eigen::Vector3f(0.f, 0.002f, 0.f)};
  spark_dsg::Mesh mesh(false, true, false, false);
  mesh.resizeVertices(3 * anchors.size());
  for (size_t a = 0; a < anchors.size(); ++a) {
    for (size_t c = 0; c < corners.size(); ++c) {
      const size_t i = 3 * a + c;
      mesh.setPos(i, anchors[a] + corners[c]);
      mesh.setTimestamp(i, 1);
    }
    mesh.faces.push_back({{3 * a, 3 * a + 1, 3 * a + 2}});
  }
  return mesh;
}

// World-frame bounding box wide enough that local == world for these points.
khronos::BoundingBox wideBox() {
  return khronos::BoundingBox(Eigen::Vector3f(-10.f, -10.f, -10.f),
                              Eigen::Vector3f(10.f, 10.f, 10.f));
}

size_t composedSize(const std::vector<Eigen::Vector3f>& inherited,
                    const std::vector<Eigen::Vector3f>& session,
                    float resolution) {
  auto inherited_mesh = makeSurface(inherited);
  auto inherited_box = wideBox();
  const auto session_mesh = makeSurface(session);
  const auto session_box = wideBox();
  khronos::composeObservationPriority(
      inherited_mesh, inherited_box, session_mesh, session_box, resolution);
  return inherited_mesh.numVertices();
}

}  // namespace

int main() {
  const float resolution = 0.05f;          // 5 cm voxel; agreement = 2.5 cm.
  const Eigen::Vector3f session_point(1.f, 1.f, 1.f);

  std::cout << "T1 agreeing memory kept" << std::endl;
  {
    // 1 cm away: inside the agreement scale.
    const Eigen::Vector3f agreeing = session_point + Eigen::Vector3f(0.01f, 0.f, 0.f);
    const size_t n = composedSize({agreeing}, {session_point}, resolution);
    check(n == 6, "the session triangle and the agreeing inherited triangle survive");
  }

  std::cout << "T2 disagreeing co-observed memory retired" << std::endl;
  {
    // 4 cm away: beyond agreement (2.5 cm), still inside the neighbouring-voxel test.
    const Eigen::Vector3f disagreeing = session_point + Eigen::Vector3f(0.04f, 0.f, 0.f);
    const size_t n = composedSize({disagreeing}, {session_point}, resolution);
    check(n == 3, "only the session triangle survives");
  }

  std::cout << "T3 unobserved memory kept" << std::endl;
  {
    // 1 m away: this session built nothing in that voxel neighbourhood.
    const Eigen::Vector3f elsewhere = session_point + Eigen::Vector3f(1.f, 0.f, 0.f);
    const size_t n = composedSize({elsewhere}, {session_point}, resolution);
    check(n == 6, "memory this session never re-observed survives");
  }

  std::cout << "T4 the session surface always reaches the composed state" << std::endl;
  {
    // Every inherited vertex disagrees and is retired; what remains must be
    // this session's measurement, not the untouched inherited mesh.
    const std::vector<Eigen::Vector3f> inherited = {
        session_point + Eigen::Vector3f(0.04f, 0.f, 0.f),
        session_point + Eigen::Vector3f(0.f, 0.04f, 0.f)};
    auto inherited_mesh = makeSurface(inherited);
    auto inherited_box = wideBox();
    khronos::composeObservationPriority(
        inherited_mesh, inherited_box, makeSurface({session_point}), wideBox(), resolution);
    check(inherited_mesh.numVertices() == 3, "only the session triangle remains");
    const bool is_session = inherited_mesh.numVertices() == 3 &&
                            (inherited_mesh.pos(0) - session_point).norm() < 1e-6f;
    check(is_session, "the surviving surface is this session's measurement");
  }

  std::cout << "T5 agreement scale follows the map resolution" << std::endl;
  {
    // At a 10 cm voxel the agreement scale is 5 cm, so the same 4 cm
    // disagreement that T2 retired is now within resolution and is kept.
    const Eigen::Vector3f disagreeing = session_point + Eigen::Vector3f(0.04f, 0.f, 0.f);
    const size_t n = composedSize({disagreeing}, {session_point}, 0.10f);
    check(n == 6, "a 4 cm difference is agreement at a 10 cm voxel");
  }

  if (failures) {
    std::cout << failures << " check(s) failed" << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "all checks passed" << std::endl;
  return EXIT_SUCCESS;
}
