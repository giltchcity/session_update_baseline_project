#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <spark_dsg/dynamic_scene_graph.h>

#include "session_core/surface/frame_archive.h"
#include "session_core/surface/present_surface.h"
#include "session_core/surface/present_tsdf.h"

namespace {

using khronos::PresentTsdf;

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << "\n";
    std::exit(EXIT_FAILURE);
  }
}

// A fronto-parallel wall at z-depth `z` over the whole image.
std::vector<float> wall(const PresentTsdf::Camera& camera, float z) {
  return std::vector<float>(static_cast<size_t>(camera.width) * camera.height, z);
}

// Same surface topology, and the largest vertex deviation [m] (+inf otherwise).
float deviation(const PresentTsdf& a, const PresentTsdf& b) {
  std::vector<Eigen::Vector3f> va, vb;
  std::vector<PresentTsdf::Face> fa, fb;
  a.extractMesh(va, fa);
  b.extractMesh(vb, fb);
  if (fa.empty() || va.size() != vb.size() || fa.size() != fb.size()) {
    return std::numeric_limits<float>::infinity();
  }
  float worst = 0.f;
  for (size_t i = 0; i < va.size(); ++i) worst = std::max(worst, (va[i] - vb[i]).norm());
  return worst;
}

// A state's evidence counts only while the state holds (README section 5): an
// object seen for three frames (a quarter of its pixels unlabelled, a different
// quarter each frame) and gone in the fourth leaves a remnant unless its ended
// state (its surface) is given; given it, no surface remains where the object
// stood, the object's own pixels elsewhere (a next state not yet known) stay,
// and moving the frames to the ended state online equals giving it from the start.
void testEndedState() {
  using khronos::FrameArchive;
  using khronos::PresentSurface;
  FrameArchive::Camera camera;
  camera.width = 64;
  camera.height = 48;
  camera.fx = camera.fy = 50.f;
  camera.cx = 32.f;
  camera.cy = 24.f;
  const PresentTsdf::Camera tsdf_camera{camera.width, camera.height, camera.fx, camera.fy, camera.cx, camera.cy};
  const auto norm = PresentTsdf::rayNorm(tsdf_camera);
  constexpr uint16_t kObject = 5;
  constexpr int64_t kSecond = 1000000000;
  auto frame = [&](int k, bool object) {
    std::vector<uint16_t> range(norm.size()), ids(norm.size(), 0);
    std::vector<FrameArchive::InstanceRun> runs;
    for (uint32_t v = 0; v < camera.height; ++v) {
      for (uint32_t u = 0; u < camera.width; ++u) {
        const size_t p = v * camera.width + u;
        const bool on_object = object && u >= 16 && u < 48 && v >= 12 && v < 36;
        const bool elsewhere = u < 10 && v < 10;  // the same identity at another site, every frame
        range[p] = static_cast<uint16_t>(1000.f * (on_object ? 1.0f : elsewhere ? 1.5f : 2.0f) * norm[p]);
        if ((on_object && (u + v + k) % 4 != 0) || elsewhere) ids[p] = kObject;
      }
    }
    for (size_t p = 1; p <= ids.size(); ++p) {
      if (p == ids.size() || ids[p] != ids[p - 1]) runs.push_back({static_cast<uint32_t>(p), ids[p - 1]});
    }
    return FrameArchive::Frame::pack(k * kSecond, Eigen::Isometry3d::Identity(), range, runs);
  };
  auto near_object = [](const spark_dsg::DynamicSceneGraph& dsg) {
    size_t n = 0;
    for (size_t i = 0; i < dsg.mesh()->numVertices(); ++i) n += std::abs(dsg.mesh()->pos(i).z() - 1.f) < 0.1f;
    return n;
  };
  auto near_elsewhere = [](const spark_dsg::DynamicSceneGraph& dsg) {
    size_t n = 0;
    for (size_t i = 0; i < dsg.mesh()->numVertices(); ++i) n += std::abs(dsg.mesh()->pos(i).z() - 1.5f) < 0.1f;
    return n;
  };
  auto near_wall = [](const spark_dsg::DynamicSceneGraph& dsg) {
    size_t n = 0;
    for (size_t i = 0; i < dsg.mesh()->numVertices(); ++i) n += std::abs(dsg.mesh()->pos(i).z() - 2.f) < 0.1f;
    return n;
  };
  auto snapshot = [] {
    auto dsg = std::make_shared<spark_dsg::DynamicSceneGraph>();
    dsg->setMesh(std::make_shared<spark_dsg::Mesh>());
    return dsg;
  };
  const PresentSurface::Inputs inputs;
  // The ended state's surface: the object plane at z = 1 over the pixels it covered.
  const float x0 = (16 - 32) / 50.f, x1 = (48 - 32) / 50.f, y0 = (12 - 24) / 50.f, y1 = (36 - 24) / 50.f;
  const std::map<size_t, khronos::TimeStamp> no_starts;
  const std::map<size_t, PresentSurface::Ended> none, ended{
      {kObject, {3 * kSecond, {{x0, y0, 1.f}, {x1, y0, 1.f}, {x1, y1, 1.f},
                               {x0, y0, 1.f}, {x1, y1, 1.f}, {x0, y1, 1.f}}}}};

  // Frames 1-3 integrated, then moved online to the ended state, then frame 4.
  PresentSurface online(0.05, 0.02, 0.04, 2);
  for (int k = 1; k <= 3; ++k) online.integrate(camera, frame(k, true));
  online.integrate(camera, frame(4, false));
  auto remnant = snapshot();
  online.compose(*remnant, 4 * kSecond, no_starts, none, inputs, false);
  require(near_object(*remnant) > 0, "without its ended state the object leaves a remnant");
  auto moved = snapshot();
  online.compose(*moved, 4 * kSecond, no_starts, ended, inputs, false);
  require(near_object(*moved) == 0, "the ended state's pixels, labelled or not, leave the TSDF");
  require(near_wall(*moved) > 0, "the wall seen behind the object stays");
  require(near_elsewhere(*moved) > 0 && near_elsewhere(*moved) == near_elsewhere(*remnant),
          "the identity's pixels away from the ended surface stay");

  // The same frames under the ended state from their first round.
  PresentSurface direct(0.05, 0.02, 0.04, 2);
  for (int k = 1; k <= 3; ++k) direct.integrate(camera, frame(k, true));
  auto first_round = snapshot();
  direct.compose(*first_round, 3 * kSecond, no_starts, ended, inputs, false);
  direct.integrate(camera, frame(4, false));
  auto second_round = snapshot();
  direct.compose(*second_round, 4 * kSecond, no_starts, ended, inputs, false);
  require(second_round->mesh()->numVertices() == moved->mesh()->numVertices() &&
              second_round->mesh()->numFaces() == moved->mesh()->numFaces(),
          "moving frames to the ended state equals integrating them under it");
  float worst = 0.f;
  for (size_t i = 0; i < moved->mesh()->numVertices(); ++i) {
    worst = std::max(worst, (moved->mesh()->pos(i) - second_round->mesh()->pos(i)).norm());
  }
  require(worst < 1e-5f, "same surface whether the ended state is given at once or online");
  std::cout << "ended state: remnant vertices " << near_object(*remnant) << " -> " << near_object(*moved)
            << ", wall vertices " << near_wall(*moved) << ", elsewhere " << near_elsewhere(*remnant) << " -> "
            << near_elsewhere(*moved) << "\n";
}

}  // namespace

int main() {
  testEndedState();
  const PresentTsdf::Camera camera{64, 48, 50.f, 50.f, 32.f, 24.f};
  const auto norm = PresentTsdf::rayNorm(camera);
  Eigen::Isometry3d near_pose = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d far_pose = Eigen::Isometry3d::Identity();
  far_pose.translation() << 0.03, -0.02, -0.05;
  const auto first = wall(camera, 1.00f);
  const auto second = wall(camera, 1.07f);
  // The second frame's free space limited on the left half of its image.
  std::vector<float> limit(first.size(), std::numeric_limits<float>::infinity());
  for (uint32_t v = 0; v < camera.height; ++v) {
    for (uint32_t u = 0; u < camera.width / 2; ++u) limit[v * camera.width + u] = 1.0f;
  }
  const PresentTsdf::Domain none, first_all{&first}, second_all{&second}, second_limited{&second, &limit};

  // Removing a frame restores the TSDF of the remaining frames (eq. 9 inverted).
  PresentTsdf only_first(0.02, 0.04, 2);
  only_first.update(camera, near_pose, norm, none, first_all);
  PresentTsdf removed_second(0.02, 0.04, 2);
  removed_second.update(camera, near_pose, norm, none, first_all);
  removed_second.update(camera, far_pose, norm, none, second_all);
  removed_second.update(camera, far_pose, norm, second_all, none);
  const float removed = deviation(only_first, removed_second);
  require(removed < 1e-5f, "removing the second frame restores the first frame's surface");

  // Moving a frame to another domain equals integrating it there.
  PresentTsdf integrated_limited(0.02, 0.04, 2);
  integrated_limited.update(camera, near_pose, norm, none, first_all);
  integrated_limited.update(camera, far_pose, norm, none, second_limited);
  PresentTsdf moved_limited(0.02, 0.04, 2);
  moved_limited.update(camera, near_pose, norm, none, first_all);
  moved_limited.update(camera, far_pose, norm, none, second_all);
  moved_limited.update(camera, far_pose, norm, second_all, second_limited);
  const float moved = deviation(integrated_limited, moved_limited);
  require(moved < 1e-5f, "moving the second frame to the limited domain equals integrating it there");

  // Removing a frame with its domain restores the remaining frames.
  moved_limited.update(camera, far_pose, norm, second_limited, none);
  const float removed_limited = deviation(only_first, moved_limited);
  require(removed_limited < 1e-5f, "removing the limited second frame restores the first frame's surface");

  // Removing the only frame leaves no surface.
  PresentTsdf emptied(0.02, 0.04, 2);
  emptied.update(camera, near_pose, norm, none, first_all);
  emptied.update(camera, near_pose, norm, first_all, none);
  std::vector<Eigen::Vector3f> vertices;
  std::vector<PresentTsdf::Face> faces;
  emptied.extractMesh(vertices, faces);
  require(faces.empty(), "removing the only frame leaves no surface");

  std::cout << "test_present_tsdf passed (max vertex deviation " << std::max({removed, moved, removed_limited})
            << " m)\n";
  return EXIT_SUCCESS;
}
