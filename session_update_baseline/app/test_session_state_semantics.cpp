#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>

#include <Eigen/Core>
#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/mesh.h>

#include "session_update_baseline/runtime/session_state.h"

namespace {

using Dsg = spark_dsg::DynamicSceneGraph;

Dsg::Ptr makeState(float marker, uint64_t stamp) {
  auto dsg = std::make_shared<Dsg>();
  auto mesh = std::make_shared<spark_dsg::Mesh>(true, true, true, true);
  mesh->resizeVertices(3);
  mesh->resizeFaces(1);
  mesh->setPos(0, Eigen::Vector3f(marker, 0.0F, 0.0F));
  mesh->setPos(1, Eigen::Vector3f(marker, 1.0F, 0.0F));
  mesh->setPos(2, Eigen::Vector3f(marker, 0.0F, 1.0F));
  for (size_t i = 0; i < 3; ++i) {
    mesh->setTimestamp(i, stamp);
    mesh->setFirstSeenTimestamp(i, stamp);
    mesh->setLabel(i, 1);
  }
  mesh->face(0) = {0, 1, 2};
  dsg->setMesh(mesh);
  return dsg;
}

bool hasMarker(const Dsg::Ptr& dsg, float marker) {
  return dsg && dsg->hasMesh() && dsg->mesh()->numVertices() == 3 &&
         std::abs(dsg->mesh()->pos(0).x() - marker) < 1.0e-6F;
}

int fail(const std::string& message) {
  std::cerr << "session-state semantics failure: " << message << "\n";
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    return fail("expected a temporary output directory");
  }

  const auto output_dir = std::filesystem::path(argv[1]);
  std::filesystem::create_directories(output_dir);
  const auto b_path = output_dir / "session_b.4dmap";
  const auto c_path = output_dir / "session_c.4dmap";
  std::filesystem::remove(b_path);
  std::filesystem::remove(c_path);

  constexpr uint64_t kPriorStamp = 100;
  constexpr uint64_t kBStamp = 200;
  constexpr uint64_t kCStamp = 300;

  khronos::SpatioTemporalMap a_map(khronos::SpatioTemporalMap::Config{});
  a_map.update(makeState(0.0F, 50), 50);
  a_map.update(makeState(1.0F, kPriorStamp), kPriorStamp);

  // P_B contains one seed (latest P_A), followed by B. Re-finalizing B at the
  // same timestamp must replace, not append to, the terminal state.
  khronos::SpatioTemporalMap b_map(khronos::SpatioTemporalMap::Config{});
  session_update::runtime::initializeSessionTimeline(
      b_map, session_update::runtime::latestSessionSeed(a_map));
  b_map.update(makeState(2.0F, kBStamp), kBStamp);
  b_map.update(makeState(3.0F, kBStamp), kBStamp);
  if (b_map.numTimeSteps() != 2 || b_map.stamps().front() != kPriorStamp ||
      b_map.stamps().back() != kBStamp) {
    return fail("terminal replacement appended or reordered a state");
  }
  if (!hasMarker(b_map.getDsgPtr(kBStamp), 3.0F)) {
    return fail("latest B state is not the reconciled replacement");
  }
  if (!b_map.save(b_path.string())) {
    return fail("could not save B map");
  }

  auto loaded_b = khronos::SpatioTemporalMap::load(b_path.string());
  if (!loaded_b || loaded_b->numTimeSteps() != 2 ||
      !hasMarker(loaded_b->getDsgPtr(kBStamp), 3.0F)) {
    return fail("B serialization changed the terminal state");
  }

  // P_C contains latest(P_B) + C, not P_A history + P_B history + C.
  khronos::SpatioTemporalMap c_map(khronos::SpatioTemporalMap::Config{});
  c_map.update(loaded_b->getDsgPtr(kBStamp)->clone(), kBStamp);
  c_map.update(makeState(4.0F, kCStamp), kCStamp);
  if (c_map.numTimeSteps() != 2 || c_map.stamps().front() != kBStamp ||
      c_map.stamps().back() != kCStamp) {
    return fail("recursive session output copied an ancestor timeline");
  }
  if (!c_map.save(c_path.string())) {
    return fail("could not save C map");
  }
  auto loaded_c = khronos::SpatioTemporalMap::load(c_path.string());
  if (!loaded_c || loaded_c->numTimeSteps() != 2 ||
      !hasMarker(loaded_c->getDsgPtr(kBStamp), 3.0F) ||
      !hasMarker(loaded_c->getDsgPtr(kCStamp), 4.0F)) {
    return fail("C output is not directly reloadable as seed + current session");
  }

  bool rejected_out_of_order = false;
  try {
    c_map.update(makeState(5.0F, kPriorStamp), kPriorStamp);
  } catch (const std::invalid_argument&) {
    rejected_out_of_order = true;
  }
  if (!rejected_out_of_order) {
    return fail("out-of-order session state was silently accepted");
  }

  std::filesystem::remove(b_path);
  std::filesystem::remove(c_path);
  std::cout << "session state semantics validated\n";
  return 0;
}
