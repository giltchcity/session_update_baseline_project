#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

#include <Eigen/Core>
#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/mesh.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>

#include "session_update_baseline/runtime/session_state.h"
#include "session_update_baseline/runtime/session_state_fingerprint.h"

namespace {

using Dsg = spark_dsg::DynamicSceneGraph;
using ObjectAttrs = spark_dsg::KhronosObjectAttributes;

constexpr std::uint64_t kStamp = 1000;

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "session-state fingerprint failure: " << message << "\n";
    std::exit(1);
  }
}

std::shared_ptr<spark_dsg::Mesh> makeMesh(float offset) {
  auto mesh = std::make_shared<spark_dsg::Mesh>(true, true, true, true);
  mesh->resizeVertices(3);
  mesh->resizeFaces(1);
  mesh->setPos(0, Eigen::Vector3f(offset, 0.0F, 1.0F));
  mesh->setPos(1, Eigen::Vector3f(offset + 0.1F, 0.0F, 1.0F));
  mesh->setPos(2, Eigen::Vector3f(offset, 0.1F, 1.0F));
  for (std::size_t i = 0; i < 3; ++i) {
    mesh->setColor(i, spark_dsg::Color(10 + i, 20 + i, 30 + i));
    mesh->setFirstSeenTimestamp(i, 100 + i);
    mesh->setTimestamp(i, 900 + i);
    mesh->setLabel(i, 70 + i);
  }
  mesh->face(0) = {0, 1, 2};
  return mesh;
}

ObjectAttrs::Ptr makeObject(std::size_t physical_id,
                            std::uint32_t semantic,
                            float offset) {
  auto attrs = std::make_unique<ObjectAttrs>();
  attrs->position = Eigen::Vector3d(offset, 0.25, 1.0);
  attrs->last_update_time_ns = 950;
  attrs->is_active = false;
  attrs->is_predicted = false;
  attrs->name = "physical_" + std::to_string(physical_id);
  attrs->color = spark_dsg::Color(1, 2, 3, 255);
  attrs->semantic_label = semantic;
  attrs->bounding_box = spark_dsg::BoundingBox(
      Eigen::Vector3f(0.5F, 0.4F, 0.8F),
      Eigen::Vector3f(offset, 0.25F, 1.0F));
  attrs->registered = true;
  attrs->world_R_object = Eigen::Quaterniond::Identity();
  attrs->first_observed_ns = {100, 700};
  attrs->last_observed_ns = {500, std::numeric_limits<std::uint64_t>::max()};
  attrs->mesh = *makeMesh(offset);
  attrs->trajectory_timestamps = {200, 300};
  attrs->trajectory_positions = {
      Eigen::Vector3f(offset - 0.2F, 0.25F, 1.0F),
      Eigen::Vector3f(offset, 0.25F, 1.0F)};
  attrs->dynamic_object_points = {
      {Eigen::Vector3f(offset - 0.2F, 0.25F, 1.0F)},
      {Eigen::Vector3f(offset, 0.25F, 1.0F)}};
  attrs->details["instance_id"] = {physical_id};
  attrs->details["observation_first_stamp_ns"] = {100};
  attrs->details["observation_last_stamp_ns"] = {900};
  return attrs;
}

Dsg::Ptr makeScene(bool reverse,
                   float global_offset = 0.0F,
                   float chair_mesh_offset = 1.0F,
                   std::uint64_t chair_presence_end =
                       std::numeric_limits<std::uint64_t>::max(),
                   std::uint64_t chair_trajectory_end = 300) {
  auto dsg = std::make_shared<Dsg>();
  dsg->setMesh(makeMesh(global_offset));

  auto insert_chair = [&]() {
    auto attrs = makeObject(10, 75, chair_mesh_offset);
    attrs->last_observed_ns.back() = chair_presence_end;
    attrs->trajectory_timestamps.back() = chair_trajectory_end;
    require(dsg->emplaceNode(spark_dsg::DsgLayers::OBJECTS,
                             spark_dsg::NodeSymbol('O', reverse ? 110 : 10),
                             std::move(attrs)),
            "could not insert chair");
  };
  auto insert_computer = [&]() {
    require(dsg->emplaceNode(spark_dsg::DsgLayers::OBJECTS,
                             spark_dsg::NodeSymbol('O', reverse ? 107 : 7),
                             makeObject(7, 74, 2.0F)),
            "could not insert computer");
  };
  if (reverse) {
    insert_chair();
    insert_computer();
  } else {
    insert_computer();
    insert_chair();
  }
  return dsg;
}

std::uint64_t fingerprint(const Dsg::Ptr& dsg) {
  require(static_cast<bool>(dsg), "cannot fingerprint a null graph");
  return session_update::runtime::canonicalCurrentSceneFingerprint(*dsg).fnv1a64;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: test_session_state_fingerprint OUTPUT_DIR\n";
    return 2;
  }
  const auto output_dir = std::filesystem::path(argv[1]);
  std::filesystem::create_directories(output_dir);
  const auto state_path = output_dir / "fingerprint_roundtrip.4dmap";
  std::filesystem::remove(state_path);

  const auto original = makeScene(false);
  const auto reordered = makeScene(true);
  const auto expected = fingerprint(original);
  require(expected == fingerprint(reordered),
          "physical scene changed under object insertion/node-ID reorder");

  khronos::SpatioTemporalMap map(khronos::SpatioTemporalMap::Config{});
  map.update(original->clone(), kStamp);
  const auto materialized_expected = fingerprint(map.getDsgPtr(kStamp));
  require(map.save(state_path.string()), "could not save round-trip fixture");
  auto loaded = khronos::SpatioTemporalMap::load(state_path.string());
  require(loaded && loaded->numTimeSteps() == 1, "could not reload fixture");
  const auto loaded_current = loaded->getDsgPtr(kStamp);
  require(materialized_expected == fingerprint(loaded_current),
          "serialize/load changed canonical current scene");

  khronos::SpatioTemporalMap recurrent(khronos::SpatioTemporalMap::Config{});
  session_update::runtime::initializeSessionTimeline(
      recurrent, session_update::runtime::latestSessionSeed(*loaded));
  require(recurrent.numTimeSteps() == 1 &&
              materialized_expected == fingerprint(recurrent.getDsgPtr(kStamp)),
          "latest-state reseed changed canonical current scene");

  require(expected != fingerprint(makeScene(false, 0.001F)),
          "global mesh geometry mutation was not detected");
  require(expected != fingerprint(makeScene(false, 0.0F, 1.001F)),
          "private mesh/object geometry mutation was not detected");
  require(expected != fingerprint(makeScene(false, 0.0F, 1.0F, 999)),
          "presence mutation was not detected");
  require(expected !=
              fingerprint(makeScene(false,
                                    0.0F,
                                    1.0F,
                                    std::numeric_limits<std::uint64_t>::max(),
                                    301)),
          "trajectory mutation was not detected");

  const auto details =
      session_update::runtime::canonicalCurrentSceneFingerprint(*original);
  require(details.object_records == 2 && details.encoded_bytes > 0,
          "canonical fingerprint diagnostics are incomplete");

  std::filesystem::remove(state_path);
  std::cout << "session state canonical fingerprint validated\n";
  return 0;
}
