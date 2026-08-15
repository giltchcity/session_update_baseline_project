#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <khronos/backend/change_detection/ray_verificator.h>
#include <khronos/common/common_types.h>
#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/mesh.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>

#include "session_update_baseline/runtime/session_state.h"

namespace {

using Dsg = spark_dsg::DynamicSceneGraph;
using ObjectAttributes = spark_dsg::KhronosObjectAttributes;

constexpr std::uint64_t kObjectFirstSeen = 100;
constexpr std::uint64_t kObjectLastSeen = 200;
constexpr std::uint64_t kBFinalStamp = 300;
constexpr std::uint64_t kCFinalStamp = 400;
const spark_dsg::NodeId kObjectId = spark_dsg::NodeSymbol('O', 1);

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << "\n";
    std::exit(1);
  }
}

std::shared_ptr<spark_dsg::Mesh> makeMesh(std::uint64_t last_seen) {
  auto mesh = std::make_shared<spark_dsg::Mesh>(true, true, true, true);
  mesh->resizeVertices(3);
  mesh->resizeFaces(1);
  mesh->setPos(0, Eigen::Vector3f(0.0F, 0.0F, 2.0F));
  mesh->setPos(1, Eigen::Vector3f(0.1F, 0.0F, 2.0F));
  mesh->setPos(2, Eigen::Vector3f(0.0F, 0.1F, 2.0F));
  for (std::size_t i = 0; i < 3; ++i) {
    mesh->setFirstSeenTimestamp(i, kObjectFirstSeen);
    mesh->setTimestamp(i, last_seen);
    mesh->setLabel(i, 1);
  }
  mesh->face(0) = {0, 1, 2};
  return mesh;
}

Dsg::Ptr makeRayGraph(std::uint64_t pose_stamp) {
  auto graph = std::make_shared<Dsg>();
  graph->setMesh(makeMesh(kObjectLastSeen));

  khronos::RayVerificator::Config config;
  const auto agents_key = graph->getLayerKey(spark_dsg::DsgLayers::AGENTS);
  require(agents_key.has_value(), "default graph has no named AGENTS layer");
  graph->addLayer(agents_key->layer, config.prefix.key);

  const auto pose_id = spark_dsg::NodeSymbol(config.prefix.key, 0);
  auto attrs = std::make_unique<spark_dsg::AgentNodeAttributes>(
      std::chrono::nanoseconds(pose_stamp),
      Eigen::Quaterniond::Identity(),
      Eigen::Vector3d::Zero(),
      pose_id);
  require(graph->emplaceNode(agents_key->layer,
                             pose_id,
                             std::move(attrs),
                             config.prefix.key),
          "failed to insert agent pose");
  return graph;
}

Dsg::Ptr makeObjectState(std::uint64_t state_stamp) {
  auto graph = std::make_shared<Dsg>();
  graph->setMesh(makeMesh(state_stamp));

  auto attrs = std::make_unique<ObjectAttributes>();
  attrs->position = Eigen::Vector3d(0.0, 0.0, 2.0);
  attrs->first_observed_ns = {kObjectFirstSeen};
  attrs->last_observed_ns = {kObjectLastSeen};
  attrs->semantic_label = 75;
  require(graph->emplaceNode(
              spark_dsg::DsgLayers::OBJECTS, kObjectId, std::move(attrs)),
          "failed to insert object node");
  return graph;
}

void testRayProvenance() {
  khronos::RayVerificator::Config config;
  config.block_size = 0.25F;
  config.radial_tolerance = 0.05F;
  config.depth_tolerance = 0.05F;
  // Middle is the production default and, before the provenance fix, selected
  // the first B pose after the requested A midpoint even when it lay entirely
  // outside the A vertex lifetime.
  config.ray_policy = khronos::RayVerificator::Config::RayPolicy::kMiddle;

  // Control: a pose inside the A vertex observation interval is a legitimate
  // source and the farther vertex is evidence that the nearer query was absent.
  khronos::RayVerificator in_interval(config);
  in_interval.setDsg(makeRayGraph(150));
  const auto control = in_interval.check(Eigen::Vector3f(0.0F, 0.0F, 1.0F));
  require(control.absent.size() == 1 && control.absent.front() == 150,
          "an in-interval pose did not produce the expected measurement ray");

  // A prior-session vertex exists only in [100, 200]. A B pose at 1000 must
  // never be selected merely because it is the nearest pose in the new graph.
  // The pre-fix nearest-pose behavior would report a fabricated absent ray here.
  khronos::RayVerificator across_session(config);
  across_session.setDsg(makeRayGraph(1000));
  khronos::RayVerificator::CheckDetails details;
  const auto result =
      across_session.check(Eigen::Vector3f(0.0F, 0.0F, 1.0F), 0, 2000, &details);
  require(result.absent.empty() && result.present.empty() && details.start.empty(),
          "a B pose outside the A vertex lifetime fabricated an A measurement ray");
}

void testPresenceAndRecursiveSeed(const std::filesystem::path& output_dir) {
  khronos::SpatioTemporalMap b_map(khronos::SpatioTemporalMap::Config{});
  b_map.update(makeObjectState(kObjectFirstSeen), kObjectFirstSeen);
  b_map.update(makeObjectState(kBFinalStamp), kBFinalStamp);

  const auto historical = b_map.getDsgPtr(150);
  require(historical && historical->hasNode(kObjectId),
          "object is missing from a historical query inside its presence interval");
  const auto current = b_map.getDsgPtr(kBFinalStamp);
  require(current && !current->hasNode(kObjectId),
          "confirmed-absent object leaked into B latest/current DSG");

  std::filesystem::create_directories(output_dir);
  const auto b_path = output_dir / "presence_b.4dmap";
  std::filesystem::remove(b_path);
  require(b_map.save(b_path.string()), "failed to serialize B presence state");

  auto loaded_b = khronos::SpatioTemporalMap::load(b_path.string());
  require(loaded_b && loaded_b->numTimeSteps() == 2,
          "failed to reload B presence state");
  require(loaded_b->getDsgPtr(150)->hasNode(kObjectId),
          "B serialization lost the historical object");
  require(!loaded_b->getDsgPtr(kBFinalStamp)->hasNode(kObjectId),
          "B serialization revived the object in the latest state");

  // Session C receives only latest(P_B). Because latest(P_B) is materialized
  // through the presence query, B's confirmed-absent object must not be copied
  // into C's seed or reappear in C's current state.
  khronos::SpatioTemporalMap c_map(khronos::SpatioTemporalMap::Config{});
  const auto seed = session_update::runtime::latestSessionSeed(*loaded_b);
  require(seed.dsg && !seed.dsg->hasNode(kObjectId),
          "latestSessionSeed returned B's raw, absent object");
  session_update::runtime::initializeSessionTimeline(c_map, seed);
  c_map.update(seed.dsg->clone(), kCFinalStamp);
  require(!c_map.getDsgPtr(seed.stamp)->hasNode(kObjectId),
          "absent B object was copied into C's initial state");
  require(!c_map.getDsgPtr(kCFinalStamp)->hasNode(kObjectId),
          "absent B object resurrected in C's latest state");

  std::filesystem::remove(b_path);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: test_d3_provenance_and_presence OUTPUT_DIR\n";
    return 2;
  }

  testRayProvenance();
  testPresenceAndRecursiveSeed(argv[1]);
  std::cout << "d3_provenance_and_presence_tests_passed\n";
  return 0;
}
