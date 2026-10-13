#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/node_symbol.h>

#include "khronos/backend/change_detection/background/ray_background_change_detector.h"
#include "khronos/backend/change_detection/objects/ray_object_change_detector.h"
#include "khronos/backend/change_detection/ray_verificator.h"
#include "khronos/backend/change_detection/sequential_change_detector.h"
#include "khronos/utils/khronos_attribute_utils.h"

namespace {

using khronos::BackgroundChanges;
using khronos::Changes;
using khronos::DynamicSceneGraph;
using khronos::KhronosObjectAttributes;
using khronos::NodeId;
using khronos::Point;
using khronos::Points;
using khronos::RayBackgroundChangeDetector;
using khronos::RayObjectChangeDetector;
using khronos::RayVerificator;
using khronos::SequentialChangeDetector;
using khronos::TimeStamp;
using spark_dsg::AgentNodeAttributes;
using spark_dsg::BoundingBox;
using spark_dsg::DsgLayers;
using spark_dsg::Mesh;
using spark_dsg::NodeSymbol;

constexpr TimeStamp kSecond = 1'000'000'000ULL;
constexpr TimeStamp kT1 = 1 * kSecond;
constexpr TimeStamp kT2 = 10 * kSecond;
constexpr TimeStamp kT3 = 20 * kSecond;

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << "\n";
    std::exit(EXIT_FAILURE);
  }
}

NodeId agentId(size_t index) { return NodeSymbol('a', index); }
NodeId objectId(size_t index) { return NodeSymbol('O', index); }

void addPose(DynamicSceneGraph& dsg,
             size_t index,
             TimeStamp stamp,
             const Eigen::Vector3d& position = Eigen::Vector3d::Zero()) {
  const auto key = dsg.getLayerKey(DsgLayers::AGENTS);
  require(key.has_value(), "default DSG has an AGENTS layer");
  const auto id = agentId(index);
  auto attrs = std::make_unique<AgentNodeAttributes>(
      std::chrono::nanoseconds(stamp), Eigen::Quaterniond::Identity(), position, id);
  require(dsg.emplaceNode(key->layer, id, std::move(attrs), 'a'),
          "agent pose is inserted");
}

void appendMeshVertex(Mesh& mesh,
                      const Point& point,
                      TimeStamp first_seen,
                      TimeStamp last_seen) {
  const size_t index = mesh.numVertices();
  mesh.resizeVertices(index + 1);
  mesh.setPos(index, point);
  mesh.setFirstSeenTimestamp(index, first_seen);
  mesh.setTimestamp(index, last_seen);
}

void addObject(DynamicSceneGraph& dsg,
               size_t index,
               const Point& center,
               TimeStamp first_seen,
               TimeStamp last_seen) {
  auto attrs = std::make_unique<KhronosObjectAttributes>();
  const Points world_points{center + Point(-0.01f, 0.0f, 0.0f),
                            center + Point(0.01f, 0.0f, 0.0f),
                            center + Point(0.0f, 0.01f, 0.0f)};
  attrs->bounding_box = BoundingBox(world_points);
  attrs->position = attrs->bounding_box.world_P_center.cast<double>();
  attrs->mesh = Mesh(false, true, false, true);
  for (const auto& point : world_points) {
    appendMeshVertex(attrs->mesh,
                     point - attrs->bounding_box.world_P_center,
                     first_seen,
                     last_seen);
  }
  attrs->first_observed_ns = {first_seen};
  attrs->last_observed_ns = {last_seen};
  khronos::setObservationBounds(*attrs, first_seen, last_seen);
  require(dsg.emplaceNode(DsgLayers::OBJECTS, objectId(index), std::move(attrs)),
          "object is inserted");
}

void setPhysicalId(DynamicSceneGraph& dsg, size_t index, size_t physical_id) {
  dsg.getNode(objectId(index))
      .attributes<KhronosObjectAttributes>()
      .details["instance_id"] = {physical_id};
}

DynamicSceneGraph::Ptr makeStageOne() {
  auto dsg = std::make_shared<DynamicSceneGraph>();
  dsg->setMesh(std::make_shared<Mesh>(false, true, false, true));
  addPose(*dsg, 0, kT1);
  appendMeshVertex(*dsg->mesh(), Point(1.0f, 0.0f, 0.0f), kT1, kT1);
  appendMeshVertex(*dsg->mesh(), Point(1.0f, 1.0f, 0.0f), kT1, kT1);
  addObject(*dsg, 1, Point(1.0f, 0.0f, 0.0f), kT1, kT1);
  addObject(*dsg, 2, Point(1.0f, 1.0f, 0.0f), kT1, kT1);
  return dsg;
}

DynamicSceneGraph::Ptr makeStageTwo(const DynamicSceneGraph& stage_one) {
  auto dsg = stage_one.clone();
  addPose(*dsg, 1, kT2);
  // The first ray passes through the old x=1 surface (absence); the second
  // terminates on the old y=1 surface (persistence).
  appendMeshVertex(*dsg->mesh(), Point(2.0f, 0.0f, 0.0f), kT2, kT2);
  appendMeshVertex(*dsg->mesh(), Point(1.0f, 1.0f, 0.0f), kT2, kT2);
  addObject(*dsg, 3, Point(2.0f, 0.0f, 0.0f), kT2, kT2);
  return dsg;
}

DynamicSceneGraph::Ptr makeStageThree(const DynamicSceneGraph& stage_two) {
  auto dsg = stage_two.clone();
  addPose(*dsg, 2, kT3);
  appendMeshVertex(*dsg->mesh(), Point(3.0f, 0.0f, 0.0f), kT3, kT3);
  appendMeshVertex(*dsg->mesh(), Point(1.0f, 1.0f, 0.0f), kT3, kT3);
  addObject(*dsg, 4, Point(3.0f, 0.0f, 0.0f), kT3, kT3);

  // Refinement of an existing private object must invalidate only that
  // object's cached result, not the entire background ray index.
  auto& attrs = dsg->getNode(objectId(2)).attributes<KhronosObjectAttributes>();
  appendMeshVertex(attrs.mesh, Point(0.0f, -0.01f, 0.0f), kT1, kT1);
  return dsg;
}

SequentialChangeDetector::Config makeConfig() {
  SequentialChangeDetector::Config config;
  config.ray_verificator.block_size = 0.25f;
  config.ray_verificator.radial_tolerance = 0.05f;
  config.ray_verificator.depth_tolerance = 0.05f;
  config.ray_verificator.ray_policy = RayVerificator::Config::RayPolicy::kAll;
  config.ray_verificator.active_window_duration = 0.0f;
  config.ray_verificator.prefix = hydra::RobotPrefixConfig(0);
  config.ray_change_detector.temporal_resolution = 0.1f;
  config.ray_change_detector.window_size = 1;
  config.ray_change_detector.use_relative_confidence = true;
  config.ray_change_detector.absence_confidence = 0.5f;
  config.ray_change_detector.presence_confidence = 0.5f;

  RayObjectChangeDetector::Config objects;
  objects.time_filtering_threshold = 0.1f;
  objects.query_subsampling = 1;
  config.objects = objects;
  RayBackgroundChangeDetector::Config background;
  background.time_filtering_threshold = 0.1f;
  config.background = background;
  return config;
}

void requireObjectChangesEqual(const khronos::ObjectChanges& actual,
                               const khronos::ObjectChanges& expected,
                               const std::string& context) {
  require(actual.size() == expected.size(), context + ": object count");
  for (size_t i = 0; i < actual.size(); ++i) {
    const auto& a = actual[i];
    const auto& e = expected[i];
    require(a.node_id == e.node_id && a.merged_id == e.merged_id &&
                a.first_absent == e.first_absent &&
                a.last_absent == e.last_absent &&
                a.first_persistent == e.first_persistent &&
                a.last_persistent == e.last_persistent,
            context + ": object record " + std::to_string(i));
  }
}

void requireChangesEqual(const Changes& actual,
                         const Changes& expected,
                         const std::string& context) {
  require(actual.background_changes == expected.background_changes,
          context + ": background changes");
  requireObjectChangesEqual(actual.object_changes, expected.object_changes, context);
}

Changes freshChanges(const SequentialChangeDetector::Config& config,
                     const DynamicSceneGraph::Ptr& dsg,
                     TimeStamp stamp) {
  SequentialChangeDetector detector(config);
  require(detector.setDsg(dsg) == RayVerificator::UpdateMode::kFullReset,
          "a fresh detector performs one full initialization");
  return detector.detectChanges({}, stamp, false);
}

void testGrowingPrefixesMatchFreshOracle() {
  const auto config = makeConfig();
  const auto stage_one = makeStageOne();
  const auto stage_two = makeStageTwo(*stage_one);
  const auto stage_three = makeStageThree(*stage_two);

  SequentialChangeDetector incremental(config);
  require(incremental.setDsg(stage_one) == RayVerificator::UpdateMode::kFullReset,
          "first prefix initializes fully");
  const Changes stage_one_incremental =
      incremental.detectChanges({}, kT1, false);
  requireChangesEqual(stage_one_incremental,
                      freshChanges(config, stage_one, kT1),
                      "stage one oracle");

  require(incremental.setDsg(stage_two) == RayVerificator::UpdateMode::kIncremental,
          "fixed stage-two prefix rebinds incrementally");
  const Changes stage_two_incremental =
      incremental.detectChanges({}, kT2, false);
  const Changes stage_two_fresh = freshChanges(config, stage_two, kT2);
  requireChangesEqual(stage_two_incremental, stage_two_fresh, "stage two oracle");
  require(stage_two_incremental.background_changes.size() == 4,
          "stage two has four background vertices");
  require(stage_two_incremental.background_changes[0] ==
              khronos::ChangeState::kAbsent,
          "new free-space ray marks the old x-axis surface absent");
  require(stage_two_incremental.background_changes[1] ==
              khronos::ChangeState::kPersistent,
          "new matching ray marks the old y-axis surface persistent");
  const auto absent_object = stage_two_incremental.object_changes.find(objectId(1));
  const auto persistent_object = stage_two_incremental.object_changes.find(objectId(2));
  require(absent_object != stage_two_incremental.object_changes.end() &&
              absent_object->last_absent != 0,
          "object-level absence is non-trivial");
  require(persistent_object != stage_two_incremental.object_changes.end() &&
              persistent_object->last_persistent != 0,
          "object-level persistence is non-trivial");

  const auto& stage_two_stats = incremental.getRayVerificatorStatistics();
  require(stage_two_stats.full_resets == 1,
          "stage two did not hide a full ray rebuild");
  require(stage_two_stats.incremental_rebinds == 1 &&
              stage_two_stats.last_new_poses == 1 &&
              stage_two_stats.last_new_vertices == 2 &&
              stage_two_stats.last_new_rays == 2,
          "stage-two counters prove suffix-only indexing");

  require(incremental.setDsg(stage_three) == RayVerificator::UpdateMode::kIncremental,
          "fixed stage-three prefix rebinds incrementally");
  const Changes stage_three_incremental =
      incremental.detectChanges({}, kT3, false);
  requireChangesEqual(stage_three_incremental,
                      freshChanges(config, stage_three, kT3),
                      "stage three oracle with object refinement");
  const auto& stage_three_stats = incremental.getRayVerificatorStatistics();
  require(stage_three_stats.full_resets == 1 &&
              stage_three_stats.incremental_rebinds == 2 &&
              stage_three_stats.last_new_vertices == 2,
          "third prefix remains suffix-only");
  require(stage_three_stats.last_reobserved_objects >= 2,
          "new rays and refined geometry select object records for recomputation");
}

void testEveryRayPolicyMatchesFreshOracle() {
  const std::vector<RayVerificator::Config::RayPolicy> policies{
      RayVerificator::Config::RayPolicy::kFirst,
      RayVerificator::Config::RayPolicy::kLast,
      RayVerificator::Config::RayPolicy::kFirstAndLast,
      RayVerificator::Config::RayPolicy::kMiddle,
      RayVerificator::Config::RayPolicy::kRandom,
      RayVerificator::Config::RayPolicy::kRandom3};
  const auto stage_one = makeStageOne();
  const auto stage_two = makeStageTwo(*stage_one);
  const auto stage_three = makeStageThree(*stage_two);
  for (const auto policy : policies) {
    auto config = makeConfig();
    config.ray_verificator.ray_policy = policy;
    SequentialChangeDetector incremental(config);
    incremental.setDsg(stage_one);
    (void)incremental.detectChanges({}, kT1, false);
    require(incremental.setDsg(stage_two) ==
                RayVerificator::UpdateMode::kIncremental,
            "every ray policy accepts the stable second prefix");
    const Changes second = incremental.detectChanges({}, kT2, false);
    requireChangesEqual(second,
                        freshChanges(config, stage_two, kT2),
                        "all-policy stage two oracle");
    require(incremental.setDsg(stage_three) ==
                RayVerificator::UpdateMode::kIncremental,
            "every ray policy accepts the stable third prefix");
    const Changes third = incremental.detectChanges({}, kT3, false);
    requireChangesEqual(third,
                        freshChanges(config, stage_three, kT3),
                        "all-policy stage three oracle");
  }
}

void testLoopClosureAndPrefixMismatchFallBackExactly() {
  const auto config = makeConfig();
  const auto stage_one = makeStageOne();
  const auto stage_two = makeStageTwo(*stage_one);
  const Changes oracle = freshChanges(config, stage_two, kT2);

  SequentialChangeDetector loopclosure(config);
  loopclosure.setDsg(stage_one);
  (void)loopclosure.detectChanges({}, kT1, false);
  require(loopclosure.setDsg(stage_two) == RayVerificator::UpdateMode::kIncremental,
          "loop-closure test first verifies the stable clone");
  const Changes forced = loopclosure.detectChanges({}, kT2, true);
  requireChangesEqual(forced, oracle, "explicit loop-closure full oracle");
  require(loopclosure.getRayVerificatorStatistics().full_resets == 2,
          "genuine loop closure forces a complete ray rebuild");

  auto pose_deformed = stage_two->clone();
  pose_deformed->getNode(agentId(0))
      .attributes<AgentNodeAttributes>()
      .position.x() += 0.25;
  SequentialChangeDetector pose_fallback(config);
  pose_fallback.setDsg(stage_two);
  (void)pose_fallback.detectChanges({}, kT2, false);
  require(pose_fallback.setDsg(pose_deformed) ==
              RayVerificator::UpdateMode::kFullReset,
          "changed pose rejects the cached prefix");
  const Changes pose_result =
      pose_fallback.detectChanges({}, kT2 + 1, false);
  requireChangesEqual(pose_result,
                      freshChanges(config, pose_deformed, kT2 + 1),
                      "pose-deformation fallback oracle");
  require(pose_fallback.getRayVerificatorStatistics().rejected_prefixes == 1,
          "pose mismatch is counted");

  auto mesh_deformed = stage_two->clone();
  mesh_deformed->mesh()->setPos(0, Point(1.1f, 0.0f, 0.0f));
  SequentialChangeDetector mesh_fallback(config);
  mesh_fallback.setDsg(stage_two);
  (void)mesh_fallback.detectChanges({}, kT2, false);
  require(mesh_fallback.setDsg(mesh_deformed) ==
              RayVerificator::UpdateMode::kFullReset,
          "changed old mesh vertex rejects the cached prefix");
  const Changes mesh_result =
      mesh_fallback.detectChanges({}, kT2 + 2, false);
  requireChangesEqual(mesh_result,
                      freshChanges(config, mesh_deformed, kT2 + 2),
                      "mesh-deformation fallback oracle");
}

void testNoDuplicatePoseAndNoFabricatedPriorSessionRay() {
  auto config = makeConfig().ray_verificator;
  const auto stage_one = makeStageOne();
  RayVerificator verifier(config);
  verifier.setDsg(stage_one);
  const auto initial = verifier.getStatistics();
  require(initial.indexed_poses == 1 && initial.rays == 2,
          "initial rays use the one real pose exactly once");
  require(verifier.updateDsg() == RayVerificator::UpdateMode::kIncremental,
          "unchanged graph is a zero-suffix update");
  require(verifier.getStatistics().indexed_poses == 1 &&
              verifier.getStatistics().rays == 2 &&
              verifier.getStatistics().last_new_poses == 0 &&
              verifier.getStatistics().last_new_rays == 0,
          "the last pose is never duplicated");
  verifier.recomputeHash();
  require(verifier.getStatistics().indexed_vertices == 2 &&
              verifier.getStatistics().indexed_objects == 2,
          "full hash rebuild retains vertex and object indexes");

  auto recurrent = std::make_shared<DynamicSceneGraph>();
  recurrent->setMesh(std::make_shared<Mesh>(false, true, false, true));
  addPose(*recurrent, 100, 100 * kSecond);
  // Persisted A target has no A pose in this independently started B graph.
  appendMeshVertex(*recurrent->mesh(), Point(1.0f, 0.0f, 0.0f), kT1, kT1);
  appendMeshVertex(*recurrent->mesh(),
                   Point(2.0f, 0.0f, 0.0f),
                   100 * kSecond,
                   100 * kSecond);
  RayVerificator recurrent_verifier(config);
  recurrent_verifier.setDsg(recurrent);
  const auto& recurrent_stats = recurrent_verifier.getStatistics();
  require(recurrent_stats.indexed_vertices == 2 && recurrent_stats.rays == 1,
          "B creates a ray only for its real B observation, never for persisted A geometry");
}

void testNewPoseInsideOldObservationIntervalFallsBack() {
  const auto config = makeConfig();
  auto first = makeStageOne();
  first->mesh()->setTimestamp(0, kT2);
  auto second = first->clone();
  addPose(*second, 1, kT2);

  SequentialChangeDetector detector(config);
  detector.setDsg(first);
  (void)detector.detectChanges({}, kT1, false);
  require(detector.setDsg(second) == RayVerificator::UpdateMode::kFullReset,
          "a new source inside an old vertex interval cannot reuse old ray selection");
  const Changes result = detector.detectChanges({}, kT2, false);
  requireChangesEqual(result,
                      freshChanges(config, second, kT2),
                      "old observation interval fallback oracle");
}

void testPhysicalSegmentsMatchIncrementalOracle() {
  const auto config = makeConfig();
  auto first = std::make_shared<DynamicSceneGraph>();
  first->setMesh(std::make_shared<Mesh>(false, true, false, true));
  addPose(*first, 0, kT1);
  appendMeshVertex(*first->mesh(), Point(1.0f, 0.0f, 0.0f), kT1, kT1);
  addObject(*first, 60, Point(0.0f, 5.0f, 0.0f), kT1, kT1);
  setPhysicalId(*first, 60, 6);
  addObject(*first, 61, Point(-5.0f, 0.0f, 0.0f), kT1, kT1);
  addObject(*first, 62, Point(-6.0f, 0.0f, 0.0f), kT1, kT1);

  auto second = first->clone();
  addPose(*second, 1, kT2);
  appendMeshVertex(*second->mesh(), Point(5.0f, 0.0f, 0.0f), kT2, kT2);
  addObject(*second, 160, Point(5.0f, 0.0f, 0.0f), kT2, kT2);
  setPhysicalId(*second, 160, 6);

  SequentialChangeDetector incremental(config);
  incremental.setDsg(first);
  const Changes initial = incremental.detectChanges({}, kT1, false);
  const auto initial_old = initial.object_changes.find(objectId(60));
  require(initial_old != initial.object_changes.end() &&
              initial_old->last_persistent == 0,
          "initial physical segment has no fabricated later direct evidence");

  require(incremental.setDsg(second) ==
              RayVerificator::UpdateMode::kIncremental,
          "new same-ID segment preserves the append-only ray prefix");
  const Changes updated = incremental.detectChanges({}, kT2, false);
  requireChangesEqual(updated,
                      freshChanges(config, second, kT2),
                      "physical-ID incremental detector oracle");
  const auto old_segment = updated.object_changes.find(objectId(60));
  const auto new_segment = updated.object_changes.find(objectId(160));
  require(old_segment != updated.object_changes.end() &&
              old_segment->last_absent == 0 &&
              old_segment->last_persistent == 0,
          "direct same-ID segment fabricated old-site ray persistence");
  require(new_segment != updated.object_changes.end() &&
              new_segment->first_absent == 0 &&
              new_segment->first_persistent == 0,
          "old same-ID segment fabricated new-site ray persistence");

  khronos::RPGOMerges proposals;
  proposals.emplace_back(objectId(160), objectId(60), true);
  proposals.emplace_back(objectId(62), objectId(61), true);
  SequentialChangeDetector proposed(config);
  proposed.setDsg(second);
  const Changes proposed_changes = proposed.detectChanges(proposals, kT2, false);
  const auto physical_proposal =
      proposed_changes.object_changes.find(objectId(160));
  const auto geometric_proposal =
      proposed_changes.object_changes.find(objectId(62));
  require(physical_proposal != proposed_changes.object_changes.end() &&
              physical_proposal->merged_id == 0,
          "physical RPGO proposal bypassed segment-wise reconciliation");
  require(geometric_proposal != proposed_changes.object_changes.end() &&
              geometric_proposal->merged_id == objectId(61),
          "non-physical verified RPGO merge path was disabled");
}

void testPhysicalIdOnlyMutationInvalidatesObjectSnapshot() {
  const auto config = makeConfig();
  auto first = std::make_shared<DynamicSceneGraph>();
  first->setMesh(std::make_shared<Mesh>(false, true, false, true));
  addPose(*first, 0, kT1);
  appendMeshVertex(*first->mesh(), Point(1.0f, 0.0f, 0.0f), kT1, kT1);
  addObject(*first, 70, Point(1.0f, 0.0f, 0.0f), kT1, kT1);
  setPhysicalId(*first, 70, 7);

  auto relabelled = first->clone();
  setPhysicalId(*relabelled, 70, 9);

  RayVerificator verifier(config.ray_verificator);
  require(verifier.setDsg(first) == RayVerificator::UpdateMode::kFullReset,
          "physical-ID snapshot fixture initializes fully");
  require(verifier.setDsg(relabelled) ==
              RayVerificator::UpdateMode::kIncremental,
          "ID-only mutation preserves the background append-only prefix");
  require(verifier.updateDsg() == RayVerificator::UpdateMode::kIncremental,
          "ID-only mutation is handled by object-level invalidation");
  require(verifier.getReobservedObjects().count(objectId(70)) == 1 &&
              verifier.getStatistics().last_reobserved_objects == 1,
          "physical ID participates in the cached object snapshot");

  SequentialChangeDetector incremental(config);
  incremental.setDsg(first);
  (void)incremental.detectChanges({}, kT1, false);
  require(incremental.setDsg(relabelled) ==
              RayVerificator::UpdateMode::kIncremental,
          "sequential detector accepts the stable relabelled prefix");
  const Changes updated = incremental.detectChanges({}, kT1 + 1, false);
  requireChangesEqual(updated,
                      freshChanges(config, relabelled, kT1 + 1),
                      "physical-ID-only mutation fresh oracle");
}

}  // namespace

int main() {
  testGrowingPrefixesMatchFreshOracle();
  testEveryRayPolicyMatchesFreshOracle();
  testLoopClosureAndPrefixMismatchFallBackExactly();
  testNoDuplicatePoseAndNoFabricatedPriorSessionRay();
  testNewPoseInsideOldObservationIntervalFallsBack();
  testPhysicalSegmentsMatchIncrementalOracle();
  testPhysicalIdOnlyMutationInvalidatesObjectSnapshot();
  return EXIT_SUCCESS;
}
