/** -----------------------------------------------------------------------------
 * Task-3 confirmation test (read-only audit follow-up).
 *
 * Question (D3): session A ends with background vertex V at (1,0,0), observed
 * [kT1,kT1]. Session B starts a new process with strictly later timestamps.
 * Can B's later rays ever verify (persistent/absent) A's seed vertex?
 *
 * Three scenarios, all starting from the same A state (pose@kT1 at origin,
 * vertex V@(1,0,0) [kT1,kT1], indexed in a fresh SequentialChangeDetector =
 * the save/load boundary):
 *
 *  S1: B observes a far surface W@(3,0,0) [kT2,kT2] from pose@kT2 at origin.
 *      B's ray origin->W passes straight through V's position with no surface
 *      there -> semantically B says "V is gone" (absent).
 *  S2: B observes W@(0,3,0) (ray does not cross V) -> B says nothing about V.
 *  S3: B has no new surface at all (only the new pose) -> B says nothing.
 *
 * The audit's claim to pin down: computeVertexSources only creates a vertex's
 * rays from poses inside [first_seen, last_seen], so A's V never receives a B
 * ray as its own source; and check() filters out V's own old ray
 * (timestamp < last_seen + threshold). Verification of A geometry in B is
 * therefore only possible if some B ray happens to cross V's block.
 * -------------------------------------------------------------------------- */

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/node_symbol.h>

#include "khronos/backend/change_detection/background/ray_background_change_detector.h"
#include "khronos/backend/change_detection/ray_change_detector.h"
#include "khronos/backend/change_detection/ray_verificator.h"
#include "khronos/backend/change_detection/sequential_change_detector.h"

namespace {

using khronos::ChangeState;
using khronos::DynamicSceneGraph;
using khronos::NodeId;
using khronos::Point;
using khronos::RayBackgroundChangeDetector;
using khronos::RayChangeDetector;
using khronos::RayVerificator;
using khronos::SequentialChangeDetector;
using khronos::TimeStamp;
using spark_dsg::AgentNodeAttributes;
using spark_dsg::DsgLayers;
using spark_dsg::Mesh;
using spark_dsg::NodeSymbol;

constexpr TimeStamp kSecond = 1'000'000'000ULL;
constexpr TimeStamp kT1 = 1 * kSecond;    // A's epoch
constexpr TimeStamp kT2 = 10 * kSecond;   // B's epoch: strictly after A

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << "\n";
    std::exit(EXIT_FAILURE);
  }
}

NodeId agentId(size_t index) { return NodeSymbol('a', index); }

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

// Session A's finished state: one pose and one background vertex V@(1,0,0).
DynamicSceneGraph::Ptr makeSessionA() {
  auto dsg = std::make_shared<DynamicSceneGraph>();
  dsg->setMesh(std::make_shared<Mesh>(false, true, false, true));
  addPose(*dsg, 0, kT1);
  appendMeshVertex(*dsg->mesh(), Point(1.0f, 0.0f, 0.0f), kT1, kT1);
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
  RayBackgroundChangeDetector::Config background;
  background.time_filtering_threshold = 0.1f;
  config.background = background;
  return config;
}

// Fresh detector = fresh process on the reloaded seed state (D3 boundary).
struct Result {
  ChangeState v_in_a;  // V's state at the end of A (before B's observations)
  ChangeState v_in_b;  // V's state after B's observations arrived
  std::string label;
};

Result runSessionB(const DynamicSceneGraph::Ptr& session_a,
                   const Point& b_surface,
                   bool has_b_surface) {
  SequentialChangeDetector detector(makeConfig());
  require(detector.setDsg(session_a) == RayVerificator::UpdateMode::kFullReset,
          "A's state is indexed as the seed (save -> load boundary)");

  // Session A's final change pass: V exists, no later rays yet.
  const auto& a_changes = detector.detectChanges({}, kT1, false);
  require(a_changes.background_changes.size() >= 1, "A's seed vertex has a state");
  const ChangeState v_in_a = a_changes.background_changes.front();

  // B's observations arrive: a new pose and possibly a new far surface.
  auto dsg = session_a->clone();
  addPose(*dsg, 1, kT2);
  if (has_b_surface) {
    appendMeshVertex(*dsg->mesh(), b_surface, kT2, kT2);
  }
  const auto b_mode = detector.setDsg(dsg);
  require(b_mode == RayVerificator::UpdateMode::kIncremental,
          "B appends to A's stable mesh prefix (incremental ray update)");

  const auto& b_changes = detector.detectChanges({}, kT2, false);
  require(b_changes.background_changes.size() >= 1,
          "background change recomputed for the seed vertex");
  return {v_in_a, b_changes.background_changes.front(), ""};
}

const char* stateName(ChangeState state) {
  switch (state) {
    case ChangeState::kUnobserved:
      return "unobserved";
    case ChangeState::kPersistent:
      return "persistent";
    case ChangeState::kAbsent:
      return "absent";
  }
  return "?";
}

}  // namespace

int main() {
  const auto session_a = makeSessionA();

  // S1: B's ray origin->(3,0,0) passes through V@(1,0,0) with no surface there.
  const Result s1 = runSessionB(session_a, Point(3.0f, 0.0f, 0.0f), true);
  std::cout << "S1 (B ray crosses V's position, no surface):\n"
            << "  V in A: " << stateName(s1.v_in_a)
            << ", V after B: " << stateName(s1.v_in_b)
            << " (D3 semantics: absent)\n";

  // S2: B observes elsewhere; no ray crosses V.
  const Result s2 = runSessionB(session_a, Point(0.0f, 3.0f, 0.0f), true);
  std::cout << "S2 (B ray does not cross V):\n"
            << "  V in A: " << stateName(s2.v_in_a)
            << ", V after B: " << stateName(s2.v_in_b)
            << " (D3 semantics: unobserved)\n";

  // S3: B has no new surface at all, only the new pose.
  const Result s3 = runSessionB(session_a, Point(0.0f, 0.0f, 0.0f), false);
  std::cout << "S3 (B observes nothing near V):\n"
            << "  V in A: " << stateName(s3.v_in_a)
            << ", V after B: " << stateName(s3.v_in_b)
            << " (D3 semantics: unobserved)\n";

  // Pin down the audit claim: A's seed vertex can only be verified by a B ray
  // that happens to cross its block; without one it stays unobserved forever.
  require(s2.v_in_b == ChangeState::kUnobserved && s3.v_in_b == ChangeState::kUnobserved,
          "without a crossing B ray, the seed vertex is unverifiable (unobserved)");
  std::cout << "PASS: A seed vertex verification in B is incidental "
               "(depends on a B ray crossing its block), not a targeted re-check\n";
  return 0;
}
