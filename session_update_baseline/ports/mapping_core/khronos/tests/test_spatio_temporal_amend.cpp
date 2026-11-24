// SpatioTemporalMap::rewriteSnapshots, as the backend uses it to show a state from its start: only the
// snapshots named are changed; every other snapshot, including the later ones stored as deltas of a
// changed one or stored again from a piece history, decodes unchanged. Run with both stores.
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/node_symbol.h>

#include "khronos/spatio_temporal_map/spatio_temporal_map.h"

namespace {

using Dsg = spark_dsg::DynamicSceneGraph;
using ObjectAttrs = spark_dsg::KhronosObjectAttributes;
using khronos::SpatioTemporalMap;
using khronos::TimeStamp;

constexpr TimeStamp kSecond = 1'000'000'000ULL;

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << "\n";
    std::exit(EXIT_FAILURE);
  }
}

std::unique_ptr<ObjectAttrs> object(size_t instance, float x) {
  auto attrs = std::make_unique<ObjectAttrs>();
  attrs->position = Eigen::Vector3d(x, 0.0, 0.0);
  attrs->first_observed_ns = {kSecond};
  attrs->last_observed_ns = {std::numeric_limits<uint64_t>::max()};
  attrs->mesh.resizeVertices(3);
  for (size_t i = 0; i < 3; ++i) attrs->mesh.setPos(i, Eigen::Vector3f(x, 0.1f * i, 0.0f));
  attrs->mesh.faces.push_back({0, 1, 2});
  attrs->details["instance_id"] = {instance};
  return attrs;
}

// Snapshot k (stamp k + 1 s): a background mesh and the agent, as the map stores them; object 10 at
// x = 0; object 20 from snapshot 3 on.
Dsg::Ptr snapshot(size_t k) {
  const TimeStamp stamp = (k + 1) * kSecond;
  auto dsg = std::make_shared<Dsg>();
  auto mesh = std::make_shared<spark_dsg::Mesh>(true, true, true, true);
  mesh->resizeVertices(3);
  for (size_t i = 0; i < 3; ++i) {
    mesh->setPos(i, Eigen::Vector3f(0.0f, 1.0f + 0.1f * i, 0.0f));
    mesh->setTimestamp(i, stamp);
    mesh->setFirstSeenTimestamp(i, kSecond);
  }
  mesh->faces.push_back({0, 1, 2});
  dsg->setMesh(mesh);
  const auto agent_key = dsg->getLayerKey(spark_dsg::DsgLayers::AGENTS);
  require(agent_key.has_value(), "default DSG has an agent layer");
  const auto agent_id = spark_dsg::NodeSymbol('a', k);
  require(dsg->emplaceNode(agent_key->layer, agent_id,
                           std::make_unique<spark_dsg::AgentNodeAttributes>(
                               std::chrono::nanoseconds(stamp), Eigen::Quaterniond::Identity(),
                               Eigen::Vector3d(0.0, 0.0, 0.0), agent_id),
                           'a'),
          "agent node insertion succeeds");
  dsg->emplaceNode(spark_dsg::DsgLayers::OBJECTS, spark_dsg::NodeSymbol('O', 10), object(10, 0.0f));
  if (k >= 3) dsg->emplaceNode(spark_dsg::DsgLayers::OBJECTS, spark_dsg::NodeSymbol('O', 20), object(20, 9.0f));
  return dsg;
}

float meshX(const Dsg& dsg, spark_dsg::NodeId id) {
  return dsg.getNode(id).attributes<ObjectAttrs>().mesh.pos(0).x();
}

void check(bool piece_history) {
  const std::string store = piece_history ? " (piece history)" : " (deltas)";
  SpatioTemporalMap map(SpatioTemporalMap::Config{});
  if (piece_history) map.usePieceHistory(1.0);
  constexpr size_t kSnapshots = 25;  // crosses the delta keyframes at 10 and 20
  for (size_t k = 0; k < kSnapshots; ++k) map.update(snapshot(k), (k + 1) * kSecond);

  const auto id10 = spark_dsg::NodeSymbol('O', 10);
  const auto id20 = spark_dsg::NodeSymbol('O', 20);
  const auto within = [](TimeStamp from, TimeStamp to) {
    return [from, to](size_t, TimeStamp t) { return t >= from && t < to; };
  };
  // Object 10 shown at x = 5 from 12 s to (not including) 16 s.
  size_t moved = 0;
  map.rewriteSnapshots(within(12 * kSecond, 16 * kSecond), [&](Dsg& past, size_t, TimeStamp) {
    auto copy = past.getNode(id10).attributes<ObjectAttrs>().clone();
    auto& c = static_cast<ObjectAttrs&>(*copy);
    for (size_t i = 0; i < 3; ++i) c.mesh.setPos(i, Eigen::Vector3f(5.0f, 0.1f * i, 0.0f));
    past.setNodeAttributes(id10, std::move(copy));
    ++moved;
    return true;
  });
  require(moved == 4, "four snapshots stamped 12-15 s are changed" + store + ", got " + std::to_string(moved));
  // Object 20 inserted where it is missing, from 1 s to 3 s.
  size_t inserted = 0;
  size_t seen_stamp_ok = 0;
  map.rewriteSnapshots(within(1 * kSecond, 3 * kSecond), [&](Dsg& past, size_t index, TimeStamp t) {
    seen_stamp_ok += (t == (index + 1) * kSecond && (t == 1 * kSecond || t == 2 * kSecond));
    if (past.hasNode(id20)) return false;
    const bool added = past.emplaceNode(spark_dsg::DsgLayers::OBJECTS, id20, object(20, 9.0f));
    inserted += added;
    return added;
  });
  require(inserted == 2 && seen_stamp_ok == 2, "object 20 is inserted into the snapshots at 1 s and 2 s" + store);

  for (size_t k = 0; k < kSnapshots; ++k) {
    const auto dsg = map.rawDsg(k);
    const TimeStamp stamp = (k + 1) * kSecond;
    const float x = meshX(*dsg, id10);
    const bool in_window = stamp >= 12 * kSecond && stamp < 16 * kSecond;
    require(x == (in_window ? 5.0f : 0.0f), "object 10 at the expected place in snapshot " + std::to_string(k) + store);
    const bool has20 = dsg->hasNode(id20);
    require(has20 == (k <= 1 || k >= 3), "object 20 present as expected in snapshot " + std::to_string(k) + store);
  }
  require(map.getDsgPtr(13 * kSecond) != nullptr, "the map still answers queries after the rewrite" + store);
}

}  // namespace

int main() {
  check(false);
  check(true);
  std::cout << "spatio_temporal_amend: ok\n";
  return EXIT_SUCCESS;
}
