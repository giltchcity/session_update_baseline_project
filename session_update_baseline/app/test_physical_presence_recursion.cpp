#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <khronos/backend/reconciliation/persistent_object_state.h>
#include <khronos/backend/reconciliation/reconciler.h>
#include <khronos/backend/update_khronos_objects_functor.h>
#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <khronos/utils/khronos_attribute_utils.h>
#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/mesh.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>

#include "session_update_baseline/runtime/session_state.h"

namespace {

using Dsg = spark_dsg::DynamicSceneGraph;
using ObjectAttrs = spark_dsg::KhronosObjectAttributes;
using Stamp = khronos::TimeStamp;

constexpr Stamp kAStamp = 1'000;
constexpr Stamp kBStamp = 2'000;
constexpr Stamp kCStamp = 3'000;
constexpr Stamp kOpenEnd = std::numeric_limits<Stamp>::max();

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << "\n";
    std::exit(EXIT_FAILURE);
  }
}

std::shared_ptr<spark_dsg::Mesh> makeBackground(Stamp stamp) {
  auto mesh = std::make_shared<spark_dsg::Mesh>(true, true, true, true);
  mesh->resizeVertices(1);
  mesh->setPos(0, Eigen::Vector3f(0.0F, -2.0F, 0.0F));
  mesh->setTimestamp(0, stamp);
  mesh->setFirstSeenTimestamp(0, stamp);
  mesh->setLabel(0, 1);
  return mesh;
}

std::unique_ptr<ObjectAttrs> makeObject(size_t physical_id,
                                        int semantic_id,
                                        float center_x,
                                        float mesh_marker,
                                        Stamp first,
                                        Stamp last) {
  auto attrs = std::make_unique<ObjectAttrs>();
  attrs->semantic_label = semantic_id;
  attrs->details["instance_id"] = {physical_id};
  attrs->details["test_marker"] = {
      static_cast<size_t>(std::lround(mesh_marker * 100.0F))};
  attrs->position = Eigen::Vector3d(center_x, 0.0, 1.0);
  attrs->bounding_box = khronos::BoundingBox(
      Eigen::Vector3f(0.4F, 0.4F, 0.4F),
      Eigen::Vector3f(center_x, 0.0F, 1.0F));
  attrs->first_observed_ns = {first};
  attrs->last_observed_ns = {last};
  attrs->mesh.resizeVertices(1);
  attrs->mesh.setPos(0, Eigen::Vector3f(mesh_marker, 0.0F, 0.0F));
  khronos::setObservationBounds(*attrs, first, last);
  return attrs;
}

khronos::Reconciler makeOptimisticReconciler() {
  khronos::Reconciler::Config config;
  config.time_estimates_conservative = false;
  config.allow_overestimation = true;
  config.merge_object_meshes = true;
  return khronos::Reconciler(config);
}

class ExposedReconciler : public khronos::Reconciler {
 public:
  explicit ExposedReconciler(const khronos::Reconciler::Config& config)
      : Reconciler(config) {}
  using Reconciler::estimatePresenceTimes;
  using Reconciler::mergeObjectAttributes;
};

void reconcileWithoutEvidence(Dsg& graph, Stamp stamp) {
  khronos::Changes changes;
  for (const auto& [node_id, unused] :
       graph.getLayer(khronos::DsgLayers::OBJECTS).nodes()) {
    (void)unused;
    changes.object_changes.emplace_back();
    changes.object_changes.back().node_id = node_id;
  }
  makeOptimisticReconciler().reconcile(graph, changes, stamp);
}

const ObjectAttrs* findPhysicalObject(const Dsg& graph, size_t physical_id) {
  if (!graph.hasLayer(khronos::DsgLayers::OBJECTS)) {
    return nullptr;
  }
  const ObjectAttrs* result = nullptr;
  for (const auto& [unused_id, node] :
       graph.getLayer(khronos::DsgLayers::OBJECTS).nodes()) {
    (void)unused_id;
    const auto& attrs = node->attributes<ObjectAttrs>();
    if (khronos::UpdateKhronosObjectsFunctor::physicalInstanceId(attrs) !=
        std::optional<size_t>(physical_id)) {
      continue;
    }
    require(result == nullptr,
            "current graph contains duplicate physical ID " +
                std::to_string(physical_id));
    result = &attrs;
  }
  return result;
}

void requireCurrentPhysicalObject(const Dsg& graph,
                                  size_t physical_id,
                                  float center_x,
                                  float mesh_marker,
                                  const std::string& context) {
  const auto* attrs = findPhysicalObject(graph, physical_id);
  require(attrs != nullptr, context + ": physical object is absent");
  require(std::abs(attrs->bounding_box.world_P_center.x() - center_x) < 1.0e-6F,
          context + ": wrong current location");
  require(attrs->mesh.numVertices() == 1 &&
              std::abs(attrs->mesh.pos(0).x() - mesh_marker) < 1.0e-6F,
          context + ": wrong current private mesh");
}

void requireOpenObservation(const Dsg& graph,
                            size_t physical_id,
                            Stamp observed_last,
                            const std::string& context) {
  const auto* attrs = findPhysicalObject(graph, physical_id);
  require(attrs && attrs->last_observed_ns.size() == 1 &&
              attrs->last_observed_ns.front() == kOpenEnd,
          context + ": presence interval is not open-ended");
  require(khronos::observationLastStamp(*attrs) == observed_last,
          context + ": real last-observation provenance changed");
}

void testTerminalUnobservedObjectsSurviveRecursion(
    const std::filesystem::path& output_dir) {
  auto terminal = std::make_shared<Dsg>();
  terminal->setMesh(makeBackground(kAStamp));
  require(terminal->emplaceNode(khronos::DsgLayers::OBJECTS,
                                spark_dsg::NodeSymbol('O', 2),
                                makeObject(2, 10, -1.0F, 0.02F, 700, 990)),
          "insert terminal I2");
  require(terminal->emplaceNode(khronos::DsgLayers::OBJECTS,
                                spark_dsg::NodeSymbol('O', 19),
                                makeObject(19, 139, 1.0F, 0.19F, 800, 980)),
          "insert terminal I19");

  reconcileWithoutEvidence(*terminal, kAStamp);
  for (const auto physical_id : {size_t{2}, size_t{19}}) {
    const auto* attrs = findPhysicalObject(*terminal, physical_id);
    require(attrs && attrs->last_observed_ns.size() == 1 &&
                attrs->last_observed_ns.front() == kOpenEnd,
            "unobserved terminal object did not receive an open interval");
    require(khronos::observationLastStamp(*attrs) < kAStamp,
            "real observation bound was replaced by its presence estimate");
  }

  khronos::SpatioTemporalMap a_map(khronos::SpatioTemporalMap::Config{});
  a_map.update(terminal, kAStamp);
  const auto a_current = a_map.getDsgPtr(kAStamp);
  requireCurrentPhysicalObject(*a_current, 2, -1.0F, 0.02F, "A latest I2");
  requireCurrentPhysicalObject(*a_current, 19, 1.0F, 0.19F, "A latest I19");
  requireOpenObservation(*a_current, 2, 990, "A latest I2");
  requireOpenObservation(*a_current, 19, 980, "A latest I19");

  std::filesystem::create_directories(output_dir);
  const auto path = output_dir / "terminal_unobserved_a.4dmap";
  std::filesystem::remove(path);
  require(a_map.save(path.string()), "save terminal unobserved A map");
  auto loaded = khronos::SpatioTemporalMap::load(path.string());
  require(loaded != nullptr, "load terminal unobserved A map");
  const auto loaded_current = loaded->getDsgPtr(kAStamp);
  requireCurrentPhysicalObject(*loaded_current, 2, -1.0F, 0.02F,
                               "loaded A latest I2");
  requireCurrentPhysicalObject(*loaded_current, 19, 1.0F, 0.19F,
                               "loaded A latest I19");
  requireOpenObservation(*loaded_current, 2, 990, "loaded A latest I2");
  requireOpenObservation(*loaded_current, 19, 980, "loaded A latest I19");

  const auto seed = session_update::runtime::latestSessionSeed(*loaded);
  requireCurrentPhysicalObject(*seed.dsg, 2, -1.0F, 0.02F, "B/C seed I2");
  requireCurrentPhysicalObject(*seed.dsg, 19, 1.0F, 0.19F, "B/C seed I19");
  khronos::SpatioTemporalMap c_map(khronos::SpatioTemporalMap::Config{});
  session_update::runtime::initializeSessionTimeline(c_map, seed);
  c_map.update(seed.dsg->clone(), kCStamp);
  const auto c_current = c_map.getDsgPtr(kCStamp);
  requireCurrentPhysicalObject(*c_current, 2, -1.0F, 0.02F, "C latest I2");
  requireCurrentPhysicalObject(*c_current, 19, 1.0F, 0.19F, "C latest I19");
  requireOpenObservation(*c_current, 2, 990, "C latest I2");
  requireOpenObservation(*c_current, 19, 980, "C latest I19");
  std::filesystem::remove(path);
}

void testMovedPhysicalObjectReplacesCurrentState(
    const std::filesystem::path& output_dir) {
  auto old_state = std::make_shared<Dsg>();
  old_state->setMesh(makeBackground(kAStamp));
  require(old_state->emplaceNode(khronos::DsgLayers::OBJECTS,
                                 spark_dsg::NodeSymbol('O', 10),
                                 makeObject(10, 75, 0.0F, 0.10F, 900, 990)),
          "insert old I10");
  reconcileWithoutEvidence(*old_state, kAStamp);

  khronos::SpatioTemporalMap a_map(khronos::SpatioTemporalMap::Config{});
  a_map.update(old_state, kAStamp);
  const auto a_seed = session_update::runtime::latestSessionSeed(a_map);

  // I10's prior-session geometry (marker 0.10 at x=0.0) is loaded as *prior current
  // state*, not as an immortal canonical shape. Seed the registry exactly as
  // session_backend.cpp's loadInputState() does
  // (persistent_objects_.initializeFromObjects(*unmerged_graph_)); B's round below must
  // still let a real relocation hand CURRENT geometry to B's own observation.
  khronos::PersistentObjectState registry;
  registry.initializeFromObjects(*a_seed.dsg);

  auto b_working = a_seed.dsg->clone();
  require(b_working->emplaceNode(khronos::DsgLayers::OBJECTS,
                                 spark_dsg::NodeSymbol('O', 110),
                                 makeObject(10, 75, 2.0F, 0.90F, 1'900, 1'990)),
          "insert moved I10 observation");
  require(khronos::UpdateKhronosObjectsFunctor::canonicalizePhysicalObjects(
              *b_working, &registry) == 1,
          "moved I10 was not canonicalized to one physical node");
  // A relocation hands CURRENT geometry to the newest valid observation: pose x=2.0 AND
  // marker 0.90 (what B actually observed at the new site). The prior-session surface
  // (marker 0.10 at x=0.0) must leave CURRENT -- it is not transported to the new pose.
  requireCurrentPhysicalObject(*b_working, 10, 2.0F, 0.90F,
                               "canonical B working I10");
  reconcileWithoutEvidence(*b_working, kBStamp);
  const auto* reconciled = findPhysicalObject(*b_working, 10);
  require(reconciled && reconciled->last_observed_ns.back() == kOpenEnd,
          "re-observed moved I10 is not current/open-ended");
  require(khronos::observationLastStamp(*reconciled) == 1'990,
          "moved I10 lost the newest real observation bound");

  khronos::SpatioTemporalMap b_map(khronos::SpatioTemporalMap::Config{});
  session_update::runtime::initializeSessionTimeline(b_map, a_seed);
  b_map.update(b_working, kBStamp);
  requireCurrentPhysicalObject(*b_map.getDsgPtr(kAStamp), 10, 0.0F, 0.10F,
                               "B historical A-state I10");
  requireCurrentPhysicalObject(*b_map.getDsgPtr(kBStamp), 10, 2.0F, 0.90F,
                               "B latest moved I10 (newest observation owns current)");

  std::filesystem::create_directories(output_dir);
  const auto path = output_dir / "moved_i10_b.4dmap";
  std::filesystem::remove(path);
  require(b_map.save(path.string()), "save moved I10 B map");
  auto loaded = khronos::SpatioTemporalMap::load(path.string());
  require(loaded != nullptr, "load moved I10 B map");
  requireCurrentPhysicalObject(*loaded->getDsgPtr(kAStamp), 10, 0.0F, 0.10F,
                               "loaded B historical I10");
  requireCurrentPhysicalObject(*loaded->getDsgPtr(kBStamp), 10, 2.0F, 0.90F,
                               "loaded B latest I10 (newest observation owns current)");
  requireOpenObservation(*loaded->getDsgPtr(kBStamp), 10, 1'990,
                         "loaded B latest I10");

  // C only reads B's output (production D3 restore): reconstruct a fresh
  // registry from the loaded B state, exactly like a new process would.
  const auto c_seed = session_update::runtime::latestSessionSeed(*loaded);
  requireCurrentPhysicalObject(*c_seed.dsg, 10, 2.0F, 0.90F, "C seed moved I10");
  khronos::PersistentObjectState c_registry;
  c_registry.initializeFromObjects(*c_seed.dsg);
  khronos::SpatioTemporalMap c_map(khronos::SpatioTemporalMap::Config{});
  session_update::runtime::initializeSessionTimeline(c_map, c_seed);
  c_map.update(c_seed.dsg->clone(), kCStamp);
  requireCurrentPhysicalObject(*c_map.getDsgPtr(kCStamp), 10, 2.0F, 0.90F,
                               "C latest moved I10 (newest observation owns current)");
  requireOpenObservation(*c_map.getDsgPtr(kCStamp), 10, 1'990,
                         "C latest moved I10");
  std::filesystem::remove(path);
}

void testSequentialPhysicalIntervalReduction(
    const std::filesystem::path& output_dir) {
  Dsg graph;
  graph.setMesh(makeBackground(kAStamp));

  auto first = makeObject(40, 75, 0.0F, 0.40F, 1'000, 1'000);
  first->first_observed_ns = {500};
  first->last_observed_ns = {kOpenEnd};
  auto middle = makeObject(40, 75, 1.0F, 0.41F, 2'000, 2'000);
  middle->first_observed_ns = {2'000};
  middle->last_observed_ns = {2'500};
  auto newest = makeObject(40, 75, 2.0F, 0.42F, 3'000, 3'000);
  newest->first_observed_ns = {3'000};
  newest->last_observed_ns = {3'500};

  require(graph.emplaceNode(khronos::DsgLayers::OBJECTS,
                            spark_dsg::NodeSymbol('O', 400),
                            std::move(first)),
          "insert first I40 segment");
  require(graph.emplaceNode(khronos::DsgLayers::OBJECTS,
                            spark_dsg::NodeSymbol('O', 401),
                            std::move(middle)),
          "insert middle I40 segment");
  require(graph.emplaceNode(khronos::DsgLayers::OBJECTS,
                            spark_dsg::NodeSymbol('O', 402),
                            std::move(newest)),
          "insert newest I40 segment");

  require(khronos::UpdateKhronosObjectsFunctor::canonicalizePhysicalObjects(
              graph) == 2,
          "three I40 segments were not canonicalized");
  const auto* attrs = findPhysicalObject(graph, 40);
  require(attrs != nullptr && attrs->first_observed_ns.size() == 2 &&
              attrs->last_observed_ns.size() == 2,
          "I40 finite middle gap was lost");
  require(attrs->first_observed_ns[0] == 500 &&
              attrs->last_observed_ns[0] == 2'500 &&
              attrs->first_observed_ns[1] == 3'000 &&
              attrs->last_observed_ns[1] == 3'500,
          "I40 pairwise interval clipping/union is incorrect");
  require(!khronos::isPresent(*attrs, 2'750) &&
              khronos::isPresent(*attrs, 3'250) &&
              !khronos::isPresent(*attrs, 4'000),
          "I40 middle gap or newest terminal right edge is incorrect");
  requireCurrentPhysicalObject(graph, 40, 2.0F, 0.42F,
                               "three-segment I40 authority");

  khronos::SpatioTemporalMap map(khronos::SpatioTemporalMap::Config{});
  map.update(graph.clone(), 2'000);
  map.update(graph.clone(), 4'000);
  const auto path = output_dir / "three_segment_presence_gap.4dmap";
  std::filesystem::remove(path);
  require(map.save(path.string()), "save three-segment I40 map");
  auto loaded = khronos::SpatioTemporalMap::load(path.string());
  require(loaded != nullptr, "load three-segment I40 map");
  const auto early = loaded->getDsgPtr(2'250);
  const auto* loaded_attrs = findPhysicalObject(*early, 40);
  require(loaded_attrs != nullptr &&
              loaded_attrs->first_observed_ns ==
                  std::vector<Stamp>({500, 3'000}) &&
              loaded_attrs->last_observed_ns ==
                  std::vector<Stamp>({2'500, 3'500}),
          "I40 full interval vectors changed across save/load");
  require(!findPhysicalObject(*loaded->getDsgPtr(2'750), 40) &&
              findPhysicalObject(*loaded->getDsgPtr(3'250), 40) &&
              !findPhysicalObject(*loaded->getDsgPtr(4'000), 40),
          "loaded I40 gap/terminal presence differs from canonical state");
  const auto c_seed = session_update::runtime::latestSessionSeed(*loaded);
  require(!findPhysicalObject(*c_seed.dsg, 40),
          "terminally absent I40 was resurrected in recursive seed");
  std::filesystem::remove(path);
}

void testSameStampAuthorityIsOrderIndependent() {
  Dsg graph;
  const auto smaller_id = spark_dsg::NodeSymbol('O', 420);
  const auto larger_id = spark_dsg::NodeSymbol('O', 421);
  require(graph.emplaceNode(khronos::DsgLayers::OBJECTS,
                            smaller_id,
                            makeObject(42, 75, 4.0F, 0.42F, kBStamp, kBStamp)),
          "insert smaller same-stamp I42 segment");
  require(graph.emplaceNode(khronos::DsgLayers::OBJECTS,
                            larger_id,
                            makeObject(42, 75, 5.0F, 0.43F, kBStamp, kBStamp)),
          "insert larger same-stamp I42 segment");

  const auto check = [&graph](const std::vector<khronos::NodeId>& order) {
    const auto result =
        khronos::UpdateKhronosObjectsFunctor::mergeObjectAttributes(graph, order);
    const auto* attrs = dynamic_cast<const ObjectAttrs*>(result.get());
    require(attrs != nullptr &&
                std::abs(attrs->bounding_box.world_P_center.x() - 5.0F) < 1.0e-6F &&
                attrs->mesh.numVertices() == 1 &&
                std::abs(attrs->mesh.pos(0).x() - 0.43F) < 1.0e-6F,
            "same-stamp I42 authority depends on input node order");
  };
  check({larger_id, smaller_id});
  check({smaller_id, larger_id});
}

void testTrajectoryOnlySegmentPreservesCanonicalMesh() {
  Dsg graph;
  auto settled = makeObject(43, 75, 0.0F, 0.43F, kAStamp, kAStamp);
  settled->first_observed_ns = {500};
  settled->last_observed_ns = {kOpenEnd};
  auto moving = makeObject(43, 75, 2.0F, 0.99F, kBStamp, kBStamp);
  moving->first_observed_ns = {kBStamp};
  moving->last_observed_ns = {kOpenEnd};
  moving->mesh.resizeVertices(0);
  moving->trajectory_timestamps = {kBStamp};
  moving->trajectory_positions = {Eigen::Vector3f(2.0F, 0.0F, 1.0F)};

  require(graph.emplaceNode(khronos::DsgLayers::OBJECTS,
                            spark_dsg::NodeSymbol('O', 430),
                            std::move(settled)),
          "insert settled I43 segment");
  require(graph.emplaceNode(khronos::DsgLayers::OBJECTS,
                            spark_dsg::NodeSymbol('O', 431),
                            std::move(moving)),
          "insert trajectory-only I43 segment");
  require(khronos::UpdateKhronosObjectsFunctor::canonicalizePhysicalObjects(
              graph) == 1,
          "trajectory-only I43 segments were not canonicalized");
  const auto* attrs = findPhysicalObject(graph, 43);
  // A trajectory-only round (motion in progress, no static mesh) does not
  // clear the established canonical mesh: the settled I43 surface remains
  // the current geometry until a real re-settled observation supersedes it.
  require(attrs != nullptr && attrs->mesh.numVertices() == 1 &&
              std::abs(attrs->mesh.pos(0).x() - 0.43F) < 1.0e-6F,
          "settled I43 canonical mesh did not survive the trajectory-only round");
  require(khronos::trajectoryHistorySize(*attrs) == 1 &&
              attrs->trajectory_timestamps.front() == kBStamp,
          "latest I43 trajectory history was lost");
  require(khronos::observationLastStamp(*attrs) == kBStamp &&
              attrs->last_observed_ns.back() == kOpenEnd,
          "trajectory-only I43 did not own observation/current right edge");
}

void testReconcilerRejectsUnpairedPresenceVectors() {
  khronos::Reconciler::Config config;
  ExposedReconciler reconciler(config);
  ObjectAttrs from;
  from.first_observed_ns = {1};
  ObjectAttrs into;
  into.first_observed_ns = {2};
  into.last_observed_ns = {3};
  bool rejected = false;
  try {
    reconciler.mergeObjectAttributes(from, into);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected,
          "Reconciler indexed an unpaired source presence vector");
}

void testPresenceMidpointsDoNotOverflow() {
  khronos::Reconciler::Config config;
  ExposedReconciler reconciler(config);
  constexpr auto kMaxStamp = std::numeric_limits<Stamp>::max();
  constexpr auto kFirstSeen = kMaxStamp - 12;
  constexpr auto kLastSeen = kMaxStamp - 8;

  ObjectAttrs attrs;
  attrs.first_observed_ns = {kFirstSeen};
  attrs.last_observed_ns = {kLastSeen};
  khronos::setObservationBounds(attrs, kFirstSeen, kLastSeen);
  khronos::ObjectChange change;
  change.first_absent = kMaxStamp - 20;
  change.last_absent = kMaxStamp - 2;
  reconciler.estimatePresenceTimes(change, attrs);

  require(attrs.first_observed_ns == std::vector<Stamp>{kMaxStamp - 16},
          "appearance midpoint overflowed near UINT64_MAX");
  require(attrs.last_observed_ns == std::vector<Stamp>{kMaxStamp - 5},
          "disappearance midpoint overflowed near UINT64_MAX");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: test_physical_presence_recursion OUTPUT_DIR\n";
    return 2;
  }
  const std::filesystem::path output_dir(argv[1]);
  testTerminalUnobservedObjectsSurviveRecursion(output_dir);
  testMovedPhysicalObjectReplacesCurrentState(output_dir);
  testSequentialPhysicalIntervalReduction(output_dir);
  testSameStampAuthorityIsOrderIndependent();
  testTrajectoryOnlySegmentPreservesCanonicalMesh();
  testReconcilerRejectsUnpairedPresenceVectors();
  testPresenceMidpointsDoNotOverflow();
  std::cout << "physical_presence_recursion_passed\n";
  return EXIT_SUCCESS;
}
