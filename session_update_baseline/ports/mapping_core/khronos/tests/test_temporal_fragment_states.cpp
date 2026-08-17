/** -----------------------------------------------------------------------------
 * Temporal-fragment state machine semantics for PersistentObjectState.
 * (Copied licence terms of the surrounding Khronos sources apply; see LICENSE.)
 * -------------------------------------------------------------------------- */

#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <string>

#include <spark_dsg/dynamic_scene_graph.h>

// Targeted semantics for the temporal-fragment state machine in PersistentObjectState.
//
// The property under test is never "did the object move?" but "what does this observation prove
// about the state we hold?". Each case pins one row of that reduction:
//
//   A  a disjoint observation with no evidence either way stays UNRESOLVED
//   A' the same observation, once CURRENT is confirmed present, is absorbed as more of one object
//   E  watched motion (D1) closes the old state and opens a new one
//   F  contradiction first, then the new-site observation
//   G  the new-site observation first, then contradiction  -- must equal F
//   J  a new fragment's geometry is what was observed there, never the old shape moved or unioned

#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/node_symbol.h>

#include "khronos/backend/reconciliation/persistent_object_state.h"
#include "khronos/backend/update_khronos_objects_functor.h"
#include "khronos/utils/khronos_attribute_utils.h"

namespace {

using khronos::DynamicSceneGraph;
using khronos::KhronosObjectAttributes;
using khronos::NodeId;
using khronos::PersistentObjectState;
using khronos::Point;
using khronos::Points;
using khronos::TimeStamp;
using spark_dsg::BoundingBox;
using spark_dsg::DsgLayers;
using spark_dsg::Mesh;
using spark_dsg::NodeSymbol;

constexpr TimeStamp kSecond = 1'000'000'000ULL;

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << "\n";
    std::exit(EXIT_FAILURE);
  }
}

NodeId objectId(size_t index) { return NodeSymbol('O', index); }

KhronosObjectAttributes::Ptr makeSegment(TimeStamp first,
                                         TimeStamp last,
                                         const Points& mesh_points,
                                         size_t instance_id,
                                         const Point& center,
                                         bool watched_moving = false) {
  auto attrs = std::make_unique<KhronosObjectAttributes>();
  attrs->mesh = Mesh(false, true, false, true);
  for (const auto& point : mesh_points) {
    const size_t index = attrs->mesh.numVertices();
    attrs->mesh.resizeVertices(index + 1);
    attrs->mesh.setPos(index, point);
    attrs->mesh.setFirstSeenTimestamp(index, first);
    attrs->mesh.setTimestamp(index, last);
  }
  attrs->bounding_box = BoundingBox(Point(1.f, 1.f, 1.f), center);
  attrs->position = center.cast<double>();
  attrs->first_observed_ns = {first};
  attrs->last_observed_ns = {last};
  khronos::setObservationBounds(*attrs, first, last);
  attrs->details["instance_id"] = {instance_id};
  if (watched_moving) {
    attrs->details[khronos::kHasDynamicHistoryDetail] = {1u};
  }
  return attrs;
}

// Feed one already-inserted segment through the registry exactly as canonicalization would.
void feed(PersistentObjectState& registry, DynamicSceneGraph& graph, NodeId node_id) {
  auto merged = khronos::UpdateKhronosObjectsFunctor::mergeObjectAttributes(graph, {node_id});
  auto* khronos_attrs = dynamic_cast<KhronosObjectAttributes*>(merged.get());
  require(khronos_attrs != nullptr, "merge result is a Khronos object");
  registry.applyPhysicalGeometry(graph, {node_id}, *khronos_attrs);
}

Points worldPointsOf(const PersistentObjectState::FragmentView& view) {
  Points world;
  world.reserve(view.geometry->numVertices());
  for (size_t i = 0; i < view.geometry->numVertices(); ++i) {
    world.push_back(view.bbox->pointToWorldFrame(view.geometry->pos(i)));
  }
  return world;
}

bool sameWorldPoints(const Points& lhs, const Points& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if ((lhs[i] - rhs[i]).norm() > 1e-4f) {
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// A: a disjoint observation, with nothing said about the state we hold, is UNRESOLVED.
//    It must neither be unioned into CURRENT nor replace it.
// ---------------------------------------------------------------------------
void testDisjointObservationStaysUnresolved() {
  constexpr size_t kInstance = 701;
  const Point center(0.f, 0.f, 0.f);
  const Points front = {Point(-0.40f, 0.f, 0.f), Point(-0.39f, 0.f, 0.f)};
  const Points back = {Point(0.40f, 0.f, 0.f), Point(0.39f, 0.f, 0.f)};

  auto dsg = std::make_shared<DynamicSceneGraph>();
  PersistentObjectState registry;
  dsg->emplaceNode(
      DsgLayers::OBJECTS, objectId(1), makeSegment(1 * kSecond, 1 * kSecond, front, kInstance, center));
  dsg->emplaceNode(
      DsgLayers::OBJECTS, objectId(2), makeSegment(3 * kSecond, 3 * kSecond, back, kInstance, center));

  feed(registry, *dsg, objectId(1));
  feed(registry, *dsg, objectId(2));

  const auto current = registry.currentFragment(kInstance);
  require(current.has_value(), "A: the ID still has exactly one CURRENT fragment");
  require(current->geometry->numVertices() == 2,
          "A: CURRENT was not grown by an observation that proved nothing (no union)");
  require(sameWorldPoints(worldPointsOf(*current), front),
          "A: CURRENT is still the surface it was established with (no replacement)");
  require(registry.historyFragments(kInstance).size() == 1,
          "A: no second fragment was opened without contradiction evidence");
  require(registry.unresolvedCandidates(kInstance).size() == 1,
          "A: the disjoint observation is held as an unresolved candidate");

  std::cout << "PASS A: a disjoint observation with no evidence stays UNRESOLVED\n";
}

// ---------------------------------------------------------------------------
// A': the same disjoint observation, once real measurements confirm CURRENT is still
//     present at that moment, is more of one object -- one ID cannot be in two places.
// ---------------------------------------------------------------------------
void testConfirmedCurrentAbsorbsDisjointView() {
  constexpr size_t kInstance = 702;
  const Point center(0.f, 0.f, 0.f);
  const Points front = {Point(-0.40f, 0.f, 0.f), Point(-0.39f, 0.f, 0.f)};
  const Points back = {Point(0.40f, 0.f, 0.f), Point(0.39f, 0.f, 0.f)};

  auto dsg = std::make_shared<DynamicSceneGraph>();
  PersistentObjectState registry;
  dsg->emplaceNode(
      DsgLayers::OBJECTS, objectId(1), makeSegment(1 * kSecond, 1 * kSecond, front, kInstance, center));
  dsg->emplaceNode(
      DsgLayers::OBJECTS, objectId(2), makeSegment(3 * kSecond, 3 * kSecond, back, kInstance, center));

  feed(registry, *dsg, objectId(1));
  // A real measurement still lands on the established surface at t=3s.
  require(registry.reportCurrentSupported(kInstance, 3 * kSecond),
          "A': support is reported against a CURRENT fragment");
  feed(registry, *dsg, objectId(2));

  const auto current = registry.currentFragment(kInstance);
  require(current.has_value(), "A': the ID has one CURRENT fragment");
  require(current->geometry->numVertices() == 4,
          "A': the confirmed-coexistent view is folded into CURRENT (2 + 2 = 4)");
  require(registry.historyFragments(kInstance).size() == 1,
          "A': folding a view in does not open a new state");
  require(registry.unresolvedCandidates(kInstance).empty(),
          "A': nothing is left unresolved once it has been absorbed");

  std::cout << "PASS A': a confirmed-present CURRENT absorbs a disjoint view of one object\n";
}

// ---------------------------------------------------------------------------
// E (D1): the object was watched moving. The old state closes, the new one opens,
//         and CURRENT is the new site only.
// ---------------------------------------------------------------------------
void testWatchedMotionOpensNewState() {
  constexpr size_t kInstance = 703;
  const Point old_center(0.f, 0.f, 0.f);
  const Point new_center(5.f, 0.f, 0.f);
  const Points old_geometry = {Point(0.f, 0.f, 0.f), Point(0.01f, 0.f, 0.f)};
  const Points new_geometry = {Point(0.f, 0.f, 0.f), Point(0.02f, 0.f, 0.f), Point(0.03f, 0.f, 0.f)};

  auto dsg = std::make_shared<DynamicSceneGraph>();
  PersistentObjectState registry;
  dsg->emplaceNode(DsgLayers::OBJECTS,
                   objectId(1),
                   makeSegment(1 * kSecond, 1 * kSecond, old_geometry, kInstance, old_center));
  dsg->emplaceNode(
      DsgLayers::OBJECTS,
      objectId(2),
      makeSegment(4 * kSecond, 4 * kSecond, new_geometry, kInstance, new_center, true));

  feed(registry, *dsg, objectId(1));
  feed(registry, *dsg, objectId(2));

  const auto history = registry.historyFragments(kInstance);
  require(history.size() == 2, "E: watched motion opened a second temporal fragment");
  require(history[0].death_time.has_value(), "E: the pre-motion fragment is closed");
  require(!history[1].death_time.has_value(), "E: the post-motion fragment is open");

  const auto current = registry.currentFragment(kInstance);
  require(current.has_value(), "E: there is a CURRENT fragment");
  Points expected_world;
  for (const auto& point : new_geometry) {
    expected_world.push_back(point + new_center);
  }
  require(sameWorldPoints(worldPointsOf(*current), expected_world),
          "E: CURRENT is the new site's own geometry only");

  std::cout << "PASS E: watched motion closes the old state and opens the new one\n";
}

// ---------------------------------------------------------------------------
// F / G / J: a relocation across an observation gap, reached from both directions.
//   F  contradiction of the old site arrives first
//   G  the new-site observation arrives first
// Both must end with the same CURRENT, the same history, and CURRENT geometry that came
// from the new-site observation (J).
// ---------------------------------------------------------------------------
struct RelocationOutcome {
  size_t fragment_count = 0;
  size_t unresolved_count = 0;
  bool old_fragment_closed = false;
  Points current_world;
  Points old_world;
};

RelocationOutcome runRelocation(size_t instance, bool contradiction_first) {
  const Point old_center(0.f, 0.f, 0.f);
  const Point new_center(5.f, 0.f, 0.f);
  const Points old_geometry = {Point(0.f, 0.f, 0.f), Point(0.01f, 0.f, 0.f)};
  const Points new_geometry = {
      Point(0.f, 0.f, 0.f), Point(0.02f, 0.f, 0.f), Point(0.03f, 0.f, 0.f)};

  auto dsg = std::make_shared<DynamicSceneGraph>();
  PersistentObjectState registry;
  dsg->emplaceNode(DsgLayers::OBJECTS,
                   objectId(1),
                   makeSegment(1 * kSecond, 1 * kSecond, old_geometry, instance, old_center));
  dsg->emplaceNode(DsgLayers::OBJECTS,
                   objectId(2),
                   makeSegment(9 * kSecond, 9 * kSecond, new_geometry, instance, new_center));

  feed(registry, *dsg, objectId(1));
  if (contradiction_first) {
    registry.reportCurrentContradicted(instance, 8 * kSecond);
    feed(registry, *dsg, objectId(2));
  } else {
    feed(registry, *dsg, objectId(2));
    registry.reportCurrentContradicted(instance, 8 * kSecond);
  }

  RelocationOutcome outcome;
  const auto history = registry.historyFragments(instance);
  outcome.fragment_count = history.size();
  outcome.unresolved_count = registry.unresolvedCandidates(instance).size();
  if (!history.empty()) {
    outcome.old_fragment_closed = history[0].death_time.has_value();
    outcome.old_world = worldPointsOf(history[0]);
  }
  const auto current = registry.currentFragment(instance);
  if (current) {
    outcome.current_world = worldPointsOf(*current);
  }
  return outcome;
}

void testRelocationIsOrderInvariantAndProvenanceClean() {
  const Point new_center(5.f, 0.f, 0.f);
  const Points old_geometry = {Point(0.f, 0.f, 0.f), Point(0.01f, 0.f, 0.f)};
  const Points new_geometry = {
      Point(0.f, 0.f, 0.f), Point(0.02f, 0.f, 0.f), Point(0.03f, 0.f, 0.f)};
  Points expected_new_world;
  for (const auto& point : new_geometry) {
    expected_new_world.push_back(point + new_center);
  }

  const auto free_first = runRelocation(704, true);
  const auto observe_first = runRelocation(705, false);

  for (const auto* outcome : {&free_first, &observe_first}) {
    const std::string tag = outcome == &free_first ? "F" : "G";
    require(outcome->fragment_count == 2, tag + ": exactly two temporal fragments exist");
    require(outcome->old_fragment_closed, tag + ": the old-site fragment is closed");
    require(outcome->unresolved_count == 0, tag + ": nothing is left unresolved");
    require(sameWorldPoints(outcome->current_world, expected_new_world),
            tag + " (J): CURRENT geometry is exactly what was observed at the new site -- "
                  "not the old shape re-anchored, not a union of both");
    require(outcome->current_world.size() != old_geometry.size() + new_geometry.size(),
            tag + " (J): CURRENT is not the union of the old and new meshes");
    require(sameWorldPoints(outcome->old_world, old_geometry),
            tag + ": the old geometry survives intact in the history");
  }

  require(free_first.fragment_count == observe_first.fragment_count &&
              sameWorldPoints(free_first.current_world, observe_first.current_world) &&
              sameWorldPoints(free_first.old_world, observe_first.old_world),
          "F == G: the outcome does not depend on which piece of evidence was processed first");

  std::cout << "PASS F/G/J: relocation is order-invariant and new geometry is observation-derived\n";
}

}  // namespace

int main() {
  testDisjointObservationStaysUnresolved();
  testConfirmedCurrentAbsorbsDisjointView();
  testWatchedMotionOpensNewState();
  testRelocationIsOrderInvariantAndProvenanceClean();
  std::cout << "ALL TEMPORAL FRAGMENT STATE TESTS PASSED\n";
  return 0;
}
