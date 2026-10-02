/** -----------------------------------------------------------------------------
 * Temporal-fragment state machine semantics for PersistentObjectState under the unified model
 * (README principles 3, 13): rounds of element verdicts drive the Persistence Filter recursion
 * (5r); commitments are made at alpha (5e).
 * (Copied licence terms of the surrounding Khronos sources apply; see LICENSE.)
 * -------------------------------------------------------------------------- */

#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/node_symbol.h>

#include "session_core/state/persistent_object_state.h"
#include "session_core/testing/registry_fixture.h"
#include "khronos/backend/update_khronos_objects_functor.h"
#include "khronos/utils/khronos_attribute_utils.h"

// The property under test is never "did the object move?" but "what does this observation prove
// about the state we hold?". Each case pins one row of that reduction:
//
//   A  a disjoint observation with no evidence either way stays UNRESOLVED
//   A' the same observation, once CURRENT is committed present, is absorbed as more of one object
//   E  watched motion (D1) closes the old state and opens a new one
//   F  contradiction first, then the new-site observation
//   G  the new-site observation first, then contradiction  -- must equal F
//   J  a new fragment's geometry is what was observed there, never the old shape moved or unioned
//   K  a placement committed changed is succeeded, not extended
//   L  with no evidence the prior decides (README (5o) (e))

namespace {

using khronos::DynamicSceneGraph;
using khronos::KhronosObjectAttributes;
using khronos::NodeId;
using khronos::PersistentObjectState;
using khronos::Point;
using khronos::Points;
using khronos::TimeStamp;
using khronos::testing::confirm;
using khronos::testing::contradict;
using khronos::testing::kSecond;
using khronos::testing::require;
using khronos::testing::trainRegistry;
using spark_dsg::BoundingBox;
using spark_dsg::DsgLayers;
using spark_dsg::Mesh;
using spark_dsg::NodeSymbol;

// The construction hits of an element for the stability of (7s).

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
    // Accepted native motion carries paired sensor times and positions.
    attrs->trajectory_timestamps = {first - kSecond, first};
    attrs->trajectory_positions = {center - Point::UnitX(), center};
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
// A': the same disjoint observation, once a committed round places CURRENT in place at that
//     moment, is more of one object -- one ID cannot be in two places (README (5o) (a)).
// ---------------------------------------------------------------------------
void testConfirmedCurrentAbsorbsDisjointView() {
  constexpr size_t kInstance = 702;
  const Point center(0.f, 0.f, 0.f);
  const Points front = {Point(-0.40f, 0.f, 0.f), Point(-0.39f, 0.f, 0.f)};
  const Points back = {Point(0.40f, 0.f, 0.f), Point(0.39f, 0.f, 0.f)};

  auto dsg = std::make_shared<DynamicSceneGraph>();
  PersistentObjectState registry;
  trainRegistry(registry);
  dsg->emplaceNode(
      DsgLayers::OBJECTS, objectId(1), makeSegment(1 * kSecond, 1 * kSecond, front, kInstance, center));
  dsg->emplaceNode(
      DsgLayers::OBJECTS, objectId(2), makeSegment(3 * kSecond, 3 * kSecond, back, kInstance, center));

  feed(registry, *dsg, objectId(1));
  feed(registry, *dsg, objectId(2));
  require(registry.unresolvedCandidates(kInstance).size() == 1, "A': precondition, one candidate");
  const auto round = confirm(registry, kInstance, 3 * kSecond);
  require(round.confirmed, "A': the round of hits commits CURRENT in place");

  const auto current = registry.currentFragment(kInstance);
  require(current.has_value(), "A': the ID has one CURRENT fragment");
  require(current->geometry->numVertices() == 4,
          "A': the coexistent view is folded into CURRENT (2 + 2 = 4)");
  require(registry.historyFragments(kInstance).size() == 1,
          "A': folding a view in does not open a new state");
  require(registry.unresolvedCandidates(kInstance).empty(),
          "A': nothing is left unresolved once it has been absorbed");

  std::cout << "PASS A': a committed-present CURRENT absorbs a disjoint view of one object\n";
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
  trainRegistry(registry);
  dsg->emplaceNode(DsgLayers::OBJECTS,
                   objectId(1),
                   makeSegment(1 * kSecond, 1 * kSecond, old_geometry, instance, old_center));
  dsg->emplaceNode(DsgLayers::OBJECTS,
                   objectId(2),
                   makeSegment(9 * kSecond, 9 * kSecond, new_geometry, instance, new_center));

  feed(registry, *dsg, objectId(1));
  if (contradiction_first) {
    require(contradict(registry, instance, 8 * kSecond).closed, "F: the old site is seen through");
    feed(registry, *dsg, objectId(2));
  } else {
    feed(registry, *dsg, objectId(2));
    require(contradict(registry, instance, 8 * kSecond).closed, "G: the old site is seen through");
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

// ---------------------------------------------------------------------------
// K / K' / K'': an observation that SHARES space with CURRENT.
//   K   CURRENT was committed changed (seen through) before the observation was made: the
//       observation succeeds it.
//   K'  CURRENT was committed in place meanwhile -> another view of the same site, folded in.
//   K'' fewer passing elements than hits never end a placement.
// ---------------------------------------------------------------------------
struct OverlapOutcome {
  size_t current_vertices = 0;
  size_t unresolved = 0;
  size_t fragments = 0;
};

OverlapOutcome runOverlap(size_t instance, size_t hits, size_t through) {
  const Point center(0.f, 0.f, 0.f);
  const Points old_site = {Point(0.00f, 0.f, 0.f), Point(0.01f, 0.f, 0.f)};
  // One point lands in the old site's map cell; the rest is 0.6 m away.
  const Points moved = {Point(0.02f, 0.f, 0.f), Point(0.60f, 0.f, 0.f), Point(0.61f, 0.f, 0.f)};

  auto dsg = std::make_shared<DynamicSceneGraph>();
  PersistentObjectState registry;
  trainRegistry(registry);
  dsg->emplaceNode(
      DsgLayers::OBJECTS, objectId(1), makeSegment(1 * kSecond, 2 * kSecond, old_site, instance, center));
  dsg->emplaceNode(
      DsgLayers::OBJECTS, objectId(2), makeSegment(5 * kSecond, 7 * kSecond, moved, instance, center));

  feed(registry, *dsg, objectId(1));
  // Round measured while the second observation is being made, before it is ingested.
  registry.resolveRound(instance,
                        khronos::testing::craftRound(registry, instance, 6 * kSecond, hits, through),
                        {}, 6 * kSecond);
  feed(registry, *dsg, objectId(2));

  OverlapOutcome outcome;
  const auto current = registry.currentFragment(instance);
  require(current.has_value(), "K: the ID keeps a CURRENT fragment");
  outcome.current_vertices = current->geometry->numVertices();
  outcome.unresolved = registry.unresolvedCandidates(instance).size();
  outcome.fragments = registry.historyFragments(instance).size();
  return outcome;
}

void testSharedSpaceIsNotConfirmation() {
  // Every judged element seen through: the round is anomalous, the placement ends (README (5e))
  // and the later observation succeeds it instead of being folded in.
  const auto contradicted = runOverlap(801, 0, 12);
  require(contradicted.current_vertices == 3 && contradicted.fragments == 2,
          "K: an observation made after CURRENT was seen empty succeeds it, it is not folded in");
  require(contradicted.unresolved == 0, "K: nothing is left unresolved");

  const auto supported = runOverlap(802, 12, 0);
  require(supported.current_vertices == 5 && supported.fragments == 1,
          "K': with CURRENT committed in place meanwhile, the overlapping view is folded in (2 + 3)");
  require(supported.unresolved == 0, "K': nothing is left unresolved");

  // Even a single passing element among twelve is an anomalous round relative to the normal
  // share: the placement is not extended by the later view unless it stays committed in place.
  const auto mostly_supported = runOverlap(804, 11, 1);
  require(mostly_supported.fragments == 1,
          "K'': one passing element in twelve does not end the placement");
  std::cout << "PASS K/K'/K'': shared space requires a commitment of CURRENT\n";
}

// ---------------------------------------------------------------------------
// L: README (5o) (e): with no evidence the odds equal the placement's own. A placement with a
// high learned hazard leaves the candidate undecided; one with a negligible hazard absorbs it.
// ---------------------------------------------------------------------------
void testNoEvidenceFollowsThePrior() {
  const Point center(0.f, 0.f, 0.f);
  const Points old_site = {Point(0.00f, 0.f, 0.f), Point(0.01f, 0.f, 0.f)};
  const Points moved = {Point(0.02f, 0.f, 0.f), Point(0.60f, 0.f, 0.f), Point(0.61f, 0.f, 0.f)};
  for (const bool trained : {false, true}) {
    const size_t instance = trained ? 806 : 805;
    auto dsg = std::make_shared<DynamicSceneGraph>();
    PersistentObjectState registry;
    if (trained) trainRegistry(registry);
    dsg->emplaceNode(DsgLayers::OBJECTS, objectId(1),
                     makeSegment(1 * kSecond, 2 * kSecond, old_site, instance, center));
    dsg->emplaceNode(DsgLayers::OBJECTS, objectId(2),
                     makeSegment(50 * kSecond, 52 * kSecond, moved, instance, center));
    feed(registry, *dsg, objectId(1));
    feed(registry, *dsg, objectId(2));
    // A round without a single source consumes nothing and decides nothing by itself; the
    // candidate is judged by the prior hazard over the time since the placement's last evidence.
    khronos::PersistentObjectState::RoundInput empty;
    empty.session_start = 1;
    registry.resolveRound(instance, empty, {}, 52 * kSecond);
    if (trained) {
      require(registry.unresolvedCandidates(instance).empty() &&
                  registry.currentFragment(instance)->geometry->numVertices() == 5,
              "L: a negligible learned hazard places the candidate in CURRENT without evidence");
    } else {
      require(registry.unresolvedCandidates(instance).size() == 1 &&
                  registry.currentFragment(instance)->geometry->numVertices() == 2,
              "L: without learned hazards the candidate stays undecided and CURRENT unchanged");
    }
  }
  std::cout << "PASS L: with no evidence the persistence prior decides\n";
}

}  // namespace

int main() {
  testDisjointObservationStaysUnresolved();
  testConfirmedCurrentAbsorbsDisjointView();
  testWatchedMotionOpensNewState();
  testRelocationIsOrderInvariantAndProvenanceClean();
  testSharedSpaceIsNotConfirmation();
  testNoEvidenceFollowsThePrior();
  std::cout << "ALL TEMPORAL FRAGMENT STATE TESTS PASSED\n";
  return 0;
}
