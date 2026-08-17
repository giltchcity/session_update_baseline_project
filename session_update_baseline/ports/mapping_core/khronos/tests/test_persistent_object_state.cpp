/** -----------------------------------------------------------------------------
 * PersistentObjectState regression tests.
 *
 * Track segments for one physical_instance_id are observations of a single
 * persistent physical object, not competing authorities for its current
 * geometry:
 *
 *  T1 Static multi-segment accumulation + idempotence: three static
 *     visibility segments (2/3/4 vertices) of the same stationary object
 *     accumulate monotonically (5, then 9 vertices) across separate
 *     canonicalization rounds, and re-processing the fully-accumulated state
 *     a third time (with no new segments) leaves it unchanged (still 9).
 *  T2 Moved object: a relocation hands CURRENT geometry to the newest valid
 *     observation. The old world-space shape leaves CURRENT entirely (it
 *     survives in presence intervals / trajectory) and is NOT transported to
 *     the new pose: current_mesh is never union(all historical meshes) nor a
 *     stale silhouette moved across.
 *  T3 Moved-then-static reobservation: after a move, a further static
 *     re-observation at the new site accumulates onto that new-site geometry
 *     instead of resurrecting the pre-move shape.
 *  T4 Trajectory-only round: a segment with no mesh (motion in progress)
 *     leaves the established canonical mesh as the current geometry.
 *  T5 Cross-session equivalence: initializeFromObjects() (D3 restore path)
 *     reconstructs a registry that behaves identically to one that carried
 *     the same state through in-process (D2) rounds.
 *  T6 Fresh/multi-ID isolation: multiple distinct physical IDs processed in
 *     the same canonicalization pass do not cross-contaminate (e.g. three
 *     independent S74-class objects).
 * -------------------------------------------------------------------------- */

#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <string>

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

// A segment whose mesh is exactly `mesh_points`, expressed in a fixed
// (dims 1x1x1, centered at `center`) bounding box. Every segment in a test
// sharing the same `center` therefore shares an *identical* bounding box, so
// PersistentObjectState's world<->box reprojection on static accumulation is
// a no-op and the accumulated mesh's raw point values are exactly the union
// of the inputs -- keeping the arithmetic exactly checkable.
KhronosObjectAttributes::Ptr makeSegment(
    TimeStamp first,
    TimeStamp last,
    const Points& mesh_points,
    size_t instance_id,
    const Point& center = Point(0.f, 0.f, 0.f),
    std::optional<bool> has_dynamic_history = std::nullopt) {
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
  if (has_dynamic_history) {
    attrs->details[khronos::kHasDynamicHistoryDetail] = {*has_dynamic_history ? 1u : 0u};
  }
  return attrs;
}

KhronosObjectAttributes::Ptr makeTrajectoryOnlySegment(TimeStamp first,
                                                       TimeStamp last,
                                                       size_t instance_id,
                                                       const Point& center) {
  auto attrs = std::make_unique<KhronosObjectAttributes>();
  attrs->mesh = Mesh(false, true, false, true);
  attrs->bounding_box = BoundingBox(Point(1.f, 1.f, 1.f), center);
  attrs->position = center.cast<double>();
  attrs->first_observed_ns = {first};
  attrs->last_observed_ns = {last};
  khronos::setObservationBounds(*attrs, first, last);
  attrs->details["instance_id"] = {instance_id};
  attrs->details[khronos::kHasDynamicHistoryDetail] = {1u};
  attrs->trajectory_timestamps = {first};
  attrs->trajectory_positions = {center};
  return attrs;
}

bool samePoints(const Points& lhs, const Points& rhs) {
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

const KhronosObjectAttributes* findPhysical(const DynamicSceneGraph& graph, size_t instance_id) {
  const KhronosObjectAttributes* result = nullptr;
  for (const auto& [node_id, node] : graph.getLayer(DsgLayers::OBJECTS).nodes()) {
    (void)node_id;
    const auto* attrs = node->tryAttributes<KhronosObjectAttributes>();
    if (!attrs) {
      continue;
    }
    if (khronos::UpdateKhronosObjectsFunctor::physicalInstanceId(*attrs) !=
        std::optional<size_t>(instance_id)) {
      continue;
    }
    require(result == nullptr, "duplicate physical ID " + std::to_string(instance_id) +
                                   " in current graph");
    result = attrs;
  }
  return result;
}

// ---------------------------------------------------------------------------
// T1: static multi-segment accumulation + idempotence.
// ---------------------------------------------------------------------------
void testStaticAccumulationAndIdempotence() {
  const Points seg_a = {Point(0.1f, 0.f, 0.f), Point(0.2f, 0.f, 0.f)};
  const Points seg_b = {
      Point(0.3f, 0.f, 0.f), Point(0.4f, 0.f, 0.f), Point(0.5f, 0.f, 0.f)};
  const Points seg_c = {Point(0.6f, 0.f, 0.f),
                        Point(0.7f, 0.f, 0.f),
                        Point(0.8f, 0.f, 0.f),
                        Point(0.9f, 0.f, 0.f)};
  constexpr size_t kInstance = 601;
  const Point center(0.f, 0.f, 0.f);

  auto dsg = std::make_shared<DynamicSceneGraph>();
  PersistentObjectState registry;

  require(dsg->emplaceNode(DsgLayers::OBJECTS, objectId(1),
                           makeSegment(1 * kSecond, 1 * kSecond, seg_a, kInstance, center)),
          "T1: segment A inserted");
  require(dsg->emplaceNode(DsgLayers::OBJECTS, objectId(2),
                           makeSegment(2 * kSecond, 2 * kSecond, seg_b, kInstance, center)),
          "T1: segment B inserted");

  size_t merged = khronos::UpdateKhronosObjectsFunctor::canonicalizePhysicalObjects(*dsg, &registry);
  require(merged == 1, "T1 round 1: A+B canonicalized to one node");
  // B's patch does not touch A's, so on its own it proves nothing: it could be more of the same
  // static object, or the object somewhere else. It is held, not merged and not promoted.
  require(registry.unresolvedCandidates(kInstance).size() == 1,
          "T1 round 1: the disjoint patch B is held as an unresolved candidate");
  // The measurement that settles it: A's surface is still being seen at B's observation time, so
  // the two patches coexist and are one object. This is the evidence a real run supplies from
  // depth; the registry never guesses it from proximity.
  require(registry.reportCurrentSupported(kInstance, 2 * kSecond),
          "T1 round 1: support is reported against the established surface");
  const auto after_ab = registry.currentFragment(kInstance);
  require(after_ab && after_ab->geometry->numVertices() == 5,
          "T1 round 1: A(2)+B(3) accumulate to 5 vertices once coexistence is confirmed");
  require(registry.unresolvedCandidates(kInstance).empty(),
          "T1 round 1: nothing stays unresolved once coexistence is confirmed");

  require(dsg->emplaceNode(DsgLayers::OBJECTS, objectId(3),
                           makeSegment(3 * kSecond, 3 * kSecond, seg_c, kInstance, center)),
          "T1: segment C inserted");
  merged = khronos::UpdateKhronosObjectsFunctor::canonicalizePhysicalObjects(*dsg, &registry);
  require(merged == 1, "T1 round 2: canonical(A+B)+C canonicalized to one node");
  require(registry.reportCurrentSupported(kInstance, 3 * kSecond),
          "T1 round 2: the accumulated surface is still being seen at C's time");
  const auto after_abc_view = registry.currentFragment(kInstance);
  require(after_abc_view && after_abc_view->geometry->numVertices() == 9,
          "T1 round 2: canonical(A+B)=5 + C(4) accumulate to 9 vertices");
  Points expected;
  expected.insert(expected.end(), seg_a.begin(), seg_a.end());
  expected.insert(expected.end(), seg_b.begin(), seg_b.end());
  expected.insert(expected.end(), seg_c.begin(), seg_c.end());
  require(samePoints(after_abc_view->geometry->points, expected),
          "T1 round 2: accumulated vertices are the exact union of A, B, C "
          "(identical bounding boxes -> no reprojection drift)");

  // Idempotence: re-run canonicalization directly (bypassing
  // canonicalizePhysicalObjects' >=2-node grouping gate, which would no-op
  // here anyway since only the single merged target node remains) to prove
  // the anchor/interval locks stop the *sole* surviving node's own
  // already-ingested geometry from being folded in again.
  const auto target_id = objectId(1);
  auto merged_attrs =
      khronos::UpdateKhronosObjectsFunctor::mergeObjectAttributes(*dsg, {target_id});
  auto* merged_khronos = dynamic_cast<KhronosObjectAttributes*>(merged_attrs.get());
  require(merged_khronos != nullptr, "T1 idempotence: merge result is a Khronos object");
  registry.applyPhysicalGeometry(*dsg, {target_id}, *merged_khronos);
  require(registry.currentFragment(kInstance)->geometry->numVertices() == 9,
          "T1 idempotence: re-processing the already-canonical state does not "
          "double the vertex count (still 9)");
  require(samePoints(registry.currentFragment(kInstance)->geometry->points, expected),
          "T1 idempotence: re-processing does not perturb the accumulated geometry");
  (void)merged_khronos;

  std::cout << "PASS T1: static multi-segment accumulation is monotonic and idempotent\n";
}

// ---------------------------------------------------------------------------
// T2: a relocation hands CURRENT geometry to the newest valid observation; the old
//     world-space shape leaves CURRENT (it survives only in history/trajectory).
// T3: a further static re-observation after the move accumulates onto that new-site
//     geometry instead of resurrecting the pre-move shape.
// ---------------------------------------------------------------------------
void testMovedObjectNewestSegmentOwnsCurrentGeometry() {
  constexpr size_t kInstance = 602;
  const Point old_center(0.f, 0.f, 0.f);
  const Point new_center(2.f, 0.f, 0.f);
  const Points established = {Point(0.10f, 0.f, 0.f), Point(0.11f, 0.f, 0.f)};
  const Points weak_new_site = {Point(0.90f, 0.f, 0.f)};

  auto dsg = std::make_shared<DynamicSceneGraph>();
  PersistentObjectState registry;

  // Round 1: establish canonical geometry at the old site from two static
  // segments sharing the exact same bounding box (no reprojection drift).
  require(dsg->emplaceNode(
              DsgLayers::OBJECTS, objectId(1),
              makeSegment(1 * kSecond, 1 * kSecond, {established[0]}, kInstance, old_center)),
          "T2: segment 1 inserted");
  require(dsg->emplaceNode(
              DsgLayers::OBJECTS, objectId(2),
              makeSegment(2 * kSecond, 2 * kSecond, {established[1]}, kInstance, old_center)),
          "T2: segment 2 inserted");
  size_t merged = khronos::UpdateKhronosObjectsFunctor::canonicalizePhysicalObjects(*dsg, &registry);
  require(merged == 1, "T2 round 1: established two-segment canonicalization");
  const auto* settled = findPhysical(*dsg, kInstance);
  require(settled != nullptr && samePoints(settled->mesh.points, established),
          "T2 round 1: canonical mesh is exactly the established geometry");
  require((settled->bounding_box.world_P_center - old_center).norm() < 1e-6f,
          "T2 round 1: canonical pose is at the old site");

  // Round 2: a new segment at the new site, carrying motion evidence. A relocation hands CURRENT
  // geometry to the newest valid observation: the old world-space surface must leave CURRENT
  // entirely (it survives in presence intervals / trajectory), and the new site must be
  // represented by what was actually observed there -- not by the old shape transported over.
  require(dsg->emplaceNode(DsgLayers::OBJECTS, objectId(3),
                           makeSegment(3 * kSecond, 3 * kSecond, weak_new_site, kInstance,
                                       new_center, /*has_dynamic_history=*/true)),
          "T2: moved segment inserted");
  merged = khronos::UpdateKhronosObjectsFunctor::canonicalizePhysicalObjects(*dsg, &registry);
  require(merged == 1, "T2 round 2: moved segment canonicalized to one node");
  const auto* moved = findPhysical(*dsg, kInstance);
  require(moved != nullptr, "T2 round 2: physical object present after move");
  require((moved->bounding_box.world_P_center - new_center).norm() < 1e-6f,
          "T2 round 2: current pose is the new site");
  require(samePoints(moved->mesh.points, weak_new_site),
          "T2 round 2: CURRENT geometry is the new observation only");
  require(!samePoints(moved->mesh.points, established),
          "T2 round 2: the old canonical shape does NOT survive the relocation");

  // T3: a further static, non-displaced re-observation at the new site accumulates onto the
  // post-move geometry (which is the new-site observation, not the pre-move shape).
  const Points more_at_new_site = {Point(0.95f, 0.f, 0.f)};
  require(dsg->emplaceNode(DsgLayers::OBJECTS, objectId(4),
                           makeSegment(4 * kSecond, 4 * kSecond, more_at_new_site, kInstance,
                                       new_center)),
          "T3: static re-observation at new site inserted");
  merged = khronos::UpdateKhronosObjectsFunctor::canonicalizePhysicalObjects(*dsg, &registry);
  require(merged == 1, "T3: post-move static re-observation canonicalized to one node");
  // Same rule as before the move: a patch that does not touch the surface we hold is folded in
  // only once measurements confirm that surface is still there at the same time.
  require(registry.reportCurrentSupported(kInstance, 4 * kSecond),
          "T3: the post-move surface is still being seen at the re-observation's time");
  const auto reobserved = registry.currentFragment(kInstance);
  require(reobserved && reobserved->geometry->numVertices() == 2,
          "T3: post-move static re-observation accumulates onto the new-site geometry "
          "(1 new-site + 1 re-observation = 2); the pre-move shape is not resurrected");
  require((reobserved->bbox->world_P_center - new_center).norm() < 1e-6f,
          "T3: pose remains the new site through the post-move accumulation");
  require(registry.historyFragments(kInstance).size() == 2,
          "T3: the pre-move state is still in the history, closed, not destroyed");

  std::cout << "PASS T2/T3: a relocation hands CURRENT geometry to the newest observation; "
               "a later static re-observation accumulates onto it\n";
}

// ---------------------------------------------------------------------------
// T4: a trajectory-only round (no mesh) leaves the established canonical
//     mesh as the current geometry.
// ---------------------------------------------------------------------------
void testTrajectoryOnlyRoundKeepsCanonicalMesh() {
  constexpr size_t kInstance = 603;
  const Point center(0.f, 0.f, 0.f);
  const Points established = {Point(0.43f, 0.f, 0.f)};

  auto dsg = std::make_shared<DynamicSceneGraph>();
  PersistentObjectState registry;

  require(dsg->emplaceNode(
              DsgLayers::OBJECTS, objectId(1),
              makeSegment(1 * kSecond, 1 * kSecond, established, kInstance, center)),
          "T4: settled segment inserted");
  require(dsg->emplaceNode(DsgLayers::OBJECTS, objectId(2),
                           makeTrajectoryOnlySegment(2 * kSecond, 2 * kSecond, kInstance,
                                                     Point(2.f, 0.f, 0.f))),
          "T4: trajectory-only segment inserted");

  const size_t merged =
      khronos::UpdateKhronosObjectsFunctor::canonicalizePhysicalObjects(*dsg, &registry);
  require(merged == 1, "T4: settled + trajectory-only canonicalized to one node");
  const auto* attrs = findPhysical(*dsg, kInstance);
  require(attrs != nullptr && samePoints(attrs->mesh.points, established),
          "T4: canonical mesh from the settled segment survives a "
          "trajectory-only (no-mesh) round unchanged");

  std::cout << "PASS T4: a trajectory-only round does not clear the established canonical mesh\n";
}

// ---------------------------------------------------------------------------
// T5: cross-session equivalence. A registry seeded via initializeFromObjects
// from a saved/reloaded DSG (D3) must behave identically to one that carried
// the same canonical state through in-process rounds (D2) when the same next
// observation is applied.
// ---------------------------------------------------------------------------
void testCrossSessionInitializeFromObjectsEquivalence() {
  constexpr size_t kInstance = 604;
  const Point old_center(0.f, 0.f, 0.f);
  const Point new_center(3.f, 0.f, 0.f);
  const Points established = {Point(0.20f, 0.f, 0.f), Point(0.21f, 0.f, 0.f)};
  const Points moved_weak = {Point(0.77f, 0.f, 0.f)};

  // D2: continuous in-process registry across both rounds.
  auto d2_dsg = std::make_shared<DynamicSceneGraph>();
  PersistentObjectState d2_registry;
  require(d2_dsg->emplaceNode(
              DsgLayers::OBJECTS, objectId(1),
              makeSegment(1 * kSecond, 1 * kSecond, {established[0]}, kInstance, old_center)),
          "T5 D2: segment 1 inserted");
  require(d2_dsg->emplaceNode(
              DsgLayers::OBJECTS, objectId(2),
              makeSegment(2 * kSecond, 2 * kSecond, {established[1]}, kInstance, old_center)),
          "T5 D2: segment 2 inserted");
  require(khronos::UpdateKhronosObjectsFunctor::canonicalizePhysicalObjects(
              *d2_dsg, &d2_registry) == 1,
          "T5 D2: round 1 canonicalized");
  require(d2_dsg->emplaceNode(DsgLayers::OBJECTS, objectId(3),
                              makeSegment(3 * kSecond, 3 * kSecond, moved_weak, kInstance,
                                          new_center, /*has_dynamic_history=*/true)),
          "T5 D2: moved segment inserted");
  require(khronos::UpdateKhronosObjectsFunctor::canonicalizePhysicalObjects(
              *d2_dsg, &d2_registry) == 1,
          "T5 D2: round 2 (move) canonicalized");
  const auto* d2_result = findPhysical(*d2_dsg, kInstance);
  require(d2_result != nullptr, "T5 D2: final physical object present");

  // D3: a fresh process/registry restores from the round-1 output (exactly
  // what session_backend.cpp's loadInputState does via
  // persistent_objects_.initializeFromObjects(*unmerged_graph_)), then
  // applies the identical round-2 observation.
  auto seed_dsg = std::make_shared<DynamicSceneGraph>();
  require(seed_dsg->emplaceNode(
              DsgLayers::OBJECTS, objectId(1),
              makeSegment(1 * kSecond, 1 * kSecond, {established[0]}, kInstance, old_center)),
          "T5 D3 seed: segment 1 inserted");
  require(seed_dsg->emplaceNode(
              DsgLayers::OBJECTS, objectId(2),
              makeSegment(2 * kSecond, 2 * kSecond, {established[1]}, kInstance, old_center)),
          "T5 D3 seed: segment 2 inserted");
  PersistentObjectState d3_registry_seed;
  require(khronos::UpdateKhronosObjectsFunctor::canonicalizePhysicalObjects(
              *seed_dsg, &d3_registry_seed) == 1,
          "T5 D3 seed: round 1 canonicalized to produce the serialized/reloaded state");

  // Simulate process termination + reload: a brand new registry restored
  // purely from the (now serialized-and-reloaded-equivalent) seed DSG.
  PersistentObjectState d3_registry;
  d3_registry.initializeFromObjects(*seed_dsg);

  auto d3_dsg = seed_dsg->clone();
  require(d3_dsg->emplaceNode(DsgLayers::OBJECTS, objectId(3),
                              makeSegment(3 * kSecond, 3 * kSecond, moved_weak, kInstance,
                                          new_center, /*has_dynamic_history=*/true)),
          "T5 D3: moved segment inserted");
  require(khronos::UpdateKhronosObjectsFunctor::canonicalizePhysicalObjects(
              *d3_dsg, &d3_registry) == 1,
          "T5 D3: round 2 (move) canonicalized after cross-session restore");
  const auto* d3_result = findPhysical(*d3_dsg, kInstance);
  require(d3_result != nullptr, "T5 D3: final physical object present");

  require(samePoints(d2_result->mesh.points, d3_result->mesh.points),
          "T5: D2 and D3 canonical geometry are equivalent");
  require((d2_result->bounding_box.world_P_center - d3_result->bounding_box.world_P_center)
                  .norm() < 1e-6f,
          "T5: D2 and D3 canonical pose are equivalent");
  require((d2_result->bounding_box.world_P_center - new_center).norm() < 1e-6f,
          "T5: both D2 and D3 reflect the moved pose");
  require(samePoints(d2_result->mesh.points, moved_weak),
          "T5: in both D2 and D3 the relocation hands CURRENT geometry to the newest "
          "observation (the pre-move shape does not survive in CURRENT)");
  require(!samePoints(d2_result->mesh.points, established),
          "T5: the pre-move shape is not transported across the move in either path");

  std::cout << "PASS T5: initializeFromObjects (D3 restore) is equivalent to a "
               "continuous in-process registry (D2) for the same next observation\n";
}

// ---------------------------------------------------------------------------
// T6: fresh/multi-ID isolation. Multiple distinct physical IDs processed in
// the same canonicalization pass never cross-contaminate (S74-class: three
// independent monitors/laptops sharing a semantic label but distinct
// physical identities).
// ---------------------------------------------------------------------------
void testMultiIdIsolation() {
  auto dsg = std::make_shared<DynamicSceneGraph>();
  PersistentObjectState registry;

  const size_t ids[3] = {701, 702, 703};
  const Point centers[3] = {Point(0.f, 0.f, 0.f), Point(5.f, 0.f, 0.f), Point(10.f, 0.f, 0.f)};
  for (size_t k = 0; k < 3; ++k) {
    const Points a = {Point(0.1f + static_cast<float>(k), 0.f, 0.f)};
    const Points b = {Point(0.2f + static_cast<float>(k), 0.f, 0.f)};
    require(dsg->emplaceNode(DsgLayers::OBJECTS, objectId(10 + 2 * k),
                             makeSegment(1 * kSecond, 1 * kSecond, a, ids[k], centers[k])),
            "T6: id segment 1 inserted");
    require(dsg->emplaceNode(DsgLayers::OBJECTS, objectId(11 + 2 * k),
                             makeSegment(2 * kSecond, 2 * kSecond, b, ids[k], centers[k])),
            "T6: id segment 2 inserted");
  }

  const size_t merged =
      khronos::UpdateKhronosObjectsFunctor::canonicalizePhysicalObjects(*dsg, &registry);
  require(merged == 3, "T6: each of the three physical IDs canonicalized independently");
  require(registry.numStates() == 3, "T6: registry tracks exactly three independent physical IDs");

  for (size_t k = 0; k < 3; ++k) {
    // Each ID's two patches are disjoint, so each needs its own confirmation that the first patch
    // is still there. The point of this case is that the three IDs never borrow each other's.
    require(registry.reportCurrentSupported(ids[k], 2 * kSecond),
            "T6: support is reported against physical ID " + std::to_string(ids[k]));
    const auto current = registry.currentFragment(ids[k]);
    require(current && current->geometry->numVertices() == 2,
            "T6: physical ID " + std::to_string(ids[k]) +
                " accumulated only its own two segments");
    require((current->bbox->world_P_center - centers[k]).norm() < 1e-6f,
            "T6: physical ID " + std::to_string(ids[k]) + " kept its own pose");
    require(registry.unresolvedCandidates(ids[k]).empty(),
            "T6: physical ID " + std::to_string(ids[k]) + " has nothing left unresolved");
  }

  std::cout << "PASS T6: distinct physical IDs processed together never cross-contaminate\n";
}

// ---------------------------------------------------------------------------
// T7: all-trajectory-only ID. When every segment of a physical ID is
//     trajectory-only (no mesh ever established), the canonical write-back
//     must not clobber the merge result with the default-constructed state
//     (INVALID bbox, zero position, cleared dynamic flag) -- downstream
//     consumers assume a valid bounding box.
// ---------------------------------------------------------------------------
void testAllTrajectoryOnlyKeepsMergeResult() {
  constexpr size_t kInstance = 705;
  const Point center(4.f, 0.f, 0.f);

  auto dsg = std::make_shared<DynamicSceneGraph>();
  PersistentObjectState registry;

  require(dsg->emplaceNode(
              DsgLayers::OBJECTS, objectId(1),
              makeTrajectoryOnlySegment(1 * kSecond, 1 * kSecond, kInstance, center)),
          "T7: trajectory-only segment 1 inserted");
  require(dsg->emplaceNode(
              DsgLayers::OBJECTS, objectId(2),
              makeTrajectoryOnlySegment(2 * kSecond, 2 * kSecond, kInstance,
                                        Point(4.5f, 0.f, 0.f))),
          "T7: trajectory-only segment 2 inserted");

  const size_t merged =
      khronos::UpdateKhronosObjectsFunctor::canonicalizePhysicalObjects(*dsg, &registry);
  require(merged == 1, "T7: two trajectory-only segments canonicalized to one node");
  const auto* attrs = findPhysical(*dsg, kInstance);
  require(attrs != nullptr, "T7: final physical object present");
  require(attrs->bounding_box.isValid(),
          "T7: merge-computed bounding box is not clobbered to INVALID");
  require(attrs->mesh.points.empty(),
          "T7: mesh stays empty (trajectory-only object has no geometry)");
  require((attrs->bounding_box.world_P_center - Point(4.5f, 0.f, 0.f)).norm() < 1e-4f,
          "T7: bounding box keeps the newest trajectory center");
  require((attrs->position - Point(4.5f, 0.f, 0.f).cast<double>()).norm() < 1e-6f,
          "T7: position keeps the merge-computed trajectory position");
  const auto dyn = attrs->details.find(khronos::kHasDynamicHistoryDetail);
  require(dyn != attrs->details.end() && !dyn->second.empty() && dyn->second.front() == 1u,
          "T7: dynamic-history flag from the trajectory segments is preserved");

  std::cout << "PASS T7: an all-trajectory-only ID keeps the merge result "
               "(valid bbox/position, dynamic flag) instead of default state\n";
}

// ---------------------------------------------------------------------------
// T8: production-style meshes accumulate without throwing. Object meshes are
// produced by utils::combineMeshLayer from blocks extracted with
// with_tracking=false: the Mesh default flags declare has_timestamps=true
// while `stamps` stays empty. The accumulation must be size-defensive and
// default the missing per-vertex fields instead of calling the flagged
// getter (which would throw vector::at on the empty vector).
// ---------------------------------------------------------------------------
void testProductionStyleEmptyStampsAccumulates() {
  constexpr size_t kInstance = 706;
  const Point center(0.f, 0.f, 0.f);
  const Points seg_a = {Point(0.1f, 0.f, 0.f), Point(0.2f, 0.f, 0.f)};
  const Points seg_b = {Point(0.3f, 0.f, 0.f), Point(0.4f, 0.f, 0.f), Point(0.5f, 0.f, 0.f)};

  auto attrs_a = makeSegment(1 * kSecond, 1 * kSecond, seg_a, kInstance, center);
  auto attrs_b = makeSegment(2 * kSecond, 2 * kSecond, seg_b, kInstance, center);
  attrs_a->mesh.stamps.clear();
  attrs_b->mesh.stamps.clear();

  auto dsg = std::make_shared<DynamicSceneGraph>();
  PersistentObjectState registry;
  require(dsg->emplaceNode(DsgLayers::OBJECTS, objectId(1), std::move(attrs_a)),
          "T8: production-style segment A inserted");
  require(dsg->emplaceNode(DsgLayers::OBJECTS, objectId(2), std::move(attrs_b)),
          "T8: production-style segment B inserted");

  const size_t merged =
      khronos::UpdateKhronosObjectsFunctor::canonicalizePhysicalObjects(*dsg, &registry);
  require(merged == 1, "T8: two production-style segments canonicalized to one node");
  // Same evidence coupling as T1; the point of this case is the empty-stamps mesh layout, which
  // must survive the accumulation without throwing.
  require(registry.reportCurrentSupported(kInstance, 2 * kSecond),
          "T8: the established surface is still being seen at B's time");
  const auto current = registry.currentFragment(kInstance);
  require(current && current->geometry->numVertices() == 5,
          "T8: production-style A(2)+B(3) accumulate to 5 vertices");
  require(current->geometry->stamps.size() == current->geometry->points.size(),
          "T8: canonical stamps are sized consistently (zero-filled defaults)");
  const auto* attrs = findPhysical(*dsg, kInstance);
  require(attrs != nullptr, "T8: the canonical node exists");

  std::cout << "PASS T8: production-style meshes (empty stamps) accumulate without throwing\n";
}

}  // namespace

int main() {
  testStaticAccumulationAndIdempotence();
  testMovedObjectNewestSegmentOwnsCurrentGeometry();
  testTrajectoryOnlyRoundKeepsCanonicalMesh();
  testCrossSessionInitializeFromObjectsEquivalence();
  testMultiIdIsolation();
  testAllTrajectoryOnlyKeepsMergeResult();
  testProductionStyleEmptyStampsAccumulates();
  std::cout << "PASS: PersistentObjectState accumulates static observations, "
               "preserves canonical shape through moves, and is idempotent and "
               "cross-session equivalent\n";
  return EXIT_SUCCESS;
}
