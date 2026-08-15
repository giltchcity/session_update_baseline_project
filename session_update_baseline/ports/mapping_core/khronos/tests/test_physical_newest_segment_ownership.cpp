/** -----------------------------------------------------------------------------
 * Physical-ID segment merge geometry gate (PersistentObjectState regression
 * test). This exercises the single-argument, fresh-registry form of
 * canonicalizePhysicalObjects: every scenario below establishes its physical
 * ID's canonical geometry for the very first time within one call, so there
 * is no previously-persisted canonical shape to protect from a move (see
 * PersistentObjectState's "established_before_round" semantics). Multi-round
 * behavior -- where an already-established canonical shape survives a later
 * move -- is covered by test_persistent_object_state.cpp instead.
 *
 * When one physical instance ID has multiple visibility segments (tracking-
 * window breaks, session boundaries), the canonical current geometry
 * combines every STATIONARY segment's surface (they are all evidence of the
 * same static object), while a real MOVE still fully hands geometry
 * ownership to the moved segment, since nothing was established yet to
 * preserve:
 *
 *  S1 stationary, weaker:   established 8-corner mesh + later 2-point sliver,
 *                           no motion evidence -> both are static evidence of
 *                           the same object, so the merged geometry
 *                           ACCUMULATES them (8 + 2 = 10 vertices) instead of
 *                           the old support-gated winner-takes-all.
 *  S2 D1 moved evidence:    later segment carries tracker-measured motion
 *                           evidence -> its reconstruction IS the new current
 *                           pose and takes over (merged = weak mesh).
 *  S3 D3 displaced:         later stationary segment, bbox displaced >= 1 m
 *                           with no intersection -> same identity, real
 *                           displacement -> takes over (merged = weak mesh).
 *  S4 trajectory-only:      newest segment has no mesh (motion in progress);
 *                           the canonical geometry established by the earlier
 *                           settled segment remains current (a trajectory-
 *                           only round never clears established geometry).
 *
 * All four also assert the presence intervals are canonicalized as before.
 * -------------------------------------------------------------------------- */

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <string>

#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/node_symbol.h>

#include "khronos/backend/update_khronos_objects_functor.h"
#include "khronos/utils/khronos_attribute_utils.h"

namespace {

using khronos::DynamicSceneGraph;
using khronos::KhronosObjectAttributes;
using khronos::NodeId;
using khronos::Point;
using khronos::Points;
using khronos::TimeStamp;
using spark_dsg::BoundingBox;
using spark_dsg::DsgLayers;
using spark_dsg::Mesh;
using spark_dsg::NodeSymbol;

constexpr TimeStamp kSecond = 1'000'000'000ULL;
constexpr TimeStamp kT1 = 1 * kSecond;
constexpr TimeStamp kT2 = 2 * kSecond;
constexpr TimeStamp kT3 = 100 * kSecond;  // B-session segment: strictly later

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << "\n";
    std::exit(EXIT_FAILURE);
  }
}

NodeId objectId(size_t index) { return NodeSymbol('O', index); }

KhronosObjectAttributes::Ptr makeSegment(
    TimeStamp first,
    TimeStamp last,
    const Points& mesh_points,
    size_t instance_id,
    std::optional<size_t> reconstruction_frames = std::nullopt,
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
  attrs->bounding_box = BoundingBox(mesh_points);
  attrs->position = attrs->bounding_box.world_P_center.cast<double>();
  attrs->first_observed_ns = {first};
  attrs->last_observed_ns = {last};
  khronos::setObservationBounds(*attrs, first, last);
  attrs->details["instance_id"] = {instance_id};
  if (reconstruction_frames) {
    attrs->details[khronos::kReconstructionFramesDetail] = {*reconstruction_frames};
  }
  if (has_dynamic_history) {
    attrs->details[khronos::kHasDynamicHistoryDetail] = {*has_dynamic_history ? 1u : 0u};
  }
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

const KhronosObjectAttributes* runMerge(DynamicSceneGraph& dsg,
                                        size_t num_segments,
                                        const std::string& scenario) {
  const size_t merged =
      khronos::UpdateKhronosObjectsFunctor::canonicalizePhysicalObjects(dsg);
  require(merged == num_segments - 1,
          scenario + ": all same-instance segments are merged into one");
  return &dsg.getNode(objectId(1)).attributes<KhronosObjectAttributes>();
}

void requireCanonicalPresence(const KhronosObjectAttributes& attrs,
                              const std::string& scenario) {
  require(!attrs.first_observed_ns.empty() && !attrs.last_observed_ns.empty(),
          scenario + ": presence intervals survive canonicalization");
  require(attrs.first_observed_ns.front() == kT1 &&
              attrs.last_observed_ns.back() == kT3,
          scenario + ": presence spans both segments (no information loss)");
}

}  // namespace

int main() {
  // A "good" established segment: full 8-corner box.
  const Points good_mesh = {Point(0.f, 0.f, 0.f), Point(1.f, 0.f, 0.f),
                            Point(0.f, 1.f, 0.f), Point(1.f, 1.f, 0.f),
                            Point(0.f, 0.f, 1.f), Point(1.f, 0.f, 1.f),
                            Point(0.f, 1.f, 1.f), Point(1.f, 1.f, 1.f)};
  // A "weak" newer segment: 2-point sliver from a partial view.
  const Points weak_mesh = {Point(0.1f, 0.1f, 0.1f), Point(0.2f, 0.1f, 0.1f)};
  // A displaced weak segment: same instance, far away (D3 move between
  // sessions, no motion bookkeeping carried over).
  const Points far_mesh = {Point(5.1f, 0.1f, 0.1f), Point(5.2f, 0.1f, 0.1f)};

  // --- S1: stationary weaker re-observation must not regress the established
  // mesh (the wardrobe flicker path) ---------------------------------------
  {
    auto dsg = std::make_shared<DynamicSceneGraph>();
    require(dsg->emplaceNode(DsgLayers::OBJECTS, objectId(1),
                             makeSegment(kT1, kT2, good_mesh, 7, 8, false)),
            "S1: segment 1 inserted");
    require(dsg->emplaceNode(DsgLayers::OBJECTS, objectId(2),
                             makeSegment(kT3, kT3, weak_mesh, 7, 1, false)),
            "S1: segment 2 inserted");
    const auto* attrs = runMerge(*dsg, 2, "S1");
    std::cout << "S1 (stationary, weak re-observation): merged current mesh has "
              << attrs->mesh.points.size() << " vertices (established had 8, weak had 2)\n";
    require(attrs->mesh.points.size() == 8 + weak_mesh.size(),
            "S1: static re-observations accumulate (8 established + 2 weak = 10), "
            "not a support-gated winner-takes-all");
    require(samePoints(Points(attrs->mesh.points.begin(), attrs->mesh.points.begin() + 8),
                       good_mesh),
            "S1: the established mesh's vertices survive the accumulation unchanged "
            "(the weak sliver's box is fully contained in the established box, so "
            "the union bounding box -- and thus the box-frame reprojection -- is a "
            "no-op)");
    // The weak sliver's own bounding box is degenerate (its two points share
    // y/z), so BoundingBox::merge() is a no-op and the union box stays
    // exactly the established box: the merged position/bbox stay there too.
    const Eigen::Vector3d good_center = BoundingBox(good_mesh).world_P_center.cast<double>();
    require((attrs->position - good_center).norm() < 1e-4,
            "S1: merged position/bbox stay with the established (union) box center");
    requireCanonicalPresence(*attrs, "S1");
  }

  // --- S2: D1 moved segment (tracker motion evidence) takes over -----------
  {
    auto dsg = std::make_shared<DynamicSceneGraph>();
    require(dsg->emplaceNode(DsgLayers::OBJECTS, objectId(1),
                             makeSegment(kT1, kT2, good_mesh, 7, 8, false)),
            "S2: segment 1 inserted");
    require(dsg->emplaceNode(DsgLayers::OBJECTS, objectId(2),
                             makeSegment(kT3, kT3, weak_mesh, 7, 1, true)),
            "S2: segment 2 inserted");
    const auto* attrs = runMerge(*dsg, 2, "S2");
    std::cout << "S2 (D1 moved evidence): merged current mesh has "
              << attrs->mesh.points.size() << " vertices (weak segment takes over)\n";
    require(samePoints(attrs->mesh.points, weak_mesh),
            "S2: the moved segment's reconstruction becomes the current pose");
    requireCanonicalPresence(*attrs, "S2");
  }

  // --- S3: D3 displaced stationary segment takes over ----------------------
  {
    auto dsg = std::make_shared<DynamicSceneGraph>();
    require(dsg->emplaceNode(DsgLayers::OBJECTS, objectId(1),
                             makeSegment(kT1, kT2, good_mesh, 7, 8, false)),
            "S3: segment 1 inserted");
    require(dsg->emplaceNode(DsgLayers::OBJECTS, objectId(2),
                             makeSegment(kT3, kT3, far_mesh, 7, 2, false)),
            "S3: segment 2 inserted");
    const auto* attrs = runMerge(*dsg, 2, "S3");
    std::cout << "S3 (D3 displaced): merged current mesh has "
              << attrs->mesh.points.size() << " vertices (displaced segment takes over)\n";
    require(samePoints(attrs->mesh.points, far_mesh),
            "S3: a displaced stationary segment of the same identity moves the "
            "current geometry");
    requireCanonicalPresence(*attrs, "S3");
  }

  // --- S4: trajectory-only newest does not clear established geometry -----
  {
    auto dsg = std::make_shared<DynamicSceneGraph>();
    require(dsg->emplaceNode(DsgLayers::OBJECTS, objectId(1),
                             makeSegment(kT1, kT2, good_mesh, 7, 8, false)),
            "S4: segment 1 inserted");
    // A terminal D1 track: trajectory only, no static mesh asserted.
    auto traj_only = std::make_unique<KhronosObjectAttributes>();
    traj_only->mesh = Mesh(false, true, false, true);
    traj_only->bounding_box =
        BoundingBox(Point(0.2f, 0.2f, 0.2f), Point(5.f, 5.f, 5.f));
    traj_only->position = traj_only->bounding_box.world_P_center.cast<double>();
    traj_only->first_observed_ns = {kT3};
    traj_only->last_observed_ns = {kT3};
    khronos::setObservationBounds(*traj_only, kT3, kT3);
    traj_only->details["instance_id"] = {7};
    traj_only->details[khronos::kHasDynamicHistoryDetail] = {1u};
    require(dsg->emplaceNode(DsgLayers::OBJECTS, objectId(2), std::move(traj_only)),
            "S4: trajectory-only segment inserted");
    const auto* attrs = runMerge(*dsg, 2, "S4");
    std::cout << "S4 (trajectory-only newest): merged current mesh has "
              << attrs->mesh.points.size() << " vertices (established mesh kept)\n";
    require(samePoints(attrs->mesh.points, good_mesh),
            "S4: a trajectory-only newest segment (no mesh, motion in progress) "
            "does not clear the established canonical geometry -- it remains "
            "the current static surface");
    requireCanonicalPresence(*attrs, "S4");
  }

  std::cout << "PASS: geometry takeover is gated by support evidence and real "
               "displacement; trajectory-only segments never borrow geometry\n";
  return 0;
}
