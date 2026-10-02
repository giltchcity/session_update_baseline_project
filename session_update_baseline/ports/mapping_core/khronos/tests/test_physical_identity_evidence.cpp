#include "khronos/backend/change_detection/objects/ray_object_change_detector.h"
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

#include <Eigen/Geometry>
#include <hydra/common/global_info.h>
#include <hydra/input/camera.h>
#include <hydra/input/input_data.h>
#include <hydra/input/sensor_extrinsics.h>
#include <opencv2/core.hpp>
#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/mesh.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>

#include "khronos/active_window/data/frame_data.h"
#include "session_core/evidence/observed_absence.h"
#include "session_core/evidence/physical_evidence_store.h"
#include "session_core/evidence/element_measurement.h"
#include "session_core/model/range_model.h"
#include "session_core/testing/fixtures.h"
#include "session_core/testing/registry_fixture.h"
#include "khronos/backend/change_detection/ray_change_detector.h"
#include "khronos/backend/change_detection/ray_verificator.h"
#include "session_core/surface/closed_object_background.h"
#include "session_core/surface/frame_archive.h"
#include "khronos/backend/reconciliation/mesh/change_merger.h"
#include "khronos/utils/khronos_attribute_utils.h"

namespace {

using khronos::EndpointClass;
using khronos::EndpointEvidence;
using khronos::FrameData;
using khronos::PhysicalEvidenceStore;
using khronos::Point;
using khronos::RayChangeDetector;
using khronos::RayVerificator;
using khronos::TimeStamp;

constexpr TimeStamp kSecond = 1'000'000'000ULL;
constexpr TimeStamp kT1 = 10 * kSecond;
constexpr TimeStamp kT2 = 20 * kSecond;
constexpr int kObjectSemantic = 42;

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << "\n";
    std::exit(EXIT_FAILURE);
  }
}

template <typename Action>
void requireInvalidArgument(Action action, const std::string& context) {
  bool rejected = false;
  try {
    action();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, context);
}

std::shared_ptr<hydra::Camera> makeCamera(const std::string& name = "evidence_camera") {
  hydra::Camera::Config config;
  config.min_range = 0.1;
  config.max_range = 10.0;
  config.width = 4;
  config.height = 2;
  config.cx = 1.0f;
  config.cy = 1.0f;
  config.fx = 1.0f;
  config.fy = 1.0f;
  config.extrinsics = hydra::ParamSensorExtrinsics::Config();
  return std::make_shared<hydra::Camera>(config, name);
}

Point pointAtPixel(int u, int v, float depth = 1.0f) {
  return Point((static_cast<float>(u) - 1.0f) * depth,
               (static_cast<float>(v) - 1.0f) * depth,
               depth);
}

FrameData makeFrame(const hydra::Sensor::ConstPtr& sensor,
                    TimeStamp stamp,
                    bool typed_pattern = false,
                    int range_type = CV_32FC1) {
  hydra::InputData input(sensor);
  input.timestamp_ns = stamp;
  input.world_T_body = Eigen::Isometry3d::Identity();
  input.range_image = cv::Mat(2, 4, range_type, cv::Scalar(1));
  input.depth_image = cv::Mat(2, 4, CV_32FC1, cv::Scalar(1.0f));
  input.label_image = cv::Mat(2, 4, CV_32SC1, cv::Scalar(1));
  input.color_image = cv::Mat(2, 4, CV_8UC3, cv::Scalar(0, 0, 0));
  if (typed_pattern) {
    input.label_image.at<int>(1, 0) = kObjectSemantic;
    input.range_image.at<float>(1, 2) = 0.0f;
    input.range_image.at<float>(1, 3) =
        std::numeric_limits<float>::quiet_NaN();
  }

  FrameData data(input);
  data.instance_image = cv::Mat::zeros(2, 4, CV_32SC1);
  data.dynamic_image = cv::Mat::zeros(2, 4, CV_32SC1);
  if (typed_pattern) {
    data.instance_image.at<int>(0, 2) = 7;
    data.instance_image.at<int>(0, 3) = 7;
    data.dynamic_image.at<int>(1, 1) = 1;
  }
  return data;
}

FrameData makeEndpointFrame(const hydra::Sensor::ConstPtr& sensor,
                            TimeStamp stamp,
                            EndpointClass endpoint_class,
                            int physical_id = 0,
                            float measured_range = 1.0f) {
  hydra::InputData input(sensor);
  input.timestamp_ns = stamp;
  input.world_T_body = Eigen::Isometry3d::Identity();
  input.range_image = cv::Mat(2, 4, CV_32FC1, cv::Scalar(1.0f));
  input.depth_image = cv::Mat(2, 4, CV_32FC1, cv::Scalar(1.0f));
  input.label_image = cv::Mat(2, 4, CV_32SC1, cv::Scalar(1));
  input.color_image = cv::Mat(2, 4, CV_8UC3, cv::Scalar(0, 0, 0));
  input.range_image.at<float>(1, 1) = measured_range;
  input.depth_image.at<float>(1, 1) = measured_range;
  if (endpoint_class == EndpointClass::kInvalid) {
    input.range_image.at<float>(1, 1) = 0.0f;
  }

  FrameData data(input);
  data.instance_image = cv::Mat::zeros(2, 4, CV_32SC1);
  data.dynamic_image = cv::Mat::zeros(2, 4, CV_32SC1);
  if (endpoint_class == EndpointClass::kPhysical) {
    data.instance_image.at<int>(1, 1) = physical_id;
  } else if (endpoint_class == EndpointClass::kUnidentifiedObject) {
    // README (6s): an object label without an instance ID is a missing label (background); only a
    // pixel of a native motion cluster without an identity is an unidentified object.
    data.dynamic_image.at<int>(1, 1) = 1;
  }
  return data;
}

void requireEvidence(const EndpointEvidence& actual,
                     EndpointClass expected_class,
                     int expected_id,
                     const std::string& context) {
  require(actual.type == expected_class && actual.physical_id == expected_id,
          context + ": endpoint class or physical ID differs");
}

void testRleProjectionAndCopyOnWrite(const hydra::Sensor::ConstPtr& camera) {
  PhysicalEvidenceStore store;
  auto frame = makeFrame(camera, kT1, true);

  // Flattened RLE is exactly (README (6s): a semantic object without an instance ID is a missing
  // label, hence background; only a motion-masked pixel is an unidentified object):
  //   background x2, physical-I7 x2, background x1, unidentified-object x1, invalid x2.
  require(store.ingest(frame), "valid typed frame is ingested");
  require(store.numFrames() == 1 && store.numRuns() == 5,
          "RLE stores five maximal runs rather than the raw raster");

  const auto first = store.snapshot();
  require(first && first.numFrames() == 1 && first.numRuns() == 5,
          "snapshot exposes immutable frame/run accounting");
  requireEvidence(first.classify(kT1, pointAtPixel(0, 0)),
                  EndpointClass::kBackground,
                  0,
                  "background projection");
  requireEvidence(first.classify(kT1, pointAtPixel(2, 0)),
                  EndpointClass::kPhysical,
                  7,
                  "physical projection");
  requireEvidence(first.classify(kT1, pointAtPixel(0, 1)),
                  EndpointClass::kBackground,
                  0,
                  "semantic object without an instance ID is a missing label, README (6s)");
  requireEvidence(first.classify(kT1, pointAtPixel(1, 1)),
                  EndpointClass::kUnidentifiedObject,
                  0,
                  "dynamic pixel without an instance ID");
  requireEvidence(first.classify(kT1, pointAtPixel(2, 1)),
                  EndpointClass::kInvalid,
                  0,
                  "zero range is invalid");
  requireEvidence(first.classify(kT1, pointAtPixel(3, 1)),
                  EndpointClass::kInvalid,
                  0,
                  "non-finite range is invalid");
  requireEvidence(first.classify(kT2, pointAtPixel(0, 0)),
                  EndpointClass::kUnavailable,
                  0,
                  "missing timestamp is unavailable");
  requireEvidence(first.classify(kT1, Point(100.0f, 0.0f, 1.0f)),
                  EndpointClass::kUnavailable,
                  0,
                  "projection outside the image is unavailable");
  requireEvidence(first.classify(
                      kT1,
                      Point(std::numeric_limits<float>::quiet_NaN(), 0.0f, 1.0f)),
                  EndpointClass::kUnavailable,
                  0,
                  "non-finite query point is unavailable");

  require(store.ingest(frame), "identical timestamp replay is accepted");
  require(store.numFrames() == 1 && store.numRuns() == 5,
          "identical replay does not append frames or runs");
  auto conflict = makeFrame(camera, kT1);
  conflict.instance_image.setTo(9);
  requireInvalidArgument([&] { store.ingest(conflict); },
                         "conflicting timestamp content is rejected");
  require(store.numFrames() == 1 && store.numRuns() == 5,
          "conflicting replay leaves the frame and run ledger unchanged");
  requireEvidence(store.snapshot().classify(kT1, pointAtPixel(0, 0)),
                  EndpointClass::kBackground, 0,
                  "rejected conflict cannot replace published evidence");

  // A different observation has its own immutable sensor timestamp.
  auto later = makeFrame(camera, kT2);
  later.instance_image.setTo(9);
  require(store.ingest(later), "second timestamp is appended");
  const auto second = store.snapshot();
  require(store.numFrames() == 2 && store.numRuns() == 6 &&
              second.numFrames() == 2 && second.numRuns() == 6,
          "new physical-I9 frame adds one run to the five original runs");
  require(first.numFrames() == 1 && first.numRuns() == 5,
          "copy-on-write preserves old snapshot accounting");
  requireEvidence(first.classify(kT1, pointAtPixel(0, 0)),
                  EndpointClass::kBackground, 0,
                  "old snapshot retains the original measurement");
  requireEvidence(first.classify(kT2, pointAtPixel(0, 0)),
                  EndpointClass::kUnavailable, 0,
                  "old snapshot cannot see a subsequently appended frame");
  requireEvidence(second.classify(kT2, pointAtPixel(0, 0)),
                  EndpointClass::kPhysical, 9,
                  "new snapshot sees the later measurement");
}

void testRejectedInputsDoNotMutate(const hydra::Sensor::ConstPtr& camera) {
  PhysicalEvidenceStore store;
  auto valid = makeFrame(camera, kT1);
  require(store.ingest(valid), "control frame is valid");
  const auto frames_before = store.numFrames();
  const auto runs_before = store.numRuns();

  auto wrong_range_type = makeFrame(camera, kT2, false, CV_16UC1);
  requireInvalidArgument([&] { store.ingest(wrong_range_type); }, "non-float range is rejected");

  auto wrong_dimensions = makeFrame(camera, kT2);
  wrong_dimensions.instance_image = cv::Mat::zeros(1, 4, CV_32SC1);
  requireInvalidArgument([&] { store.ingest(wrong_dimensions); }, "mismatched identity raster is rejected");

  auto wrong_identity_type = makeFrame(camera, kT2);
  wrong_identity_type.instance_image = cv::Mat::zeros(2, 4, CV_8UC1);
  requireInvalidArgument([&] { store.ingest(wrong_identity_type); }, "non-int identity raster is rejected");

  const auto unavailable_camera = makeCamera("unregistered_evidence_camera");
  auto unavailable_sensor = makeFrame(unavailable_camera, kT2);
  requireInvalidArgument([&] { store.ingest(unavailable_sensor); }, "unregistered projection sensor is rejected");

  require(store.numFrames() == frames_before && store.numRuns() == runs_before,
          "rejected inputs cannot mutate the published snapshot");
}

void addPose(spark_dsg::DynamicSceneGraph& graph, TimeStamp stamp, size_t index = 0) {
  const hydra::RobotPrefixConfig prefix(0);
  const auto layer_key = graph.getLayerKey(spark_dsg::DsgLayers::AGENTS);
  require(layer_key.has_value(), "default graph has an agent layer key");
  if (!graph.findLayer(layer_key->layer, prefix.key)) {
    graph.addLayer(layer_key->layer, prefix.key);
  }
  const auto id = spark_dsg::NodeSymbol(prefix.key, index);
  auto attrs = std::make_unique<spark_dsg::AgentNodeAttributes>(
      std::chrono::nanoseconds(stamp),
      Eigen::Quaterniond::Identity(),
      Eigen::Vector3d::Zero(),
      id);
  require(graph.emplaceNode(layer_key->layer, id, std::move(attrs), prefix.key),
          "ray source pose is inserted");
}

spark_dsg::DynamicSceneGraph::Ptr makeRayGraph(TimeStamp stamp,
                                               const Point& endpoint) {
  auto graph = std::make_shared<spark_dsg::DynamicSceneGraph>();
  auto mesh = std::make_shared<spark_dsg::Mesh>(false, true, false, true);
  mesh->resizeVertices(1);
  mesh->setPos(0, endpoint);
  mesh->setFirstSeenTimestamp(0, stamp);
  mesh->setTimestamp(0, stamp);
  graph->setMesh(mesh);
  addPose(*graph, stamp);
  return graph;
}

RayVerificator::Config makeVerifierConfig() {
  RayVerificator::Config config;
  // A half-meter hash block keeps a surface just in front of the query in the
  // candidate set so the geometric-occlusion branch is exercised directly.
  config.block_size = 0.5f;
  config.radial_tolerance = 0.05f;
  config.depth_tolerance = 0.05f;
  config.ray_policy = RayVerificator::Config::RayPolicy::kAll;
  config.active_window_duration = 0.0f;
  config.prefix = hydra::RobotPrefixConfig(0);
  return config;
}

RayVerificator::CheckResult checkOnePhysicalRay(
    const hydra::Sensor::ConstPtr& camera,
    EndpointClass endpoint_class,
    int endpoint_physical_id,
    const Point& measured_endpoint,
    size_t query_physical_id = 7,
    bool ingest_frame = true) {
  auto store = std::make_shared<PhysicalEvidenceStore>();
  if (ingest_frame) {
    auto frame =
        makeEndpointFrame(camera, kT1, endpoint_class, endpoint_physical_id,
                          measured_endpoint.norm());
    require(store->ingest(frame), "typed endpoint frame is ingested");
  }

  RayVerificator verifier(makeVerifierConfig());
  verifier.setPhysicalEvidenceStore(store); verifier.observedAbsenceModel().setInitialRangeModel(khronos::testing::fixedRangeModel(0.02));
  require(verifier.setDsg(makeRayGraph(kT1, measured_endpoint)) ==
              RayVerificator::UpdateMode::kFullReset,
          "physical ray fixture initializes exactly once");
  return verifier.checkPhysical(Point(0.0f, 0.0f, 1.0f), query_physical_id);
}

void requireReasons(const RayVerificator::CheckResult& result,
                    size_t present,
                    size_t absent,
                    size_t inconclusive,
                    const std::string& context) {
  require(result.present.size() == present && result.absent.size() == absent &&
              result.inconclusive.size() == inconclusive,
          context + ": decisive/inconclusive vote counts differ");
}

void testTypedPhysicalReasons(const hydra::Sensor::ConstPtr& camera) {
  const Point near_endpoint(0.0f, 0.0f, 1.0f);

  const auto same = checkOnePhysicalRay(
      camera, EndpointClass::kPhysical, 7, near_endpoint);
  requireReasons(same, 1, 0, 0, "same physical ID");
  require(same.reasons.same_id == 1,
          "same-ID vote retains its typed reason");

  const auto different = checkOnePhysicalRay(
      camera, EndpointClass::kPhysical, 9, near_endpoint);
  requireReasons(different, 1, 0, 0, "co-located different physical ID: geometry is present");
  require(different.reasons.different_id == 1,
          "different-ID replacement retains its typed reason");

  const auto unidentified = checkOnePhysicalRay(
      camera, EndpointClass::kUnidentifiedObject, 0, near_endpoint);
  requireReasons(unidentified, 1, 0, 0, "co-located unidentified object: geometry is present");
  require(unidentified.reasons.unidentified_object == 1,
          "unidentified-object occluder retains its typed reason");

  const auto background = checkOnePhysicalRay(
      camera, EndpointClass::kBackground, 0, near_endpoint);
  requireReasons(background, 1, 0, 0, "co-located background: geometry is present");
  require(background.reasons.background_replacement == 1,
          "the identity of a coincident echo is recorded, not used for the decision");

  const auto through = checkOnePhysicalRay(
      camera, EndpointClass::kBackground, 0, Point(0.0f, 0.0f, 2.0f));
  requireReasons(through, 0, 1, 0, "free-space through");
  require(through.reasons.free_space == 1,
          "farther background endpoint is typed free-space absence");

  const auto closer = checkOnePhysicalRay(
      camera, EndpointClass::kBackground, 0, Point(0.0f, 0.0f, 0.9f));
  requireReasons(closer, 0, 0, 1, "closer geometric occlusion");
  require(closer.reasons.geometric_occlusion == 1,
          "nearer surface never deletes the hidden physical object");

  const auto invalid = checkOnePhysicalRay(
      camera, EndpointClass::kInvalid, 0, near_endpoint);
  requireReasons(invalid, 0, 0, 1, "invalid typed endpoint");
  require(invalid.reasons.invalid == 1,
          "invalid endpoint remains explicitly inconclusive");

  const auto unavailable = checkOnePhysicalRay(
      camera, EndpointClass::kUnavailable, 0, near_endpoint, 7, false);
  requireReasons(unavailable, 0, 0, 0, "unavailable typed endpoint");
  require(unavailable.reasons.unavailable == 0,
          "a missing sensor frame cannot manufacture a measurement vote");
  const auto legacy_through = checkOnePhysicalRay(camera,
                                                  EndpointClass::kUnavailable,
                                                  0,
                                                  Point(0.0f, 0.0f, 2.0f),
                                                  7,
                                                  false);
  requireReasons(legacy_through, 0, 0, 0,
                 "mesh free-space without an actual sensor frame");
  require(legacy_through.reasons.free_space == 0,
          "a geometric proxy cannot establish unmeasured physical absence");

  auto numeric_store = std::make_shared<PhysicalEvidenceStore>();
  auto numeric_frame =
      makeEndpointFrame(camera, kT1, EndpointClass::kBackground);
  require(numeric_store->ingest(numeric_frame),
          "numeric-boundary evidence is ingested");
  RayVerificator numeric_verifier(makeVerifierConfig());
  numeric_verifier.setPhysicalEvidenceStore(numeric_store); numeric_verifier.observedAbsenceModel().setInitialRangeModel(khronos::testing::fixedRangeModel(0.02));
  numeric_verifier.setDsg(makeRayGraph(kT1, near_endpoint));
  const auto zero_depth = numeric_verifier.checkPhysical(Point::Zero(), 7);
  requireReasons(zero_depth, 0, 0, 0, "query equals ray source");
  require(zero_depth.reasons.free_space == 0 && zero_depth.reasons.same_id == 0,
          "an unprojectable zero-depth query cannot manufacture a verdict");
  const auto nonfinite = numeric_verifier.checkPhysical(
      Point(std::numeric_limits<float>::quiet_NaN(), 0.0f, 1.0f), 7);
  requireReasons(nonfinite, 0, 0, 0, "non-finite physical query");
  require(nonfinite.reasons.invalid == 1,
          "non-finite query is rejected before grid/ray lookup");

  auto empty_graph = std::make_shared<spark_dsg::DynamicSceneGraph>();
  empty_graph->setMesh(
      std::make_shared<spark_dsg::Mesh>(false, true, false, true));
  addPose(*empty_graph, kT1);
  RayVerificator no_ray_verifier(makeVerifierConfig());
  no_ray_verifier.setPhysicalEvidenceStore(numeric_store); no_ray_verifier.observedAbsenceModel().setInitialRangeModel(khronos::testing::fixedRangeModel(0.02));
  no_ray_verifier.setDsg(empty_graph);
  const auto no_ray =
      no_ray_verifier.checkPhysical(Point(0.0f, 0.0f, 1.0f), 7);
  requireReasons(no_ray, 1, 0, 0, "actual coincident pixel without a mesh ray");
  require(no_ray.reasons.background_replacement == 1,
          "physical evidence depends on sensor coverage, not mesh-ray publication");

  // Production ordering regression: I9 need not have materialized as a DSG
  // object yet. Its frame-local typed endpoint still protects old I7 from a
  // false background-replacement decision.
  auto active_other_graph = makeRayGraph(kT1, near_endpoint);
  require(!active_other_graph->hasLayer(spark_dsg::DsgLayers::OBJECTS) ||
              active_other_graph->getLayer(spark_dsg::DsgLayers::OBJECTS)
                      .nodes().empty(),
          "active I9 is deliberately absent from the terminal object layer");
  auto active_store = std::make_shared<PhysicalEvidenceStore>();
  auto active_i9 = makeEndpointFrame(camera, kT1, EndpointClass::kPhysical, 9, .9f);
  require(active_store->ingest(active_i9), "active I9 evidence is stored");
  RayVerificator active_verifier(makeVerifierConfig());
  active_verifier.setPhysicalEvidenceStore(active_store); active_verifier.observedAbsenceModel().setInitialRangeModel(khronos::testing::fixedRangeModel(0.02));
  active_verifier.setDsg(active_other_graph);
  const auto protected_i7 =
      active_verifier.checkPhysical(Point(0.0f, 0.0f, 1.0f), 7);
  requireReasons(protected_i7, 0, 0, 1,
                 "active other-ID before terminal materialization");
  require(protected_i7.reasons.geometric_occlusion == 1,
          "measured nearer I9 protects I7 independently of terminal DSG objects");
}

// README principle 9: a background element that coincides with the surface of a placement committed
// changed has that placement's closure odds as its prior; the frames after its last support are
// its rounds; it is marked absent once the posterior odds reach (1-alpha)/alpha.
void testClosedObjectBackground(const hydra::Sensor::ConstPtr& camera) {
  const auto run = [&](EndpointClass endpoint_class, int endpoint_id, bool close_old_state,
                       float endpoint_depth, bool have_frame) {
    auto graph = makeRayGraph(kT2, Point(0.f, 0.f, endpoint_depth));
    auto& background = *graph->mesh();
    const auto append = [&](const Point& point, TimeStamp stamp) {
      const size_t i = background.numVertices();
      background.resizeVertices(i + 1);
      background.setPos(i, point);
      background.setFirstSeenTimestamp(i, stamp);
      background.setTimestamp(i, stamp);
    };
    append(Point(0.f, 0.f, 1.f), kT1);      // the old object's duplicate in the background
    append(Point(0.f, 0.f, 1.2f), kT1);     // Older wall, 20 cm behind the TV: beyond the hit band (6e).
    append(Point(0.f, 0.f, 1.005f), kT2);   // New surface within the old voxel.
    addPose(*graph, kT1, 1);

    auto old = std::make_unique<khronos::KhronosObjectAttributes>();
    old->mesh = spark_dsg::Mesh(false, true, false, true);
    old->mesh.resizeVertices(1);
    old->mesh.setPos(0, Point::Zero());
    old->mesh.setTimestamp(0, kT1);
    old->mesh.setFirstSeenTimestamp(0, kT1);
    old->bounding_box = spark_dsg::BoundingBox(Point::Ones(), Point(0, 0, 1));
    old->position = Eigen::Vector3d(0, 0, 1);
    old->first_observed_ns = {kT1};
    old->last_observed_ns = {std::numeric_limits<TimeStamp>::max()};
    khronos::setObservationBounds(*old, kT1, kT1);
    old->details["instance_id"] = {7};
    require(graph->emplaceNode(spark_dsg::DsgLayers::OBJECTS,
                              spark_dsg::NodeSymbol('O', 0), std::move(old)),
            "old physical state is inserted");
    khronos::PersistentObjectState objects;
    // The round statistics a quiet history has taught: one see-through reading is evidence of
    // absence and one hit is evidence of presence (without statistics the likelihood ratio is
    // neutral and the posterior is the closure prior alone).
    khronos::testing::trainRegistry(objects);
    objects.initializeFromObjects(*graph, kT1);
    if (close_old_state) {
      // Visible motion (D1) is committed at the odds of alpha: a trajectory-only segment.
      auto motion = std::make_unique<khronos::KhronosObjectAttributes>();
      motion->mesh = spark_dsg::Mesh(false, true, false, true);
      motion->bounding_box = spark_dsg::BoundingBox(Point::Ones(), Point(3, 0, 1));
      motion->first_observed_ns = {kT2};
      motion->last_observed_ns = {kT2};
      khronos::setObservationBounds(*motion, kT2, kT2);
      motion->trajectory_timestamps = {kT2 - kSecond, kT2};
      motion->trajectory_positions = {Point(0, 0, 1), Point(3, 0, 1)};
      motion->details["instance_id"] = {7};
      auto moving = graph->clone();
      require(moving->emplaceNode(spark_dsg::DsgLayers::OBJECTS, spark_dsg::NodeSymbol('O', 5),
                                  std::move(motion)),
              "motion-only segment is inserted");
      objects.ingestObjects(*moving);
      require(!objects.currentFragment(7).has_value(), "committed motion closes the old state");
    }
    const auto historical = graph->clone();

    auto store = std::make_shared<PhysicalEvidenceStore>();
    if (have_frame) {
      auto frame = makeEndpointFrame(camera, kT2, endpoint_class, endpoint_id, endpoint_depth);
      require(store->ingest(frame), "replacement evidence is captured");
    }
    auto config = makeVerifierConfig();
    config.depth_tolerance = 0.3f;
    RayVerificator verifier(config);
    verifier.setPhysicalEvidenceStore(store);
    verifier.observedAbsenceModel().setInitialRangeModel(khronos::testing::fixedRangeModel(0.02));
    verifier.observedAbsenceModel().statistics()->noteFrame(kT1, 10.0);
    verifier.setDsg(graph);
    khronos::BackgroundChanges changes;
    changes.assign(background.numVertices(), khronos::ChangeState::kPersistent);
    const float map_resolution = 0.05f;
    const size_t removed = khronos::markClosedObjectBackground(
        background, objects, verifier, map_resolution, kT2, changes);
    // Only a closed placement makes the prior (odds 99 of alpha); a reading behind the old
    // surface (see-through) is evidence of absence, one at the surface (a hit) is evidence of
    // presence, and a missing reading is no evidence.
    const bool should_clear = close_old_state && have_frame && endpoint_depth > 1.f + 0.5f * map_resolution;
    require(removed == (should_clear ? 1u : 0u),
            "only a closed, observed-replaced old surface is marked absent: endpoint=" +
                std::to_string(static_cast<int>(endpoint_class)) +
                " depth=" + std::to_string(endpoint_depth) +
                " captured=" + std::to_string(have_frame));
    require(changes[0] != khronos::ChangeState::kAbsent || should_clear,
            "the background duplicate is removed only with the closure and the evidence");
    khronos::ChangeMerger::Config merge_config;
    khronos::ChangeMerger merger(merge_config);
    merger.merge(*graph, changes);
    require(historical->mesh()->numVertices() == 4 || historical->mesh()->numVertices() == 3,
            "old snapshots remain intact");
  };
  run(EndpointClass::kBackground, 0, true, 1.2f, true);
  run(EndpointClass::kBackground, 0, false, 1.2f, true);
  run(EndpointClass::kBackground, 0, true, 1.f, true);
  run(EndpointClass::kBackground, 0, true, 1.2f, false);
}


void testProjectedCoverageWithoutMeshRays(const hydra::Sensor::ConstPtr& camera) {
  auto graph = std::make_shared<spark_dsg::DynamicSceneGraph>();
  graph->setMesh(std::make_shared<spark_dsg::Mesh>(false, true, false, true));
  addPose(*graph, kT2);
  auto store = std::make_shared<PhysicalEvidenceStore>();
  // A wall 20 cm behind the old site: beyond the hit band delta_+* (about 9 cm at sigma_s = 2 cm, (6e)).
  auto frame = makeEndpointFrame(camera, kT2, EndpointClass::kBackground, 0, 1.2f);
  require(store->ingest(frame), "real RGB-D wall observation is captured without any mesh ray");
  RayVerificator verifier(makeVerifierConfig());
  verifier.setPhysicalEvidenceStore(store); verifier.observedAbsenceModel().setInitialRangeModel(khronos::testing::fixedRangeModel(0.02));
  verifier.setDsg(graph);
  require(verifier.check(Point(0,0,1)).absent.empty(),
          "empty native mesh ray index has no geometric absence witness");
  require(verifier.checkPhysical(Point(0,0,1), 7).absent.size() == 1,
          "physical compatibility query still sees the actual background pixel");
  const auto snapshot = verifier.physicalEvidenceSnapshot();
  const auto projected = verifier.checkProjectedPhysical(Point(0,0,1), 7, snapshot, kT1, kT2, kT1);
  require(!projected.absent.empty(), "actual observed old site is absent even without a mesh endpoint");
  require(verifier.checkProjectedPhysical(Point(0,0,1), 7, snapshot, kT1, kT2-1, kT1).absent.empty(),
          "future frames cannot explain an earlier snapshot");
  require(verifier.checkProjectedPhysical(Point(100,0,1), 7, snapshot, kT1, kT2, kT1).absent.empty(),
          "outside-FOV surfaces remain unobserved");

  spark_dsg::Mesh surface(false, true, false, true);
  surface.resizeVertices(3);
  surface.setPos(0, Point(0,0,1));
  surface.setPos(1, Point(0,0,1));
  surface.setPos(2, Point(.001f,0,1));
  const spark_dsg::BoundingBox box(Point::Ones(), Point::Zero());
  const khronos::ElementSource source{.05f, 0, kT1};
  const auto psi = khronos::testing::fixedRangeModel(0.02);
  auto round = khronos::measureElements(*snapshot, 7, surface, box, psi, source, kT1, kT2);
  require(round.num_elements == 1 && round.verdicts.size() == 1 && round.verdicts[0].through,
          "duplicate surfaces and a repeated image pixel are one element with one verdict");

  const TimeStamp occlusion_stamp = kT2 + kSecond;
  auto closer = makeEndpointFrame(camera, occlusion_stamp, EndpointClass::kBackground, 0, .5f);
  require(store->ingest(closer), "later occluding frame has its own sensor timestamp");
  const auto occluded = verifier.checkProjectedPhysical(Point(0,0,1), 7,
      verifier.physicalEvidenceSnapshot(), occlusion_stamp, occlusion_stamp, kT2);
  require(occluded.absent.empty() && occluded.reasons.geometric_occlusion == 1,
          "direct projection preserves genuinely occluded old geometry");
  const TimeStamp moved_stamp = kT2 + 2 * kSecond;
  auto same_farther = makeEndpointFrame(camera, moved_stamp, EndpointClass::kPhysical, 7, 2.f);
  require(store->ingest(same_farther), "same identity is seen farther along the ray");
  const auto moved = verifier.checkProjectedPhysical(Point(0,0,1), 7,
      verifier.physicalEvidenceSnapshot(), moved_stamp, moved_stamp, kT2);
  require(moved.present.empty() && moved.reasons.free_space == 1,
          "same identity farther away cannot support its empty old surface");
}


void testMeasuredShortVertexInterval(const hydra::Sensor::ConstPtr& camera) {
  // Reduced from Synthetic B I49, frame 1872. The real background endpoint
  // exists, but last_seen - 3 s precedes first_seen, eliminating its mesh ray.
  constexpr TimeStamp first=201733333333ULL;
  constexpr TimeStamp observation=203400000000ULL;
  constexpr TimeStamp last=203866666667ULL;
  auto graph=makeRayGraph(observation,Point(0,0,2.876f));
  graph->mesh()->setFirstSeenTimestamp(0,first);
  graph->mesh()->setTimestamp(0,last);
  auto config=makeVerifierConfig();
  config.active_window_duration=3.f;
  config.depth_tolerance=.3f;
  config.ray_policy=RayVerificator::Config::RayPolicy::kMiddle;
  auto store=std::make_shared<PhysicalEvidenceStore>();
  require(store->ingest(makeEndpointFrame(camera,observation,
      EndpointClass::kBackground,0,2.876f)), "real short-lived wall observation stored");
  RayVerificator verifier(config);
  verifier.setPhysicalEvidenceStore(store); verifier.observedAbsenceModel().setInitialRangeModel(khronos::testing::fixedRangeModel(0.02));
  verifier.setDsg(graph);
  require(graph->mesh()->numVertices()==1 && verifier.getStatistics().rays==0,
          "existing background vertex loses its ray to the inverted adjusted interval");
  spark_dsg::Mesh surface(false,true,false,true);
  surface.resizeVertices(1);surface.setPos(0,Point(0,0,2.574172f));
  const spark_dsg::BoundingBox box(Point::Ones(),Point::Zero());
  auto snapshot=verifier.physicalEvidenceSnapshot();
  const auto old=verifier.check(Point(0,0,2.574172f),first,last);
  const khronos::ElementSource source{.05f, 0, first};
  const auto psi = khronos::testing::fixedRangeModel(0.02);
  auto measured = khronos::measureElements(*snapshot,49,surface,box,psi,source,first,last);
  require(old.present.empty() && old.absent.empty(),
          "original mesh-index query reproduces the missed old-cabinet evidence");
  require(measured.verdicts.size()==1 && measured.verdicts[0].through && measured.occluded==0,
          "measured fallback recovers the actual free-space ray without changing its timestamp");
  auto occluded_store=std::make_shared<PhysicalEvidenceStore>();
  require(occluded_store->ingest(makeEndpointFrame(camera,observation,
      EndpointClass::kBackground,0,2.0f)), "independent nearer-occlusion observation stored");
  verifier.setPhysicalEvidenceStore(occluded_store); verifier.observedAbsenceModel().setInitialRangeModel(khronos::testing::fixedRangeModel(0.02));
  measured=khronos::measureElements(*verifier.physicalEvidenceSnapshot(),49,surface,box,psi,source,first,last);
  require(measured.verdicts.empty() && measured.occluded==1,
          "the same fallback never treats an occluder as an empty old site");
}

void testFrozenEvidenceSnapshot(const hydra::Sensor::ConstPtr& camera) {
  const Point endpoint(0.0f, 0.0f, 1.0f);
  auto store = std::make_shared<PhysicalEvidenceStore>();
  auto same = makeEndpointFrame(camera, kT1, EndpointClass::kPhysical, 7);
  require(store->ingest(same), "same-ID snapshot source is ingested");

  RayVerificator verifier(makeVerifierConfig());
  verifier.setPhysicalEvidenceStore(store); verifier.observedAbsenceModel().setInitialRangeModel(khronos::testing::fixedRangeModel(0.02));
  verifier.setDsg(makeRayGraph(kT1, endpoint));
  const auto frozen = verifier.physicalEvidenceSnapshot();

  auto other = makeEndpointFrame(camera, kT2, EndpointClass::kPhysical, 9, .9f);
  require(store->ingest(other), "later other-ID occlusion is appended");
  const auto frozen_result = verifier.checkPhysical(
      Point(0.0f, 0.0f, 1.0f), 7, frozen);
  const auto live_result =
      verifier.checkPhysical(Point(0.0f, 0.0f, 1.0f), 7, kT2, kT2);
  requireReasons(frozen_result, 1, 0, 0, "caller-frozen evidence snapshot");
  requireReasons(live_result, 0, 0, 1, "new live evidence snapshot");
  require(frozen_result.reasons.same_id == 1 &&
              live_result.reasons.geometric_occlusion == 1,
          "one detector update cannot be split across store versions");
  requireReasons(verifier.checkPhysical(endpoint, 7, frozen, kT2, kT2), 0, 0, 0,
                 "caller-frozen snapshot excludes a later appended observation");
}

RayChangeDetector makeChangeDetector() {
  RayChangeDetector::Config config;
  config.temporal_resolution = 1.0f;
  config.window_size = 1;
  config.use_relative_confidence = true;
  config.absence_confidence = 0.6f;
  config.presence_confidence = 0.5f;
  return RayChangeDetector(config);
}

RayVerificator::CheckResult physicalVotes(size_t absent,
                                          size_t inconclusive,
                                          TimeStamp stamp) {
  RayVerificator::CheckResult result;
  result.absent.assign(absent, stamp);
  result.inconclusive.assign(inconclusive, stamp);
  result.reasons.background_replacement = absent;
  result.reasons.different_id = inconclusive;
  return result;
}

void testPhysicalReducerRequiresActualCoverage(const hydra::Sensor::ConstPtr& camera) {
  for (int mode=0; mode<3; ++mode) {
    const bool outside=mode==0, missing=mode==1;
    auto graph=makeRayGraph(kT2, outside?Point(6,0,2):Point(0,0,2));
    auto object=std::make_unique<khronos::KhronosObjectAttributes>();
    object->mesh=spark_dsg::Mesh(false,true,false,true);
    object->mesh.resizeVertices(1);
    object->mesh.setPos(0,outside?Point(3,0,1):Point(0,0,1));
    object->mesh.setTimestamp(0,kT1);object->mesh.setFirstSeenTimestamp(0,kT1);
    object->bounding_box=spark_dsg::BoundingBox(Point::Ones(),Point::Zero());
    object->first_observed_ns={kT1};object->last_observed_ns={kT1};
    khronos::setObservationBounds(*object,kT1,kT1);
    object->details["instance_id"]={49};
    require(graph->emplaceNode(spark_dsg::DsgLayers::OBJECTS,spark_dsg::NodeSymbol('O',0),std::move(object)),"physical reducer fixture inserted");
    auto store=std::make_shared<PhysicalEvidenceStore>();
    // Missing later coverage uses a genuinely pre-observation frame, not a
    // contradictory background measurement at the object support timestamp.
    require(store->ingest(makeEndpointFrame(camera,missing?kT1-kSecond:kT2,EndpointClass::kBackground,0,2)),"sensor frame stored");
    auto verifier=std::make_shared<RayVerificator>(makeVerifierConfig());
    verifier->setPhysicalEvidenceStore(store); verifier->observedAbsenceModel().setInitialRangeModel(khronos::testing::fixedRangeModel(0.02));verifier->setDsg(graph);
    RayChangeDetector::Config tc;tc.temporal_resolution=1;tc.window_size=1;
    auto temporal=std::make_shared<RayChangeDetector>(tc);
    khronos::RayObjectChangeDetector::Config dc;dc.time_filtering_threshold=0;dc.query_subsampling=1;
    khronos::RayObjectChangeDetector detector(dc,verifier,temporal);
    khronos::ObjectChanges changes;detector.detectChanges(*graph,{},changes);
    require(changes.size()==1,"physical reducer emits one change record");
    require((changes.begin()->last_absent!=0)==(mode==2),
      "physical object reducer must reject unavailable pixels and accept actual later free space; mode="+std::to_string(mode));
  }
}

void testSensorTimeCutoff(const hydra::Sensor::ConstPtr& camera) {
  const Point old_surface(0.f, 0.f, 1.f);
  auto store = std::make_shared<PhysicalEvidenceStore>();
  require(store->ingest(makeEndpointFrame(camera, kT1, EndpointClass::kPhysical, 49)),
          "earlier actual object support is stored");
  const auto before_append = store->snapshot(kT1);
  require(store->ingest(makeEndpointFrame(camera, kT2, EndpointClass::kBackground, 0, 2.f)),
          "future actual departure is already ingested by the active window");
  const auto cutoff = store->snapshot(kT1);
  const auto maximum = std::numeric_limits<TimeStamp>::max();
  require(cutoff.numFrames() == 1 && cutoff.numRuns() == before_append.numRuns(),
          "cutoff accounting excludes already-ingested future frames");
  require(cutoff.timestamps(0, maximum) == std::vector<TimeStamp>{kT1} &&
              cutoff.timestamps(kT2, maximum).empty(),
          "even an unbounded caller cannot enumerate past the sensor cutoff");
  requireEvidence(cutoff.classify(kT1, old_surface), EndpointClass::kPhysical, 49,
                  "measurement at the inclusive cutoff is available");
  requireEvidence(cutoff.project(kT2, old_surface).endpoint, EndpointClass::kUnavailable, 0,
                  "direct projection cannot bypass the cutoff");
  uint32_t width = 0, height = 0;
  Eigen::Isometry3f sensor_T_world;
  hydra::Sensor::ConstPtr sensor;
  std::vector<uint16_t> ranges;
  require(!cutoff.denseRange(kT2, width, height, sensor_T_world, sensor, ranges),
          "dense sensor range cannot bypass the cutoff");
  require(cutoff.denseRange(kT1, width, height, sensor_T_world, sensor, ranges) &&
              width == 4 && height == 2 && ranges.at(5) == 1000,
          "dense measurement at the cutoff remains readable");
  require(store->snapshot(kT2).timestamps(0, maximum) ==
              std::vector<TimeStamp>{kT1, kT2},
          "advancing the sensor boundary exposes the actual later frame");

  auto graph = makeRayGraph(kT2, Point(0.f, 0.f, 2.f));
  auto object = std::make_unique<khronos::KhronosObjectAttributes>();
  object->mesh = spark_dsg::Mesh(false, true, false, true);
  object->mesh.resizeVertices(1);
  object->mesh.setPos(0, old_surface);
  object->mesh.setTimestamp(0, kT1);
  object->mesh.setFirstSeenTimestamp(0, kT1);
  object->bounding_box = spark_dsg::BoundingBox(Point::Ones(), Point::Zero());
  object->first_observed_ns = {kT1};
  object->last_observed_ns = {kT1};
  khronos::setObservationBounds(*object, kT1, kT1);
  object->details["instance_id"] = {49};
  require(graph->emplaceNode(spark_dsg::DsgLayers::OBJECTS,
                            spark_dsg::NodeSymbol('O', 0), std::move(object)),
          "cutoff reducer has one actually observed physical surface");
  auto verifier = std::make_shared<RayVerificator>(makeVerifierConfig());
  verifier->setPhysicalEvidenceStore(store); verifier->observedAbsenceModel().setInitialRangeModel(khronos::testing::fixedRangeModel(0.02));
  verifier->setDsg(graph);
  verifier->setPhysicalEvidenceCutoff(kT1);
  const auto frozen = verifier->physicalEvidenceSnapshot();
  requireReasons(verifier->checkPhysicalObserved(old_surface, 49, frozen, 0, maximum, kT1),
                 1, 0, 0, "unbounded native physical query obeys the decision cutoff");
  auto temporal = std::make_shared<RayChangeDetector>(makeChangeDetector());
  khronos::RayObjectChangeDetector::Config config;
  config.time_filtering_threshold = 0;
  config.query_subsampling = 1;
  khronos::RayObjectChangeDetector detector(config, verifier, temporal);
  khronos::ObjectChanges changes;
  detector.detectChanges(*graph, {}, changes);
  require(changes.size() == 1 && changes.begin()->last_absent == 0,
          "native D2 cannot use the future departure for the earlier decision");
  verifier->setPhysicalEvidenceCutoff(kT2);
  requireReasons(verifier->checkPhysicalObserved(
                     old_surface, 49, verifier->physicalEvidenceSnapshot(), kT2, maximum, kT1),
                 0, 1, 0, "departure enters evidence exactly at its sensor timestamp");
  detector.detectChanges(*graph, {}, changes);
  require(changes.size() == 1 && changes.begin()->last_absent == kT2,
          "native D2 reports the actual departure timestamp after cutoff advances");
  requireReasons(verifier->checkPhysicalObserved(old_surface, 49, frozen, kT2, maximum, kT1),
                 0, 0, 0, "previous decision snapshot stays bounded after cutoff advances");
}

void testArchivePreservesPhysicalDynamicCandidates(const hydra::Sensor::ConstPtr& camera) {
  auto frame = makeFrame(camera, kT1);
  frame.dynamic_image.at<int>(1, 1) = 1;
  frame.instance_image.at<int>(1, 1) = 7;
  frame.dynamic_image.at<int>(1, 2) = 1;
  khronos::FrameArchive archive;
  archive.offer(frame);
  archive.offer(frame);
  require(archive.numFrames() == 1,
          "identical archive replay preserves one immutable sensor observation");
  khronos::FrameArchive::Camera archived_camera;
  const auto frames = archive.release(&archived_camera);
  require(frames.size() == 1 && frames.front().stamp == kT1 &&
              archived_camera.width == 4 && archived_camera.height == 2,
          "archive preserves the candidate sensor timestamp and raster");
  std::vector<uint16_t> ranges, identities, classes;
  std::vector<uint8_t> motion;
  require(frames.front().decode(8, ranges, identities, &classes, &motion),
          "archived physical dynamic candidates decode successfully");
  require(ranges.at(5) == 1000 && identities.at(5) == 7,
          "physical dynamic candidate keeps its measured range for later static authorization");
  // README principle 5: the archive keeps every reading with its identity, class and native motion
  // mask; the session-end refusion excludes the anonymous motion cluster with the integration mask.
  require(ranges.at(6) == 1000 && identities.at(6) == 0 && motion.at(6) == 1 && motion.at(4) == 0,
          "anonymous dynamic pixel is archived with its motion mask for the terminal judgement");
  require(ranges.at(4) == 1000 && identities.at(4) == 0,
          "ordinary background range is preserved alongside the candidate");
}

void testPhysicalConfidenceBoundary() {
  const auto detector = makeChangeDetector();
  const auto physical = RayChangeDetector::CoverageMode::kPhysical;

  const auto below = detector.detectChanges(physicalVotes(5, 5, kT1), true, physical);
  require(!below.closest_absent,
          "physical absence confidence below 0.6 is inconclusive");

  const auto equal = detector.detectChanges(physicalVotes(6, 4, kT1), true, physical);
  require(!equal.closest_absent,
          "physical absence confidence exactly 0.6 respects strict threshold");

  const auto above = detector.detectChanges(physicalVotes(7, 3, kT1), true, physical);
  require(above.closest_absent == kT1,
          "physical absence confidence above 0.6 is accepted");

  const auto decisive = detector.detectChanges(
      physicalVotes(5, 5, kT1),
      true,
      RayChangeDetector::CoverageMode::kDecisiveOnly);
  require(decisive.closest_absent == kT1,
          "generic decisive-only mode preserves its original denominator");
}

void testBackwardWindowUsesPastBins() {
  RayChangeDetector::Config config;
  config.temporal_resolution = 1.0f;
  config.window_size = 2;
  config.use_relative_confidence = true;
  config.absence_confidence = 0.6f;
  config.presence_confidence = 0.5f;
  const RayChangeDetector detector(config);

  RayVerificator::CheckResult votes;
  votes.absent = {10 * kSecond, 9 * kSecond, 9 * kSecond};
  votes.inconclusive = {10 * kSecond};
  const auto result = detector.detectChanges(
      votes, false, RayChangeDetector::CoverageMode::kPhysical);
  require(result.closest_absent == 10 * kSecond,
          "backward window must aggregate the current bin toward the past");
}

}  // namespace

// README principle 6, (6d): a round keeps the latest verdict of every element among the new
// frames; an element is one cell of the map resolution however many samples fall into it.
void testElementVerdictIsTheLatestOfTheRound(const hydra::Sensor::ConstPtr& camera) {
  RayVerificator verifier(makeVerifierConfig());
  verifier.observedAbsenceModel().setInitialRangeModel(khronos::testing::fixedRangeModel(0.02));
  auto store = std::make_shared<PhysicalEvidenceStore>();
  verifier.setPhysicalEvidenceStore(store);
  spark_dsg::Mesh surface(false, true, false, true);
  surface.resizeVertices(1);
  const Point query_point = pointAtPixel(1, 1, 2.0f);
  surface.setPos(0, query_point);
  const spark_dsg::BoundingBox box(Point::Ones(), Point::Zero());
  const float query = query_point.norm();
  for (TimeStamp t = kT1; t <= kT1 + 2 * kSecond; t += kSecond)
    require(store->ingest(makeEndpointFrame(camera, t, EndpointClass::kPhysical, 7, query)),
            "frame with an echo on the surface is stored");
  const TimeStamp through = kT1 + 2 * kSecond;
  const khronos::ElementSource source{.05f, 0, kT1};
  const auto psi = khronos::testing::fixedRangeModel(0.02);
  auto round = khronos::measureElements(*verifier.physicalEvidenceSnapshot(), 7, surface, box, psi,
                                        source, kT1, through);
  require(round.num_elements == 1 && round.verdicts.size() == 1 && !round.verdicts[0].through &&
              round.verdicts[0].label == khronos::ElementVerdict::kOwn &&
              round.latest_hit == through,
          "three echoes on the surface are one hit verdict of its own identity");
  require(store->ingest(makeEndpointFrame(camera, through + kSecond, EndpointClass::kBackground, 0,
                                          query + 1.0f)),
          "frame with an echo behind the surface is stored");
  round = khronos::measureElements(*verifier.physicalEvidenceSnapshot(), 7, surface, box, psi, source,
                                   kT1, through + kSecond);
  require(round.verdicts.size() == 1 && round.verdicts[0].through &&
              round.first_through == through + kSecond,
          "the latest echo passes the surface: the element's verdict of the round is see-through");
  round = khronos::measureElements(*verifier.physicalEvidenceSnapshot(), 7, surface, box, psi, source,
                                   through + 2 * kSecond, through + 3 * kSecond);
  require(round.verdicts.empty(), "a window without inputs carries no verdict");
  auto foreign = std::make_shared<PhysicalEvidenceStore>();
  require(foreign->ingest(makeEndpointFrame(camera, kT1, EndpointClass::kPhysical, 9, query)),
          "frame whose echo carries another identity");
  round = khronos::measureElements(foreign->snapshot(), 7, surface, box, psi, source, kT1, kT1);
  require(round.verdicts.size() == 1 && round.verdicts[0].label == khronos::ElementVerdict::kForeign,
          "a hit carrying another physical label is foreign, README (6s)");
}

int main() {
  hydra::PipelineConfig config;
  config.label_space.total_labels = 256;
  config.label_space.object_labels.insert(kObjectSemantic);
  hydra::GlobalInfo::init(config);
  const auto camera = makeCamera();
  require(hydra::GlobalInfo::instance().setSensor(camera),
          "test projection sensor is registered");

  testRleProjectionAndCopyOnWrite(camera);
  testRejectedInputsDoNotMutate(camera);
  testTypedPhysicalReasons(camera);
  testClosedObjectBackground(camera);
  testProjectedCoverageWithoutMeshRays(camera);
  testMeasuredShortVertexInterval(camera);
  testPhysicalReducerRequiresActualCoverage(camera);
  testElementVerdictIsTheLatestOfTheRound(camera);
  testFrozenEvidenceSnapshot(camera);
  testSensorTimeCutoff(camera);
  testArchivePreservesPhysicalDynamicCandidates(camera);
  testPhysicalConfidenceBoundary();
  testBackwardWindowUsesPastBins();

  hydra::GlobalInfo::reset();
  return EXIT_SUCCESS;
}
