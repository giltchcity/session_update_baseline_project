#include "khronos/backend/change_detection/objects/ray_object_change_detector.h"
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

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
#include "khronos/backend/change_detection/physical_evidence_store.h"
#include "khronos/backend/change_detection/ray_change_detector.h"
#include "khronos/backend/change_detection/ray_verificator.h"
#include "khronos/backend/reconciliation/closed_object_background.h"
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
  } else if (endpoint_class == EndpointClass::kUnidentifiedObject) {
    input.label_image.at<int>(1, 1) = kObjectSemantic;
  }

  FrameData data(input);
  data.instance_image = cv::Mat::zeros(2, 4, CV_32SC1);
  data.dynamic_image = cv::Mat::zeros(2, 4, CV_32SC1);
  if (endpoint_class == EndpointClass::kPhysical) {
    data.instance_image.at<int>(1, 1) = physical_id;
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

  // Flattened RLE is exactly:
  //   background x2, physical-I7 x2, unidentified-object x2, invalid x2.
  require(store.ingest(frame), "valid typed frame is ingested");
  require(store.numFrames() == 1 && store.numRuns() == 4,
          "RLE stores four maximal runs rather than the raw raster");

  const auto first = store.snapshot();
  require(first && first.numFrames() == 1 && first.numRuns() == 4,
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
                  EndpointClass::kUnidentifiedObject,
                  0,
                  "semantic object without an instance ID");
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

  // Replacing an equal timestamp must be atomic and leave an older snapshot
  // readable. The replacement collapses to one physical-I9 run.
  auto replacement = makeFrame(camera, kT1);
  replacement.instance_image.setTo(9);
  require(store.ingest(replacement), "same-timestamp replacement is accepted");
  const auto second = store.snapshot();
  require(store.numFrames() == 1 && store.numRuns() == 1 &&
              second.numFrames() == 1 && second.numRuns() == 1,
          "same timestamp replaces accounting rather than appending");
  requireEvidence(first.classify(kT1, pointAtPixel(0, 0)),
                  EndpointClass::kBackground,
                  0,
                  "old snapshot remains immutable");
  requireEvidence(second.classify(kT1, pointAtPixel(0, 0)),
                  EndpointClass::kPhysical,
                  9,
                  "new snapshot observes replacement");

  auto later = makeFrame(camera, kT2);
  require(store.ingest(later), "second timestamp is appended");
  require(store.numFrames() == 2 && store.numRuns() == 2,
          "frame and RLE accounting includes the new timestamp");
}

void testRejectedInputsDoNotMutate(const hydra::Sensor::ConstPtr& camera) {
  PhysicalEvidenceStore store;
  auto valid = makeFrame(camera, kT1);
  require(store.ingest(valid), "control frame is valid");
  const auto frames_before = store.numFrames();
  const auto runs_before = store.numRuns();

  auto wrong_range_type = makeFrame(camera, kT2, false, CV_16UC1);
  require(!store.ingest(wrong_range_type), "non-float range is rejected");

  auto wrong_dimensions = makeFrame(camera, kT2);
  wrong_dimensions.instance_image = cv::Mat::zeros(1, 4, CV_32SC1);
  require(!store.ingest(wrong_dimensions), "mismatched identity raster is rejected");

  auto wrong_identity_type = makeFrame(camera, kT2);
  wrong_identity_type.instance_image = cv::Mat::zeros(2, 4, CV_8UC1);
  require(!store.ingest(wrong_identity_type), "non-int identity raster is rejected");

  const auto unavailable_camera = makeCamera("unregistered_evidence_camera");
  auto unavailable_sensor = makeFrame(unavailable_camera, kT2);
  require(!store.ingest(unavailable_sensor), "unregistered projection sensor is rejected");

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
        makeEndpointFrame(camera, kT1, endpoint_class, endpoint_physical_id);
    require(store->ingest(frame), "typed endpoint frame is ingested");
  }

  RayVerificator verifier(makeVerifierConfig());
  verifier.setPhysicalEvidenceStore(store);
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
  requireReasons(different, 0, 0, 1, "different physical ID");
  require(different.reasons.different_id == 1,
          "different-ID occluder retains its typed reason");

  const auto unidentified = checkOnePhysicalRay(
      camera, EndpointClass::kUnidentifiedObject, 0, near_endpoint);
  requireReasons(unidentified, 0, 0, 1, "unidentified semantic object");
  require(unidentified.reasons.unidentified_object == 1,
          "unidentified-object occluder retains its typed reason");

  const auto background = checkOnePhysicalRay(
      camera, EndpointClass::kBackground, 0, near_endpoint);
  requireReasons(background, 0, 1, 0, "background replacement");
  require(background.reasons.background_replacement == 1,
          "background replacement cannot masquerade as anonymous persistence");

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
  requireReasons(unavailable, 0, 0, 1, "unavailable typed endpoint");
  require(unavailable.reasons.unavailable == 1,
          "missing session-local evidence remains explicitly inconclusive");
  const auto legacy_through = checkOnePhysicalRay(camera,
                                                  EndpointClass::kUnavailable,
                                                  0,
                                                  Point(0.0f, 0.0f, 2.0f),
                                                  7,
                                                  false);
  requireReasons(legacy_through, 0, 1, 0,
                 "legacy free-space without typed endpoint");
  require(legacy_through.reasons.unavailable == 1,
          "legacy through remains absence while preserving unavailable provenance");

  auto numeric_store = std::make_shared<PhysicalEvidenceStore>();
  auto numeric_frame =
      makeEndpointFrame(camera, kT1, EndpointClass::kBackground);
  require(numeric_store->ingest(numeric_frame),
          "numeric-boundary evidence is ingested");
  RayVerificator numeric_verifier(makeVerifierConfig());
  numeric_verifier.setPhysicalEvidenceStore(numeric_store);
  numeric_verifier.setDsg(makeRayGraph(kT1, near_endpoint));
  const auto zero_depth = numeric_verifier.checkPhysical(Point::Zero(), 7);
  requireReasons(zero_depth, 0, 0, 1, "query equals ray source");
  require(zero_depth.reasons.invalid == 1,
          "zero-depth query cannot manufacture a finite verdict");
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
  no_ray_verifier.setPhysicalEvidenceStore(numeric_store);
  no_ray_verifier.setDsg(empty_graph);
  const auto no_ray =
      no_ray_verifier.checkPhysical(Point(0.0f, 0.0f, 1.0f), 7);
  requireReasons(no_ray, 0, 0, 0, "no measurement ray");

  // Production ordering regression: I9 need not have materialized as a DSG
  // object yet. Its frame-local typed endpoint still protects old I7 from a
  // false background-replacement decision.
  auto active_other_graph = makeRayGraph(kT1, near_endpoint);
  require(!active_other_graph->hasLayer(spark_dsg::DsgLayers::OBJECTS) ||
              active_other_graph->getLayer(spark_dsg::DsgLayers::OBJECTS)
                      .nodes().empty(),
          "active I9 is deliberately absent from the terminal object layer");
  auto active_store = std::make_shared<PhysicalEvidenceStore>();
  auto active_i9 = makeEndpointFrame(camera, kT1, EndpointClass::kPhysical, 9);
  require(active_store->ingest(active_i9), "active I9 evidence is stored");
  RayVerificator active_verifier(makeVerifierConfig());
  active_verifier.setPhysicalEvidenceStore(active_store);
  active_verifier.setDsg(active_other_graph);
  const auto protected_i7 =
      active_verifier.checkPhysical(Point(0.0f, 0.0f, 1.0f), 7);
  requireReasons(protected_i7, 0, 0, 1,
                 "active other-ID before terminal materialization");
  require(protected_i7.reasons.different_id == 1,
          "store evidence protects I7 independently of terminal DSG objects");
}

void testClosedObjectBackground(const hydra::Sensor::ConstPtr& camera) {
  const auto run = [&](EndpointClass endpoint_class, int endpoint_id,
                       bool close_old_state, float endpoint_depth,
                       TimeStamp query_time, bool have_frame, bool reobserved_old = false, bool present_again = false, float association_offset = 0.f, float map_resolution = 0.05f) {
    auto graph = makeRayGraph(kT2, Point(0.f, 0.f, endpoint_depth));
    auto& background = *graph->mesh();
    const auto append = [&](const Point& point, TimeStamp stamp) {
      const size_t i = background.numVertices();
      background.resizeVertices(i + 1);
      background.setPos(i, point);
      background.setFirstSeenTimestamp(i, stamp);
      background.setTimestamp(i, stamp);
    };
    append(Point(0.f, 0.f, 1.f), reobserved_old ? kT1 + (kT2-kT1)/2 : kT1);
    // The inherited object can retain A's timestamp while its background
    // duplicate was re-observed during B, before the actual departure.
    append(Point(0.f, 0.f, 1.055f), kT1);   // Older wall, 5.5 cm behind TV.
    append(Point(0.f, 0.f, 1.005f), kT2);   // New surface within the old voxel.
    addPose(*graph, kT1, 1);

    auto old = std::make_unique<khronos::KhronosObjectAttributes>();
    old->mesh = spark_dsg::Mesh(false, true, false, true);
    old->mesh.resizeVertices(1);
    old->mesh.setPos(0, Point::Zero());
    old->mesh.setTimestamp(0, kT1);
    old->mesh.setFirstSeenTimestamp(0, kT1);
    old->bounding_box = spark_dsg::BoundingBox(Point::Ones(), Point(-association_offset, 0, 1));
    old->position = Eigen::Vector3d(0, 0, 1);
    old->first_observed_ns = {kT1};
    old->last_observed_ns = {kT1};
    khronos::setObservationBounds(*old, kT1, kT1);
    old->details["instance_id"] = {7};
    require(graph->emplaceNode(spark_dsg::DsgLayers::OBJECTS,
                              spark_dsg::NodeSymbol('O', 0), std::move(old)),
            "old physical state is inserted");
    khronos::PersistentObjectState objects;
    objects.initializeFromObjects(*graph);
    if (close_old_state) {
      require(objects.reportCurrentContradicted(7, kT2),
              "measurement closes the old physical state");
    }
    const auto historical = graph->clone();

    auto store = std::make_shared<PhysicalEvidenceStore>();
    if (have_frame) {
      auto frame = makeEndpointFrame(
          camera, kT2, endpoint_class, endpoint_id, endpoint_depth);
      require(store->ingest(frame), "replacement evidence is captured");
    }
    if (present_again) {
      auto replacement_surface = makeEndpointFrame(camera, query_time,
          EndpointClass::kBackground, 0, 1.f);
      require(store->ingest(replacement_surface), "later actual surface is observed after the empty interval");
    }
    auto config = makeVerifierConfig();
    config.depth_tolerance = 0.3f;  // Actual production tolerance, not a tighter test value.
    RayVerificator verifier(config);
    verifier.setPhysicalEvidenceStore(store);
    verifier.setDsg(graph);
    RayChangeDetector::Config temporal;
    temporal.temporal_resolution = 5.f;
    temporal.window_size = 5;
    temporal.absence_confidence = 0.6f;
    temporal.presence_confidence = 0.5f;
    const RayChangeDetector detector(temporal);
    khronos::BackgroundChanges changes;
    changes.assign(background.numVertices(), khronos::ChangeState::kPersistent);
    if (endpoint_class == EndpointClass::kBackground && endpoint_depth == 1.055f &&
        query_time == kT2 && have_frame) {
      const auto ordinary = verifier.check(Point(0, 0, 1), kT1 + 1, kT2);
      require(!ordinary.present.empty() && ordinary.absent.empty(),
              "ordinary geometry preserves the thin old surface (production regression)");
    }
    const size_t removed = khronos::markClosedObjectBackground(
        background, objects, verifier, detector, map_resolution, query_time, changes);
    const bool should_clear = close_old_state && have_frame && query_time == kT2 &&
                              (endpoint_class == EndpointClass::kBackground ||
                               (endpoint_class == EndpointClass::kPhysical && endpoint_id != 7)) &&
                              endpoint_depth > 1.f + 0.5f * map_resolution + .001f &&
                              association_offset <= std::sqrt(3.f) * map_resolution;
    require(removed == (should_clear ? 1u : 0u),
            "only a closed, observed-replaced old surface is marked absent: endpoint=" +
                std::to_string(static_cast<int>(endpoint_class)) +
                " depth=" + std::to_string(endpoint_depth) +
                " captured=" + std::to_string(have_frame));
    require((changes[1] == khronos::ChangeState::kAbsent) == should_clear,
            "old TV background duplicate follows its confirmed physical-state closure");
    require(changes[0] != khronos::ChangeState::kAbsent &&
                changes[2] != khronos::ChangeState::kAbsent &&
                changes[3] != khronos::ChangeState::kAbsent,
            "new wall, older nearby wall and later same-voxel surface are preserved");
    khronos::ChangeMerger::Config merge_config;
    khronos::ChangeMerger merger(merge_config);
    merger.merge(*graph, changes);
    require(graph->mesh()->numVertices() == (should_clear ? 3u : 4u),
            "ordinary ChangeMerger applies the additional physical absence verdict");
    require(historical->mesh()->numVertices() == 4 &&
                objects.historyFragments(7).front().geometry->numVertices() == 1,
            "old snapshots and closed object history remain intact");
  };
  run(EndpointClass::kBackground, 0, true, 1.055f, kT2, true);
  run(EndpointClass::kBackground, 0, true, 1.055f, kT2, true, true);
  run(EndpointClass::kBackground, 0, true, 1.055f, kT2 + 1000000000ULL, true, true, true);
  run(EndpointClass::kPhysical, 7, true, 1.f, kT2, true, true);
  run(EndpointClass::kBackground, 0, true, 0.55f, kT2, true, true);
  run(EndpointClass::kBackground, 0, false, 1.055f, kT2, true);
  run(EndpointClass::kPhysical, 7, true, 1.f, kT2, true);
  run(EndpointClass::kPhysical, 9, true, 1.f, kT2, true);
  run(EndpointClass::kBackground, 0, true, 1.f, kT2, true);
  run(EndpointClass::kPhysical, 9, true, 1.055f, kT2, true);
  run(EndpointClass::kPhysical, 9, true, 0.55f, kT2, true);
  run(EndpointClass::kUnidentifiedObject, 0, true, 1.f, kT2, true);
  run(EndpointClass::kInvalid, 0, true, 1.f, kT2, true);
  run(EndpointClass::kBackground, 0, true, 0.55f, kT2, true);
  run(EndpointClass::kBackground, 0, true, 1.055f, kT2, false);
  run(EndpointClass::kBackground, 0, true, 1.055f, kT2 - 1, true);
  // Independent coarse/fine reconstructions differ by more than half a cell.
  // Expanding candidates must not expand the depth absence tolerance.
  for (float offset : {.06f, .104f, .2f}) {
    run(EndpointClass::kBackground, 0, true, 1.055f, kT2, true, false, false, offset, .1f);
    run(EndpointClass::kBackground, 0, true, 1.f, kT2, true, false, false, offset, .1f);
    run(EndpointClass::kBackground, 0, true, .55f, kT2, true, false, false, offset, .1f);
    run(EndpointClass::kBackground, 0, true, 1.055f, kT2, false, false, false, offset, .1f);
    run(EndpointClass::kBackground, 0, true, 1.055f, kT2 + kSecond, true, true, true, offset, .1f);
  }

}


void testProjectedCoverageWithoutMeshRays(const hydra::Sensor::ConstPtr& camera) {
  auto graph = std::make_shared<spark_dsg::DynamicSceneGraph>();
  graph->setMesh(std::make_shared<spark_dsg::Mesh>(false, true, false, true));
  addPose(*graph, kT2);
  auto store = std::make_shared<PhysicalEvidenceStore>();
  auto frame = makeEndpointFrame(camera, kT2, EndpointClass::kBackground, 0, 1.055f);
  require(store->ingest(frame), "real RGB-D wall observation is captured without any mesh ray");
  RayVerificator verifier(makeVerifierConfig());
  verifier.setPhysicalEvidenceStore(store);
  verifier.setDsg(graph);
  require(verifier.checkPhysical(Point(0,0,1), 7).absent.empty(),
          "empty mesh ray index reproduces missed physical visibility");
  const auto snapshot = verifier.physicalEvidenceSnapshot();
  const auto projected = verifier.checkProjectedPhysical(Point(0,0,1), 7, snapshot, kT1, kT2);
  require(!projected.absent.empty(), "actual observed old site is absent even without a mesh endpoint");
  require(verifier.checkProjectedPhysical(Point(0,0,1), 7, snapshot, kT1, kT2-1).absent.empty(),
          "future frames cannot explain an earlier snapshot");
  require(verifier.checkProjectedPhysical(Point(100,0,1), 7, snapshot, kT1, kT2).absent.empty(),
          "outside-FOV surfaces remain unobserved");

  spark_dsg::Mesh surface(false, true, false, true);
  surface.resizeVertices(3);
  surface.setPos(0, Point(0,0,1));
  surface.setPos(1, Point(0,0,1));
  surface.setPos(2, Point(.001f,0,1));
  const spark_dsg::BoundingBox box(Point::Ones(), Point::Zero());
  auto counts = verifier.countProjectedPhysicalSurface(7, surface, box, snapshot, .05f, kT1, kT2);
  require(counts.surface_samples == 1 && counts.contradiction_rays == 1,
          "duplicate surfaces and a repeated image pixel are counted once");

  auto closer = makeEndpointFrame(camera, kT2, EndpointClass::kBackground, 0, .5f);
  require(store->ingest(closer), "occluding frame replaces the exact timestamp");
  const auto occluded = verifier.checkProjectedPhysical(Point(0,0,1), 7,
      verifier.physicalEvidenceSnapshot(), kT1, kT2);
  require(occluded.absent.empty() && occluded.reasons.geometric_occlusion == 1,
          "direct projection preserves genuinely occluded old geometry");
  auto same_farther = makeEndpointFrame(camera, kT2, EndpointClass::kPhysical, 7, 2.f);
  require(store->ingest(same_farther), "same identity is seen farther along the ray");
  const auto moved = verifier.checkProjectedPhysical(Point(0,0,1), 7,
      verifier.physicalEvidenceSnapshot(), kT1, kT2);
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
  verifier.setPhysicalEvidenceStore(store);
  verifier.setDsg(graph);
  require(graph->mesh()->numVertices()==1 && verifier.getStatistics().rays==0,
          "existing background vertex loses its ray to the inverted adjusted interval");
  spark_dsg::Mesh surface(false,true,false,true);
  surface.resizeVertices(1);surface.setPos(0,Point(0,0,2.574172f));
  const spark_dsg::BoundingBox box(Point::Ones(),Point::Zero());
  auto snapshot=verifier.physicalEvidenceSnapshot();
  auto old=verifier.countPhysicalSurface(49,surface,box,snapshot,first,last);
  auto measured=verifier.countProjectedPhysicalSurface(49,surface,box,snapshot,.05f,first,last);
  require(old.support_rays==0 && old.contradiction_rays==0,
          "original mesh-index query reproduces the missed old-cabinet evidence");
  require(measured.support_rays==0 && measured.contradiction_rays==1 && measured.free_space_votes==1,
          "measured fallback recovers the actual free-space ray without changing its timestamp");
  require(store->ingest(makeEndpointFrame(camera,observation,
      EndpointClass::kBackground,0,2.0f)), "replace witness with nearer occlusion");
  measured=verifier.countProjectedPhysicalSurface(49,surface,box,
      verifier.physicalEvidenceSnapshot(),.05f,first,last);
  require(measured.contradiction_rays==0 && measured.occluded_votes==1,
          "the same fallback never treats an occluder as an empty old site");
}

void testFrozenEvidenceSnapshot(const hydra::Sensor::ConstPtr& camera) {
  const Point endpoint(0.0f, 0.0f, 1.0f);
  auto store = std::make_shared<PhysicalEvidenceStore>();
  auto same = makeEndpointFrame(camera, kT1, EndpointClass::kPhysical, 7);
  require(store->ingest(same), "same-ID snapshot source is ingested");

  RayVerificator verifier(makeVerifierConfig());
  verifier.setPhysicalEvidenceStore(store);
  verifier.setDsg(makeRayGraph(kT1, endpoint));
  const auto frozen = verifier.physicalEvidenceSnapshot();

  auto other = makeEndpointFrame(camera, kT1, EndpointClass::kPhysical, 9);
  require(store->ingest(other), "live evidence is replaced with other ID");
  const auto frozen_result = verifier.checkPhysical(
      Point(0.0f, 0.0f, 1.0f), 7, frozen);
  const auto live_result =
      verifier.checkPhysical(Point(0.0f, 0.0f, 1.0f), 7);
  requireReasons(frozen_result, 1, 0, 0, "caller-frozen evidence snapshot");
  requireReasons(live_result, 0, 0, 1, "new live evidence snapshot");
  require(frozen_result.reasons.same_id == 1 &&
              live_result.reasons.different_id == 1,
          "one detector update cannot be split across store versions");
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

void testWholeObjectAbsenceNeedsSpatialCoverage(const hydra::Sensor::ConstPtr& camera) {
  RayVerificator verifier(makeVerifierConfig());
  auto store = std::make_shared<PhysicalEvidenceStore>();
  verifier.setPhysicalEvidenceStore(store);
  verifier.setDsg(makeRayGraph(kT2, Point(0,0,2)));
  spark_dsg::Mesh surface(false,true,false,true);
  surface.resizeVertices(8);
  for(int v=0;v<2;++v) for(int u=0;u<4;++u) surface.setPos(v*4+u,pointAtPixel(u,v));
  const spark_dsg::BoundingBox box(Point::Ones(),Point::Zero());
  // Many repeated views of ONE empty tip cannot become full spatial coverage.
  for (TimeStamp t=kT1; t<=kT2; t+=kSecond) {
    auto frame=makeEndpointFrame(camera,t,EndpointClass::kBackground,0,2);
    require(store->ingest(frame),"partial empty view stored");
  }
  auto counts=verifier.countCurrentPhysicalSurface(7,surface,box,
      verifier.physicalEvidenceSnapshot(),.05f,kT1-1,kT2);
  require(counts.contradiction_rays>0 && counts.contradicted_surface_samples==1,
          "raw empty-tip evidence is preserved and counted spatially once");
  require(!counts.absence_coverage_sufficient,
          "one of eight samples cannot close the whole object, even with repeated frames");
  auto empty=makeEndpointFrame(camera,kT2+kSecond,EndpointClass::kBackground,0,4);
  cv::Mat ranges = empty.input.range_image; ranges.setTo(4.f);
  require(store->ingest(empty),"actual broad empty site stored");
  counts=verifier.countCurrentPhysicalSurface(7,surface,box,
      verifier.physicalEvidenceSnapshot(),.05f,kT1-1,kT2+kSecond);
  require(counts.absence_coverage_sufficient && counts.contradicted_surface_samples==8,
          "a genuinely observed empty site still permits disappearance");
}

void testCurrentStateRejectsPreArrivalAbsence(const hydra::Sensor::ConstPtr& camera) {
  auto config = makeVerifierConfig();
  RayVerificator verifier(config);
  auto store = std::make_shared<PhysicalEvidenceStore>();
  verifier.setPhysicalEvidenceStore(store);
  verifier.setDsg(makeRayGraph(kT1, Point(0, 0, 2)));
  spark_dsg::Mesh surface(false, true, false, true);
  surface.resizeVertices(1); surface.setPos(0, Point(0, 0, 1));
  const spark_dsg::BoundingBox box(Point::Ones(), Point::Zero());
  for (TimeStamp t : {kT1, kT1 + kSecond, kT1 + 2*kSecond}) {
    require(store->ingest(makeEndpointFrame(camera, t, EndpointClass::kBackground, 0, 2)),
            "pre-arrival free space stored");
  }
  require(store->ingest(makeEndpointFrame(camera, kT2, EndpointClass::kPhysical, 49, 1)),
          "new position observed at arrival");
  const auto old = verifier.countProjectedPhysicalSurface(
      49, surface, box, verifier.physicalEvidenceSnapshot(), .05f, 0, kT2);
  require(old.contradiction_rays > old.support_rays,
          "unbounded history reproduces deletion of the newly arrived object");
  auto query = [&](TimeStamp end) {
    return verifier.countCurrentPhysicalSurface(
        49, surface, box, verifier.physicalEvidenceSnapshot(), .05f, kT2, end);
  };
  auto current = query(kT2 + kSecond);
  require(current.contradiction_rays == 0 && current.support_rays == 0,
          "no post-support observation means preserve the new state");
  require(query(kT2).contradiction_rays == 0,
          "equal support/check times have an empty evidence window");
  require(store->ingest(makeEndpointFrame(camera, kT2+kSecond,
                                         EndpointClass::kBackground, 0, .5f)),
          "subsequent occluder stored");
  require(query(kT2+kSecond).contradiction_rays == 0,
          "a later occluder cannot erase the state");
  require(store->ingest(makeEndpointFrame(camera, kT2+2*kSecond,
                                         EndpointClass::kBackground, 0, 2)),
          "actual later departure stored");
  require(query(kT2+kSecond).contradiction_rays == 0,
          "future departure cannot leak into an earlier decision");
  current = query(kT2+2*kSecond);
  require(current.contradiction_rays == 1 && current.support_rays == 0,
          "later measured free space still establishes disappearance");
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
    require(store->ingest(makeEndpointFrame(camera,missing?kT1:kT2,EndpointClass::kBackground,0,2)),"sensor frame stored");
    auto verifier=std::make_shared<RayVerificator>(makeVerifierConfig());
    verifier->setPhysicalEvidenceStore(store);verifier->setDsg(graph);
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

void testIndexedCandidateRequiresRealCoverage(const hydra::Sensor::ConstPtr& camera) {
  auto cfg = makeVerifierConfig();
  RayVerificator verifier(cfg);
  auto store = std::make_shared<PhysicalEvidenceStore>();
  verifier.setPhysicalEvidenceStore(store);
  require(store->ingest(makeEndpointFrame(camera, kT1, EndpointClass::kBackground, 0, 2)),
          "unrelated visible background observation stored");
  spark_dsg::Mesh surface(false,true,false,true);
  surface.resizeVertices(1); surface.setPos(0,Point(3,0,1));
  const spark_dsg::BoundingBox box(Point::Ones(),Point::Zero());
  verifier.setDsg(makeRayGraph(kT1, Point(6,0,2)));
  require(verifier.getStatistics().rays > 0, "out-of-view mesh ray candidate exists");
  auto counts = verifier.countPhysicalSurface(
      49,surface,box,verifier.physicalEvidenceSnapshot(),kT1,kT1);
  require(counts.contradiction_rays == 0 && counts.support_rays == 0,
          "a candidate ray outside actual camera coverage cannot establish absence");
  const auto replacement=verifier.checkPhysicalReplacement(
      Point(3,0,1),49,verifier.physicalEvidenceSnapshot(),kT1,kT1);
  require(replacement.absent.empty(),
          "closed-object background cleanup also requires actual camera coverage");
  surface.setPos(0,Point(0,0,1));
  verifier.setDsg(makeRayGraph(kT2, Point(0,0,2)));
  counts = verifier.countPhysicalSurface(
      49,surface,box,verifier.physicalEvidenceSnapshot(),kT2,kT2);
  require(counts.contradiction_rays == 0,
          "a mesh ray with no corresponding measured frame is not free-space evidence");
  require(store->ingest(makeEndpointFrame(camera,kT2,EndpointClass::kPhysical,49,2)),
          "same identity observed farther down the ray");
  counts=verifier.countPhysicalSurface(
      49,surface,box,verifier.physicalEvidenceSnapshot(),kT2,kT2);
  require(counts.support_rays == 0 && counts.contradiction_rays == 1,
          "same identity at a farther depth confirms vacated old position");
}

void testSparseAbsenceRequiresMeasuredConfirmation(const hydra::Sensor::ConstPtr& camera) {
  auto cfg=makeVerifierConfig();
  RayVerificator verifier(cfg);
  auto store=std::make_shared<PhysicalEvidenceStore>();
  verifier.setPhysicalEvidenceStore(store);
  verifier.setDsg(makeRayGraph(kT1,Point(0,0,2)));
  spark_dsg::Mesh surface(false,true,false,true);
  surface.resizeVertices(1);surface.setPos(0,Point(0,0,1));
  const spark_dsg::BoundingBox box(Point::Ones(),Point::Zero());
  require(store->ingest(makeEndpointFrame(camera,kT1,EndpointClass::kBackground,0,2)),
          "one sparse mesh-indexed absence frame stored");
  require(store->ingest(makeEndpointFrame(camera,kT2,EndpointClass::kPhysical,49,1)),
          "first actual support frame stored");
  require(store->ingest(makeEndpointFrame(camera,kT2+kSecond,EndpointClass::kPhysical,49,1)),
          "second actual support frame stored");
  const auto snap=verifier.physicalEvidenceSnapshot();
  const auto sparse=verifier.countPhysicalSurface(49,surface,box,snap,0,kT2+kSecond);
  require(sparse.contradiction_rays == 1 && sparse.support_rays == 0,
          "sparse geometry index alone reproduces the wrong absence majority");
  bool projected=false;
  const auto measured=verifier.countCurrentPhysicalSurface(
      49,surface,box,snap,.05f,0,kT2+kSecond,&projected);
  require(projected && measured.support_rays == 2 && measured.contradiction_rays == 1,
          "actual sensor support prevents a deletion proposed by the sparse proxy");
  const auto absent=verifier.countCurrentPhysicalSurface(
      49,surface,box,snap,.05f,0,kT1,&projected);
  require(projected && absent.support_rays == 0 && absent.contradiction_rays == 1,
          "the same guard still accepts true measured absence in its own time window");
}

void testNearBackgroundIsOcclusion(const hydra::Sensor::ConstPtr& camera) {
  auto cfg=makeVerifierConfig();cfg.depth_tolerance=.3f;
  RayVerificator verifier(cfg);
  auto store=std::make_shared<PhysicalEvidenceStore>();verifier.setPhysicalEvidenceStore(store);
  require(store->ingest(makeEndpointFrame(camera,kT1,EndpointClass::kBackground,0,5.279f)),
          "real I49 nearer background depth stored");
  verifier.setDsg(makeRayGraph(kT1,Point(0,0,6.0f)));
  spark_dsg::Mesh surface(false,true,false,true);
  surface.resizeVertices(1);surface.setPos(0,Point(0,0,5.5785384f));
  const spark_dsg::BoundingBox box(Point::Ones(),Point::Zero());
  auto snapshot=verifier.physicalEvidenceSnapshot();
  auto indexed=verifier.countPhysicalSurface(49,surface,box,snapshot,0,kT1);
  auto projected=verifier.countProjectedPhysicalSurface(49,surface,box,snapshot,.05f,0,kT1);
  require(indexed.contradiction_rays==0 && indexed.occluded_votes>0,
          "nearer background within matching tolerance is an occluder in the ray index");
  require(projected.contradiction_rays==0 && projected.occluded_votes>0,
          "direct RGB-D projection also preserves the occluded cabinet");
  auto replacement=verifier.checkPhysicalReplacement(Point(0,0,5.5785384f),49,snapshot,0,kT1);
  require(replacement.absent.empty() && !replacement.inconclusive.empty(),
          "occluding background cannot authorize deleting a closed object's background copy");
  require(store->ingest(makeEndpointFrame(camera,kT1,EndpointClass::kPhysical,68,5.279f)),
          "different physical object placed in front");
  projected=verifier.countCurrentPhysicalSurface(
      49,surface,box,verifier.physicalEvidenceSnapshot(),.05f,0,kT1);
  require(projected.contradiction_rays==0,
          "a different nearer object is occlusion, not proof that the hidden old object left");
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
  testCurrentStateRejectsPreArrivalAbsence(camera);
  testWholeObjectAbsenceNeedsSpatialCoverage(camera);
  testIndexedCandidateRequiresRealCoverage(camera);
  testPhysicalReducerRequiresActualCoverage(camera);
  testSparseAbsenceRequiresMeasuredConfirmation(camera);
  testNearBackgroundIsOcclusion(camera);
  testFrozenEvidenceSnapshot(camera);
  testPhysicalConfidenceBoundary();
  testBackwardWindowUsesPastBins();

  hydra::GlobalInfo::reset();
  return EXIT_SUCCESS;
}
