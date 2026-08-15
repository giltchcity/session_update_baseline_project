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
                            int physical_id = 0) {
  hydra::InputData input(sensor);
  input.timestamp_ns = stamp;
  input.world_T_body = Eigen::Isometry3d::Identity();
  input.range_image = cv::Mat(2, 4, CV_32FC1, cv::Scalar(1.0f));
  input.depth_image = cv::Mat(2, 4, CV_32FC1, cv::Scalar(1.0f));
  input.label_image = cv::Mat(2, 4, CV_32SC1, cv::Scalar(1));
  input.color_image = cv::Mat(2, 4, CV_8UC3, cv::Scalar(0, 0, 0));
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

void addPose(spark_dsg::DynamicSceneGraph& graph, TimeStamp stamp) {
  const hydra::RobotPrefixConfig prefix(0);
  const auto layer_key = graph.getLayerKey(spark_dsg::DsgLayers::AGENTS);
  require(layer_key.has_value(), "default graph has an agent layer key");
  if (!graph.findLayer(layer_key->layer, prefix.key)) {
    graph.addLayer(layer_key->layer, prefix.key);
  }
  const auto id = spark_dsg::NodeSymbol(prefix.key, 0);
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
  testFrozenEvidenceSnapshot(camera);
  testPhysicalConfidenceBoundary();
  testBackwardWindowUsesPastBins();

  hydra::GlobalInfo::reset();
  return EXIT_SUCCESS;
}
