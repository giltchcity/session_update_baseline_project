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
#include "session_core/evidence/physical_evidence_store.h"
#include "session_core/state/persistent_object_state.h"
#include "session_core/surface/closed_object_background.h"
#include "khronos/utils/khronos_attribute_utils.h"
#include <session_core/evidence/surface_evidence.h>
#include <session_core/objects/physical_presence.h>

namespace {

using khronos::EndpointClass;
using khronos::EndpointEvidence;
using khronos::FrameData;
using khronos::PhysicalEvidenceStore;
using khronos::Point;
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

khronos::session::SurfaceEvidence::Config makeEvidenceConfig() {
  khronos::session::SurfaceEvidence::Config config;
  config.depth_tolerance = 0.05f;
  return config;
}

khronos::PhysicalEvidenceSnapshot snap(const std::shared_ptr<PhysicalEvidenceStore>& store) {
  return store->snapshot();
}




void testClosedObjectBackground(const hydra::Sensor::ConstPtr&) {
  const auto run = [](bool close_old_state, float duplicate_offset,
                      TimeStamp duplicate_stamp, float association_offset,
                      float map_resolution, size_t expected_removed,
                      const char* label) {
    auto graph = makeRayGraph(kT2, Point(0.f, 0.f, 1.f));
    auto& background = *graph->mesh();
    const auto append = [&](const Point& point, TimeStamp stamp) {
      const size_t i = background.numVertices();
      background.resizeVertices(i + 1);
      background.setPos(i, point);
      background.setFirstSeenTimestamp(i, stamp);
      background.setTimestamp(i, stamp);
    };
    append(Point(0.f, 0.f, 1.f + duplicate_offset), duplicate_stamp);
    append(Point(0.f, 0.f, 1.055f), kT1);   // wall 5.5 cm behind the surface
    append(Point(0.f, 0.f, 1.005f), kT2);   // replacement measured after the move
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
    const auto closed_surfaces = khronos::closedStateSurfaces(objects, kT2);
    const auto owned = khronos::closedStateBackground(background, closed_surfaces, map_resolution);
    const auto owns = [&](size_t i) { return i < owned.size() && owned[i] != 0; };
    size_t removed = 0;
    for (size_t i = 0; i < owned.size(); ++i) removed += owns(i);
    require(removed == expected_removed,
            std::string(label) + ": owned=" + std::to_string(removed));
    require(owns(1) == (expected_removed >= 1),
            std::string(label) + ": the state's coincident background duplicate follows closure");
    require(!owns(0) && !owns(2) && !owns(3),
            std::string(label) + ": measured endpoint, wall behind and later replacement stay");
  };
  run(true, 0.f, kT1, 0.f, 0.05f, 1, "closed duplicate is removed atomically");
  run(false, 0.f, kT1, 0.f, 0.05f, 0, "open state owns nothing");
  run(true, 0.f, kT2, 0.f, 0.05f, 0, "surface measured after the last support is not owned");
  run(true, 0.06f, kT1, 0.f, 0.05f, 0, "duplicate outside one cell is not owned");
  run(true, 0.f, kT1, 0.06f, 0.05f, 0, "object shifted away from the duplicate");
}


void testProjectedCoverageWithoutMeshRays(const hydra::Sensor::ConstPtr& camera) {
  auto graph = std::make_shared<spark_dsg::DynamicSceneGraph>();
  graph->setMesh(std::make_shared<spark_dsg::Mesh>(false, true, false, true));
  addPose(*graph, kT2);
  auto store = std::make_shared<PhysicalEvidenceStore>();
  auto frame = makeEndpointFrame(camera, kT2, EndpointClass::kBackground, 0, 1.055f);
  require(store->ingest(frame), "real RGB-D wall observation is captured without any mesh ray");
  khronos::session::SurfaceEvidence evidence(makeEvidenceConfig());
  const auto snapshot = snap(store);
  // The presence start of a one-point surface first observed after kT2, one look ending at kT2:
  // the site seen through at kT2 (the wall 5.5 cm behind it) is an empty site.
  spark_dsg::Mesh site(false, true, false, true);
  site.resizeVertices(1);
  site.setPos(0, Point(0,0,1));
  spark_dsg::Mesh outside(false, true, false, true);
  outside.resizeVertices(1);
  outside.setPos(0, Point(100,0,1));
  const spark_dsg::BoundingBox at_origin(Point::Ones(), Point::Zero());
  // A fresh evidence instance per question: an identity's presence record keeps the looks it took.
  const auto empty_before = [&](const spark_dsg::Mesh& m, const std::vector<uint64_t>& looks, uint64_t first_seen) {
    const khronos::session::SurfaceEvidence fresh(makeEvidenceConfig());
    return fresh.presenceStart(7, m, at_origin, snap(store), looks, first_seen, first_seen).absent;
  };
  require(empty_before(site, {kT2}, kT2 + 1), "actual observed old site is empty even without a mesh endpoint");
  require(!empty_before(site, {kT2 - 1}, kT2), "future frames cannot explain an earlier snapshot");
  require(!empty_before(outside, {kT2}, kT2 + 1), "outside-FOV surfaces remain unobserved");

  spark_dsg::Mesh surface(false, true, false, true);
  surface.resizeVertices(3);
  surface.setPos(0, Point(0,0,1));
  surface.setPos(1, Point(0,0,1));
  surface.setPos(2, Point(.001f,0,1));
  const spark_dsg::BoundingBox box(Point::Ones(), Point::Zero());
  auto counts = evidence.countProjected(7, surface, box, snapshot, .05f, kT1, kT2);
  require(counts.surface_samples == 1 && counts.contradiction_rays == 1,
          "duplicate surfaces and a repeated image pixel are counted once");

  // Each frame below replaces the one at kT2. A reading that stops in front of the point, or no
  // reading within range, did not observe it; a reading on its surface, whatever the label, finds
  // the site taken; the same identity seen farther along the ray sees through it.
  require(store->ingest(makeEndpointFrame(camera, kT2, EndpointClass::kBackground, 0, .5f)),
          "occluding frame replaces the exact timestamp");
  require(!empty_before(site, {kT2}, kT2 + 1), "a reading in front of the point did not observe it");
  require(store->ingest(makeEndpointFrame(camera, kT2, EndpointClass::kInvalid)),
          "a frame without a reading at the pixel replaces the exact timestamp");
  require(!empty_before(site, {kT2}, kT2 + 1), "a pixel without a reading within range did not observe it");
  require(store->ingest(makeEndpointFrame(camera, kT2, EndpointClass::kUnidentifiedObject, 0, 1.f)),
          "an unidentified object on the surface replaces the exact timestamp");
  require(!empty_before(site, {kT2}, kT2 + 1), "a reading on the surface finds the site taken, whatever its label");
  require(store->ingest(makeEndpointFrame(camera, kT2, EndpointClass::kPhysical, 7, 2.f)),
          "same identity is seen farther along the ray");
  require(empty_before(site, {kT2}, kT2 + 1), "same identity farther away sees through the site");
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
  auto config=makeEvidenceConfig();
  config.depth_tolerance=.3f;
  auto store=std::make_shared<PhysicalEvidenceStore>();
  require(store->ingest(makeEndpointFrame(camera,observation,
      EndpointClass::kBackground,0,2.876f)), "real short-lived wall observation stored");
  khronos::session::SurfaceEvidence evidence(config);
  spark_dsg::Mesh surface(false,true,false,true);
  surface.resizeVertices(1);surface.setPos(0,Point(0,0,2.574172f));
  const spark_dsg::BoundingBox box(Point::Ones(),Point::Zero());
  auto snapshot=snap(store);
  auto measured=evidence.countProjected(49,surface,box,snapshot,.05f,first,last);
  require(measured.support_rays==0 && measured.contradiction_rays==1 && measured.free_space_votes==1,
          "measured fallback recovers the actual free-space ray without changing its timestamp");
  require(store->ingest(makeEndpointFrame(camera,observation,
      EndpointClass::kBackground,0,2.0f)), "replace witness with nearer occlusion");
  measured=evidence.countProjected(49,surface,box,
      snap(store),.05f,first,last);
  require(measured.contradiction_rays==0 && measured.occluded_votes==1,
          "the same fallback never treats an occluder as an empty old site");
}




void testWholeObjectAbsenceNeedsSpatialCoverage(const hydra::Sensor::ConstPtr& camera) {
  khronos::session::SurfaceEvidence evidence(makeEvidenceConfig());
  auto store = std::make_shared<PhysicalEvidenceStore>();
  spark_dsg::Mesh surface(false,true,false,true);
  surface.resizeVertices(8);
  for(int v=0;v<2;++v) for(int u=0;u<4;++u) surface.setPos(v*4+u,pointAtPixel(u,v));
  const spark_dsg::BoundingBox box(Point::Ones(),Point::Zero());
  // Many repeated views of ONE empty tip cannot become full spatial coverage.
  for (TimeStamp t=kT1; t<=kT2; t+=kSecond) {
    auto frame=makeEndpointFrame(camera,t,EndpointClass::kBackground,0,2);
    require(store->ingest(frame),"partial empty view stored");
  }
  auto counts=evidence.countCurrent(7,surface,box,
      snap(store),.05f,kT1-1,kT2);
  require(counts.contradiction_rays>0 && counts.contradicted_surface_samples==1,
          "raw empty-tip evidence is preserved and counted spatially once");
  require(!counts.absence_coverage_sufficient,
          "one of eight samples cannot close the whole object, even with repeated frames");
  auto empty=makeEndpointFrame(camera,kT2+kSecond,EndpointClass::kBackground,0,4);
  cv::Mat ranges = empty.input.range_image; ranges.setTo(4.f);
  require(store->ingest(empty),"actual broad empty site stored");
  counts=evidence.countCurrent(7,surface,box,
      snap(store),.05f,kT1-1,kT2+kSecond);
  require(counts.absence_coverage_sufficient && counts.contradicted_surface_samples==8,
          "a genuinely observed empty site still permits disappearance");
}

void testCurrentStateRejectsPreArrivalAbsence(const hydra::Sensor::ConstPtr& camera) {
  auto config = makeEvidenceConfig();
  khronos::session::SurfaceEvidence evidence(config);
  auto store = std::make_shared<PhysicalEvidenceStore>();
  spark_dsg::Mesh surface(false, true, false, true);
  surface.resizeVertices(1); surface.setPos(0, Point(0, 0, 1));
  const spark_dsg::BoundingBox box(Point::Ones(), Point::Zero());
  for (TimeStamp t : {kT1, kT1 + kSecond, kT1 + 2*kSecond}) {
    require(store->ingest(makeEndpointFrame(camera, t, EndpointClass::kBackground, 0, 2)),
            "pre-arrival free space stored");
  }
  require(store->ingest(makeEndpointFrame(camera, kT2, EndpointClass::kPhysical, 49, 1)),
          "new position observed at arrival");
  const auto old = evidence.countProjected(
      49, surface, box, snap(store), .05f, 0, kT2);
  require(old.contradiction_rays > old.support_rays,
          "unbounded history reproduces deletion of the newly arrived object");
  auto query = [&](TimeStamp end) {
    return evidence.countCurrent(
        49, surface, box, snap(store), .05f, kT2, end);
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
  // The object is first seen at kT2. Mode 2: a frame at kT1 sees through its site, so its presence
  // starts halfway between; mode 0: that site is outside the frame; mode 1: the only frame is the
  // observation's own time, no earlier evidence. Its presence is never ended here: the state layer
  // ends an identity's presence when its last state closes.
  for (int mode=0; mode<3; ++mode) {
    const bool outside=mode==0, missing=mode==1;
    auto graph=makeRayGraph(kT2, outside?Point(6,0,2):Point(0,0,2));
    auto object=std::make_unique<khronos::KhronosObjectAttributes>();
    object->mesh=spark_dsg::Mesh(false,true,false,true);
    object->mesh.resizeVertices(1);
    object->mesh.setPos(0,outside?Point(3,0,1):Point(0,0,1));
    object->mesh.setTimestamp(0,kT2);object->mesh.setFirstSeenTimestamp(0,kT2);
    object->bounding_box=spark_dsg::BoundingBox(Point::Ones(),Point::Zero());
    object->first_observed_ns={kT2};object->last_observed_ns={kT2};
    khronos::setObservationBounds(*object,kT2,kT2);
    object->details["instance_id"]={49};
    require(graph->emplaceNode(spark_dsg::DsgLayers::OBJECTS,spark_dsg::NodeSymbol('O',0),std::move(object)),"physical presence fixture inserted");
    auto store=std::make_shared<PhysicalEvidenceStore>();
    require(store->ingest(makeEndpointFrame(camera,missing?kT2:kT1,EndpointClass::kBackground,0,2)),"sensor frame stored");
    khronos::session::SurfaceEvidence evidence(makeEvidenceConfig());
    auto attrs=graph->getNode(spark_dsg::NodeSymbol('O',0)).attributes<khronos::KhronosObjectAttributes>().clone();
    auto& object_attrs=static_cast<khronos::KhronosObjectAttributes&>(*attrs);
    // One look ends at kT1, the next at the observation kT2 (the rounds of the session).
    const auto first_seen=khronos::session::openPresence(object_attrs);
    const auto dated=khronos::session::presenceStart(evidence,store->snapshot(),{kT1,kT2},49,object_attrs,first_seen,kT2);
    if (dated) object_attrs.first_observed_ns={*dated};
    const auto start=object_attrs.first_observed_ns.front();
    require(mode==2 ? start==kT1+(kT2-kT1)/2 : start==0,
      "the presence start must reject unavailable pixels and accept actual earlier free space; mode="+std::to_string(mode));
    require(object_attrs.last_observed_ns==std::vector<uint64_t>{std::numeric_limits<uint64_t>::max()},
      "no evidence ends a physical object's presence on its own; mode="+std::to_string(mode));
  }
}


void testSparseAbsenceRequiresMeasuredConfirmation(const hydra::Sensor::ConstPtr& camera) {
  auto cfg=makeEvidenceConfig();
  khronos::session::SurfaceEvidence evidence(cfg);
  auto store=std::make_shared<PhysicalEvidenceStore>();
  spark_dsg::Mesh surface(false,true,false,true);
  surface.resizeVertices(1);surface.setPos(0,Point(0,0,1));
  const spark_dsg::BoundingBox box(Point::Ones(),Point::Zero());
  require(store->ingest(makeEndpointFrame(camera,kT1,EndpointClass::kBackground,0,2)),
          "one sparse mesh-indexed absence frame stored");
  require(store->ingest(makeEndpointFrame(camera,kT2,EndpointClass::kPhysical,49,1)),
          "first actual support frame stored");
  require(store->ingest(makeEndpointFrame(camera,kT2+kSecond,EndpointClass::kPhysical,49,1)),
          "second actual support frame stored");
  const auto stored=snap(store);
  bool projected=false;
  const auto measured=evidence.countCurrent(
      49,surface,box,stored,.05f,0,kT2+kSecond,&projected);
  require(projected && measured.support_rays == 2 && measured.contradiction_rays == 1,
          "actual sensor support prevents a deletion a single empty frame would propose");
  const auto absent=evidence.countCurrent(
      49,surface,box,stored,.05f,0,kT1,&projected);
  require(projected && absent.support_rays == 0 && absent.contradiction_rays == 1,
          "the same guard still accepts true measured absence in its own time window");
}

void testNearBackgroundIsOcclusion(const hydra::Sensor::ConstPtr& camera) {
  auto cfg=makeEvidenceConfig();cfg.depth_tolerance=.3f;
  khronos::session::SurfaceEvidence evidence(cfg);
  auto store=std::make_shared<PhysicalEvidenceStore>();
  require(store->ingest(makeEndpointFrame(camera,kT1,EndpointClass::kBackground,0,5.279f)),
          "real I49 nearer background depth stored");
  spark_dsg::Mesh surface(false,true,false,true);
  surface.resizeVertices(1);surface.setPos(0,Point(0,0,5.5785384f));
  const spark_dsg::BoundingBox box(Point::Ones(),Point::Zero());
  auto snapshot=snap(store);
  auto projected=evidence.countProjected(49,surface,box,snapshot,.05f,0,kT1);
  require(projected.contradiction_rays==0 && projected.occluded_votes>0,
          "direct RGB-D projection also preserves the occluded cabinet");
  require(store->ingest(makeEndpointFrame(camera,kT1,EndpointClass::kPhysical,68,5.279f)),
          "different physical object placed in front");
  projected=evidence.countCurrent(
      49,surface,box,snap(store),.05f,0,kT1);
  require(projected.contradiction_rays==0,
          "a different nearer object is occlusion, not proof that the hidden old object left");
}



// A ray through a background point is absence evidence only if it faces the surface there
// (P23, the incidence rule of the surface evidence); without a normal
// every ray counts.

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
  testClosedObjectBackground(camera);
  testProjectedCoverageWithoutMeshRays(camera);
  testMeasuredShortVertexInterval(camera);
  testCurrentStateRejectsPreArrivalAbsence(camera);
  testWholeObjectAbsenceNeedsSpatialCoverage(camera);
  testPhysicalReducerRequiresActualCoverage(camera);
  testSparseAbsenceRequiresMeasuredConfirmation(camera);
  testNearBackgroundIsOcclusion(camera);

  hydra::GlobalInfo::reset();
  return EXIT_SUCCESS;
}
