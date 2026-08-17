#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <hydra/input/input_data.h>
#include <hydra/input/camera.h>
#include <hydra/input/sensor_extrinsics.h>
#include <hydra/reconstruction/projection_interpolators.h>
#include <opencv2/core.hpp>
#include <spark_dsg/dynamic_scene_graph.h>

#include "khronos/active_window/active_window.h"
#include "khronos/active_window/data/frame_data_buffer.h"
#include "khronos/active_window/object_detection/instance_forwarding.h"
#include "khronos/active_window/object_extraction/mesh_object_extractor.h"
#include "khronos/active_window/tracking/external_tracker.h"
#include "khronos/spatio_temporal_map/spatio_temporal_map.h"
#include "khronos/backend/reconciliation/persistent_object_state.h"
#include "khronos/backend/update_khronos_objects_functor.h"
#include "khronos/utils/geometry_utils.h"
#include "khronos/utils/khronos_attribute_utils.h"

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << "\n";
    std::exit(1);
  }
}

hydra::Sensor::ConstPtr makeCamera() {
  hydra::Camera::Config config;
  config.min_range = 0.1;
  config.max_range = 10.0;
  config.width = 3;
  config.height = 2;
  config.cx = 1.0f;
  config.cy = 1.0f;
  config.fx = 1.0f;
  config.fy = 1.0f;
  config.extrinsics = hydra::ParamSensorExtrinsics::Config();
  return std::make_shared<hydra::Camera>(config, "test_camera");
}

hydra::Sensor::ConstPtr makeDenseCamera() {
  hydra::Camera::Config config;
  config.min_range = 0.1;
  config.max_range = 10.0;
  config.width = 48;
  config.height = 36;
  config.cx = 24.0f;
  config.cy = 18.0f;
  config.fx = 40.0f;
  config.fy = 40.0f;
  config.extrinsics = hydra::ParamSensorExtrinsics::Config();
  return std::make_shared<hydra::Camera>(config, "dense_test_camera");
}

hydra::InputData makeInput(std::uint64_t stamp) {
  static const auto camera = makeCamera();
  hydra::InputData input(camera);
  input.timestamp_ns = stamp;
  input.world_T_body = Eigen::Isometry3d::Identity();
  input.label_image = cv::Mat::zeros(2, 3, CV_32SC1);
  input.vertex_map = cv::Mat(2, 3, CV_32FC3);
  input.depth_image = cv::Mat::ones(2, 3, CV_32FC1);
  input.range_image = cv::Mat::ones(2, 3, CV_32FC1);
  input.color_image = cv::Mat(2, 3, CV_8UC3, cv::Scalar(128, 128, 128));
  for (int v = 0; v < input.vertex_map.rows; ++v) {
    for (int u = 0; u < input.vertex_map.cols; ++u) {
      input.vertex_map.at<cv::Vec3f>(v, u) =
          cv::Vec3f(static_cast<float>(u), static_cast<float>(v), 1.0f);
    }
  }
  return input;
}

hydra::InputData makeInvalidDepthInstanceInput(std::uint64_t stamp,
                                               bool include_valid_patch) {
  static const auto camera = makeDenseCamera();
  hydra::InputData input(camera);
  input.timestamp_ns = stamp;
  input.world_T_body = Eigen::Isometry3d::Identity();
  input.depth_image = cv::Mat(36, 48, CV_32FC1);
  input.color_image = cv::Mat(36, 48, CV_8UC3, cv::Scalar(128, 128, 128));
  input.label_image = cv::Mat(36, 48, CV_32SC1, cv::Scalar(999));

  // Every pixel carries I6 in the instance raster, but these five depth cases
  // are all unusable for object geometry.
  for (int v = 0; v < input.depth_image.rows; ++v) {
    for (int u = 0; u < input.depth_image.cols; ++u) {
      switch (v % 5) {
        case 0:
          input.depth_image.at<float>(v, u) = 0.0f;
          break;
        case 1:
          input.depth_image.at<float>(v, u) =
              std::numeric_limits<float>::quiet_NaN();
          break;
        case 2:
          input.depth_image.at<float>(v, u) =
              std::numeric_limits<float>::infinity();
          break;
        case 3:
          input.depth_image.at<float>(v, u) = 0.05f;
          break;
        default:
          input.depth_image.at<float>(v, u) = 6.0f;
          break;
      }
    }
  }

  if (include_valid_patch) {
    for (int v = 13; v < 23; ++v) {
      for (int u = 18; u < 30; ++u) {
        input.depth_image.at<float>(v, u) =
            1.0f + 0.01f * static_cast<float>(u - 18) +
            0.005f * static_cast<float>(v - 13);
        input.label_image.at<std::int32_t>(v, u) = 35;
      }
    }
  }

  require(input.getSensor().finalizeRepresentations(input),
          "invalid-depth fixture representations are available");
  return input;
}

khronos::MeasurementCluster makeCluster(
    int id,
    std::initializer_list<khronos::Pixel> pixels,
    int semantic_id = -1) {
  khronos::MeasurementCluster cluster;
  cluster.id = id;
  cluster.pixels.assign(pixels.begin(), pixels.end());
  if (semantic_id >= 0) {
    cluster.semantics = khronos::SemanticClusterInfo(semantic_id);
  }
  return cluster;
}

void testExplicitLabelProtocol() {
  cv::Mat semantic = (cv::Mat_<std::int32_t>(1, 3) << 75, 7, 131);
  cv::Mat untouched = semantic.clone();
  cv::Mat instances;
  require(khronos::decodeSemanticInstanceLabels(semantic, instances, false),
          "semantic-only decode succeeds");
  require(instances.empty(), "semantic-only input creates no instance channel");
  require(cv::countNonZero(semantic != untouched) == 0,
          "semantic-only CV_32SC1 values are not mistaken for packed labels");

  cv::Mat packed(1, 3, CV_32SC1);
  packed.at<std::int32_t>(0, 0) = (75 << 16) | 10;
  packed.at<std::int32_t>(0, 1) = (7 << 16) | 1;
  packed.at<std::int32_t>(0, 2) = (131 << 16) | 18;
  require(khronos::decodeSemanticInstanceLabels(packed, instances, true),
          "explicit packed decode succeeds");
  require(packed.at<std::int32_t>(0, 0) == 75 &&
              packed.at<std::int32_t>(0, 1) == 7 &&
              packed.at<std::int32_t>(0, 2) == 131,
          "packed semantic IDs decode correctly");
  require(instances.at<std::int32_t>(0, 0) == 10 &&
              instances.at<std::int32_t>(0, 1) == 1 &&
              instances.at<std::int32_t>(0, 2) == 18,
          "packed physical IDs decode correctly");

  auto input = makeInput(10'000'000'000ULL);
  input.label_image.at<std::int32_t>(0, 0) = 75;
  input.label_image.at<std::int32_t>(0, 1) = 75;
  khronos::FrameData data(input);
  data.instance_image = cv::Mat::zeros(2, 3, CV_32SC1);
  data.instance_image.at<std::int32_t>(0, 0) = 10;
  data.instance_image.at<std::int32_t>(0, 1) = 10;
  khronos::InstanceForwarding::Config forwarding_config;
  forwarding_config.min_cluster_size = 1;
  khronos::InstanceForwarding forwarding(forwarding_config);
  forwarding.extractSemanticClusters(data);
  require(data.semantic_clusters.size() == 1,
          "one physical instance produces one semantic cluster");
  require(data.semantic_clusters.front().id == 10,
          "InstanceForwarding retains the physical ID");
  require(data.semantic_clusters.front().semantics &&
              data.semantic_clusters.front().semantics->category_id == 75,
          "InstanceForwarding retains the independent semantic ID");
}

void testInvalidDepthPixelsCannotInflatePhysicalObject() {
  constexpr std::uint64_t kStamp = 20'000'000'000ULL;
  constexpr int kValidPixels = 12 * 10;

  auto input = makeInvalidDepthInstanceInput(kStamp, true);
  auto frame = std::make_shared<khronos::FrameData>(input);
  frame->instance_image = cv::Mat(36, 48, CV_32SC1, cv::Scalar(6));

  khronos::InstanceForwarding::Config forwarding_config;
  forwarding_config.max_range = 5.0f;
  forwarding_config.min_cluster_size = 50;
  forwarding_config.max_object_volume = 10.0;
  khronos::InstanceForwarding forwarding(forwarding_config);
  forwarding.extractSemanticClusters(*frame);

  require(frame->semantic_clusters.size() == 1,
          "valid I6 depth patch survives invalid-depth filtering");
  require(frame->semantic_clusters.front().id == 6,
          "filtered cluster retains physical identity I6");
  require(frame->semantic_clusters.front().pixels.size() == kValidPixels,
          "zero/NaN/Inf/near/far I6 pixels never enter the cluster");
  require(frame->semantic_clusters.front().semantics &&
              frame->semantic_clusters.front().semantics->category_id == 35,
          "invalid pixels cannot outvote the valid I6 semantic class");
  require(cv::countNonZero(frame->object_image) == kValidPixels,
          "object raster exposes only geometrically valid I6 pixels");

  khronos::ExternalTracker::Config tracker_config;
  tracker_config.min_num_observations = 1;
  khronos::ExternalTracker tracker(tracker_config);
  tracker.processInput(*frame);
  require(tracker.getTracks().size() == 1,
          "filtered I6 cluster creates exactly one physical track");
  const auto& track = tracker.getTracks().front();
  require(track.physical_instance_id && *track.physical_instance_id == 6,
          "I6 identity survives forwarding and tracking");
  require(std::isfinite(track.last_bounding_box.volume()) &&
              track.last_bounding_box.volume() > 0.0f &&
              track.last_bounding_box.volume() < 1.0f,
          "I6 bounding box is finite and follows only the compact valid patch");

  khronos::FrameDataBuffer::Config buffer_config;
  buffer_config.max_buffer_size = 2;
  khronos::FrameDataBuffer buffer(buffer_config);
  buffer.storeData(frame);

  khronos::MeshObjectExtractor::Config extractor_config;
  extractor_config.verbosity = 0;
  extractor_config.min_object_allocation_confidence = 0.0f;
  extractor_config.min_object_volume = 0.0f;
  extractor_config.max_object_volume = 10.0f;
  extractor_config.object_reconstruction_resolution = 0.05f;
  extractor_config.only_extract_reconstructed_objects = false;
  extractor_config.min_object_reconstruction_confidence = 0.0f;
  extractor_config.min_object_reconstruction_observations = 0;
  extractor_config.projective_integrator.num_threads = 1;
  extractor_config.projective_integrator.interpolation_method =
      hydra::InterpolatorNearest::Config();
  extractor_config.mesh_integrator.integrator_threads = 1;
  khronos::MeshObjectExtractor extractor(extractor_config);
  auto object = extractor.extractObject(track, buffer);
  require(object != nullptr,
          "terminal extraction retains I6 after invalid-depth filtering");
  require(object->details.at("instance_id").front() == 6 &&
              object->semantic_label == 35,
          "extracted object retains physical I6 and semantic S35");
  require(std::isfinite(object->bounding_box.volume()) &&
              object->bounding_box.volume() < extractor_config.max_object_volume,
          "extracted I6 geometry stays below the unchanged production volume limit");

  auto invalid_input = makeInvalidDepthInstanceInput(kStamp + 1, false);
  khronos::FrameData invalid_frame(invalid_input);
  invalid_frame.instance_image = cv::Mat(36, 48, CV_32SC1, cv::Scalar(6));
  forwarding.extractSemanticClusters(invalid_frame);
  require(invalid_frame.semantic_clusters.empty(),
          "an I6 mask with no valid depth produces no cluster");
  require(cv::countNonZero(invalid_frame.object_image) == 0,
          "an all-invalid I6 mask never reaches the object raster");
}

void testUnifiedExternalTracker() {
  khronos::ExternalTracker::Config config;
  config.min_num_observations = 1;
  config.min_cross_iou = 0.1f;
  config.max_dynamic_distance = 1.5f;
  config.settle_time = 0.5f;
  khronos::ExternalTracker tracker(config);

  {
    auto input = makeInput(10'000'000'000ULL);
    khronos::FrameData data(input);
    data.semantic_clusters.push_back(
        makeCluster(10, {{0, 0}, {1, 0}}, 75));
    data.dynamic_clusters.push_back(
        makeCluster(1, {{0, 0}, {1, 0}}));
    tracker.processInput(data);
  }

  require(tracker.getTracks().size() == 1,
          "overlapping motion does not duplicate a physical object (tracks=" +
              std::to_string(tracker.getTracks().size()) + ")");
  const auto& chair = tracker.getTracks().front();
  require(chair.id == 10, "physical track keeps external ID I10");
  require(chair.physical_instance_id && *chair.physical_instance_id == 10,
          "physical track explicitly records I10 as persistent identity");
  require(chair.is_dynamic, "physical track is marked dynamic when motion is observed");
  require(chair.semantics && chair.semantics->category_id == 75,
          "dynamic physical track keeps its semantic class");
  require(chair.observations.back().semantic_cluster_id == 10 &&
              chair.observations.back().dynamic_cluster_id == 1,
          "one observation records both physical and motion evidence");

  {
    auto input = makeInput(11'000'000'000ULL);
    khronos::FrameData data(input);
    data.semantic_clusters.push_back(
        makeCluster(10, {{0, 0}, {1, 0}}, 75));
    tracker.processInput(data);
  }
  require(tracker.getTracks().size() == 1,
          "the same physical ID remains one track on the next frame");
  require(tracker.getTracks().front().observations.size() == 2,
          "physical track receives the next observation");
  require(!tracker.getTracks().front().is_dynamic,
          "a moved physical object settles back into current static reconstruction");
  require(tracker.getTracks().front().has_dynamic_history,
          "settling does not erase its D1 dynamic history");

  {
    auto input = makeInput(12'000'000'000ULL);
    khronos::FrameData data(input);
    data.semantic_clusters.push_back(makeCluster(7, {{0, 1}}, 74));
    data.dynamic_clusters.push_back(makeCluster(4, {{2, 1}}, 12));
    tracker.processInput(data);
  }
  require(tracker.getTracks().size() == 3,
          "static physical and dynamic-only observations are both retained");
  const auto static_it = std::find_if(
      tracker.getTracks().begin(), tracker.getTracks().end(), [](const auto& track) {
        return track.id == 7;
      });
  require(static_it != tracker.getTracks().end() && !static_it->is_dynamic,
          "a physical object without motion remains static");
  const auto dynamic_it = std::find_if(
      tracker.getTracks().begin(), tracker.getTracks().end(), [](const auto& track) {
        return track.id >= (1 << 16);
      });
  require(dynamic_it != tracker.getTracks().end() && dynamic_it->is_dynamic,
          "unmatched free-space/dynamic-semantic input creates a dynamic track");
  require(!dynamic_it->physical_instance_id,
          "runtime dynamic track is not mislabeled as a physical instance");
  require(dynamic_it->semantics && dynamic_it->semantics->category_id == 12,
          "dynamic-only track retains a supplied semantic class");
  const int dynamic_track_id = dynamic_it->id;

  {
    auto input = makeInput(13'000'000'000ULL);
    khronos::FrameData data(input);
    data.dynamic_clusters.push_back(makeCluster(9, {{2, 1}}, 12));
    tracker.processInput(data);
  }
  require(tracker.getTracks().size() == 3,
          "nearby dynamic observations associate instead of duplicating");
  const auto updated_dynamic = std::find_if(
      tracker.getTracks().begin(), tracker.getTracks().end(), [&](const auto& track) {
        return track.id == dynamic_track_id;
      });
  require(updated_dynamic != tracker.getTracks().end() &&
              updated_dynamic->observations.size() == 2,
          "dynamic-only track is updated across frames");
}

void testSettledPhysicalObjectKeepsHistoryAndCurrentGeometry() {
  constexpr std::uint64_t kMovingStamp = 10'000'000'000ULL;
  constexpr std::uint64_t kSettledStamp = 11'000'000'000ULL;

  khronos::Track chair;
  chair.id = 10;
  chair.physical_instance_id = 10;
  chair.first_seen = kMovingStamp;
  chair.last_seen = kSettledStamp;
  chair.last_motion_seen = kMovingStamp;
  chair.has_dynamic_history = true;
  chair.is_dynamic = false;
  chair.confidence = 1.0f;
  chair.semantics = khronos::SemanticClusterInfo(75);
  chair.observations.emplace_back(kMovingStamp, 10, 1);
  chair.observations.emplace_back(kSettledStamp, 10, -1);

  khronos::FrameDataBuffer::Config buffer_config;
  buffer_config.max_buffer_size = 4;
  khronos::FrameDataBuffer buffer(buffer_config);

  auto moving_input = makeInput(kMovingStamp);
  moving_input.vertex_map.at<cv::Vec3f>(0, 0) = cv::Vec3f(-2.f, 0.f, 1.f);
  auto moving = std::make_shared<khronos::FrameData>(moving_input);
  moving->semantic_clusters.push_back(makeCluster(10, {{0, 0}}, 75));
  moving->dynamic_clusters.push_back(makeCluster(1, {{0, 0}}, 75));
  moving->dynamic_clusters.front().bounding_box =
      khronos::BoundingBox(khronos::Points{khronos::Point(-2.f, 0.f, 1.f)});
  buffer.storeData(moving);

  hydra::InputData settled_input(makeDenseCamera());
  settled_input.timestamp_ns = kSettledStamp;
  settled_input.world_T_body = Eigen::Isometry3d::Identity();
  settled_input.depth_image = cv::Mat(36, 48, CV_32FC1);
  for (int v = 0; v < settled_input.depth_image.rows; ++v) {
    for (int u = 0; u < settled_input.depth_image.cols; ++u) {
      settled_input.depth_image.at<float>(v, u) =
          0.9f + 0.2f * static_cast<float>(u) /
                     static_cast<float>(settled_input.depth_image.cols - 1);
    }
  }
  settled_input.color_image = cv::Mat(36, 48, CV_8UC3, cv::Scalar(128, 128, 128));
  settled_input.label_image = cv::Mat(36, 48, CV_32SC1, cv::Scalar(75));
  require(settled_input.getSensor().finalizeRepresentations(settled_input),
          "dense settled frame representations are valid");
  auto settled = std::make_shared<khronos::FrameData>(settled_input);
  khronos::MeasurementCluster settled_cluster;
  settled_cluster.id = 10;
  settled_cluster.semantics = khronos::SemanticClusterInfo(75);
  for (int v = 0; v < settled_input.vertex_map.rows; ++v) {
    for (int u = 0; u < settled_input.vertex_map.cols; ++u) {
      settled_cluster.pixels.emplace_back(u, v);
    }
  }
  settled->semantic_clusters.push_back(std::move(settled_cluster));
  settled->semantic_clusters.front().bounding_box =
      khronos::BoundingBox(khronos::utils::VertexMapAdaptor(
          settled->semantic_clusters.front().pixels, settled_input.vertex_map));
  settled->object_image = cv::Mat(36, 48, CV_32SC1, cv::Scalar(10));
  buffer.storeData(settled);

  const auto current_frames =
      khronos::MeshObjectExtractor::collectSemanticFrames(chair, buffer, chair.last_motion_seen);
  require(current_frames.size() == 1 &&
              current_frames.front().first->input.timestamp_ns == kSettledStamp,
          "I10 current reconstruction excludes every pre-settle semantic frame");

  khronos::MeshObjectExtractor::Config extractor_config;
  extractor_config.verbosity = 0;
  extractor_config.min_object_allocation_confidence = 0.f;
  extractor_config.min_object_volume = 0.f;
  extractor_config.max_object_volume = 10.f;
  // This fixture has only one dynamic sample and tests preservation of an
  // already accepted history, not displacement qualification. The dedicated
  // low-motion regression exercises the production 1 m threshold.
  extractor_config.min_dynamic_displacement = 0.f;
  extractor_config.object_reconstruction_resolution = 0.05f;
  extractor_config.only_extract_reconstructed_objects = false;
  extractor_config.preserve_settled_dynamic_history = true;
  extractor_config.min_object_reconstruction_confidence = 0.f;
  extractor_config.min_object_reconstruction_observations = 0;
  extractor_config.projective_integrator.num_threads = 1;
  extractor_config.projective_integrator.interpolation_method =
      hydra::InterpolatorNearest::Config();
  extractor_config.mesh_integrator.integrator_threads = 1;
  khronos::MeshObjectExtractor extractor(extractor_config);
  auto attrs = extractor.extractObject(chair, buffer);
  require(attrs != nullptr, "settled I10 produces current object attributes");
  require(attrs->details.at("instance_id").front() == 10,
          "settled I10 retains physical identity");
  require(khronos::hasCurrentObjectMesh(*attrs),
          "settled I10 materializes current private geometry at its new pose");
  require(khronos::hasTrajectoryHistory(*attrs),
          "settled I10 simultaneously retains D1 trajectory history");
  require(attrs->trajectory_timestamps.size() == 1 &&
              attrs->trajectory_timestamps.front() == kMovingStamp,
          "I10 trajectory contains the motion observation without treating it as current mesh");
  require(std::abs(attrs->trajectory_positions.front().x() + 2.f) < 0.1f,
          "I10 trajectory retains its old moving position");
  require((attrs->bounding_box.world_P_center - khronos::Point(0.f, 0.f, 1.f)).norm() < 0.25f,
          "I10 current geometry is centered at the settled position");
  // No absence evidence was observed after the settled frame. The backend
  // reconciler represents that open interval with max(), rather than treating
  // tracker.last_seen as a confirmed disappearance.
  attrs->last_observed_ns = {std::numeric_limits<std::uint64_t>::max()};

  // A later motion-history sample in the source snapshot must not leak into
  // an earlier time query. It remains metadata alongside the current mesh.
  attrs->trajectory_timestamps.push_back(kSettledStamp);
  attrs->trajectory_positions.push_back(attrs->bounding_box.world_P_center);

  auto dsg = std::make_shared<khronos::DynamicSceneGraph>();
  auto background = std::make_shared<spark_dsg::Mesh>(true, true, true, true);
  background->resizeVertices(2);
  background->setPos(0, khronos::Point::Zero());
  background->setPos(1, khronos::Point(0.1f, 0.f, 0.f));
  background->setTimestamp(0, kMovingStamp);
  background->setTimestamp(1, kSettledStamp);
  background->setFirstSeenTimestamp(0, kMovingStamp);
  background->setFirstSeenTimestamp(1, kSettledStamp);
  background->setLabel(0, 0);
  background->setLabel(1, 0);
  dsg->setMesh(background);
  require(dsg->emplaceNode(khronos::DsgLayers::OBJECTS,
                           khronos::NodeSymbol('O', 10),
                           std::move(attrs)),
          "I10 can be inserted into the persistent object layer");

  // Optimistic reconciliation uses zero to mean present from the beginning.
  // This current object is intentionally far from the background mesh: its
  // visibility must come from the explicit interval, not a TSDF proximity
  // heuristic (object-labelled surfaces are absent from the background TSDF).
  auto always_present = std::make_unique<khronos::KhronosObjectAttributes>();
  always_present->semantic_label = 74;
  always_present->details["instance_id"] = {7};
  always_present->first_observed_ns = {0};
  always_present->last_observed_ns = {
      std::numeric_limits<std::uint64_t>::max()};
  always_present->bounding_box = khronos::BoundingBox(
      khronos::Point(0.5f, 0.5f, 0.5f), khronos::Point(100.f, 0.f, 1.f));
  always_present->mesh.resizeVertices(1);
  always_present->mesh.setPos(0, khronos::Point::Zero());
  require(dsg->emplaceNode(khronos::DsgLayers::OBJECTS,
                           khronos::NodeSymbol('O', 7),
                           std::move(always_present)),
          "always-present I7 can be inserted away from background geometry");

  auto disappeared = std::make_unique<khronos::KhronosObjectAttributes>();
  disappeared->semantic_label = 74;
  disappeared->details["instance_id"] = {8};
  disappeared->first_observed_ns = {0};
  disappeared->last_observed_ns = {kMovingStamp};
  disappeared->bounding_box = khronos::BoundingBox(
      khronos::Point(0.5f, 0.5f, 0.5f), khronos::Point(-4.f, 0.f, 1.f));
  disappeared->mesh.resizeVertices(1);
  disappeared->mesh.setPos(0, khronos::Point::Zero());
  require(dsg->emplaceNode(khronos::DsgLayers::OBJECTS,
                           khronos::NodeSymbol('O', 8),
                           std::move(disappeared)),
          "disappeared I8 can remain as a historical DSG node");

  khronos::SpatioTemporalMap map(khronos::SpatioTemporalMap::Config{});
  map.update(dsg, kSettledStamp);

  const auto at_motion = map.getDsgPtr(kMovingStamp);
  const auto* motion_node = at_motion->findNode(khronos::NodeSymbol('O', 10));
  require(motion_node != nullptr, "earlier query retains visible I10");
  const auto& motion_attrs =
      motion_node->attributes<khronos::KhronosObjectAttributes>();
  require(khronos::trajectoryHistorySize(motion_attrs) == 1 &&
              motion_attrs.trajectory_timestamps.front() == kMovingStamp,
          "earlier query crops future I10 trajectory history");
  require(at_motion->findNode(khronos::NodeSymbol('O', 7)) != nullptr,
          "zero-start current object is visible without nearby background TSDF");

  const auto current = map.getDsgPtr(kSettledStamp);
  const auto* node = current->findNode(khronos::NodeSymbol('O', 10));
  require(node != nullptr, "latest current query retains settled I10");
  const auto& current_attrs = node->attributes<khronos::KhronosObjectAttributes>();
  require(khronos::hasCurrentObjectMesh(current_attrs) &&
              khronos::hasTrajectoryHistory(current_attrs),
          "latest current query exposes I10 mesh and trajectory together");
  require(khronos::trajectoryHistorySize(current_attrs) == 2,
          "latest query restores complete I10 trajectory history");

  std::size_t physical_i10_nodes = 0;
  for (const auto& [unused_id, object] :
       current->getLayer(khronos::DsgLayers::OBJECTS).nodes()) {
    (void)unused_id;
    const auto& object_attrs =
        object->attributes<khronos::KhronosObjectAttributes>();
    const auto it = object_attrs.details.find("instance_id");
    if (it != object_attrs.details.end() && !it->second.empty() &&
        it->second.front() == 10) {
      ++physical_i10_nodes;
    }
  }
  require(physical_i10_nodes == 1,
          "current DSG contains exactly one logical I10 node");

  const auto canonical_mesh = khronos::composeCurrentSceneMesh(*current);
  require(canonical_mesh &&
              canonical_mesh->numVertices() ==
                  current->mesh()->numVertices() +
                      current_attrs.mesh.numVertices() + 1,
          "canonical current geometry includes settled I10 and I7 private meshes");
  bool old_pose_materialized = false;
  for (std::size_t i = 0; i < canonical_mesh->numVertices(); ++i) {
    old_pose_materialized |= canonical_mesh->pos(i).x() < -1.0f;
  }
  require(!old_pose_materialized,
          "I10 old moving location remains history and is not current geometry");

  const auto source_current_mesh =
      khronos::composeCurrentSceneMesh(*dsg, kSettledStamp);
  bool disappeared_pose_materialized = false;
  for (std::size_t i = 0; i < source_current_mesh->numVertices(); ++i) {
    disappeared_pose_materialized |= source_current_mesh->pos(i).x() < -3.f;
  }
  require(!disappeared_pose_materialized,
          "canonical current geometry excludes a historical absent object");
}

std::unique_ptr<khronos::KhronosObjectAttributes> makePhysicalObject(
    int instance_id,
    std::uint64_t first,
    std::uint64_t last,
    float center_x,
    float mesh_marker,
    std::uint64_t trajectory_stamp) {
  auto attrs = std::make_unique<khronos::KhronosObjectAttributes>();
  attrs->semantic_label = 75;
  attrs->details["instance_id"] = {static_cast<std::size_t>(instance_id)};
  attrs->first_observed_ns = {first};
  attrs->last_observed_ns = {last};
  attrs->position = Eigen::Vector3d(center_x, 0.0, 1.0);
  attrs->bounding_box = khronos::BoundingBox(
      khronos::Point(0.5f, 0.5f, 0.5f), khronos::Point(center_x, 0.0f, 1.0f));
  attrs->mesh.resizeVertices(1);
  attrs->mesh.setPos(0, khronos::Point(mesh_marker, 0.0f, 0.0f));
  attrs->trajectory_timestamps = {trajectory_stamp};
  attrs->trajectory_positions = {khronos::Point(center_x, 0.0f, 1.0f)};
  attrs->dynamic_object_points = {
      {khronos::Point(center_x, 0.0f, 1.0f)}};
  return attrs;
}

void testPhysicalIdentityMergeKeepsNewestCurrentState() {
  khronos::DynamicSceneGraph graph;
  const auto old_id = khronos::NodeSymbol('O', 100);
  const auto new_id = khronos::NodeSymbol('O', 101);
  require(graph.emplaceNode(khronos::DsgLayers::OBJECTS,
                            old_id,
                            makePhysicalObject(10, 100, 200, 0.0f, 1.0f, 150)),
          "insert old I10 segment");
  require(graph.emplaceNode(khronos::DsgLayers::OBJECTS,
                            new_id,
                            makePhysicalObject(10, 300, 400, 2.0f, 9.0f, 350)),
          "insert new I10 segment");

  auto merged = khronos::UpdateKhronosObjectsFunctor::mergeObjectAttributes(
      graph, {old_id, new_id});
  const auto* attrs =
      dynamic_cast<const khronos::KhronosObjectAttributes*>(merged.get());
  require(attrs != nullptr, "physical merge returns Khronos object attributes");
  require(attrs->details.at("instance_id").front() == 10,
          "physical merge preserves I10 identity");
  require(attrs->mesh.numVertices() == 1 &&
              std::abs(attrs->mesh.pos(0).x() - 9.0f) < 1.0e-6f,
          "moved I10 keeps only newest current mesh instead of old/new union");
  require(std::abs(attrs->bounding_box.world_P_center.x() - 2.0f) < 1.0e-6f,
          "moved I10 current bounding box is its newest location");
  require(attrs->first_observed_ns.size() == 2 &&
              attrs->first_observed_ns.front() == 100 &&
              attrs->first_observed_ns.back() == 300,
          "physical merge retains both historical presence intervals");
  require(attrs->trajectory_timestamps == std::vector<std::uint64_t>({150, 350}),
          "physical merge retains ordered unique D1 history");
  require(attrs->dynamic_object_points.size() == attrs->trajectory_timestamps.size() &&
              attrs->dynamic_object_points.front().front().x() == 0.0f &&
              attrs->dynamic_object_points.back().front().x() == 2.0f,
          "physical merge keeps optional point frames aligned with D1 history");

  // Different S74 computers must never be treated as the same physical object.
  const auto other_id = khronos::NodeSymbol('O', 102);
  require(graph.emplaceNode(khronos::DsgLayers::OBJECTS,
                            other_id,
                            makePhysicalObject(7, 300, 400, 2.0f, 7.0f, 350)),
          "insert I7 monitor");
  auto not_physical = khronos::UpdateKhronosObjectsFunctor::mergeObjectAttributes(
      graph, {new_id, other_id});
  const auto* unmerged =
      dynamic_cast<const khronos::KhronosObjectAttributes*>(not_physical.get());
  require(unmerged && unmerged->details.at("instance_id").front() == 10,
          "different S74 physical IDs are not fused by physical merge semantics");

  // The chair carries no tracker motion evidence: nobody watched it move. Under the frozen
  // contract the only thing that can end its old state is a real measurement passing through the
  // surface that state claims -- never a bounding-box separation, which is how this used to be
  // decided. Supply that measurement, as a session that revisits the old site would.
  khronos::PersistentObjectState registry;
  {
    auto establish = khronos::UpdateKhronosObjectsFunctor::mergeObjectAttributes(graph, {old_id});
    auto* establish_attrs = dynamic_cast<khronos::KhronosObjectAttributes*>(establish.get());
    require(establish_attrs != nullptr, "old I10 segment merges to Khronos attributes");
    registry.applyPhysicalGeometry(graph, {old_id}, *establish_attrs);
  }
  require(registry.reportCurrentContradicted(10, 250),
          "the old I10 site is later seen through, closing that state");
  require(khronos::UpdateKhronosObjectsFunctor::canonicalizePhysicalObjects(graph, &registry) == 1,
          "two I10 temporal segments collapse into one logical graph node");
  std::size_t i10_count = 0;
  const khronos::KhronosObjectAttributes* canonical_i10 = nullptr;
  for (const auto& [unused_id, node] :
       graph.getLayer(khronos::DsgLayers::OBJECTS).nodes()) {
    (void)unused_id;
    const auto& object_attrs =
        node->attributes<khronos::KhronosObjectAttributes>();
    const auto instance = object_attrs.details.find("instance_id");
    if (instance != object_attrs.details.end() && !instance->second.empty() &&
        instance->second.front() == 10) {
      ++i10_count;
      canonical_i10 = &object_attrs;
    }
  }
  require(i10_count == 1 && canonical_i10,
          "canonical current DSG contains exactly one physical I10");
  require(std::abs(canonical_i10->mesh.pos(0).x() - 9.0f) < 1.0e-6f &&
              khronos::trajectoryHistorySize(*canonical_i10) == 2,
          "canonical I10 keeps newest current mesh and complete D1 history");
}

void testPresenceIntervalUnionMatrix() {
  using Attrs = khronos::KhronosObjectAttributes;
  using Interval = std::pair<std::uint64_t, std::uint64_t>;
  const auto run = [](const std::vector<Interval>& initial,
                      Interval added,
                      const std::vector<Interval>& expected,
                      const std::string& context) {
    Attrs attrs;
    for (const auto& interval : initial) {
      attrs.first_observed_ns.push_back(interval.first);
      attrs.last_observed_ns.push_back(interval.second);
    }
    khronos::addPresenceDuration(attrs, added.first, added.second);
    require(attrs.first_observed_ns.size() == expected.size() &&
                attrs.last_observed_ns.size() == expected.size(),
            context + ": interval count");
    for (size_t i = 0; i < expected.size(); ++i) {
      require(attrs.first_observed_ns[i] == expected[i].first &&
                  attrs.last_observed_ns[i] == expected[i].second,
              context + ": interval " + std::to_string(i));
    }
  };

  run({}, {5, 10}, {{5, 10}}, "empty");
  run({{20, 30}}, {5, 10}, {{5, 10}, {20, 30}}, "left disjoint");
  run({{5, 10}}, {20, 30}, {{5, 10}, {20, 30}}, "right disjoint");
  run({{10, 20}}, {5, 10}, {{5, 20}}, "left touching");
  run({{10, 20}}, {20, 25}, {{10, 25}}, "right touching");
  run({{10, 20}}, {5, 15}, {{5, 20}}, "left overlap");
  run({{10, 20}}, {15, 25}, {{10, 25}}, "right overlap");
  run({{10, 20}}, {12, 18}, {{10, 20}}, "contained");
  run({{10, 20}}, {5, 25}, {{5, 25}}, "engulfing");
  run({{10, 20}, {30, 40}}, {15, 35}, {{10, 40}}, "bridge");
  run({{10, 20}, {30, 40}}, {21, 29},
      {{10, 20}, {21, 29}, {30, 40}}, "finite gaps");
  run({{10, 20}}, {30, std::numeric_limits<std::uint64_t>::max()},
      {{10, 20}, {30, std::numeric_limits<std::uint64_t>::max()}},
      "open right");
  run({{10, 20}}, {0, 5}, {{0, 5}, {10, 20}}, "zero start");

  Attrs reversed;
  bool rejected = false;
  try {
    khronos::addPresenceDuration(reversed, 10, 5);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "inverted new presence interval was accepted");

  Attrs unpaired;
  unpaired.first_observed_ns = {1};
  rejected = false;
  try {
    khronos::addPresenceDuration(unpaired, 5, 10);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "unpaired existing presence vectors were accepted");

  Attrs inverted_existing;
  inverted_existing.first_observed_ns = {20};
  inverted_existing.last_observed_ns = {10};
  rejected = false;
  try {
    khronos::addPresenceDuration(inverted_existing, 30, 40);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "inverted existing presence interval was accepted");
}

}  // namespace

int main() {
  testExplicitLabelProtocol();
  testInvalidDepthPixelsCannotInflatePhysicalObject();
  testUnifiedExternalTracker();
  testSettledPhysicalObjectKeepsHistoryAndCurrentGeometry();
  testPhysicalIdentityMergeKeepsNewestCurrentState();
  testPresenceIntervalUnionMatrix();
  std::cout << "explicit_label_and_external_tracker_tests_passed\n";
  return 0;
}
