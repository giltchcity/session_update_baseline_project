#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include <hydra/input/camera.h>
#include <hydra/input/input_data.h>
#include <hydra/input/sensor_extrinsics.h>
#include <hydra/reconstruction/projection_interpolators.h>
#include <opencv2/core.hpp>

#include "khronos/active_window/data/frame_data_buffer.h"
#include "khronos/active_window/object_extraction/mesh_object_extractor.h"
#include "khronos/active_window/tracking/external_tracker.h"
#include "khronos/utils/khronos_attribute_utils.h"

namespace {

constexpr int kPhysicalId = 2;
constexpr int kSemanticId = 10;
constexpr int kDynamicId = 1;

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
  config.width = 48;
  config.height = 36;
  config.cx = 24.0f;
  config.cy = 18.0f;
  config.fx = 40.0f;
  config.fy = 40.0f;
  config.extrinsics = hydra::ParamSensorExtrinsics::Config();
  return std::make_shared<hydra::Camera>(config, "low_motion_test_camera");
}

khronos::FrameData::Ptr makeFrame(std::uint64_t stamp,
                                  float centroid_offset_x,
                                  bool with_physical,
                                  bool with_motion) {
  static const auto camera = makeCamera();
  hydra::InputData input(camera);
  input.timestamp_ns = stamp;
  input.world_T_body = Eigen::Isometry3d::Identity();
  input.depth_image = cv::Mat(36, 48, CV_32FC1);
  for (int v = 0; v < input.depth_image.rows; ++v) {
    for (int u = 0; u < input.depth_image.cols; ++u) {
      input.depth_image.at<float>(v, u) =
          0.9f + 0.2f * static_cast<float>(u) /
                     static_cast<float>(input.depth_image.cols - 1);
    }
  }
  input.color_image = cv::Mat(36, 48, CV_8UC3, cv::Scalar(128, 128, 128));
  input.label_image = cv::Mat(36, 48, CV_32SC1, cv::Scalar(kSemanticId));
  require(input.getSensor().finalizeRepresentations(input),
          "dense input representations are valid");
  for (int v = 0; v < input.vertex_map.rows; ++v) {
    for (int u = 0; u < input.vertex_map.cols; ++u) {
      input.vertex_map.at<cv::Vec3f>(v, u)[0] += centroid_offset_x;
    }
  }

  auto frame = std::make_shared<khronos::FrameData>(input);
  frame->object_image = cv::Mat::zeros(36, 48, CV_32SC1);
  frame->dynamic_image = cv::Mat::zeros(36, 48, CV_32SC1);

  khronos::Pixels pixels;
  pixels.reserve(36 * 48);
  for (int v = 0; v < 36; ++v) {
    for (int u = 0; u < 48; ++u) {
      pixels.emplace_back(u, v);
    }
  }

  if (with_physical) {
    khronos::MeasurementCluster physical;
    physical.id = kPhysicalId;
    physical.pixels = pixels;
    physical.semantics = khronos::SemanticClusterInfo(kSemanticId);
    frame->semantic_clusters.emplace_back(std::move(physical));
    frame->object_image.setTo(kPhysicalId);
  }
  if (with_motion) {
    khronos::MeasurementCluster motion;
    motion.id = kDynamicId;
    motion.pixels = std::move(pixels);
    motion.semantics = khronos::SemanticClusterInfo(kSemanticId);
    frame->dynamic_clusters.emplace_back(std::move(motion));
    frame->dynamic_image.setTo(kDynamicId);
  }
  return frame;
}

khronos::ExternalTracker makeTracker() {
  khronos::ExternalTracker::Config config;
  config.min_num_observations = 1;
  config.min_cross_iou = 0.1f;
  config.max_dynamic_distance = 2.0f;
  config.settle_time = 5.0f;
  return khronos::ExternalTracker(config);
}

khronos::MeshObjectExtractor makeExtractor() {
  khronos::MeshObjectExtractor::Config config;
  config.verbosity = 0;
  config.min_object_allocation_confidence = 0.0f;
  config.min_object_volume = 0.0f;
  config.max_object_volume = 10.0f;
  config.min_dynamic_displacement = 1.0f;
  config.object_reconstruction_resolution = 0.05f;
  config.only_extract_reconstructed_objects = true;
  config.preserve_settled_dynamic_history = true;
  config.min_object_reconstruction_confidence = 0.0f;
  config.min_object_reconstruction_observations = 0;
  config.projective_integrator.num_threads = 1;
  config.projective_integrator.interpolation_method =
      hydra::InterpolatorNearest::Config();
  config.mesh_integrator.integrator_threads = 1;
  return khronos::MeshObjectExtractor(config);
}

void processAndStore(khronos::ExternalTracker& tracker,
                     khronos::FrameDataBuffer& buffer,
                     const khronos::FrameData::Ptr& frame) {
  tracker.processInput(*frame);
  buffer.storeData(frame);
}

void requireStaticPhysicalCurrentObject(
    const khronos::KhronosObjectAttributes::Ptr& attrs,
    const std::string& phase) {
  require(attrs != nullptr, phase + ": physical I2 is not deleted");
  require(attrs->semantic_label == kSemanticId, phase + ": semantic S10 is retained");
  const auto instance = attrs->details.find("instance_id");
  require(instance != attrs->details.end() && instance->second.size() == 1 &&
              instance->second.front() == kPhysicalId,
          phase + ": physical identity I2 is retained");
  require(khronos::hasCurrentObjectMesh(*attrs),
          phase + ": a static current private mesh is reconstructed");
  require(!khronos::hasTrajectoryHistory(*attrs) &&
              attrs->trajectory_timestamps.empty() &&
              attrs->trajectory_positions.empty() &&
              attrs->dynamic_object_points.empty(),
          phase + ": rejected low motion does not fabricate D1 history");
}

void testPhysicalTransientLowMotionFallsBackToStaticCurrent() {
  constexpr std::uint64_t kFirstStamp = 10'000'000'000ULL;
  constexpr std::uint64_t kTerminalStamp = 11'000'000'000ULL;
  constexpr std::uint64_t kSettledStamp = 17'000'000'000ULL;

  auto tracker = makeTracker();
  khronos::FrameDataBuffer::Config buffer_config;
  buffer_config.max_buffer_size = 4;
  khronos::FrameDataBuffer buffer(buffer_config);
  processAndStore(tracker, buffer, makeFrame(kFirstStamp, 0.0f, true, true));
  processAndStore(tracker, buffer, makeFrame(kTerminalStamp, 0.175f, true, true));

  require(tracker.getTracks().size() == 1,
          "overlap remains one physical track rather than a duplicate");
  const auto& terminal_track = tracker.getTracks().front();
  require(terminal_track.physical_instance_id &&
              *terminal_track.physical_instance_id == kPhysicalId &&
              terminal_track.semantics &&
              terminal_track.semantics->category_id == kSemanticId,
          "terminal track carries I2/S10");
  require(terminal_track.is_dynamic,
          "terminal overlap has not reached the tracker settle branch");
  require(std::abs(khronos::MeshObjectExtractor::computeDynamicDisplacement(
                       terminal_track, buffer) -
                   0.175f) < 1.0e-4f,
          "regression fixture measures the observed 0.175 m displacement");

  auto extractor = makeExtractor();
  auto terminal = extractor.extractObject(terminal_track, buffer);
  requireStaticPhysicalCurrentObject(terminal, "terminal low-motion fallback");

  processAndStore(tracker, buffer, makeFrame(kSettledStamp, 0.175f, true, false));
  require(!tracker.getTracks().front().is_dynamic &&
              tracker.getTracks().front().has_dynamic_history,
          "tracker settle bookkeeping still records candidate motion history");
  auto settled = extractor.extractObject(tracker.getTracks().front(), buffer);
  requireStaticPhysicalCurrentObject(settled, "settled low-motion fallback");
}

void testHighMotionPhysicalTrackRemainsDynamicHistory() {
  auto tracker = makeTracker();
  khronos::FrameDataBuffer::Config buffer_config;
  buffer_config.max_buffer_size = 2;
  khronos::FrameDataBuffer buffer(buffer_config);
  processAndStore(tracker, buffer,
                  makeFrame(20'000'000'000ULL, 0.0f, true, true));
  processAndStore(tracker, buffer,
                  makeFrame(21'000'000'000ULL, 1.25f, true, true));

  auto extractor = makeExtractor();
  auto attrs = extractor.extractObject(tracker.getTracks().front(), buffer);
  require(attrs != nullptr, "high-motion physical track is extracted");
  require(attrs->details.at("instance_id").front() == kPhysicalId &&
              attrs->semantic_label == kSemanticId,
          "high-motion D1 keeps I2/S10");
  require(khronos::hasTrajectoryHistory(*attrs) &&
              attrs->trajectory_timestamps.size() == 2 &&
              attrs->trajectory_positions.size() == 2,
          "true 1.25 m motion remains dynamic D1 history");
  require(!khronos::hasCurrentObjectMesh(*attrs),
          "an unsettled high-motion D1 is not mislabeled as static current mesh");
}

void testNonPhysicalLowMotionStillDrops() {
  auto tracker = makeTracker();
  khronos::FrameDataBuffer::Config buffer_config;
  buffer_config.max_buffer_size = 2;
  khronos::FrameDataBuffer buffer(buffer_config);
  processAndStore(tracker, buffer,
                  makeFrame(30'000'000'000ULL, 0.0f, false, true));
  processAndStore(tracker, buffer,
                  makeFrame(31'000'000'000ULL, 0.175f, false, true));

  require(tracker.getTracks().size() == 1 &&
              !tracker.getTracks().front().physical_instance_id,
          "free-space low motion creates no physical identity");
  auto extractor = makeExtractor();
  require(extractor.extractObject(tracker.getTracks().front(), buffer) == nullptr,
          "non-physical low motion does not invent a static entity");
}

}  // namespace

int main() {
  testPhysicalTransientLowMotionFallsBackToStaticCurrent();
  testHighMotionPhysicalTrackRemainsDynamicHistory();
  testNonPhysicalLowMotionStillDrops();
  std::cout << "physical_low_motion_fallback_tests_passed\n";
  return 0;
}
