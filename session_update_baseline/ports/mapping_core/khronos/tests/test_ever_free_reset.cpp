// A measured point occupies its voxel, also where the TSDF says free; an ever-free voxel that keeps
// holding measured points for temporal_buffer (gaps up to temporal_buffer) is static again, while
// a measurement that only passes through leaves it ever-free.
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include <hydra/common/global_info.h>
#include <hydra/input/camera.h>
#include <hydra/input/input_data.h>
#include <hydra/input/sensor_extrinsics.h>
#include <hydra/reconstruction/volumetric_map.h>
#include <opencv2/core.hpp>
#include <spatial_hash/grid.h>

#include "khronos/active_window/data/frame_data.h"
#include "khronos/active_window/integration/tracking_integrator.h"

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << "\n";
    std::exit(1);
  }
}

// One frame whose single pixel measures `point` (identity pose, world = sensor frame).
khronos::FrameData frame(std::uint64_t stamp, const Eigen::Vector3f& point) {
  static const auto camera = [] {
    hydra::Camera::Config config;
    config.min_range = 0.1;
    config.max_range = 10.0;
    config.width = 1;
    config.height = 1;
    config.fx = 1.0f;
    config.fy = 1.0f;
    config.cx = 0.5f;
    config.cy = 0.5f;
    config.extrinsics = hydra::ParamSensorExtrinsics::Config();
    return std::make_shared<hydra::Camera>(config, "test_camera");
  }();
  hydra::InputData input(camera);
  input.timestamp_ns = stamp;
  input.world_T_body = Eigen::Isometry3d::Identity();
  input.vertex_map = cv::Mat(1, 1, CV_32FC3);
  input.vertex_map.at<cv::Vec3f>(0, 0) = cv::Vec3f(point.x(), point.y(), point.z());
  input.range_image = cv::Mat(1, 1, CV_32FC1, cv::Scalar(point.norm()));
  return khronos::FrameData(input);
}

hydra::GlobalIndex voxelIndex(hydra::VolumetricMap& map, const Eigen::Vector3f& point) {
  const auto index = spatial_hash::indexFromPoint<hydra::GlobalIndex>(point, 1.f / map.config.voxel_size);
  map.allocateBlock(spatial_hash::blockIndexFromGlobalIndex(index, map.config.voxels_per_side));
  return index;
}

hydra::TrackingVoxel& trackingVoxel(hydra::VolumetricMap& map, const Eigen::Vector3f& point) {
  const auto index = voxelIndex(map, point);
  const size_t vps = map.config.voxels_per_side;
  return map.getTrackingLayer()
      ->getBlockPtr(spatial_hash::blockIndexFromGlobalIndex(index, vps))
      ->getVoxel(spatial_hash::localIndexFromGlobalIndex(index, vps));
}

hydra::TsdfVoxel& tsdfVoxel(hydra::VolumetricMap& map, const Eigen::Vector3f& point) {
  const auto index = voxelIndex(map, point);
  const size_t vps = map.config.voxels_per_side;
  return map.getTsdfLayer()
      .getBlockPtr(spatial_hash::blockIndexFromGlobalIndex(index, vps))
      ->getVoxel(spatial_hash::localIndexFromGlobalIndex(index, vps));
}

constexpr std::uint64_t kStep = 100'000'000ULL;  // 0.1 s between frames

}  // namespace

int main() {
  hydra::PipelineConfig pipeline;
  pipeline.default_num_threads = 1;
  hydra::GlobalInfo::init(pipeline);
  hydra::VolumetricMap::Config map_config;
  map_config.voxel_size = 0.05f;
  map_config.with_tracking = true;
  khronos::TrackingIntegrator::Config config;
  config.temporal_buffer = 1.f;
  config.num_threads = 1;

  {
    // The TSDF says free here (it never received this surface), the point says occupied.
    hydra::VolumetricMap map(map_config);
    khronos::TrackingIntegrator integrator(config);
    const Eigen::Vector3f site(1.02f, 0.02f, 0.02f), elsewhere(2.02f, 0.02f, 0.02f);
    tsdfVoxel(map, site).distance = 0.1f;
    tsdfVoxel(map, site).weight = 1.f;
    trackingVoxel(map, site).last_occupied = 0;
    integrator.updateBlocks(frame(1'000'000'000ULL, elsewhere), map);
    require(trackingVoxel(map, site).last_occupied == 0, "a free voxel without points stays free");
    integrator.updateBlocks(frame(1'100'000'000ULL, site), map);
    require(trackingVoxel(map, site).last_occupied == 1'100'000'000ULL,
            "a voxel holding a measured point is occupied");
  }
  {
    // A static surface measured inside ever-free space: ever-free until it has held points for
    // temporal_buffer, then static.
    hydra::VolumetricMap map(map_config);
    khronos::TrackingIntegrator integrator(config);
    const Eigen::Vector3f surface(1.02f, 0.02f, 0.02f);
    trackingVoxel(map, surface).ever_free = true;
    for (int k = 0; k <= 10; ++k) {
      const auto data = frame(1'000'000'000ULL + k * kStep, surface);
      integrator.updateBlocks(data, map);
      require(trackingVoxel(map, surface).ever_free == (k < 10),
              "a voxel holding points stays ever-free for exactly temporal_buffer (frame " +
                  std::to_string(k) + ")");
    }
  }
  {
    // Something passing through: points for 0.5 s, then away for longer than temporal_buffer,
    // then back for 0.5 s. No uninterrupted temporal_buffer of occupancy: still ever-free.
    hydra::VolumetricMap map(map_config);
    khronos::TrackingIntegrator integrator(config);
    const Eigen::Vector3f site(1.02f, 0.02f, 0.02f), elsewhere(2.02f, 0.02f, 0.02f);
    trackingVoxel(map, site).ever_free = true;
    std::uint64_t stamp = 1'000'000'000ULL;
    for (int k = 0; k < 5; ++k, stamp += kStep) integrator.updateBlocks(frame(stamp, site), map);
    for (int k = 0; k < 12; ++k, stamp += kStep) integrator.updateBlocks(frame(stamp, elsewhere), map);
    for (int k = 0; k < 5; ++k, stamp += kStep) integrator.updateBlocks(frame(stamp, site), map);
    require(trackingVoxel(map, site).ever_free,
            "occupancy interrupted for longer than temporal_buffer starts over");
  }
  hydra::GlobalInfo::reset();
  std::cout << "ever_free_reset_tests_passed\n";
  return 0;
}
