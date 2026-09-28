#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include "khronos/common/common_types.h"

namespace khronos {

struct FrameData;

/**
 * @brief The session's own depth frames, kept for the session-end
 * re-integration of the present (SessionRefusion).
 *
 * Every frame the active window processes is kept at the mapper's input
 * resolution: its stamp, the sensor pose the mapper integrated it with, the
 * range image in millimetres (0 where the reading is invalid, outside the
 * sensor's (min_range, max_range], on a dynamic / invalid semantic class, or on
 * a motion / dynamic cluster of the frame: the pixels the active window's
 * reconstruction rejects)
 * and the physical instance id per pixel (0 = none). Range and ids are held
 * compressed (row-wise range differences and id runs, zstd), about 1/6 of the
 * raw range on real data.
 *
 * The archive is session-local and never serialized with the map. It is
 * released (moved out) at session end.
 */
class FrameArchive {
 public:
  using Ptr = std::shared_ptr<FrameArchive>;

  struct Camera {
    uint32_t width = 0;
    uint32_t height = 0;
    float fx = 0.f, fy = 0.f, cx = 0.f, cy = 0.f;
    float min_range = 0.f, max_range = 0.f;
    bool valid() const { return width > 0 && height > 0 && fx > 0.f && fy > 0.f; }
    bool sameAs(const Camera& other) const;
  };

  struct InstanceRun {
    uint32_t end = 0;  // exclusive flattened-pixel end offset
    uint16_t id = 0;
  };

  struct Frame {
    TimeStamp stamp = 0;
    // camera_to_world: the pose the mapper integrated this frame with.
    Eigen::Isometry3d world_T_sensor = Eigen::Isometry3d::Identity();
    std::vector<uint8_t> packed;  // zstd(range differences, instance runs)

    static Frame pack(TimeStamp stamp,
                      const Eigen::Isometry3d& world_T_sensor,
                      const std::vector<uint16_t>& range_mm,
                      const std::vector<InstanceRun>& instances);
    /** Range [mm] and physical id per pixel (num_pixels each). False if corrupt. */
    bool decode(size_t num_pixels, std::vector<uint16_t>& range_mm, std::vector<uint16_t>& ids) const;
  };

  FrameArchive() = default;

  /** Keep a processed frame. Thread-safe. */
  void offer(const FrameData& data);

  /** Move the kept frames out (session end). */
  std::vector<Frame> release(Camera* camera);

  size_t numOffered() const;
  size_t numFrames() const;
  size_t numBytes() const;

  /** Binary dump / load (diagnostics and offline replay only). */
  static bool save(const std::string& path, const std::vector<Frame>& frames, const Camera& camera);
  static bool load(const std::string& path, std::vector<Frame>& frames, Camera& camera);

 private:
  mutable std::mutex mutex_;
  size_t offered_ = 0;
  size_t skipped_ = 0;
  size_t bytes_ = 0;
  size_t raw_bytes_ = 0;
  Camera camera_;
  std::vector<Frame> frames_;
};

}  // namespace khronos
