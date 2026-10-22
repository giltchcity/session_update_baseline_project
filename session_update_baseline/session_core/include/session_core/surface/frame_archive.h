#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include "khronos/common/common_types.h"

namespace khronos {

struct FrameData;

/**
 * @brief The session's own depth frames as the present integrated them
 * (PresentSurface), kept so that a frame's contribution can be removed exactly
 * when the legal contribution domain (README section 5, eq. 8) changes.
 *
 * Every frame the active window processes is kept at the mapper's input
 * resolution: its stamp, the sensor pose the mapper integrated it with, the
 * range image in millimetres (0 where the reading is invalid, outside the
 * sensor's (min_range, max_range], on a dynamic / invalid semantic class, or on
 * a motion / dynamic cluster of the frame: the pixels the active window's
 * reconstruction rejects) and the physical instance id per pixel (0 = none).
 * Range and ids are held compressed (row-wise range differences and id runs,
 * LZ4), about 1/4 of the raw range on real data.
 *
 * Session-local, never serialized with the map. Not thread-safe: its owner
 * serializes all access.
 */
class FrameArchive {
 public:
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
    std::vector<uint8_t> packed;  // LZ4(range differences, instance runs)
    // What the frame reads: the ids with a reading on one of their pixels (sorted), the farthest
    // reading [mm], and the farthest reading of each tile of kTile x kTile pixels (row-major).
    std::vector<uint16_t> seen;
    uint16_t farthest_mm = 0;
    std::vector<uint16_t> tile_farthest_mm;
    // The free limit the present counts the frame with (packLimit; empty: none), kept from when
    // the frame was integrated or last moved to a new domain.
    std::vector<uint8_t> limit;

    static Frame pack(TimeStamp stamp,
                      const Eigen::Isometry3d& world_T_sensor,
                      uint32_t width,
                      const std::vector<uint16_t>& range_mm,
                      const std::vector<InstanceRun>& instances);
    /** Range [mm] per pixel (num_pixels) and the instance runs, as packed. False if corrupt. */
    bool decode(size_t num_pixels, std::vector<uint16_t>& range_mm, std::vector<InstanceRun>& runs) const;
    /** Range [mm] per pixel, 0 on the pixels of the `excluded` ids (sorted). False if corrupt. */
    bool decodeExcluding(size_t num_pixels, const std::vector<uint16_t>& excluded,
                         std::vector<uint16_t>& range_mm) const;
    /** The kept free limit (num_pixels values; empty: none). False if corrupt. */
    bool keptLimit(size_t num_pixels, std::vector<float>& free_limit) const;
    /** Whether the frame has a reading on a pixel of `id`. */
    bool sees(uint16_t id) const { return std::binary_search(seen.begin(), seen.end(), id); }
    /** Bound [mm] of the readings on columns [u0, u1] by rows [v0, v1]: the farthest of their tiles. */
    uint16_t farthestIn(uint32_t width, int u0, int u1, int v0, int v1) const;
  };
  // Pixels per side of the tiles a frame bounds its readings by (any size bounds them; smaller is tighter).
  static constexpr uint32_t kTile = 16;

  /**
   * Keep a processed frame. False (nothing kept) without a pinhole
   * camera and range image, or for a camera that differs from the first frame's.
   */
  bool offer(const FrameData& data);

  /** Keep an already packed frame of `camera`; false for a camera that differs from the first frame's. */
  bool keep(const Camera& camera, Frame frame);

  const Frame& frame(size_t index) const { return frames_.at(index); }
  /** A free limit (one value per pixel; empty: none) as a frame keeps it. */
  static std::vector<uint8_t> packLimit(const std::vector<float>& free_limit);
  /** Keep the packed free limit (packLimit) kept frame `index` is counted with. */
  void keepLimit(size_t index, std::vector<uint8_t> packed_limit);
  const std::vector<Frame>& frames() const { return frames_; }
  size_t numFrames() const { return frames_.size(); }
  size_t numBytes() const { return bytes_; }
  const Camera& camera() const { return camera_; }

  /** Binary dump / load (diagnostics and offline replay only). */
  static bool save(const std::string& path, const std::vector<Frame>& frames, const Camera& camera);
  static bool load(const std::string& path, std::vector<Frame>& frames, Camera& camera);

 private:
  size_t bytes_ = 0;
  Camera camera_;
  std::vector<Frame> frames_;
};

}  // namespace khronos
