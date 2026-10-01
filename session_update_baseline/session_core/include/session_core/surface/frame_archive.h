#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
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
 * sensor's (min_range, max_range]). Physical-ID measurements are retained for
 * terminal state authorization (README 8a); anonymous invalid-label pixels and
 * anonymous motion clusters retain the input exclusion mask (no class branch).
 * The physical instance id is stored per pixel (0 = none), and so are the semantic class
 * (README principle 5: it is the grouping key of the persistence prior and decides at session end,
 * with the class's learned survival, whether a reading without identity is fused) and the native
 * motion mask. Range, ids and attributes are held compressed (row-wise range differences and runs,
 * zstd), about 1/6 of the raw range on real data.
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
    bool valid() const {
      return width > 0 && height > 0 &&
          static_cast<uint64_t>(width) * height <= std::numeric_limits<uint32_t>::max() &&
          width <= static_cast<uint32_t>(std::numeric_limits<int>::max()) &&
          height <= static_cast<uint32_t>(std::numeric_limits<int>::max()) &&
          std::isfinite(fx) && std::isfinite(fy) && fx > 0.f && fy > 0.f &&
          std::isfinite(cx) && std::isfinite(cy) && std::isfinite(min_range) &&
          min_range >= 0.f && max_range > min_range;
    }
    bool sameAs(const Camera& other) const;
  };

  struct InstanceRun {
    uint32_t end = 0;  // exclusive flattened-pixel end offset
    uint16_t id = 0;
  };

  // Per-pixel attribute word of a run: the semantic class + 1 in the low 15 bits (0: no class) and
  // the native motion mask in the top bit.
  static constexpr uint16_t kMotionBit = 0x8000;
  static constexpr uint16_t kClassMask = 0x7FFF;
  static uint16_t attributeWord(int32_t semantic_label, bool motion) {
    const uint16_t code = semantic_label >= 0 && semantic_label < kClassMask - 1
        ? static_cast<uint16_t>(semantic_label + 1) : 0;
    return static_cast<uint16_t>(code | (motion ? kMotionBit : 0));
  }

  struct Frame {
    TimeStamp stamp = 0;
    // camera_to_world: the pose the mapper integrated this frame with.
    Eigen::Isometry3d world_T_sensor = Eigen::Isometry3d::Identity();
    std::vector<uint8_t> packed;  // zstd(range differences, instance runs[, attribute runs])
    // 2: range differences and instance runs; 3: with the attribute runs after a run count.
    uint8_t layout = 3;

    static Frame pack(TimeStamp stamp,
                      const Eigen::Isometry3d& world_T_sensor,
                      const std::vector<uint16_t>& range_mm,
                      const std::vector<InstanceRun>& instances,
                      const std::vector<InstanceRun>* attributes = nullptr);
    /** Range [mm] and physical id per pixel (num_pixels each); optionally the semantic class + 1
     * (0: none) and the native motion mask. A layout without attributes decodes them as empty.
     * False if corrupt. */
    bool decode(size_t num_pixels, std::vector<uint16_t>& range_mm, std::vector<uint16_t>& ids,
                std::vector<uint16_t>* classes = nullptr, std::vector<uint8_t>* motion = nullptr) const;
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
  std::unordered_map<TimeStamp, size_t> frame_indices_;
};

}  // namespace khronos
