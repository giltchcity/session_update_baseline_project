#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace hydra {
class PointNeighborSearch;
}

namespace khronos {

/**
 * @brief Per surface element attributes a session hands to the next one for
 * the session-end memory rules (SessionRefusion): the nearest range at which
 * the element's own session reached it, that session's measured depth scale,
 * its physical label and its layer. Stored next to the chain state
 * (consolidation_memory.attr) for every vertex of the consolidated final map.
 */
struct MemoryAttributeRecord {
  float x = 0.f, y = 0.f, z = 0.f;
  float q = std::numeric_limits<float>::infinity();  // nearest reaching range [m]
  float s = 0.f;                                     // depth scale of that session
  uint32_t label = 0;                                // physical id, 0 = background
  uint8_t layer = 0;                                 // 0 background, 1 objects
};

bool writeMemoryAttributes(const std::string& path, const std::vector<MemoryAttributeRecord>& records);
bool readMemoryAttributes(const std::string& path, std::vector<MemoryAttributeRecord>& records);

/** Nearest record lookup (world frame). */
class MemoryAttributeLookup {
 public:
  explicit MemoryAttributeLookup(std::vector<MemoryAttributeRecord> records);
  ~MemoryAttributeLookup();
  size_t size() const { return records_.size(); }
  /** The nearest record within max_distance, or nullptr. */
  const MemoryAttributeRecord* find(const Eigen::Vector3f& p, float max_distance) const;

 private:
  std::vector<MemoryAttributeRecord> records_;
  std::vector<Eigen::Vector3f> points_;
  std::unique_ptr<hydra::PointNeighborSearch> search_;
};

}  // namespace khronos
