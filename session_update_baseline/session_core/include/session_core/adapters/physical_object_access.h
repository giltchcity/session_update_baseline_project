#pragma once
#include <limits>
#include <optional>
#include "khronos/utils/khronos_attribute_utils.h"
namespace khronos::session_detail {

inline std::optional<size_t> getPhysicalInstanceId(const KhronosObjectAttributes& attrs) {
  const auto iter = attrs.details.find("instance_id");
  if (iter == attrs.details.end() || iter->second.size() != 1 || iter->second.front() == 0) {
    return std::nullopt;
  }
  return iter->second.front();
}

inline uint64_t firstObservation(const KhronosObjectAttributes& attrs) {
  const auto stamp = observationFirstStamp(attrs);
  return stamp == 0 ? std::numeric_limits<uint64_t>::max() : stamp;
}

}  // namespace
