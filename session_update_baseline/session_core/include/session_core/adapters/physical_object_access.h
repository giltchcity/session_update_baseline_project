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

// Raw segment provenance stays separate from materialized CURRENT timestamps.
inline constexpr auto kInputFirst = "session_input_first";
inline constexpr auto kInputLast = "session_input_last";
inline TimeStamp inputFirstStamp(const KhronosObjectAttributes& attrs) {
  const auto it = attrs.details.find(kInputFirst);
  return it != attrs.details.end() && it->second.size() == 1
      ? it->second.front() : observationFirstStamp(attrs);
}
inline TimeStamp inputLastStamp(const KhronosObjectAttributes& attrs) {
  const auto it = attrs.details.find(kInputLast);
  return it != attrs.details.end() && it->second.size() == 1
      ? it->second.front() : observationLastStamp(attrs);
}
inline void setInputBounds(KhronosObjectAttributes& attrs, TimeStamp first, TimeStamp last) {
  attrs.details[kInputFirst] = {first};
  attrs.details[kInputLast] = {last};
}

inline uint64_t firstObservation(const KhronosObjectAttributes& attrs) {
  const auto stamp = observationFirstStamp(attrs);
  return stamp == 0 ? std::numeric_limits<uint64_t>::max() : stamp;
}

}  // namespace
