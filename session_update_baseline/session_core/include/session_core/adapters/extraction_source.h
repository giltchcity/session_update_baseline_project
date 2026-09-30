#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include "khronos/common/common_types.h"

namespace khronos::session_detail {

inline constexpr char kExtractionSourceScope[] = "session_source_scope";
inline constexpr char kExtractionSourceGeneration[] = "session_source_generation";
inline constexpr char kExtractionSourceInputThrough[] = "session_source_input_through";

struct ExtractionSource {
  std::array<uint64_t, 2> scope{};
  uint64_t generation = 0;
  // Processing watermark of the immutable request, never a surface support time.
  TimeStamp input_through = 0;
};

std::array<uint64_t, 2> makeExtractionScope();
void setExtractionSource(KhronosObjectAttributes& attrs, const ExtractionSource& source);
// Untagged legacy inputs return nullopt; partial or invalid tags are errors.
std::optional<ExtractionSource> extractionSource(const KhronosObjectAttributes& attrs);

}  // namespace khronos::session_detail
