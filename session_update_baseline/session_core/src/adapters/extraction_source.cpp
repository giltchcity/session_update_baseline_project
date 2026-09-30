#include "session_core/adapters/extraction_source.h"

#include <limits>
#include <stdexcept>
#include <utility>

#include <openssl/rand.h>

#include "khronos/active_window/object_extraction/object_worker_pool.h"

namespace khronos::session_detail {
namespace {
void validateSource(const ExtractionSource& source) {
  if ((!source.scope[0] && !source.scope[1]) || !source.generation) {
    throw std::invalid_argument("Invalid extraction source identity");
  }
}
}  // namespace

std::array<uint64_t, 2> makeExtractionScope() {
  std::array<uint64_t, 2> result{};
  do {
    if (RAND_bytes(reinterpret_cast<unsigned char*>(result.data()),
                   static_cast<int>(sizeof(uint64_t) * result.size())) != 1) {
      throw std::runtime_error("Unable to generate extraction source namespace");
    }
  } while (!result[0] && !result[1]);
  return result;
}

std::optional<ExtractionSource> extractionSource(const KhronosObjectAttributes& attrs) {
  const auto scope = attrs.details.find(kExtractionSourceScope);
  const auto generation = attrs.details.find(kExtractionSourceGeneration);
  const auto through = attrs.details.find(kExtractionSourceInputThrough);
  const auto end = attrs.details.end();
  if (scope == end && generation == end && through == end) return std::nullopt;
  if (scope == end || generation == end || through == end ||
      scope->second.size() != 2 || generation->second.size() != 1 || through->second.size() != 1) {
    throw std::invalid_argument("Incomplete extraction source metadata");
  }
  ExtractionSource source{{scope->second[0], scope->second[1]},
                          generation->second[0], through->second[0]};
  validateSource(source);
  return source;
}

void setExtractionSource(KhronosObjectAttributes& attrs, const ExtractionSource& source) {
  static_assert(std::numeric_limits<size_t>::digits >= 64,
                "Extraction source metadata requires 64-bit attribute detail values");
  validateSource(source);
  if (const auto previous = extractionSource(attrs)) {
    if (previous->scope != source.scope || previous->generation != source.generation ||
        previous->input_through != source.input_through) {
      throw std::logic_error("Cannot overwrite extraction source identity");
    }
    return;
  }
  attrs.details[kExtractionSourceScope] = {source.scope[0], source.scope[1]};
  attrs.details[kExtractionSourceGeneration] = {source.generation};
  attrs.details[kExtractionSourceInputThrough] = {source.input_through};
}

}  // namespace khronos::session_detail

namespace khronos {

// Project-owned worker completion policy; the upstream extractor stays unchanged.
void ObjectWorkerPool::markCompleted(uint64_t generation,
                                     spark_dsg::NodeAttributes::Ptr attrs,
                                     std::exception_ptr failure) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (curr_workers_ == 0) {
    completions_.failInvalidCompletion();
  } else {
    --curr_workers_;
  }
  completions_.complete(generation, std::move(attrs), failure);
  // stop() must not observe the finished frontier and destroy the pool while
  // a detached worker still has a pending access to its condition variable.
  state_cv_.notify_all();
}

}  // namespace khronos
