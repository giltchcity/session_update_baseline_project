#include "session_core/runtime/ordered_completion.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace khronos::session_detail {

OrderedExtractionCompletion::OrderedExtractionCompletion()
    : invalid_completion_(std::make_exception_ptr(
          std::logic_error("Invalid object extraction completion protocol"))) {}

void OrderedExtractionCompletion::reserve(uint64_t generation) {
  rethrowFailure();
  const auto last = pending_.empty() ? completed_through_ : pending_.rbegin()->first;
  if (last == std::numeric_limits<uint64_t>::max() || generation != last + 1) {
    std::rethrow_exception(invalid_completion_);
  }
  // Both the lookup node and its transferable output node are allocated before
  // the producer accepts the request. A failed allocation leaves no reservation.
  pending_.try_emplace(generation);
}

void OrderedExtractionCompletion::cancelReservation(uint64_t generation) noexcept {
  const auto found = pending_.find(generation);
  if (found == pending_.end() || found->second.completed || pending_.rbegin()->first != generation) {
    failInvalidCompletion();
    return;
  }
  pending_.erase(found);
}

void OrderedExtractionCompletion::recordFailure(std::exception_ptr failure) noexcept {
  if (!failure || failure_) return;
  failure_ = failure;
  ready_.clear();
  for (auto& entry : pending_) entry.second.output.front().reset();
}

void OrderedExtractionCompletion::failInvalidCompletion() noexcept {
  recordFailure(invalid_completion_);
}

void OrderedExtractionCompletion::complete(uint64_t generation, Output output,
                                           std::exception_ptr failure) noexcept {
  const auto completed = pending_.find(generation);
  if (completed == pending_.end() || completed->second.completed) {
    failInvalidCompletion();
    return;
  }
  recordFailure(failure);
  completed->second.completed = true;
  if (!failure_) completed->second.output.front() = std::move(output);
  // Empty and failed tasks still advance the drain frontier. Splicing the
  // preallocated output node never allocates and preserves generation order.
  while (completed_through_ != std::numeric_limits<uint64_t>::max()) {
    const auto next = pending_.find(completed_through_ + 1);
    if (next == pending_.end() || !next->second.completed) break;
    if (!failure_ && next->second.output.front()) {
      ready_.splice(ready_.end(), next->second.output);
    }
    pending_.erase(next);
    ++completed_through_;
  }
}

void OrderedExtractionCompletion::rethrowFailure() const {
  if (failure_) std::rethrow_exception(failure_);
}

OrderedExtractionCompletion::Outputs OrderedExtractionCompletion::takeReady() {
  rethrowFailure();
  Outputs result;
  result.splice(result.end(), ready_);
  return result;
}

}  // namespace khronos::session_detail
