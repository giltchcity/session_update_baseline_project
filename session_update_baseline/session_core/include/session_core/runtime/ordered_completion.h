#pragma once

#include <cstdint>
#include <exception>
#include <list>
#include <map>

#include <spark_dsg/node_attributes.h>

namespace khronos::session_detail {

// The owner serializes every call under its state mutex. Reserve before making
// a request visible to workers, so completing an accepted request cannot allocate.
class OrderedExtractionCompletion {
 public:
  using Output = spark_dsg::NodeAttributes::Ptr;
  using Outputs = std::list<Output>;

  OrderedExtractionCompletion();
  void reserve(uint64_t generation);
  // Only withdraw the last reservation, before its request has been queued.
  void cancelReservation(uint64_t generation) noexcept;
  void complete(uint64_t generation, Output output, std::exception_ptr failure) noexcept;
  void failInvalidCompletion() noexcept;
  void recordFailure(std::exception_ptr failure) noexcept;
  uint64_t completedThrough() const { return completed_through_; }
  std::exception_ptr failure() const { return failure_; }
  void rethrowFailure() const;
  Outputs takeReady();

 private:
  struct Slot {
    Slot() : output(1) {}
    bool completed = false;
    Outputs output;
  };

  uint64_t completed_through_ = 0;
  std::map<uint64_t, Slot> pending_;
  Outputs ready_;
  std::exception_ptr failure_;
  const std::exception_ptr invalid_completion_;
};

}  // namespace khronos::session_detail
