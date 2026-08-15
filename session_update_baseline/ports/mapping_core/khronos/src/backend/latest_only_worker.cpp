#include "khronos/backend/latest_only_worker.h"

#include <stdexcept>
#include <utility>

namespace khronos {

LatestOnlyWorker::LatestOnlyWorker(std::function<void()> callback)
    : callback_(std::move(callback)) {
  if (!callback_) {
    throw std::invalid_argument("LatestOnlyWorker requires a callback");
  }
  worker_ = std::thread(&LatestOnlyWorker::spin, this);
}

LatestOnlyWorker::~LatestOnlyWorker() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  work_cv_.notify_one();
  if (worker_.joinable()) {
    worker_.join();
  }
}

void LatestOnlyWorker::request() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (failure_) {
      std::rethrow_exception(failure_);
    }
    if (stopping_) {
      throw std::logic_error("cannot request work after shutdown");
    }
    ++stats_.requests;
    if (pending_ || executing_) {
      ++stats_.coalesced_requests;
    }
    pending_ = true;
  }
  work_cv_.notify_one();
}

void LatestOnlyWorker::waitUntilIdle() {
  std::unique_lock<std::mutex> lock(mutex_);
  idle_cv_.wait(lock, [this] { return failure_ || (!pending_ && !executing_); });
  if (failure_) {
    std::rethrow_exception(failure_);
  }
}

LatestOnlyWorker::Stats LatestOnlyWorker::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_;
}

void LatestOnlyWorker::spin() {
  while (true) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      work_cv_.wait(lock, [this] { return pending_ || stopping_; });
      if (stopping_ && !pending_) {
        return;
      }
      pending_ = false;
      executing_ = true;
      ++stats_.executions;
    }

    try {
      callback_();
    } catch (...) {
      std::lock_guard<std::mutex> lock(mutex_);
      failure_ = std::current_exception();
      pending_ = false;
      executing_ = false;
      stopping_ = true;
      idle_cv_.notify_all();
      return;
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      executing_ = false;
      if (!pending_) {
        idle_cv_.notify_all();
      }
    }
  }
}

}  // namespace khronos
