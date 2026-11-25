#pragma once

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>

namespace khronos {

/**
 * A single persistent worker for expensive state refreshes.
 *
 * Requests are level-triggered: while work is executing, any number of new
 * requests collapse into one subsequent execution. This bounds both threads
 * and queued snapshots without losing the newest state. The callback itself is
 * responsible for taking a snapshot when it begins, so coalesced requests do
 * not clone stale graphs.
 */
class LatestOnlyWorker {
 public:
  struct Stats {
    size_t requests = 0;
    size_t executions = 0;
    size_t coalesced_requests = 0;
  };

  explicit LatestOnlyWorker(std::function<void()> callback);
  ~LatestOnlyWorker();

  LatestOnlyWorker(const LatestOnlyWorker&) = delete;
  LatestOnlyWorker& operator=(const LatestOnlyWorker&) = delete;

  void request();
  void waitUntilIdle();
  Stats stats() const;

 private:
  void spin();

  const std::function<void()> callback_;
  mutable std::mutex mutex_;
  std::condition_variable work_cv_;
  std::condition_variable idle_cv_;
  std::thread worker_;
  bool pending_ = false;
  bool executing_ = false;
  bool stopping_ = false;
  std::exception_ptr failure_;
  Stats stats_;
};

}  // namespace khronos
