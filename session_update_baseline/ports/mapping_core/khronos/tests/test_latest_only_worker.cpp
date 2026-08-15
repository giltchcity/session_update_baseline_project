#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "khronos/backend/latest_only_worker.h"

int main() {
  std::mutex gate_mutex;
  std::condition_variable gate_cv;
  bool release_first = false;
  std::atomic<size_t> calls{0};
  std::atomic<size_t> in_flight{0};
  std::atomic<size_t> max_in_flight{0};
  std::atomic<bool> pending_loopclosure{false};
  std::vector<bool> observed_loopclosures;

  khronos::LatestOnlyWorker worker([&] {
    observed_loopclosures.push_back(pending_loopclosure.exchange(false));
    const size_t active = ++in_flight;
    max_in_flight = std::max(max_in_flight.load(), active);
    const size_t call = ++calls;
    if (call == 1) {
      std::unique_lock<std::mutex> lock(gate_mutex);
      gate_cv.wait(lock, [&] { return release_first; });
    }
    --in_flight;
  });

  worker.request();
  while (calls.load() == 0) {
    std::this_thread::yield();
  }
  // All of these arrive while the first callback is blocked. They must become
  // exactly one follow-up callback, never N snapshots or N threads. The sticky
  // loop-closure flag mirrors Backend's coalesced request latch: one true input
  // must survive all the false periodic requests until the follow-up snapshot.
  pending_loopclosure.store(true);
  for (size_t i = 0; i < 100; ++i) {
    worker.request();
  }
  {
    std::lock_guard<std::mutex> lock(gate_mutex);
    release_first = true;
  }
  gate_cv.notify_one();
  worker.waitUntilIdle();

  const auto stats = worker.stats();
  if (calls != 2 || max_in_flight != 1 || stats.requests != 101 ||
      stats.executions != 2 || stats.coalesced_requests != 100 ||
      observed_loopclosures != std::vector<bool>({false, true})) {
    std::cerr << "calls=" << calls << " max_in_flight=" << max_in_flight
              << " requests=" << stats.requests << " executions=" << stats.executions
              << " coalesced=" << stats.coalesced_requests
              << " loopclosure_observations=" << observed_loopclosures.size() << "\n";
    return EXIT_FAILURE;
  }

  // A worker failure must be delivered to the waiter, never leave terminal
  // finalization stuck waiting on an executing flag that can no longer clear.
  khronos::LatestOnlyWorker failing_worker([] { throw std::runtime_error("expected"); });
  failing_worker.request();
  try {
    failing_worker.waitUntilIdle();
    std::cerr << "worker exception was not propagated\n";
    return EXIT_FAILURE;
  } catch (const std::runtime_error& error) {
    if (std::string(error.what()) != "expected") {
      std::cerr << "unexpected worker error: " << error.what() << "\n";
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}
