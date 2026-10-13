#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <vector>

#include "khronos/active_window/object_extraction/object_worker_pool.h"

namespace {

using namespace std::chrono_literals;

struct Gate {
  std::mutex mutex;
  std::condition_variable cv;
  std::atomic<size_t> started{0};
  bool release_slow = false;
};

class DelayedExtractor : public khronos::ObjectExtractor {
 public:
  explicit DelayedExtractor(std::shared_ptr<Gate> gate) : gate_(std::move(gate)) {}

  spark_dsg::KhronosObjectAttributes::Ptr extractObject(
      const khronos::Track& track,
      const khronos::FrameDataBuffer&) override {
    ++gate_->started;
    gate_->cv.notify_all();
    if (track.id <= 2) {
      std::unique_lock<std::mutex> lock(gate_->mutex);
      gate_->cv.wait(lock, [this] { return gate_->release_slow; });
    }

    auto attrs = std::make_unique<spark_dsg::KhronosObjectAttributes>();
    attrs->details["instance_id"] = {static_cast<size_t>(track.id)};
    return attrs;
  }

 private:
  std::shared_ptr<Gate> gate_;
};

bool require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << "\n";
  }
  return condition;
}

std::set<size_t> takeIds(khronos::ObjectWorkerPool& pool) {
  hydra::LayerUpdate update(2);
  pool.fill(update);
  std::set<size_t> result;
  for (const auto& attrs : update.attributes) {
    const auto* object = dynamic_cast<const spark_dsg::KhronosObjectAttributes*>(attrs.get());
    if (!object) {
      continue;
    }
    const auto iter = object->details.find("instance_id");
    if (iter != object->details.end() && iter->second.size() == 1) {
      result.insert(iter->second.front());
    }
  }
  return result;
}

}  // namespace

int main() {
  auto gate = std::make_shared<Gate>();
  khronos::ObjectWorkerPool::Config config;
  config.num_workers = 2;
  config.poll_time_us = 100;
  khronos::ObjectWorkerPool pool(config, std::make_unique<DelayedExtractor>(gate));
  khronos::FrameDataBuffer buffer(khronos::FrameDataBuffer::Config{});

  for (int id = 1; id <= 5; ++id) {
    khronos::Track track;
    track.id = id;
    pool.submit(static_cast<khronos::TimeStamp>(id), track, buffer);
  }

  {
    std::unique_lock<std::mutex> lock(gate->mutex);
    if (!gate->cv.wait_for(lock, 2s, [&] { return gate->started.load() == 2; })) {
      std::cerr << "two configured workers did not start\n";
      return EXIT_FAILURE;
    }
  }

  auto joined = std::async(std::launch::async, [&pool] { pool.join(); });
  if (!require(joined.wait_for(50ms) == std::future_status::timeout,
               "join returned while two workers were blocked and three requests were queued")) {
    return EXIT_FAILURE;
  }

  {
    std::lock_guard<std::mutex> lock(gate->mutex);
    gate->release_slow = true;
  }
  gate->cv.notify_all();
  if (!require(joined.wait_for(2s) == std::future_status::ready,
               "join did not reach the accepted generation")) {
    return EXIT_FAILURE;
  }
  joined.get();

  const auto first_ids = takeIds(pool);
  if (!require(first_ids == std::set<size_t>({1, 2, 3, 4, 5}),
               "terminal fill did not contain every accepted object exactly once")) {
    return EXIT_FAILURE;
  }

  // The barrier is reusable: a later generation must not be confused with the
  // already-completed prefix or leak into the earlier fill.
  khronos::Track track;
  track.id = 6;
  pool.submit(6, track, buffer);
  pool.join();
  if (!require(takeIds(pool) == std::set<size_t>({6}),
               "a later generation was not isolated from the completed prefix")) {
    return EXIT_FAILURE;
  }

  const auto elapsed = hydra::timing::ElapsedTimeRecorder::instance().getLastElapsed(
      "active_window/extract_object");
  if (!require(elapsed && *elapsed > 0.0,
               "object extraction timing must be recorded with a positive duration")) {
    return EXIT_FAILURE;
  }

  pool.stop();
  return EXIT_SUCCESS;
}
