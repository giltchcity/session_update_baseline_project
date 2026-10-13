#include <algorithm>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "khronos/active_window/data/frame_data_buffer.h"

namespace {

using khronos::FrameData;
using khronos::FrameDataBuffer;
using khronos::Observation;
using khronos::TimeStamp;
using khronos::Track;
using khronos::Tracks;

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << "\n";
    std::exit(EXIT_FAILURE);
  }
}

FrameData::Ptr makeFrame(TimeStamp stamp) {
  hydra::InputData input(nullptr);
  input.timestamp_ns = stamp;
  return std::make_shared<FrameData>(input);
}

// This is the pre-optimization implementation, retained in the test as a
// behavioral oracle. It intentionally performs the original linear searches.
class ReferenceBuffer {
 public:
  explicit ReferenceBuffer(const FrameDataBuffer::Config& config) : config_(config) {}

  void storeData(const FrameData::Ptr& data) {
    const bool store_data = input_counter_ == 0;
    input_counter_ = (input_counter_ + 1) % config_.store_every_n_frames;
    if (!store_data) {
      if (!buffer_.empty()) {
        buffer_.pop_back();
      }
      buffer_.emplace_back(data);
      return;
    }

    buffer_.emplace_back(data);
    if (buffer_.size() > config_.max_buffer_size) {
      buffer_.pop_front();
      oldest_time_stamp_ = buffer_.front()->input.timestamp_ns;
    }
  }

  void trimBuffer(const Tracks& tracks) {
    auto iter = buffer_.begin();
    while (iter != buffer_.end()) {
      const auto stamp = (*iter)->input.timestamp_ns;
      bool referenced = false;
      for (const auto& track : tracks) {
        referenced = std::find_if(track.observations.begin(),
                                  track.observations.end(),
                                  [stamp](const Observation& observation) {
                                    return observation.stamp == stamp;
                                  }) != track.observations.end();
        if (referenced) {
          break;
        }
      }

      if (referenced) {
        ++iter;
      } else {
        iter = buffer_.erase(iter);
      }
    }

    if (!buffer_.empty()) {
      oldest_time_stamp_ = buffer_.front()->input.timestamp_ns;
    }
  }

  FrameData::Ptr getData(TimeStamp stamp) const {
    if (stamp < oldest_time_stamp_) {
      return nullptr;
    }
    const auto iter = std::find_if(buffer_.begin(), buffer_.end(), [stamp](const auto& data) {
      return data->input.timestamp_ns == stamp;
    });
    return iter == buffer_.end() ? nullptr : *iter;
  }

  const std::deque<FrameData::Ptr>& data() const { return buffer_; }

 private:
  FrameDataBuffer::Config config_;
  std::deque<FrameData::Ptr> buffer_;
  TimeStamp oldest_time_stamp_ = 0;
  int input_counter_ = 0;
};

std::vector<FrameData::Ptr> contents(const FrameDataBuffer& buffer) {
  return {buffer.begin(), buffer.end()};
}

void requireEquivalent(const ReferenceBuffer& reference,
                       const FrameDataBuffer& actual,
                       const std::vector<TimeStamp>& queries,
                       const std::string& context) {
  const std::vector<FrameData::Ptr> expected(reference.data().begin(), reference.data().end());
  require(contents(actual) == expected, context + ": ordered buffer contents differ");
  require(actual.size() == expected.size(), context + ": buffer sizes differ");
  for (const auto stamp : queries) {
    require(actual.getData(stamp) == reference.getData(stamp),
            context + ": getData differs for stamp " + std::to_string(stamp));
  }
}

Track makeTrack(int id, const std::vector<TimeStamp>& stamps) {
  Track track;
  track.id = id;
  for (const auto stamp : stamps) {
    track.observations.emplace_back(stamp);
  }
  return track;
}

void testStoreEveryAndCapacityAgainstOracle() {
  FrameDataBuffer::Config config;
  config.max_buffer_size = 3;
  config.store_every_n_frames = 3;
  ReferenceBuffer reference(config);
  FrameDataBuffer actual(config);
  const std::vector<TimeStamp> stamps = {10, 20, 20, 30, 40, 50, 60, 60, 70, 80};
  const std::vector<TimeStamp> queries = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90};
  for (size_t i = 0; i < stamps.size(); ++i) {
    const auto frame = makeFrame(stamps[i]);
    reference.storeData(frame);
    actual.storeData(frame);
    requireEquivalent(reference, actual, queries, "store step " + std::to_string(i));
  }
}

void testDuplicateStampUsesFirstBufferedFrame() {
  FrameDataBuffer::Config config;
  config.max_buffer_size = 3;
  config.store_every_n_frames = 1;
  ReferenceBuffer reference(config);
  FrameDataBuffer actual(config);
  const auto first = makeFrame(10);
  const auto second = makeFrame(10);
  const auto third = makeFrame(20);
  for (const auto& frame : {first, second, third}) {
    reference.storeData(frame);
    actual.storeData(frame);
  }
  require(actual.getData(10) == first, "duplicate stamp must resolve to its first buffered frame");

  const Tracks tracks = {makeTrack(1, {10})};
  reference.trimBuffer(tracks);
  actual.trimBuffer(tracks);
  requireEquivalent(reference, actual, {10, 20}, "duplicate trim");
  require(actual.size() == 2, "all buffered occurrences of a referenced stamp must survive trim");

  const auto later = makeFrame(30);
  reference.storeData(later);
  actual.storeData(later);
  const auto newest = makeFrame(40);
  reference.storeData(newest);
  actual.storeData(newest);
  requireEquivalent(reference, actual, {10, 20, 30, 40}, "duplicate capacity eviction");
  require(actual.getData(10) == second,
          "evicting the first duplicate must expose the next buffered occurrence");
}

void testLongHistoryTrimAgainstOracle() {
  FrameDataBuffer::Config config;
  config.max_buffer_size = 100;
  config.store_every_n_frames = 1;
  ReferenceBuffer reference(config);
  FrameDataBuffer actual(config);
  for (TimeStamp stamp = 9900; stamp < 10000; ++stamp) {
    const auto frame = makeFrame(stamp);
    reference.storeData(frame);
    actual.storeData(frame);
  }

  std::vector<TimeStamp> long_history;
  long_history.reserve(10000);
  for (TimeStamp stamp = 0; stamp < 10000; ++stamp) {
    if (stamp % 7 == 0 || stamp == 9999) {
      long_history.push_back(stamp);
    }
  }
  Tracks tracks;
  tracks.emplace_back(makeTrack(1, long_history));
  tracks.emplace_back(makeTrack(2, {9951, 9973, 9991}));
  reference.trimBuffer(tracks);
  actual.trimBuffer(tracks);

  std::vector<TimeStamp> queries;
  for (TimeStamp stamp = 9895; stamp < 10005; ++stamp) {
    queries.push_back(stamp);
  }
  requireEquivalent(reference, actual, queries, "long-history trim");
}

void testCopiedBufferPreservesWorkerLookupSnapshot() {
  FrameDataBuffer::Config config;
  config.max_buffer_size = 4;
  config.store_every_n_frames = 1;
  FrameDataBuffer live(config);
  const auto first = makeFrame(10);
  const auto duplicate = makeFrame(10);
  live.storeData(first);
  live.storeData(duplicate);
  live.storeData(makeFrame(20));

  const FrameDataBuffer worker_snapshot = live;
  live.storeData(makeFrame(30));
  live.storeData(makeFrame(40));
  require(worker_snapshot.size() == 3, "worker snapshot size changed with live buffer");
  require(worker_snapshot.getData(10) == first,
          "worker snapshot changed duplicate timestamp lookup result");
  require(worker_snapshot.getData(20) != nullptr, "worker snapshot lost retained frame");
  require(worker_snapshot.getData(30) == nullptr, "worker snapshot observed a later live frame");
}

}  // namespace

int main() {
  testStoreEveryAndCapacityAgainstOracle();
  testDuplicateStampUsesFirstBufferedFrame();
  testLongHistoryTrimAgainstOracle();
  testCopiedBufferPreservesWorkerLookupSnapshot();
  return EXIT_SUCCESS;
}
