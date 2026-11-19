#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <glog/logging.h>

#include "khronos/common/common_types.h"

namespace khronos {

/**
 * @brief The trajectory of a physical object's move, written to the object's node of the map.
 *
 * A move ends where the hand-over finds the boundary of the object's current static surface
 * (selectStaticFrames, STATIC_SURFACE_BOUNDARY: the last frame rejected as not that surface). Its
 * positions are the tracker's per-frame centre of the object (Track::last_centroid, the quantity
 * the live observer records). The move starts at the last frame before the end whose smoothed
 * centre (running median) is still farther from the settled position than a fraction of the
 * largest such distance within the time before the end; the settled position is the median
 * centre of the time after the end. This is evaluation/single_run/ours_motion_from_run.py, value
 * for value, computed in the mapper instead of from its log.
 *
 * An output only: the samples are written to the snapshots of the 4D map (Backend), never to the
 * graph the mapping works on, so no reconstruction and no judgement reads them. The centres are
 * kept from the oldest frame a boundary can still be found at, less the time before an end, and
 * no longer.
 */
class MoveTrajectories {
 public:
  struct Config {
    // Whether moves are written at all.
    bool enabled = true;
    // The time before the end of a move its start is looked for in [s].
    double before_s = 10.0;
    // The time after the end whose median centre is the settled position [s].
    double settled_s = 2.0;
    // Frames of the running median the centres before the end are smoothed with.
    int smoothing_frames = 9;
    // The move starts at the last frame still farther from the settled position than this
    // fraction of the largest distance before the end.
    double start_fraction = 0.8;
    // Fewer tracked frames than these after / before the end: no trajectory.
    int min_settled_frames = 5;
    int min_before_frames = 10;
  };

  struct Sample {
    TimeStamp stamp = 0;
    Point position = Point::Zero();
  };

  static void configure(const Config& config) {
    auto& self = instance();
    std::lock_guard<std::mutex> lock(self.mutex_);
    self.config_ = config;
  }

  //! The tracker's centre of a physical instance in a frame.
  static void record(int instance_id, TimeStamp stamp, const Point& centre) {
    auto& self = instance();
    std::lock_guard<std::mutex> lock(self.mutex_);
    if (!self.config_.enabled) return;
    self.centres_[instance_id][stamp] = centre;
  }

  //! A hand-over found the end of a move of the instance at this frame.
  static void markEnd(int instance_id, TimeStamp end) {
    auto& self = instance();
    std::lock_guard<std::mutex> lock(self.mutex_);
    if (!self.config_.enabled || self.written_[instance_id].count(end)) return;
    self.pending_[instance_id].insert(end);
  }

  //! Tell the moves whose time after the end has passed at `now` (or after which no frame comes
  //! any more), each once.
  static void resolve(TimeStamp now) {
    auto& self = instance();
    std::lock_guard<std::mutex> lock(self.mutex_);
    const int64_t settled_ns = nanoseconds(self.config_.settled_s);
    for (auto pending = self.pending_.begin(); pending != self.pending_.end();) {
      for (auto end = pending->second.begin(); end != pending->second.end();) {
        if (!self.finished_ && signedStamp(now) < signedStamp(*end) + settled_ns) {
          ++end;
          continue;
        }
        auto& samples = self.moves_[pending->first];
        const size_t had = samples.size();
        self.move(pending->first, *end, samples);
        if (samples.size() == had && had == 0) self.moves_.erase(pending->first);
        self.written_[pending->first].insert(*end);
        end = pending->second.erase(end);
      }
      pending = pending->second.empty() ? self.pending_.erase(pending) : std::next(pending);
    }
    for (auto& [id, samples] : self.moves_) {
      std::stable_sort(samples.begin(), samples.end(),
                       [](const Sample& a, const Sample& b) { return a.stamp < b.stamp; });
    }
  }

  //! The samples of the moves told so far, by instance, in order of time.
  static std::map<int, std::vector<Sample>> moves() {
    auto& self = instance();
    std::lock_guard<std::mutex> lock(self.mutex_);
    return self.moves_;
  }

  //! No frame comes any more: every end still waiting for its time after is taken as it is.
  static void finish() {
    auto& self = instance();
    std::lock_guard<std::mutex> lock(self.mutex_);
    self.finished_ = true;
  }

  //! The ends still waiting for their time after (instance, end).
  static std::vector<std::pair<int, TimeStamp>> waiting() {
    auto& self = instance();
    std::lock_guard<std::mutex> lock(self.mutex_);
    std::vector<std::pair<int, TimeStamp>> result;
    for (const auto& [id, ends] : self.pending_) {
      for (const TimeStamp end : ends) result.emplace_back(id, end);
    }
    return result;
  }

  //! A boundary is found among the buffered frames only: centres older than the oldest of them
  //! (or the oldest end still waiting) less the time before an end serve no move any more.
  static void forgetBefore(TimeStamp oldest_buffered_frame) {
    auto& self = instance();
    std::lock_guard<std::mutex> lock(self.mutex_);
    int64_t oldest = signedStamp(oldest_buffered_frame);
    for (const auto& [id, ends] : self.pending_) {
      if (!ends.empty()) oldest = std::min(oldest, signedStamp(*ends.begin()));
    }
    const int64_t keep_from = oldest - nanoseconds(self.config_.before_s);
    if (keep_from <= 0) return;
    const TimeStamp first = static_cast<TimeStamp>(keep_from);
    for (auto& [id, centres] : self.centres_) {
      centres.erase(centres.begin(), centres.lower_bound(first));
    }
    for (auto& [id, ends] : self.written_) {
      ends.erase(ends.begin(), ends.lower_bound(first));
    }
  }

 private:
  static MoveTrajectories& instance() {
    static MoveTrajectories self;
    return self;
  }
  static int64_t signedStamp(TimeStamp stamp) { return static_cast<int64_t>(stamp); }
  static int64_t nanoseconds(double seconds) { return std::llround(seconds * 1e9); }
  // The median of each coordinate (of an even number: the mean of the middle two).
  static Eigen::Vector3d median(const std::vector<Eigen::Vector3d>& points, size_t first, size_t last) {
    Eigen::Vector3d result = Eigen::Vector3d::Zero();
    std::vector<double> values(last - first + 1);
    for (int axis = 0; axis < 3; ++axis) {
      for (size_t i = first; i <= last; ++i) values[i - first] = points[i](axis);
      std::sort(values.begin(), values.end());
      const size_t n = values.size();
      result(axis) = n % 2 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
    }
    return result;
  }

  // The samples of the move of the instance that ends at `end`, appended (mutex held).
  void move(int instance_id, TimeStamp end, std::vector<Sample>& samples) const {
    const auto found = centres_.find(instance_id);
    const int64_t e = signedStamp(end);
    const int64_t before_ns = nanoseconds(config_.before_s), settled_ns = nanoseconds(config_.settled_s);
    std::vector<TimeStamp> stamps;       // the frames before the end, in order of time
    std::vector<Eigen::Vector3d> before, after;
    std::vector<Point> measured;
    if (found != centres_.end()) {
      for (const auto& [stamp, centre] : found->second) {
        const int64_t t = signedStamp(stamp);
        if (t < e && t >= e - before_ns) {
          stamps.push_back(stamp);
          before.push_back(centre.cast<double>());
          measured.push_back(centre);
        } else if (t >= e && t <= e + settled_ns) {
          after.push_back(centre.cast<double>());
        }
      }
    }
    if (static_cast<int>(after.size()) < config_.min_settled_frames ||
        static_cast<int>(before.size()) < config_.min_before_frames) {
      LOG(INFO) << "MOVE_TRAJECTORY inst=" << instance_id << " end=" << end
                << " written=0 why=too_few_tracked_frames before=" << before.size()
                << " after=" << after.size();
      return;
    }
    const Eigen::Vector3d settled = median(after, 0, after.size() - 1);
    const size_t half = static_cast<size_t>(std::max(0, config_.smoothing_frames / 2));
    std::vector<double> distance(before.size());
    double largest = 0.0;
    for (size_t i = 0; i < before.size(); ++i) {
      const size_t first = i >= half ? i - half : 0;
      const size_t last = std::min(before.size() - 1, i + half);
      distance[i] = (median(before, first, last) - settled).norm();
      largest = std::max(largest, distance[i]);
    }
    size_t start = 0;
    for (size_t i = 0; i < before.size(); ++i) {
      if (distance[i] >= config_.start_fraction * largest) start = i;
    }
    for (size_t i = start; i < before.size(); ++i) {
      samples.push_back({stamps[i], measured[i]});
    }
    LOG(INFO) << "MOVE_TRAJECTORY inst=" << instance_id << " end=" << end << " written=1 start="
              << stamps[start] << " frames=" << before.size() - start << " distance=" << largest;
  }

  std::mutex mutex_;
  Config config_;
  bool finished_ = false;
  std::map<int, std::map<TimeStamp, Point>> centres_;   // instance -> frame -> the tracker's centre
  std::map<int, std::set<TimeStamp>> pending_;          // ends waiting for their time after
  std::map<int, std::set<TimeStamp>> written_;          // ends told
  std::map<int, std::vector<Sample>> moves_;            // the samples of the moves told
};

}  // namespace khronos
