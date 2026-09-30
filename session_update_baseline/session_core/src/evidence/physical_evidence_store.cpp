#include "session_core/evidence/range_encoding.h"
/** -----------------------------------------------------------------------------
 * Copyright (c) 2024 Massachusetts Institute of Technology.
 * All Rights Reserved.
 * -------------------------------------------------------------------------- */

#include "session_core/evidence/physical_evidence_store.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include <yaml-cpp/yaml.h>
#include <hydra/common/global_info.h>
#include <hydra/input/sensor.h>

#include "khronos/active_window/data/frame_data.h"

namespace khronos {
namespace {

constexpr int32_t kInvalidCode = std::numeric_limits<int32_t>::min();
constexpr int32_t kUnidentifiedObjectCode = -1;
constexpr int32_t kBackgroundCode = 0;

struct Run {
  // Exclusive flattened-pixel end offset.
  uint32_t end = 0;
  int32_t value = kInvalidCode;
};

struct DepthRun {
  // Exclusive flattened-pixel end offset.
  uint32_t end = 0;
  // Depth in millimetres. 0 means invalid/unavailable.
  uint16_t depth_mm = 0;
};

struct FrameEvidence {
  uint32_t width = 0;
  uint32_t height = 0;
  // Keep the input pose for exact duplicate comparison before float projection.
  Eigen::Isometry3d world_T_sensor = Eigen::Isometry3d::Identity();
  Eigen::Isometry3f sensor_T_world = Eigen::Isometry3f::Identity();
  hydra::Sensor::ConstPtr sensor;
  std::vector<Run> runs;
  std::vector<DepthRun> depth_runs;
};

// Sensors retain immutable configuration. Normally both submissions share the
// same registered sensor; equivalent reconstructed camera objects may also be
// accepted using Hydra's complete virtual configuration serialization.
bool sameSensor(const hydra::Sensor::ConstPtr& first, const hydra::Sensor::ConstPtr& second) {
  if (first == second) return true;
  return first && second && typeid(*first) == typeid(*second) && first->name == second->name &&
      YAML::Dump(first->dump()) == YAML::Dump(second->dump());
}

bool sameMeasurement(const FrameEvidence& first, const FrameEvidence& second) {
  return first.width == second.width && first.height == second.height &&
      (first.world_T_sensor.matrix().array() == second.world_T_sensor.matrix().array()).all() &&
      sameSensor(first.sensor,second.sensor) &&
      std::equal(first.runs.begin(),first.runs.end(),second.runs.begin(),second.runs.end(),
                 [](const Run& a, const Run& b) { return a.end == b.end && a.value == b.value; }) &&
      std::equal(first.depth_runs.begin(),first.depth_runs.end(),
                 second.depth_runs.begin(),second.depth_runs.end(),
                 [](const DepthRun& a, const DepthRun& b) {
                   return a.end == b.end && a.depth_mm == b.depth_mm;
                 });
}

bool sameSize(const cv::Mat& image, int rows, int cols) {
  return image.empty() || (image.rows == rows && image.cols == cols);
}

bool isFinitePoint(const Point& point) {
  return point.array().isFinite().all();
}

}  // namespace

struct PhysicalEvidenceStore::Storage {
  std::map<TimeStamp, std::shared_ptr<const FrameEvidence>> frames;
  size_t num_runs = 0;
  size_t num_depth_runs = 0;
};

PhysicalEvidenceStore::Snapshot::Snapshot(std::shared_ptr<const Storage> storage, TimeStamp latest)
    : storage_(std::move(storage)), latest_(latest) {}

EndpointEvidence PhysicalEvidenceStore::Snapshot::classify(
    TimeStamp stamp, const Point& world_point) const {
  return project(stamp, world_point).endpoint;
}

std::vector<TimeStamp> PhysicalEvidenceStore::Snapshot::timestamps(
    TimeStamp earliest, TimeStamp latest) const {
  std::vector<TimeStamp> result;
  latest = std::min(latest, latest_);
  if (!storage_ || earliest > latest) return result;
  for (auto it = storage_->frames.lower_bound(earliest);
       it != storage_->frames.end() && it->first <= latest; ++it) {
    result.push_back(it->first);
  }
  return result;
}

bool PhysicalEvidenceStore::Snapshot::denseRange(TimeStamp stamp,
                                                  uint32_t& width,
                                                  uint32_t& height,
                                                  Eigen::Isometry3f& sensor_T_world,
                                                  hydra::Sensor::ConstPtr& sensor,
                                                  std::vector<uint16_t>& range_mm) const {
  if (!storage_ || stamp > latest_) {
    return false;
  }
  const auto frame_it = storage_->frames.find(stamp);
  if (frame_it == storage_->frames.end()) {
    return false;
  }
  const auto& frame = *frame_it->second;
  width = frame.width;
  height = frame.height;
  sensor_T_world = frame.sensor_T_world;
  sensor = frame.sensor;
  const size_t n = static_cast<size_t>(width) * height;
  range_mm.assign(n, 0);
  size_t start = 0;
  for (const auto& run : frame.depth_runs) {
    const size_t end = std::min<size_t>(run.end, n);
    if (end > start) {
      std::fill(range_mm.begin() + start, range_mm.begin() + end, run.depth_mm);
    }
    start = std::max(start, end);
  }
  return true;
}

ProjectedEndpointEvidence PhysicalEvidenceStore::Snapshot::project(
    TimeStamp stamp, const Point& world_point) const {
  if (!storage_ || stamp > latest_) {
    return {};
  }

  const auto frame_it = storage_->frames.find(stamp);
  if (frame_it == storage_->frames.end()) {
    return {};
  }

  const auto& frame = *frame_it->second;
  if (!frame.sensor || !isFinitePoint(world_point)) {
    return {};
  }

  const Eigen::Vector3f sensor_point = frame.sensor_T_world * world_point;
  if (!sensor_point.array().isFinite().all()) {
    return {};
  }

  int u = -1;
  int v = -1;
  if (!frame.sensor->projectPointToImagePlane(sensor_point, u, v) || u < 0 || v < 0 ||
      static_cast<uint32_t>(u) >= frame.width ||
      static_cast<uint32_t>(v) >= frame.height) {
    return {};
  }

  const uint32_t index = static_cast<uint32_t>(v) * frame.width +
                         static_cast<uint32_t>(u);
  const auto run_it = std::upper_bound(
      frame.runs.begin(),
      frame.runs.end(),
      index,
      [](uint32_t pixel, const Run& run) { return pixel < run.end; });
  if (run_it == frame.runs.end()) {
    return {};
  }

  float measured_depth = std::numeric_limits<float>::quiet_NaN();
  const auto depth_it = std::upper_bound(
      frame.depth_runs.begin(),
      frame.depth_runs.end(),
      index,
      [](uint32_t pixel, const DepthRun& run) { return pixel < run.end; });
  if (depth_it != frame.depth_runs.end() && depth_it->depth_mm > 0) {
    measured_depth = static_cast<float>(depth_it->depth_mm) / 1000.0f;
  }

  ProjectedEndpointEvidence projection;
  projection.query_range_m = sensor_point.norm();
  projection.view_direction_world =
      frame.sensor_T_world.linear().transpose() * sensor_point.normalized();
  projection.pixel_index = index;
  projection.sensor_min_range = frame.sensor->min_range();
  projection.sensor_max_range = frame.sensor->max_range();
  auto& result = projection.endpoint;
  result.measured_depth_m = measured_depth;
  if (run_it->value == kInvalidCode) {
    result.type = EndpointClass::kInvalid;
    return projection;
  }
  if (run_it->value == kUnidentifiedObjectCode) {
    result.type = EndpointClass::kUnidentifiedObject;
    return projection;
  }
  if (run_it->value == kBackgroundCode) {
    result.type = EndpointClass::kBackground;
    return projection;
  }
  if (run_it->value > 0) {
    result.type = EndpointClass::kPhysical;
    result.physical_id = run_it->value;
  }
  return projection;
}

size_t PhysicalEvidenceStore::Snapshot::numFrames() const {
  if (!storage_) return 0;
  return static_cast<size_t>(
      std::distance(storage_->frames.begin(), storage_->frames.upper_bound(latest_)));
}

size_t PhysicalEvidenceStore::Snapshot::numRuns() const {
  if (!storage_) return 0;
  size_t count = 0;
  for (auto it = storage_->frames.begin();
       it != storage_->frames.end() && it->first <= latest_; ++it) {
    count += it->second->runs.size();
  }
  return count;
}

PhysicalEvidenceStore::PhysicalEvidenceStore()
    : storage_(std::make_shared<Storage>()) {}

bool PhysicalEvidenceStore::ingest(const FrameData& data) {
  const auto& input = data.input;
  const cv::Mat& ranges = input.range_image;
  if (ranges.empty() || ranges.type() != CV_32FC1 || ranges.rows <= 0 ||
      ranges.cols <= 0) {
    throw std::invalid_argument("Evidence frame requires a non-empty CV_32FC1 range image");
  }

  const uint64_t pixel_count = static_cast<uint64_t>(ranges.rows) *
      static_cast<uint64_t>(ranges.cols);
  // UINT32_MAX is the unavailable-pixel sentinel; the exclusive RLE end may
  // equal it, while every real pixel index must remain strictly smaller.
  if (pixel_count > std::numeric_limits<uint32_t>::max())
    throw std::length_error("Evidence image exceeds the uint32 pixel domain");

  if (!sameSize(input.label_image, ranges.rows, ranges.cols) ||
      !sameSize(data.instance_image, ranges.rows, ranges.cols) ||
      !sameSize(data.dynamic_image, ranges.rows, ranges.cols)) {
    throw std::invalid_argument("Evidence image dimensions do not match the range image");
  }
  if ((!input.label_image.empty() && input.label_image.type() != CV_32SC1) ||
      (!data.instance_image.empty() && data.instance_image.type() != CV_32SC1) ||
      (!data.dynamic_image.empty() && data.dynamic_image.type() != CV_32SC1)) {
    throw std::invalid_argument("Evidence label, instance, and dynamic images must be CV_32SC1");
  }

  auto sensor = hydra::GlobalInfo::instance().getSensor(input.getSensor().name);
  if (!sensor) {
    throw std::invalid_argument("Evidence sensor is unavailable: " + input.getSensor().name);
  }

  auto frame = std::make_shared<FrameEvidence>();
  frame->width = static_cast<uint32_t>(ranges.cols);
  frame->height = static_cast<uint32_t>(ranges.rows);
  frame->world_T_sensor = input.getSensorPose();
  if (!frame->world_T_sensor.matrix().allFinite())
    throw std::invalid_argument("Evidence sensor pose is non-finite");
  frame->sensor_T_world = frame->world_T_sensor.cast<float>().inverse();
  if (!frame->sensor_T_world.matrix().allFinite())
    throw std::invalid_argument("Evidence sensor pose exceeds the projection numeric range");
  frame->sensor = std::move(sensor);
  frame->runs.reserve(static_cast<size_t>(pixel_count) / 8 + 1);

  int32_t previous = 0;
  uint16_t previous_depth_mm = 0;
  bool have_previous = false;
  bool have_previous_depth = false;
  uint32_t offset = 0;
  for (int v = 0; v < ranges.rows; ++v) {
    for (int u = 0; u < ranges.cols; ++u, ++offset) {
      const float range = ranges.at<float>(v, u);
      const uint16_t depth_mm = measurement::encodeRange(
          range, input.getSensor().min_range(), input.getSensor().max_range());
      if (have_previous_depth && depth_mm != previous_depth_mm) {
        frame->depth_runs.push_back({offset, previous_depth_mm});
      }
      previous_depth_mm = depth_mm;
      have_previous_depth = true;

      int32_t code = kInvalidCode;
      if (depth_mm) {
        const int physical_id = data.instance_image.empty()
                                    ? 0
                                    : data.instance_image.at<FrameData::InstanceImageType>(v, u);
        if (physical_id > 0) {
          code = measurement::encodeIdentity(physical_id);
        } else {
          // README (6s) assumption 5: a hit without a physical label is a missing label, whether
          // it is background or an unidentified moving pixel; the semantic class plays no role.
          const bool dynamic = !data.dynamic_image.empty() &&
              data.dynamic_image.at<FrameData::DynamicImageType>(v, u) != 0;
          code = dynamic ? kUnidentifiedObjectCode : kBackgroundCode;
        }
      }

      if (have_previous && code != previous) {
        frame->runs.push_back({offset, previous});
      }
      previous = code;
      have_previous = true;
    }
  }
  if (have_previous) {
    frame->runs.push_back({offset, previous});
  }
  if (have_previous_depth) {
    frame->depth_runs.push_back({offset, previous_depth_mm});
  }

  IngestObserver observer;
  {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto existing = storage_->frames.find(input.timestamp_ns);
  if (existing != storage_->frames.end()) {
    if (!sameMeasurement(*existing->second,*frame))
      throw std::invalid_argument("Conflicting evidence measurement at timestamp " +
                                  std::to_string(input.timestamp_ns));
    // Normal output and terminal extraction can submit the same latest frame.
    // Preserve its original payload, snapshots and counters on an exact replay.
    return true;
  }
  // With no live snapshot the index has one owner and can be updated in place.
  // Otherwise copy only the index; immutable frame payloads remain shared.
  auto next = storage_.unique() ? storage_ : std::make_shared<Storage>(*storage_);
  next->frames.emplace(input.timestamp_ns,frame);
  next->num_runs += frame->runs.size();
  next->num_depth_runs += frame->depth_runs.size();
  storage_ = std::move(next);
  observer = observer_;
  }
  if (observer) observer(snapshot(input.timestamp_ns), input.timestamp_ns);
  return true;
}

void PhysicalEvidenceStore::setIngestObserver(IngestObserver observer) {
  std::lock_guard<std::mutex> lock(mutex_);
  observer_ = std::move(observer);
}

void PhysicalEvidenceStore::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  storage_ = std::make_shared<Storage>();
}

PhysicalEvidenceStore::Snapshot PhysicalEvidenceStore::snapshot(TimeStamp latest) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return Snapshot(storage_, latest);
}

size_t PhysicalEvidenceStore::numFrames() const { return snapshot().numFrames(); }

size_t PhysicalEvidenceStore::numRuns() const { return snapshot().numRuns(); }

}  // namespace khronos
