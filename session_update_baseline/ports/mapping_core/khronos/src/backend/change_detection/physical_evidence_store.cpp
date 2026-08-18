/** -----------------------------------------------------------------------------
 * Copyright (c) 2024 Massachusetts Institute of Technology.
 * All Rights Reserved.
 * -------------------------------------------------------------------------- */

#include "khronos/backend/change_detection/physical_evidence_store.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <utility>
#include <vector>

#include <glog/logging.h>
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
  Eigen::Isometry3f sensor_T_world = Eigen::Isometry3f::Identity();
  hydra::Sensor::ConstPtr sensor;
  std::vector<Run> runs;
  std::vector<DepthRun> depth_runs;
};

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

PhysicalEvidenceStore::Snapshot::Snapshot(std::shared_ptr<const Storage> storage)
    : storage_(std::move(storage)) {}

EndpointEvidence PhysicalEvidenceStore::Snapshot::classify(
    TimeStamp stamp, const Point& world_point) const {
  if (!storage_) {
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

  EndpointEvidence result;
  result.measured_depth_m = measured_depth;
  if (run_it->value == kInvalidCode) {
    result.type = EndpointClass::kInvalid;
    return result;
  }
  if (run_it->value == kUnidentifiedObjectCode) {
    result.type = EndpointClass::kUnidentifiedObject;
    return result;
  }
  if (run_it->value == kBackgroundCode) {
    result.type = EndpointClass::kBackground;
    return result;
  }
  if (run_it->value > 0) {
    result.type = EndpointClass::kPhysical;
    result.physical_id = run_it->value;
  }
  return result;
}

size_t PhysicalEvidenceStore::Snapshot::numFrames() const {
  return storage_ ? storage_->frames.size() : 0;
}

size_t PhysicalEvidenceStore::Snapshot::numRuns() const {
  return storage_ ? storage_->num_runs : 0;
}

PhysicalEvidenceStore::PhysicalEvidenceStore()
    : storage_(std::make_shared<const Storage>()) {}

bool PhysicalEvidenceStore::ingest(const FrameData& data) {
  const auto& input = data.input;
  const cv::Mat& ranges = input.range_image;
  if (ranges.empty() || ranges.type() != CV_32FC1 || ranges.rows <= 0 ||
      ranges.cols <= 0) {
    LOG(WARNING) << "[PhysicalEvidenceStore] Cannot ingest frame " << input.timestamp_ns
                 << ": expected a non-empty CV_32FC1 range image.";
    return false;
  }

  if (!sameSize(input.label_image, ranges.rows, ranges.cols) ||
      !sameSize(data.instance_image, ranges.rows, ranges.cols) ||
      !sameSize(data.dynamic_image, ranges.rows, ranges.cols)) {
    LOG(WARNING) << "[PhysicalEvidenceStore] Cannot ingest frame " << input.timestamp_ns
                 << ": evidence image dimensions do not match the range image.";
    return false;
  }
  if ((!input.label_image.empty() && input.label_image.type() != CV_32SC1) ||
      (!data.instance_image.empty() && data.instance_image.type() != CV_32SC1) ||
      (!data.dynamic_image.empty() && data.dynamic_image.type() != CV_32SC1)) {
    LOG(WARNING) << "[PhysicalEvidenceStore] Cannot ingest frame " << input.timestamp_ns
                 << ": label, instance, and dynamic images must be CV_32SC1.";
    return false;
  }

  auto sensor = hydra::GlobalInfo::instance().getSensor(input.getSensor().name);
  if (!sensor) {
    LOG(WARNING) << "[PhysicalEvidenceStore] Cannot ingest frame " << input.timestamp_ns
                 << ": sensor '" << input.getSensor().name << "' is unavailable.";
    return false;
  }

  auto frame = std::make_shared<FrameEvidence>();
  frame->width = static_cast<uint32_t>(ranges.cols);
  frame->height = static_cast<uint32_t>(ranges.rows);
  frame->sensor_T_world = input.getSensorPose().cast<float>().inverse();
  frame->sensor = std::move(sensor);
  frame->runs.reserve(static_cast<size_t>(ranges.rows) * ranges.cols / 8 + 1);

  const auto& label_space = hydra::GlobalInfo::instance().getLabelSpaceConfig();
  int32_t previous = 0;
  uint16_t previous_depth_mm = 0;
  bool have_previous = false;
  bool have_previous_depth = false;
  uint32_t offset = 0;
  for (int v = 0; v < ranges.rows; ++v) {
    for (int u = 0; u < ranges.cols; ++u, ++offset) {
      const float range = ranges.at<float>(v, u);
      uint16_t depth_mm = 0;
      if (std::isfinite(range) && range > 0.0f && input.inRange(range)) {
        const uint32_t mm = static_cast<uint32_t>(std::lround(range * 1000.0f));
        if (mm > 0 && mm <= std::numeric_limits<uint16_t>::max()) {
          depth_mm = static_cast<uint16_t>(mm);
        }
      }
      if (have_previous_depth && depth_mm != previous_depth_mm) {
        frame->depth_runs.push_back({offset, previous_depth_mm});
      }
      previous_depth_mm = depth_mm;
      have_previous_depth = true;

      int32_t code = kInvalidCode;
      if (std::isfinite(range) && range > 0.0f && input.inRange(range)) {
        const int physical_id = data.instance_image.empty()
                                    ? 0
                                    : data.instance_image.at<FrameData::InstanceImageType>(v, u);
        if (physical_id > 0) {
          code = physical_id;
        } else {
          const bool dynamic = !data.dynamic_image.empty() &&
              data.dynamic_image.at<FrameData::DynamicImageType>(v, u) != 0;
          const int semantic_id = input.label_image.empty()
                                      ? 0
                                      : input.label_image.at<InputData::LabelType>(v, u);
          const bool semantic_object = semantic_id >= 0 &&
              (label_space.isObject(static_cast<uint32_t>(semantic_id)) ||
               label_space.isDynamic(static_cast<uint32_t>(semantic_id)));
          code = (dynamic || semantic_object) ? kUnidentifiedObjectCode
                                              : kBackgroundCode;
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

  std::lock_guard<std::mutex> lock(mutex_);
  auto next = std::make_shared<Storage>(*storage_);
  const auto existing = next->frames.find(input.timestamp_ns);
  if (existing != next->frames.end()) {
    next->num_runs -= existing->second->runs.size();
    next->num_depth_runs -= existing->second->depth_runs.size();
  }
  next->frames[input.timestamp_ns] = frame;
  next->num_runs += frame->runs.size();
  next->num_depth_runs += frame->depth_runs.size();
  storage_ = std::move(next);
  return true;
}

PhysicalEvidenceStore::Snapshot PhysicalEvidenceStore::snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return Snapshot(storage_);
}

size_t PhysicalEvidenceStore::numFrames() const { return snapshot().numFrames(); }

size_t PhysicalEvidenceStore::numRuns() const { return snapshot().numRuns(); }

}  // namespace khronos
