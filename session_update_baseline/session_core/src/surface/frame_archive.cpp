#include "session_core/evidence/range_encoding.h"
#include "session_core/surface/frame_archive.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

#include <glog/logging.h>
#include <hydra/common/global_info.h>
#include <hydra/input/camera.h>
#include <zstd.h>

#include "khronos/active_window/data/frame_data.h"

namespace khronos {

namespace {

constexpr int kZstdLevel = 1;
constexpr size_t kRunBytes = sizeof(uint32_t) + sizeof(uint16_t);

}  // namespace

bool FrameArchive::Camera::sameAs(const Camera& o) const {
  return width == o.width && height == o.height && fx == o.fx && fy == o.fy && cx == o.cx &&
         cy == o.cy && min_range == o.min_range && max_range == o.max_range;
}

FrameArchive::Frame FrameArchive::Frame::pack(TimeStamp stamp,
                                              const Eigen::Isometry3d& world_T_sensor,
                                              const std::vector<uint16_t>& range_mm,
                                              const std::vector<InstanceRun>& instances,
                                              const std::vector<InstanceRun>* attributes) {
  // Raw layout: range differences to the previous pixel (uint16, wrapping), then the instance runs
  // (uint32 end, uint16 id) and, in layout 3, a uint32 instance-run count in front of them and the
  // attribute runs (uint32 end, uint16 word) after them.
  const size_t n = range_mm.size();
  const bool with_attributes = attributes != nullptr;
  std::vector<uint8_t> raw(n * sizeof(uint16_t) + (with_attributes ? sizeof(uint32_t) : 0) +
                           (instances.size() + (with_attributes ? attributes->size() : 0)) * kRunBytes);
  uint16_t previous = 0;
  for (size_t p = 0; p < n; ++p) {
    const uint16_t delta = static_cast<uint16_t>(range_mm[p] - previous);
    std::memcpy(&raw[2 * p], &delta, sizeof(delta));
    previous = range_mm[p];
  }
  uint8_t* out = raw.data() + n * sizeof(uint16_t);
  if (with_attributes) {
    const uint32_t count = static_cast<uint32_t>(instances.size());
    std::memcpy(out, &count, sizeof(count));
    out += sizeof(count);
  }
  const auto write_runs = [&out](const std::vector<InstanceRun>& runs) {
    for (const auto& run : runs) {
      std::memcpy(out, &run.end, sizeof(run.end));
      std::memcpy(out + sizeof(run.end), &run.id, sizeof(run.id));
      out += kRunBytes;
    }
  };
  write_runs(instances);
  if (with_attributes) write_runs(*attributes);
  Frame frame;
  frame.stamp = stamp;
  frame.world_T_sensor = world_T_sensor;
  frame.layout = with_attributes ? 3 : 2;
  frame.packed.resize(ZSTD_compressBound(raw.size()));
  const size_t size =
      ZSTD_compress(frame.packed.data(), frame.packed.size(), raw.data(), raw.size(), kZstdLevel);
  CHECK(!ZSTD_isError(size)) << "[FrameArchive] zstd: " << ZSTD_getErrorName(size);
  frame.packed.resize(size);
  frame.packed.shrink_to_fit();
  return frame;
}

bool FrameArchive::Frame::decode(size_t num_pixels,
                                 std::vector<uint16_t>& range_mm,
                                 std::vector<uint16_t>& ids,
                                 std::vector<uint16_t>* classes,
                                 std::vector<uint8_t>* motion) const {
  thread_local std::vector<uint8_t> raw;
  const unsigned long long size = ZSTD_getFrameContentSize(packed.data(), packed.size());
  const size_t header = layout >= 3 ? sizeof(uint32_t) : 0;
  if (size == ZSTD_CONTENTSIZE_ERROR || size == ZSTD_CONTENTSIZE_UNKNOWN ||
      size < num_pixels * sizeof(uint16_t) + header ||
      (size - num_pixels * sizeof(uint16_t) - header) % kRunBytes) {
    return false;
  }
  raw.resize(size);
  if (ZSTD_decompress(raw.data(), raw.size(), packed.data(), packed.size()) != size) return false;
  range_mm.resize(num_pixels);
  uint16_t value = 0;
  for (size_t p = 0; p < num_pixels; ++p) {
    uint16_t delta;
    std::memcpy(&delta, &raw[2 * p], sizeof(delta));
    value = static_cast<uint16_t>(value + delta);
    range_mm[p] = value;
  }
  size_t cursor = num_pixels * sizeof(uint16_t);
  size_t instance_bytes = raw.size() - cursor - header;
  if (layout >= 3) {
    uint32_t count = 0;
    std::memcpy(&count, &raw[cursor], sizeof(count));
    cursor += sizeof(count);
    if (static_cast<uint64_t>(count) * kRunBytes > instance_bytes) return false;
    instance_bytes = static_cast<size_t>(count) * kRunBytes;
  }
  ids.assign(num_pixels, 0);
  const auto read_runs = [&](size_t begin, size_t bytes, auto&& assign) {
    size_t start = 0;
    for (size_t k = begin; k < begin + bytes; k += kRunBytes) {
      InstanceRun run;
      std::memcpy(&run.end, &raw[k], sizeof(run.end));
      std::memcpy(&run.id, &raw[k + sizeof(run.end)], sizeof(run.id));
      const size_t end = std::min<size_t>(run.end, num_pixels);
      if (end > start) assign(start, end, run.id);
      start = std::max(start, end);
    }
  };
  read_runs(cursor, instance_bytes, [&](size_t begin, size_t end, uint16_t id) {
    if (id) std::fill(ids.begin() + begin, ids.begin() + end, id);
  });
  if (classes) classes->assign(num_pixels, 0);
  if (motion) motion->assign(num_pixels, 0);
  if (layout >= 3 && (classes || motion)) {
    const size_t attribute_begin = cursor + instance_bytes;
    read_runs(attribute_begin, raw.size() - attribute_begin,
              [&](size_t begin, size_t end, uint16_t word) {
                if (classes && (word & kClassMask)) {
                  std::fill(classes->begin() + begin, classes->begin() + end,
                            static_cast<uint16_t>(word & kClassMask));
                }
                if (motion && (word & kMotionBit)) {
                  std::fill(motion->begin() + begin, motion->begin() + end, uint8_t{1});
                }
              });
  }
  return true;
}

void FrameArchive::offer(const FrameData& data) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ++offered_;
  }
  const auto& input = data.input;
  const cv::Mat& ranges = input.range_image;
  const auto* camera = dynamic_cast<const hydra::Camera*>(&input.getSensor());
  if (!camera || ranges.empty() || ranges.type() != CV_32FC1)
    throw std::invalid_argument("Session frame requires a pinhole camera and float range image");
  const auto& cc = camera->getConfig();
  Camera cam;
  cam.width = static_cast<uint32_t>(ranges.cols);
  cam.height = static_cast<uint32_t>(ranges.rows);
  cam.fx = cc.fx;
  cam.fy = cc.fy;
  cam.cx = cc.cx;
  cam.cy = cc.cy;
  cam.min_range = input.getSensor().min_range();
  cam.max_range = input.getSensor().max_range();
  if (!cam.valid() || !input.getSensorPose().matrix().allFinite())
    throw std::invalid_argument("Invalid session frame camera or pose");
  const bool have_labels = !input.label_image.empty() && input.label_image.type() == CV_32SC1 &&
                           input.label_image.rows == ranges.rows &&
                           input.label_image.cols == ranges.cols;
  // The motion clusters of this frame (the active window's integration mask excludes them).
  const bool have_dynamic = !data.dynamic_image.empty() && data.dynamic_image.type() == CV_32SC1 &&
                            data.dynamic_image.rows == ranges.rows &&
                            data.dynamic_image.cols == ranges.cols;
  const bool have_instances = !data.instance_image.empty() &&
                              data.instance_image.type() == CV_32SC1 &&
                              data.instance_image.rows == ranges.rows &&
                              data.instance_image.cols == ranges.cols;
  const auto& labels = hydra::GlobalInfo::instance().getLabelSpaceConfig();
  // Invalid semantic labels (no measurement), looked up once per label value. Dynamic classes are
  // not excluded: the class is only the grouping key of the persistence prior (README section 3).
  std::vector<int8_t> excluded(1024, -1);
  auto isExcluded = [&](int s) {
    if (s < 0) return false;
    const auto label = static_cast<uint32_t>(s);
    if (label >= excluded.size()) return labels.invalid_labels.count(label) > 0;
    if (excluded[label] < 0) excluded[label] = labels.invalid_labels.count(label) > 0;
    return excluded[label] > 0;
  };

  const size_t n = static_cast<size_t>(cam.width) * cam.height;
  std::vector<uint16_t> range_mm(n, 0);
  std::vector<InstanceRun> instances, attributes;
  uint32_t offset = 0;
  uint16_t previous = 0, previous_word = 0;
  for (int v = 0; v < ranges.rows; ++v) {
    const float* row = ranges.ptr<float>(v);
    const int* label_row = have_labels ? input.label_image.ptr<int>(v) : nullptr;
    const int* id_row = have_instances ? data.instance_image.ptr<int>(v) : nullptr;
    const int* dynamic_row = have_dynamic ? data.dynamic_image.ptr<int>(v) : nullptr;
    for (int u = 0; u < ranges.cols; ++u, ++offset) {
      const float r = row[u];
      const uint16_t id16 = measurement::encodeIdentity(id_row ? id_row[u] : 0);
      // README (8): physical observations stay available until the terminal registry authorizes
      // their state, and (principle 5) a reading is never dropped by its class at the time it is
      // archived: identity, semantic class and the native motion mask are stored per pixel, and the
      // session-end refusion judges each reading from them. Only a reading that carries no
      // measurement (an invalid semantic label) is not stored.
      if (id16 || !(label_row && isExcluded(label_row[u]))) {
        range_mm[offset] = measurement::encodeRange(r, cam.min_range, cam.max_range);
      }
      if (offset > 0 && id16 != previous) instances.push_back({offset, previous});
      previous = id16;
      const uint16_t word = attributeWord(label_row ? label_row[u] : -1,
                                          dynamic_row && dynamic_row[u] != 0);
      if (offset > 0 && word != previous_word) attributes.push_back({offset, previous_word});
      previous_word = word;
    }
  }
  if (offset > 0) {
    instances.push_back({offset, previous});
    attributes.push_back({offset, previous_word});
  }
  Frame frame = Frame::pack(input.timestamp_ns, input.getSensorPose(), range_mm, instances, &attributes);

  std::lock_guard<std::mutex> lock(mutex_);
  if (!frames_.empty() && !camera_.sameAs(cam))
    throw std::invalid_argument("Session frame camera differs from the archive");
  // A timestamp names one immutable observation. Retransmission is idempotent;
  // a revised measurement requires a new source instead of extra integration.
  const auto [entry, inserted] = frame_indices_.emplace(frame.stamp, frames_.size());
  if (!inserted) {
    const auto& existing = frames_.at(entry->second);
    if (!(existing.world_T_sensor.matrix().array() == frame.world_T_sensor.matrix().array()).all() ||
        existing.packed != frame.packed)
      throw std::invalid_argument("Conflicting session observations share a timestamp");
    return;
  }
  if (frames_.empty()) camera_ = cam;
  const size_t packed_bytes = frame.packed.size();
  try {
    frames_.push_back(std::move(frame));
  } catch (...) {
    frame_indices_.erase(entry);
    throw;
  }
  bytes_ += packed_bytes + sizeof(Frame);
  raw_bytes_ += n * sizeof(uint16_t);
}

std::vector<FrameArchive::Frame> FrameArchive::release(Camera* camera) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (camera) *camera = camera_;
  std::vector<Frame> out = std::move(frames_);
  frames_.clear();
  frame_indices_.clear();
  std::stable_sort(out.begin(), out.end(),
                   [](const Frame& a, const Frame& b) { return a.stamp < b.stamp; });
  LOG(INFO) << "[FrameArchive] released " << out.size() << " frames of " << offered_
            << " processed (skipped " << skipped_ << "), " << (bytes_ >> 20) << " MB for "
            << (raw_bytes_ >> 20) << " MB of range.";
  bytes_ = 0;
  raw_bytes_ = 0;
  return out;
}

size_t FrameArchive::numOffered() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return offered_;
}

size_t FrameArchive::numFrames() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return frames_.size();
}

size_t FrameArchive::numBytes() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return bytes_;
}

namespace {
constexpr char kMagic[4] = {'K', 'F', 'A', '1'};
}

// Version 1 holds the raw range and runs per frame (offline converters write
// it); version 2 holds the packed frames; version 3 adds the layout byte of each packed frame
// (the semantic-class and motion attributes).
bool FrameArchive::save(const std::string& path,
                        const std::vector<Frame>& frames,
                        const Camera& camera) {
  std::ofstream out(path, std::ios::binary);
  if (!out) return false;
  out.write(kMagic, 4);
  const uint32_t version = 3;
  out.write(reinterpret_cast<const char*>(&version), sizeof(version));
  out.write(reinterpret_cast<const char*>(&camera.width), sizeof(uint32_t));
  out.write(reinterpret_cast<const char*>(&camera.height), sizeof(uint32_t));
  const float k[6] = {camera.fx, camera.fy, camera.cx, camera.cy, camera.min_range,
                      camera.max_range};
  out.write(reinterpret_cast<const char*>(k), sizeof(k));
  const uint64_t n = frames.size();
  out.write(reinterpret_cast<const char*>(&n), sizeof(n));
  for (const auto& f : frames) {
    const uint64_t stamp = f.stamp;
    out.write(reinterpret_cast<const char*>(&stamp), sizeof(stamp));
    const Eigen::Matrix4d m = f.world_T_sensor.matrix();  // column-major
    out.write(reinterpret_cast<const char*>(m.data()), 16 * sizeof(double));
    out.write(reinterpret_cast<const char*>(&f.layout), sizeof(f.layout));
    const uint64_t size = f.packed.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    out.write(reinterpret_cast<const char*>(f.packed.data()), size);
  }
  return static_cast<bool>(out);
}

bool FrameArchive::load(const std::string& path, std::vector<Frame>& frames, Camera& camera) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  char magic[4];
  uint32_t version = 0;
  if (!in.read(magic, 4) || std::memcmp(magic, kMagic, 4) != 0) return false;
  if (!in.read(reinterpret_cast<char*>(&version), sizeof(version)) ||
      (version != 1 && version != 2 && version != 3)) {
    return false;
  }
  in.read(reinterpret_cast<char*>(&camera.width), sizeof(uint32_t));
  in.read(reinterpret_cast<char*>(&camera.height), sizeof(uint32_t));
  float k[6];
  in.read(reinterpret_cast<char*>(k), sizeof(k));
  camera.fx = k[0];
  camera.fy = k[1];
  camera.cx = k[2];
  camera.cy = k[3];
  camera.min_range = k[4];
  camera.max_range = k[5];
  uint64_t n = 0;
  if (!in.read(reinterpret_cast<char*>(&n), sizeof(n)) || !camera.valid()) return false;
  frames.clear();
  frames.resize(n);
  std::vector<uint16_t> range;
  std::vector<InstanceRun> runs;
  TimeStamp previous_stamp = 0;
  bool first_frame = true;
  for (auto& f : frames) {
    uint64_t stamp = 0;
    Eigen::Matrix4d m;
    in.read(reinterpret_cast<char*>(&stamp), sizeof(stamp));
    in.read(reinterpret_cast<char*>(m.data()), 16 * sizeof(double));
    if (!in || !m.allFinite() || (!first_frame && stamp <= previous_stamp)) return false;
    previous_stamp = stamp;
    first_frame = false;
    Eigen::Isometry3d pose;
    pose.matrix() = m;
    if (version >= 2) {
      uint8_t layout = 2;
      if (version >= 3) in.read(reinterpret_cast<char*>(&layout), sizeof(layout));
      uint64_t size = 0;
      in.read(reinterpret_cast<char*>(&size), sizeof(size));
      f.stamp = stamp;
      f.layout = layout;
      f.world_T_sensor = pose;
      f.packed.resize(size);
      in.read(reinterpret_cast<char*>(f.packed.data()), size);
    } else {
      uint64_t num_range = 0, num_runs = 0;
      in.read(reinterpret_cast<char*>(&num_range), sizeof(num_range));
      in.read(reinterpret_cast<char*>(&num_runs), sizeof(num_runs));
      if (!in) return false;
      range.resize(num_range);
      in.read(reinterpret_cast<char*>(range.data()), num_range * sizeof(uint16_t));
      runs.resize(num_runs);
      for (auto& run : runs) {
        uint32_t rec[2];
        in.read(reinterpret_cast<char*>(rec), sizeof(rec));
        run.end = rec[0];
        run.id = static_cast<uint16_t>(rec[1]);
      }
      f = Frame::pack(stamp, pose, range, runs);
    }
    if (!in) return false;
  }
  return true;
}

}  // namespace khronos
