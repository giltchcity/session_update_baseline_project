#include "khronos/backend/reconciliation/frame_archive.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

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
                                              const std::vector<InstanceRun>& instances) {
  // Raw layout: range differences to the previous pixel (uint16, wrapping),
  // then the instance runs (uint32 end, uint16 id).
  const size_t n = range_mm.size();
  std::vector<uint8_t> raw(n * sizeof(uint16_t) + instances.size() * kRunBytes);
  uint16_t previous = 0;
  for (size_t p = 0; p < n; ++p) {
    const uint16_t delta = static_cast<uint16_t>(range_mm[p] - previous);
    std::memcpy(&raw[2 * p], &delta, sizeof(delta));
    previous = range_mm[p];
  }
  uint8_t* out = raw.data() + n * sizeof(uint16_t);
  for (const auto& run : instances) {
    std::memcpy(out, &run.end, sizeof(run.end));
    std::memcpy(out + sizeof(run.end), &run.id, sizeof(run.id));
    out += kRunBytes;
  }
  Frame frame;
  frame.stamp = stamp;
  frame.world_T_sensor = world_T_sensor;
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
                                 std::vector<uint16_t>& ids) const {
  thread_local std::vector<uint8_t> raw;
  const unsigned long long size = ZSTD_getFrameContentSize(packed.data(), packed.size());
  if (size == ZSTD_CONTENTSIZE_ERROR || size == ZSTD_CONTENTSIZE_UNKNOWN ||
      size < num_pixels * sizeof(uint16_t) || (size - num_pixels * sizeof(uint16_t)) % kRunBytes) {
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
  ids.assign(num_pixels, 0);
  size_t start = 0;
  for (size_t k = num_pixels * sizeof(uint16_t); k < raw.size(); k += kRunBytes) {
    InstanceRun run;
    std::memcpy(&run.end, &raw[k], sizeof(run.end));
    std::memcpy(&run.id, &raw[k + sizeof(run.end)], sizeof(run.id));
    const size_t end = std::min<size_t>(run.end, num_pixels);
    if (end > start && run.id) std::fill(ids.begin() + start, ids.begin() + end, run.id);
    start = std::max(start, end);
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
  if (!camera || ranges.empty() || ranges.type() != CV_32FC1) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++skipped_;
    LOG_EVERY_N(WARNING, 100) << "[FrameArchive] frame " << input.timestamp_ns
                              << " skipped: no pinhole camera or no CV_32FC1 range image.";
    return;
  }
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
  const bool have_labels = !input.label_image.empty() && input.label_image.type() == CV_32SC1 &&
                           input.label_image.rows == ranges.rows &&
                           input.label_image.cols == ranges.cols;
  // The motion / dynamic clusters of this frame (the active window's
  // integration mask excludes them as it excludes dynamic semantics).
  const bool have_dynamic = !data.dynamic_image.empty() && data.dynamic_image.type() == CV_32SC1 &&
                            data.dynamic_image.rows == ranges.rows &&
                            data.dynamic_image.cols == ranges.cols;
  const bool have_instances = !data.instance_image.empty() &&
                              data.instance_image.type() == CV_32SC1 &&
                              data.instance_image.rows == ranges.rows &&
                              data.instance_image.cols == ranges.cols;
  const auto& labels = hydra::GlobalInfo::instance().getLabelSpaceConfig();
  // Dynamic / invalid semantic classes, looked up once per label value.
  std::vector<int8_t> excluded(1024, -1);
  auto isExcluded = [&](int s) {
    if (s < 0) return false;
    const auto label = static_cast<uint32_t>(s);
    if (label >= excluded.size()) return labels.isDynamic(label) || labels.invalid_labels.count(label) > 0;
    if (excluded[label] < 0) {
      excluded[label] = labels.isDynamic(label) || labels.invalid_labels.count(label) > 0;
    }
    return excluded[label] > 0;
  };

  const size_t n = static_cast<size_t>(cam.width) * cam.height;
  std::vector<uint16_t> range_mm(n, 0);
  std::vector<InstanceRun> instances;
  uint32_t offset = 0;
  uint16_t previous = 0;
  for (int v = 0; v < ranges.rows; ++v) {
    const float* row = ranges.ptr<float>(v);
    const int* label_row = have_labels ? input.label_image.ptr<int>(v) : nullptr;
    const int* id_row = have_instances ? data.instance_image.ptr<int>(v) : nullptr;
    const int* dynamic_row = have_dynamic ? data.dynamic_image.ptr<int>(v) : nullptr;
    for (int u = 0; u < ranges.cols; ++u, ++offset) {
      const float r = row[u];
      if (std::isfinite(r) && r > cam.min_range && r <= cam.max_range &&
          !(label_row && isExcluded(label_row[u])) && !(dynamic_row && dynamic_row[u] != 0)) {
        const float mm = r * 1000.f;  // truncation, as the offline archive (numpy astype)
        range_mm[offset] = mm >= static_cast<float>(std::numeric_limits<uint16_t>::max())
                               ? std::numeric_limits<uint16_t>::max()
                               : static_cast<uint16_t>(mm);
      }
      const int id = id_row ? id_row[u] : 0;
      const uint16_t id16 =
          (id > 0 && id <= std::numeric_limits<uint16_t>::max()) ? static_cast<uint16_t>(id) : 0;
      if (offset > 0 && id16 != previous) instances.push_back({offset, previous});
      previous = id16;
    }
  }
  if (offset > 0) instances.push_back({offset, previous});
  Frame frame = Frame::pack(input.timestamp_ns, input.getSensorPose(), range_mm, instances);

  std::lock_guard<std::mutex> lock(mutex_);
  if (!frames_.empty() && !camera_.sameAs(cam)) {
    ++skipped_;
    LOG(ERROR) << "[FrameArchive] frame " << frame.stamp
               << " skipped: its camera differs from the archive's.";
    return;
  }
  if (frames_.empty()) camera_ = cam;
  bytes_ += frame.packed.size() + sizeof(Frame);
  raw_bytes_ += n * sizeof(uint16_t);
  // Frames arrive in processing (stamp) order; keep that order.
  frames_.push_back(std::move(frame));
}

std::vector<FrameArchive::Frame> FrameArchive::release(Camera* camera) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (camera) *camera = camera_;
  std::vector<Frame> out = std::move(frames_);
  frames_.clear();
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
// it); version 2 holds the packed frames.
bool FrameArchive::save(const std::string& path,
                        const std::vector<Frame>& frames,
                        const Camera& camera) {
  std::ofstream out(path, std::ios::binary);
  if (!out) return false;
  out.write(kMagic, 4);
  const uint32_t version = 2;
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
      (version != 1 && version != 2)) {
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
  if (!in.read(reinterpret_cast<char*>(&n), sizeof(n))) return false;
  frames.clear();
  frames.resize(n);
  std::vector<uint16_t> range;
  std::vector<InstanceRun> runs;
  for (auto& f : frames) {
    uint64_t stamp = 0;
    Eigen::Matrix4d m;
    in.read(reinterpret_cast<char*>(&stamp), sizeof(stamp));
    in.read(reinterpret_cast<char*>(m.data()), 16 * sizeof(double));
    if (!in) return false;
    Eigen::Isometry3d pose;
    pose.matrix() = m;
    if (version == 2) {
      uint64_t size = 0;
      in.read(reinterpret_cast<char*>(&size), sizeof(size));
      f.stamp = stamp;
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
