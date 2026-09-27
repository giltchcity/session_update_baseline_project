#include "khronos/backend/reconciliation/frame_archive.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

#include <glog/logging.h>
#include <hydra/common/global_info.h>
#include <hydra/input/camera.h>

#include "khronos/active_window/data/frame_data.h"

namespace khronos {

bool FrameArchive::Camera::sameAs(const Camera& o) const {
  return width == o.width && height == o.height && fx == o.fx && fy == o.fy && cx == o.cx &&
         cy == o.cy && min_range == o.min_range && max_range == o.max_range;
}

void FrameArchive::Frame::decodeInstances(std::vector<uint16_t>& out, size_t num_pixels) const {
  out.assign(num_pixels, 0);
  size_t start = 0;
  for (const auto& run : instances) {
    const size_t end = std::min<size_t>(run.end, num_pixels);
    if (end > start && run.id) std::fill(out.begin() + start, out.begin() + end, run.id);
    start = std::max(start, end);
  }
}

void FrameArchive::offer(const FrameData& data) {
  size_t index = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    index = offered_++;
  }
  if (index % kKeepEvery != 0) return;

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
  const bool have_instances = !data.instance_image.empty() &&
                              data.instance_image.type() == CV_32SC1 &&
                              data.instance_image.rows == ranges.rows &&
                              data.instance_image.cols == ranges.cols;
  const auto& labels = hydra::GlobalInfo::instance().getLabelSpaceConfig();

  Frame frame;
  frame.stamp = input.timestamp_ns;
  frame.world_T_sensor = input.getSensorPose();
  const size_t n = static_cast<size_t>(cam.width) * cam.height;
  frame.range_mm.assign(n, 0);
  uint32_t offset = 0;
  uint16_t previous = 0;
  bool have_previous = false;
  for (int v = 0; v < ranges.rows; ++v) {
    for (int u = 0; u < ranges.cols; ++u, ++offset) {
      const float r = ranges.at<float>(v, u);
      bool valid = std::isfinite(r) && r > cam.min_range && r <= cam.max_range;
      if (valid && have_labels) {
        const int s = input.label_image.at<int>(v, u);
        if (s >= 0 && (labels.isDynamic(static_cast<uint32_t>(s)) ||
                       labels.invalid_labels.count(static_cast<uint32_t>(s)))) {
          valid = false;
        }
      }
      if (valid) {
        const float mm = r * 1000.f;  // truncation, as the offline archive (numpy astype)
        frame.range_mm[offset] =
            mm >= static_cast<float>(std::numeric_limits<uint16_t>::max())
                ? std::numeric_limits<uint16_t>::max()
                : static_cast<uint16_t>(mm);
      }
      int id = have_instances ? data.instance_image.at<int>(v, u) : 0;
      const uint16_t id16 =
          (id > 0 && id <= std::numeric_limits<uint16_t>::max()) ? static_cast<uint16_t>(id) : 0;
      if (have_previous && id16 != previous) frame.instances.push_back({offset, previous});
      previous = id16;
      have_previous = true;
    }
  }
  if (have_previous) frame.instances.push_back({offset, previous});
  frame.instances.shrink_to_fit();

  std::lock_guard<std::mutex> lock(mutex_);
  if (!frames_.empty() && !camera_.sameAs(cam)) {
    ++skipped_;
    LOG(ERROR) << "[FrameArchive] frame " << frame.stamp
               << " skipped: its camera differs from the archive's.";
    return;
  }
  if (frames_.empty()) camera_ = cam;
  bytes_ += frame.range_mm.size() * sizeof(uint16_t) +
            frame.instances.size() * sizeof(InstanceRun) + sizeof(Frame);
  // Frames arrive in processing (stamp) order; keep that order.
  frames_.push_back(std::move(frame));
}

std::vector<FrameArchive::Frame> FrameArchive::release(Camera* camera) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (camera) *camera = camera_;
  std::vector<Frame> out = std::move(frames_);
  frames_.clear();
  bytes_ = 0;
  std::stable_sort(out.begin(), out.end(),
                   [](const Frame& a, const Frame& b) { return a.stamp < b.stamp; });
  LOG(INFO) << "[FrameArchive] released " << out.size() << " frames of " << offered_
            << " processed (skipped " << skipped_ << ").";
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

bool FrameArchive::save(const std::string& path,
                        const std::vector<Frame>& frames,
                        const Camera& camera) {
  std::ofstream out(path, std::ios::binary);
  if (!out) return false;
  out.write(kMagic, 4);
  const uint32_t version = 1;
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
    const uint64_t num_range = f.range_mm.size(), num_runs = f.instances.size();
    out.write(reinterpret_cast<const char*>(&num_range), sizeof(num_range));
    out.write(reinterpret_cast<const char*>(&num_runs), sizeof(num_runs));
    out.write(reinterpret_cast<const char*>(f.range_mm.data()), num_range * sizeof(uint16_t));
    for (const auto& run : f.instances) {
      const uint32_t rec[2] = {run.end, run.id};
      out.write(reinterpret_cast<const char*>(rec), sizeof(rec));
    }
  }
  return static_cast<bool>(out);
}

bool FrameArchive::load(const std::string& path, std::vector<Frame>& frames, Camera& camera) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  char magic[4];
  uint32_t version = 0;
  if (!in.read(magic, 4) || std::memcmp(magic, kMagic, 4) != 0) return false;
  if (!in.read(reinterpret_cast<char*>(&version), sizeof(version)) || version != 1) return false;
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
  for (auto& f : frames) {
    uint64_t stamp = 0, num_range = 0, num_runs = 0;
    Eigen::Matrix4d m;
    in.read(reinterpret_cast<char*>(&stamp), sizeof(stamp));
    in.read(reinterpret_cast<char*>(m.data()), 16 * sizeof(double));
    in.read(reinterpret_cast<char*>(&num_range), sizeof(num_range));
    in.read(reinterpret_cast<char*>(&num_runs), sizeof(num_runs));
    if (!in) return false;
    f.stamp = stamp;
    f.world_T_sensor.matrix() = m;
    f.range_mm.resize(num_range);
    in.read(reinterpret_cast<char*>(f.range_mm.data()), num_range * sizeof(uint16_t));
    f.instances.resize(num_runs);
    for (auto& run : f.instances) {
      uint32_t rec[2];
      in.read(reinterpret_cast<char*>(rec), sizeof(rec));
      run.end = rec[0];
      run.id = static_cast<uint16_t>(rec[1]);
    }
    if (!in) return false;
  }
  return true;
}

}  // namespace khronos
