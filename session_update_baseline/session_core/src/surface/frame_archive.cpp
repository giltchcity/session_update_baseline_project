#include "session_core/surface/frame_archive.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

#include <glog/logging.h>
#include <hydra/common/global_info.h>
#include <hydra/input/camera.h>
#include <lz4.h>

#include "khronos/active_window/data/frame_data.h"

namespace khronos {

namespace {

constexpr size_t kRunBytes = sizeof(uint32_t) + sizeof(uint16_t);

// An LZ4 block behind the raw size it holds.
std::vector<uint8_t> compress(const void* raw, size_t bytes) {
  const uint32_t size = static_cast<uint32_t>(bytes);
  std::vector<uint8_t> packed(sizeof(size) + LZ4_compressBound(static_cast<int>(bytes)));
  std::memcpy(packed.data(), &size, sizeof(size));
  const int n = LZ4_compress_default(static_cast<const char*>(raw), reinterpret_cast<char*>(packed.data() + sizeof(size)),
                                     static_cast<int>(bytes), static_cast<int>(packed.size() - sizeof(size)));
  CHECK(n > 0) << "[FrameArchive] lz4 cannot compress " << bytes << " bytes";
  packed.resize(sizeof(size) + static_cast<size_t>(n));
  packed.shrink_to_fit();
  return packed;
}

// The raw size a packed block holds (0: none).
size_t rawSize(const std::vector<uint8_t>& packed) {
  uint32_t size = 0;
  if (packed.size() >= sizeof(size)) std::memcpy(&size, packed.data(), sizeof(size));
  return size;
}

bool decompress(const std::vector<uint8_t>& packed, void* raw, size_t bytes) {
  return rawSize(packed) == bytes &&
         LZ4_decompress_safe(reinterpret_cast<const char*>(packed.data() + sizeof(uint32_t)), static_cast<char*>(raw),
                             static_cast<int>(packed.size() - sizeof(uint32_t)), static_cast<int>(bytes)) ==
             static_cast<int>(bytes);
}

}  // namespace

bool FrameArchive::Camera::sameAs(const Camera& o) const {
  return width == o.width && height == o.height && fx == o.fx && fy == o.fy && cx == o.cx &&
         cy == o.cy && min_range == o.min_range && max_range == o.max_range;
}

FrameArchive::Frame FrameArchive::Frame::pack(TimeStamp stamp,
                                              const Eigen::Isometry3d& world_T_sensor,
                                              uint32_t width,
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
  size_t begin = 0;
  for (const auto& run : instances) {
    const size_t end = std::min<size_t>(run.end, n);
    if (run.id && std::any_of(range_mm.begin() + begin, range_mm.begin() + std::max(begin, end),
                              [](uint16_t r) { return r > 0; })) {
      frame.seen.push_back(run.id);
    }
    begin = std::max(begin, end);
  }
  std::sort(frame.seen.begin(), frame.seen.end());
  frame.seen.erase(std::unique(frame.seen.begin(), frame.seen.end()), frame.seen.end());
  const size_t height = width ? n / width : 0;
  const size_t tiles_x = (width + kTile - 1) / kTile;
  frame.tile_farthest_mm.assign(tiles_x * ((height + kTile - 1) / kTile), 0);
  for (size_t v = 0; v < height; ++v) {
    uint16_t* tiles = frame.tile_farthest_mm.data() + (v / kTile) * tiles_x;
    for (size_t u = 0; u < width; ++u) tiles[u / kTile] = std::max(tiles[u / kTile], range_mm[v * width + u]);
  }
  frame.farthest_mm = frame.tile_farthest_mm.empty()
                          ? 0
                          : *std::max_element(frame.tile_farthest_mm.begin(), frame.tile_farthest_mm.end());
  frame.packed = compress(raw.data(), raw.size());
  return frame;
}

bool FrameArchive::Frame::decode(size_t num_pixels, std::vector<uint16_t>& range_mm,
                                 std::vector<InstanceRun>& runs) const {
  thread_local std::vector<uint8_t> raw;
  const size_t size = rawSize(packed);
  if (size < num_pixels * sizeof(uint16_t) || (size - num_pixels * sizeof(uint16_t)) % kRunBytes) return false;
  raw.resize(size);
  if (!decompress(packed, raw.data(), size)) return false;
  range_mm.resize(num_pixels);
  uint16_t value = 0;
  for (size_t p = 0; p < num_pixels; ++p) {
    uint16_t delta;
    std::memcpy(&delta, &raw[2 * p], sizeof(delta));
    value = static_cast<uint16_t>(value + delta);
    range_mm[p] = value;
  }
  // Runs ending in order within the image.
  runs.clear();
  uint32_t start = 0;
  for (size_t k = num_pixels * sizeof(uint16_t); k < raw.size(); k += kRunBytes) {
    InstanceRun r;
    std::memcpy(&r.end, &raw[k], sizeof(r.end));
    std::memcpy(&r.id, &raw[k + sizeof(r.end)], sizeof(r.id));
    r.end = std::max(start, std::min<uint32_t>(r.end, static_cast<uint32_t>(num_pixels)));
    runs.push_back(r);
    start = r.end;
  }
  return true;
}

bool FrameArchive::Frame::decodeExcluding(size_t num_pixels, const std::vector<uint16_t>& excluded,
                                          std::vector<uint16_t>& range_mm) const {
  thread_local std::vector<InstanceRun> runs;
  if (!decode(num_pixels, range_mm, runs)) return false;
  uint32_t begin = 0;
  for (const auto& run : runs) {
    if (run.id && std::binary_search(excluded.begin(), excluded.end(), run.id)) {
      std::fill(range_mm.begin() + begin, range_mm.begin() + run.end, 0);
    }
    begin = run.end;
  }
  return true;
}

uint16_t FrameArchive::Frame::farthestIn(uint32_t width, int u0, int u1, int v0, int v1) const {
  const size_t tiles_x = (width + kTile - 1) / kTile;
  if (!tiles_x || tile_farthest_mm.size() < tiles_x || u0 > u1 || v0 > v1) return 0;
  const size_t tiles_y = tile_farthest_mm.size() / tiles_x;
  uint16_t farthest = 0;
  for (size_t ty = std::max(0, v0) / kTile; ty <= std::min<size_t>(v1 / kTile, tiles_y - 1); ++ty) {
    for (size_t tx = std::max(0, u0) / kTile; tx <= std::min<size_t>(u1 / kTile, tiles_x - 1); ++tx) {
      farthest = std::max(farthest, tile_farthest_mm[ty * tiles_x + tx]);
    }
  }
  return farthest;
}

bool FrameArchive::Frame::keptLimit(size_t num_pixels, std::vector<float>& free_limit) const {
  if (limit.empty()) {
    free_limit.clear();
    return true;
  }
  free_limit.resize(num_pixels);
  return decompress(limit, free_limit.data(), num_pixels * sizeof(float));
}

std::vector<uint8_t> FrameArchive::packLimit(const std::vector<float>& free_limit) {
  if (free_limit.empty()) return {};
  return compress(free_limit.data(), free_limit.size() * sizeof(float));
}

void FrameArchive::keepLimit(size_t index, std::vector<uint8_t> packed_limit) {
  auto& kept = frames_.at(index).limit;
  bytes_ -= kept.size();
  kept = std::move(packed_limit);
  bytes_ += kept.size();
}

bool FrameArchive::pack(const FrameData& data, Camera& cam, Frame& frame) {
  const auto& input = data.input;
  const cv::Mat& ranges = input.range_image;
  const auto* camera = dynamic_cast<const hydra::Camera*>(&input.getSensor());
  if (!camera || ranges.empty() || ranges.type() != CV_32FC1) {
    LOG_EVERY_N(WARNING, 100) << "[FrameArchive] frame " << input.timestamp_ns
                              << " skipped: no pinhole camera or no CV_32FC1 range image.";
    return false;
  }
  const auto& cc = camera->getConfig();
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
  frame = Frame::pack(input.timestamp_ns, input.getSensorPose(), cam.width, range_mm, instances);
  return true;
}

bool FrameArchive::keep(const Camera& camera, Frame frame) {
  if (!frames_.empty() && !camera_.sameAs(camera)) {
    LOG(ERROR) << "[FrameArchive] frame " << frame.stamp
               << " skipped: its camera differs from the archive's.";
    return false;
  }
  if (frames_.empty()) camera_ = camera;
  bytes_ += frame.packed.size() + (frame.seen.size() + frame.tile_farthest_mm.size()) * sizeof(uint16_t) + sizeof(Frame);
  frames_.push_back(std::move(frame));
  return true;
}

namespace {

constexpr char kMagic[4] = {'K', 'F', 'A', '1'};
}

// Version 1 holds the raw range and runs per frame (offline converters write
// it); version 3 holds the packed frames (LZ4).
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
      (version != 1 && version != 3)) {
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
    if (version == 3) {
      uint64_t size = 0;
      in.read(reinterpret_cast<char*>(&size), sizeof(size));
      f.packed.resize(size);
      in.read(reinterpret_cast<char*>(f.packed.data()), size);
      if (!in || !f.decode(static_cast<size_t>(camera.width) * camera.height, range, runs)) return false;
      f = Frame::pack(stamp, pose, camera.width, range, runs);
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
      f = Frame::pack(stamp, pose, camera.width, range, runs);
    }
    if (!in) return false;
  }
  return true;
}

}  // namespace khronos
