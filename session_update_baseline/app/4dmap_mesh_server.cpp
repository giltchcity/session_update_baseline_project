// Long-lived .4dmap mesh server: loads one or two maps once, then answers
// per-time-step current-scene mesh queries over stdin/stdout as raw binary.
// Produces no files; the Python viewer drives it via a subprocess pipe.
//
// Protocol (stdout):
//   line 1: JSON metadata {"a": {"frames": N, "stamps": [...]}, "b": {...}}
//   then per stdin request "A <index>" | "B <index>" | "QUIT":
//     uint32 magic (0x314D4241)
//     uint32 n_vertices
//     uint32 n_faces
//     uint64 timestamp_ns
//     float32 vertices[3 * n_vertices]
//     uint32  faces[3 * n_faces]
//     uint8   colors[3 * n_vertices]   (RGB)
//
// stderr carries diagnostics so the stdout pipe stays binary-clean.
//
// Latency strategy: composeCurrentSceneMesh costs seconds per frame, so a
// worker thread computes requested (and prefetched neighbour) frames into an
// LRU mesh cache; the main loop serves cache hits instantly and only waits on
// a cache miss. SpatioTemporalMap::getDsgPtr is stateful (snapshot spill
// cache), so all such calls are serialized behind one mutex; composition is a
// pure read of the returned DSG and runs outside the lock.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <khronos/utils/khronos_attribute_utils.h>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

namespace {

constexpr std::uint32_t kMagic = 0x314D4241;  // "ABM1"
constexpr std::size_t kCacheCap = 32;

struct MapView {
  std::string name;  // "a" | "b"
  std::unique_ptr<khronos::SpatioTemporalMap> map;
  std::vector<khronos::TimeStamp> stamps;
};

struct Key {
  std::size_t view = 0;
  std::size_t index = 0;

  bool operator==(const Key& other) const {
    return view == other.view && index == other.index;
  }
};

struct KeyHash {
  std::size_t operator()(const Key& key) const {
    return key.view * 73856093u ^ key.index;
  }
};

// FrameResult: 0-vertex means "no mesh at this snapshot" and is cached too.
struct FrameResult {
  khronos::TimeStamp stamp = 0;
  std::vector<float> vertices;
  std::vector<std::uint32_t> faces;
  std::vector<std::uint8_t> colors;
};

class FrameCache {
 public:
  void put(Key key, FrameResult frame) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = by_key_.find(key);
    if (it != by_key_.end()) {
      order_.erase(it->second);
    }
    store_[key] = std::move(frame);
    order_.push_back(key);
    by_key_[key] = std::prev(order_.end());
    while (order_.size() > kCacheCap) {
      const Key victim = order_.front();
      order_.pop_front();
      by_key_.erase(victim);
      store_.erase(victim);
    }
  }

  // Returns nullptr if the key is not cached.
  const FrameResult* get(Key key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = store_.find(key);
    if (it == store_.end()) {
      return nullptr;
    }
    // Move to back (LRU touch).
    order_.erase(by_key_[key]);
    order_.push_back(key);
    by_key_[key] = std::prev(order_.end());
    return &it->second;
  }

 private:
  std::mutex mutex_;
  std::unordered_map<Key, FrameResult, KeyHash> store_;
  std::list<Key> order_;
  std::unordered_map<Key, std::list<Key>::iterator, KeyHash> by_key_;
};

class MeshService {
 public:
  MeshService(std::vector<MapView> views, std::size_t stride)
      : views_(std::move(views)), stride_(stride) {}

  void start() { worker_ = std::thread(&MeshService::run, this); }

  // Serves the requested frame, blocking until it is in the cache.
  void serve(Key key, std::ostream& out) {
    if (cache_.get(key) != nullptr) {
      writeFrame(key, out);
      prefetchNeighbours(key);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      if (!pending_.count(key)) {
        pending_.insert(key);
        queue_.push_back(key);
        queue_cv_.notify_all();
      }
    }
    waitFor(key);
    writeFrame(key, out);
    prefetchNeighbours(key);
  }

  void shutdown() {
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      quit_ = true;
    }
    queue_cv_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  // One-line JSON listing the snapshot's OBJECTS layer: id, semantic label,
  // private-mesh size, trajectory length. Diagnoses what the pipeline built.
  std::string inspect(std::size_t view, std::size_t index) {
    std::lock_guard<std::mutex> lock(map_mutex_);
    auto dsg = views_[view].map->getDsgPtr(views_[view].stamps[index]);
    nlohmann::json out = nlohmann::json::array();
    if (dsg && dsg->hasLayer(khronos::DsgLayers::OBJECTS)) {
      for (const auto& [node_id, node] :
           dsg->getLayer(khronos::DsgLayers::OBJECTS).nodes()) {
        const auto* attrs =
            node->tryAttributes<khronos::KhronosObjectAttributes>();
        if (!attrs) {
          continue;
        }
        std::size_t dynamic_points = 0;
        for (const auto& sample : attrs->dynamic_object_points) {
          dynamic_points += sample.size();
        }
        out.push_back(
            {{"id", node_id},
             {"label", static_cast<int>(attrs->semantic_label)},
             {"mesh_vertices", attrs->mesh.numVertices()},
             {"trajectory_len", khronos::trajectoryHistorySize(*attrs)},
             {"dynamic_points", dynamic_points}});
      }
    }
    return out.dump();
  }

  // Dynamic-object payload for one snapshot: per OBJECTS node with any
  // trajectory, emit id, label, full trajectory positions (world frame) and
  // the point cloud of the sample measured at or before the snapshot time
  // (world frame). The scene mesh never contains dynamic objects, so the
  // viewer renders this temporal layer on top. Historical samples are not
  // serialized -- the trajectory line already conveys where the object went.
  // Dynamic history comes from the final snapshot of the session, matching
  // the legacy export_dynamic_history() path used by the last working
  // visualization: every track carries its full timestamped trajectory and
  // per-timestamp point clouds, and the viewer selects the sample for the
  // queried time itself (person at time t -> point cloud at time t).
  std::string dynamicPayload(std::size_t view, std::size_t /*index*/) {
    std::lock_guard<std::mutex> lock(map_mutex_);
    const auto& stamps = views_[view].stamps;
    if (stamps.empty()) {
      return "[]";
    }
    auto dsg = views_[view].map->getDsgPtr(stamps.back());
    nlohmann::json out = nlohmann::json::array();
    if (dsg && dsg->hasLayer(khronos::DsgLayers::OBJECTS)) {
      for (const auto& [node_id, node] :
           dsg->getLayer(khronos::DsgLayers::OBJECTS).nodes()) {
        const auto* attrs =
            node->tryAttributes<khronos::KhronosObjectAttributes>();
        if (!attrs) {
          continue;
        }
        const std::size_t traj_len = khronos::trajectoryHistorySize(*attrs);
        if (traj_len == 0) {
          continue;
        }
        nlohmann::json traj = nlohmann::json::array();
        nlohmann::json traj_ts = nlohmann::json::array();
        nlohmann::json frames = nlohmann::json::array();
        const std::size_t n_ts = attrs->trajectory_timestamps.size();
        for (std::size_t i = 0; i < traj_len; ++i) {
          if (i < attrs->trajectory_positions.size()) {
            const auto& p = attrs->trajectory_positions[i];
            traj.push_back({p.x(), p.y(), p.z()});
          } else {
            traj.push_back({0.0, 0.0, 0.0});
          }
          traj_ts.push_back(i < n_ts ? attrs->trajectory_timestamps[i] : 0);
          // Per-timestamp point cloud, stride-capped like the legacy exporter
          // (<=1500 points per frame keeps the payload reasonable).
          nlohmann::json frame = nlohmann::json::array();
          if (i < attrs->dynamic_object_points.size()) {
            const auto& frame_points = attrs->dynamic_object_points[i];
            const std::size_t stride =
                std::max<std::size_t>(1, (frame_points.size() + 1499) / 1500);
            for (std::size_t j = 0; j < frame_points.size(); j += stride) {
              const auto& p = frame_points[j];
              frame.push_back({p.x(), p.y(), p.z()});
            }
          }
          frames.push_back(std::move(frame));
        }
        const auto& dims = attrs->bounding_box.dimensions;
        out.push_back({{"id", node_id},
                       {"label", static_cast<int>(attrs->semantic_label)},
                       {"bbox_dimensions",
                        {dims.x(), dims.y(), dims.z()}},
                       {"timestamps_ns", std::move(traj_ts)},
                       {"positions", std::move(traj)},
                       {"point_frames", std::move(frames)}});
      }
    }
    return out.dump();
  }

  const std::string& nameOf(std::size_t view) const {
    return views_[view].name;
  }

  std::size_t frameCount(std::size_t view) const {
    return views_[view].stamps.size();
  }

  std::size_t viewCount() const { return views_.size(); }

 private:
  std::vector<MapView> views_;
  std::size_t stride_ = 1;
  FrameCache cache_;
  std::thread worker_;
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::deque<Key> queue_;
  std::atomic<bool> quit_{false};
  std::unordered_set<Key, KeyHash> pending_;

  void prefetchNeighbours(Key key) {
    requestPrefetch(key, -1);
    requestPrefetch(key, +1);
  }

  void requestPrefetch(Key key, long delta) {
    const std::size_t frame_count = views_[key.view].stamps.size();
    if (delta < 0 && key.index == 0) {
      return;
    }
    if (delta > 0 && key.index + 1 >= frame_count) {
      return;
    }
    Key neighbour{key.view, static_cast<std::size_t>(key.index + delta)};
    if (cache_.get(neighbour) != nullptr) {
      return;
    }
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (pending_.count(neighbour)) {
      return;
    }
    pending_.insert(neighbour);
    queue_.push_back(neighbour);
    queue_cv_.notify_all();
  }

  void run() {
    while (true) {
      Key key;
      {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait(lock, [this]() { return quit_ || !queue_.empty(); });
        if (quit_) {
          return;
        }
        key = queue_.front();
        queue_.pop_front();
      }
      compute(key);
    }
  }

  void compute(Key key) {
    auto frame = std::make_unique<FrameResult>();
    frame->stamp = views_[key.view].stamps[key.index];
    {
      // getDsgPtr is stateful; serialize all map access.
      std::lock_guard<std::mutex> lock(map_mutex_);
      auto dsg = views_[key.view].map->getDsgPtr(frame->stamp);
      if (dsg && dsg->hasMesh() && dsg->mesh()) {
        auto display = khronos::composeCurrentSceneMesh(*dsg);
        const auto& mesh = *display;
        frame->vertices.resize(3 * mesh.numVertices());
        frame->colors.resize(3 * mesh.numVertices());
        for (std::size_t i = 0; i < mesh.numVertices(); ++i) {
          const auto& p = mesh.pos(i);
          frame->vertices[3 * i + 0] = static_cast<float>(p.x());
          frame->vertices[3 * i + 1] = static_cast<float>(p.y());
          frame->vertices[3 * i + 2] = static_cast<float>(p.z());
          const auto color =
              mesh.has_colors && i < mesh.colors.size()
                  ? mesh.colors[i]
                  : spark_dsg::Color(180, 180, 180);
          frame->colors[3 * i + 0] = color.r;
          frame->colors[3 * i + 1] = color.g;
          frame->colors[3 * i + 2] = color.b;
        }
        frame->faces.resize(3 * mesh.numFaces());
        for (std::size_t i = 0; i < mesh.numFaces(); ++i) {
          const auto& face = mesh.face(i);
          frame->faces[3 * i + 0] = static_cast<std::uint32_t>(face[0]);
          frame->faces[3 * i + 1] = static_cast<std::uint32_t>(face[1]);
          frame->faces[3 * i + 2] = static_cast<std::uint32_t>(face[2]);
        }
      }
    }
    // Optional vertex-stride decimation: keeps every stride-th vertex, drops
    // faces that reference a removed vertex. Cuts pipe transfer and client
    // GPU upload ~stride^2 so 1cm maps stay draggable. The composition above
    // still runs at full resolution; this only lightens the wire payload.
    if (stride_ > 1 && !frame->vertices.empty()) {
      const std::size_t full_verts = frame->vertices.size() / 3;
      std::vector<std::int64_t> remap(full_verts, -1);
      std::size_t kept = 0;
      for (std::size_t i = 0; i < full_verts; ++i) {
        if (i % stride_ == 0) {
          remap[i] = static_cast<std::int64_t>(kept++);
        }
      }
      std::vector<float> verts(3 * kept);
      std::vector<std::uint8_t> colors(3 * kept);
      for (std::size_t i = 0; i < full_verts; ++i) {
        if (remap[i] < 0) {
          continue;
        }
        const std::size_t dst = static_cast<std::size_t>(remap[i]);
        verts[3 * dst + 0] = frame->vertices[3 * i + 0];
        verts[3 * dst + 1] = frame->vertices[3 * i + 1];
        verts[3 * dst + 2] = frame->vertices[3 * i + 2];
        colors[3 * dst + 0] = frame->colors[3 * i + 0];
        colors[3 * dst + 1] = frame->colors[3 * i + 1];
        colors[3 * dst + 2] = frame->colors[3 * i + 2];
      }
      std::vector<std::uint32_t> faces;
      faces.reserve(frame->faces.size() / stride_);
      for (std::size_t i = 0; i + 2 < frame->faces.size(); i += 3) {
        const auto f0 = remap[frame->faces[i]];
        const auto f1 = remap[frame->faces[i + 1]];
        const auto f2 = remap[frame->faces[i + 2]];
        if (f0 >= 0 && f1 >= 0 && f2 >= 0) {
          faces.push_back(static_cast<std::uint32_t>(f0));
          faces.push_back(static_cast<std::uint32_t>(f1));
          faces.push_back(static_cast<std::uint32_t>(f2));
        }
      }
      frame->vertices = std::move(verts);
      frame->colors = std::move(colors);
      frame->faces = std::move(faces);
    }
    cache_.put(key, std::move(*frame));
    std::cerr << "computed " << views_[key.view].name << " " << key.index
              << " (" << frame->vertices.size() / 3 << "v)\n";
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      pending_.erase(key);
      // Wake any thread waiting for this key.
    }
    wait_cv_.notify_all();
  }

  void waitFor(Key key) {
    while (true) {
      if (cache_.get(key) != nullptr) {
        return;
      }
      std::unique_lock<std::mutex> lock(queue_mutex_);
      if (cache_.get(key) != nullptr) {
        return;
      }
      wait_cv_.wait(lock);
    }
  }

  void writeFrame(Key key, std::ostream& out) {
    const FrameResult* frame = cache_.get(key);
    const std::uint32_t magic = kMagic;
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    if (!frame) {
      const std::uint32_t zero = 0;
      out.write(reinterpret_cast<const char*>(&zero), sizeof(zero));
      out.write(reinterpret_cast<const char*>(&zero), sizeof(zero));
      const std::uint64_t ts = 0;
      out.write(reinterpret_cast<const char*>(&ts), sizeof(ts));
      out.flush();
      return;
    }
    const std::uint32_t n_vertices =
        static_cast<std::uint32_t>(frame->vertices.size() / 3);
    const std::uint32_t n_faces =
        static_cast<std::uint32_t>(frame->faces.size() / 3);
    out.write(reinterpret_cast<const char*>(&n_vertices), sizeof(n_vertices));
    out.write(reinterpret_cast<const char*>(&n_faces), sizeof(n_faces));
    out.write(reinterpret_cast<const char*>(&frame->stamp), sizeof(frame->stamp));
    if (n_vertices > 0) {
      out.write(reinterpret_cast<const char*>(frame->vertices.data()),
                static_cast<std::streamsize>(frame->vertices.size() * sizeof(float)));
      out.write(reinterpret_cast<const char*>(frame->faces.data()),
                static_cast<std::streamsize>(frame->faces.size() * sizeof(std::uint32_t)));
      out.write(reinterpret_cast<const char*>(frame->colors.data()),
                static_cast<std::streamsize>(frame->colors.size()));
    }
    out.flush();
  }

  std::mutex map_mutex_;
  std::condition_variable wait_cv_;
};

}  // namespace

int main(int argc, char** argv) {
  std::string map_a;
  std::string map_b;
  std::size_t stride = 1;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto value = [&]() {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + key);
      }
      return std::string(argv[++i]);
    };
    if (key == "--map_a" || key == "--map-a") {
      map_a = value();
    } else if (key == "--map_b" || key == "--map-b") {
      map_b = value();
    } else if (key == "--stride") {
      stride = std::stoull(value());
    } else {
      throw std::runtime_error("Unknown argument: " + key);
    }
  }
  if (map_a.empty() && map_b.empty()) {
    throw std::runtime_error("at least one of --map_a / --map_b is required");
  }
  if (stride == 0) {
    throw std::runtime_error("--stride must be >= 1");
  }

  std::vector<MapView> views;
  auto load = [&](const std::string& path, const std::string& name) {
    if (path.empty()) {
      return;
    }
    std::uintmax_t bytes = 0;
    std::error_code ec;
    const auto size = fs::file_size(path, ec);
    if (!ec) {
      bytes = size;
    }
    std::cerr << "loading map " << name << ": " << path << " ("
              << (bytes / (1024.0 * 1024.0 * 1024.0)) << " GiB)\n";
    std::cerr.flush();
    auto map = khronos::SpatioTemporalMap::load(path);
    if (!map) {
      throw std::runtime_error("failed to load map: " + path);
    }
    views.emplace_back();
    views.back().name = name;
    views.back().map = std::move(map);
    views.back().stamps = views.back().map->stamps();
    std::cerr << "loaded map " << name << " (" << views.back().stamps.size()
              << " frames)\n";
    std::cerr.flush();
  };
  load(map_a, "a");
  load(map_b, "b");

  nlohmann::json metadata = nlohmann::json::object();
  for (const auto& view : views) {
    nlohmann::json stamps = nlohmann::json::array();
    for (const auto stamp : view.stamps) {
      stamps.push_back(stamp);
    }
    metadata[view.name] = {{"frames", view.stamps.size()}, {"stamps", stamps}};
  }
  std::cout << metadata.dump() << "\n";
  std::cout.flush();
  std::cerr << "server ready maps=" << views.size() << "\n";

  MeshService service(std::move(views), stride);
  service.start();

  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream request(line);
    std::string session;
    std::size_t index = 0;
    request >> session >> index;
    if (session == "QUIT") {
      break;
    }
    if (session == "INFO") {
      std::size_t view = static_cast<std::size_t>(-1);
      for (std::size_t i = 0; i < service.viewCount(); ++i) {
        if (service.nameOf(i) == request.str().substr(5, 1)) {
          view = i;
          break;
        }
      }
      // Request is "INFO <session> <index>"; re-parse the session token.
      std::istringstream info_request(line);
      std::string info_cmd;
      std::string info_session;
      std::size_t info_index = 0;
      info_request >> info_cmd >> info_session >> info_index;
      for (std::size_t i = 0; i < service.viewCount(); ++i) {
        if (service.nameOf(i) == info_session) {
          view = i;
          break;
        }
      }
      if (view == static_cast<std::size_t>(-1) ||
          info_index >= service.frameCount(view)) {
        std::cout << "{\"error\":\"bad info request\"}\n";
        std::cout.flush();
        continue;
      }
      std::cout << service.inspect(view, info_index) << "\n";
      std::cout.flush();
      continue;
    }
    if (session == "DYNPTS") {
      std::istringstream dyn_request(line);
      std::string dyn_cmd;
      std::string dyn_session;
      std::size_t dyn_index = 0;
      dyn_request >> dyn_cmd >> dyn_session >> dyn_index;
      std::size_t dyn_view = static_cast<std::size_t>(-1);
      for (std::size_t i = 0; i < service.viewCount(); ++i) {
        if (service.nameOf(i) == dyn_session) {
          dyn_view = i;
          break;
        }
      }
      if (dyn_view == static_cast<std::size_t>(-1) ||
          dyn_index >= service.frameCount(dyn_view)) {
        std::cout << "{\"error\":\"bad dynpts request\"}\n";
        std::cout.flush();
        continue;
      }
      std::cout << service.dynamicPayload(dyn_view, dyn_index) << "\n";
      std::cout.flush();
      continue;
    }
    std::size_t view = static_cast<std::size_t>(-1);
    for (std::size_t i = 0; i < service.viewCount(); ++i) {
      if (service.nameOf(i) == session) {
        view = i;
        break;
      }
    }
    if (view == static_cast<std::size_t>(-1) ||
        index >= service.frameCount(view)) {
      std::cerr << "bad request: " << line << "\n";
      const std::uint32_t magic = kMagic;
      std::cout.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
      const std::uint32_t zero = 0;
      std::cout.write(reinterpret_cast<const char*>(&zero), sizeof(zero));
      std::cout.write(reinterpret_cast<const char*>(&zero), sizeof(zero));
      const std::uint64_t ts = 0;
      std::cout.write(reinterpret_cast<const char*>(&ts), sizeof(ts));
      std::cout.flush();
      continue;
    }
    service.serve(Key{view, index}, std::cout);
  }
  service.shutdown();
  return 0;
}
