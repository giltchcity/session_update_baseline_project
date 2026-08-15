#include <sys/types.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/node_symbol.h>
#include <spark_dsg/serialization/binary_serialization.h>
#include <spark_dsg/serialization/graph_binary_serialization.h>
#include <spark_dsg/serialization/versioning.h>

#include "khronos/spatio_temporal_map/spatio_temporal_map.h"
#include "khronos/utils/khronos_attribute_utils.h"

namespace {

using Dsg = spark_dsg::DynamicSceneGraph;
using Mesh = spark_dsg::Mesh;
using ObjectAttrs = spark_dsg::KhronosObjectAttributes;
using khronos::SpatioTemporalMap;
using khronos::TimeStamp;

constexpr TimeStamp kBaseStamp = 1'000'000'000ULL;

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << "\n";
    std::exit(EXIT_FAILURE);
  }
}

Dsg::Ptr makeRichSnapshot(size_t step) {
  const TimeStamp stamp = kBaseStamp * (step + 1);
  auto dsg = std::make_shared<Dsg>();
  auto mesh = std::make_shared<Mesh>(true, true, true, true);
  const size_t num_vertices = 5 + 3 * step;
  mesh->resizeVertices(num_vertices);
  for (size_t i = 0; i < num_vertices; ++i) {
    mesh->setPos(i,
                 Eigen::Vector3f(static_cast<float>(step),
                                 0.01f * static_cast<float>(i),
                                 1.0f + 0.001f * static_cast<float>(i)));
    mesh->setColor(i,
                   spark_dsg::Color(static_cast<uint8_t>(10 + step),
                                    static_cast<uint8_t>(20 + i),
                                    static_cast<uint8_t>(30 + step + i),
                                    255));
    mesh->setTimestamp(i, stamp + i);
    mesh->setFirstSeenTimestamp(i, stamp);
    mesh->setLabel(i, static_cast<uint32_t>(70 + step));
  }
  for (size_t i = 2; i < num_vertices; ++i) {
    mesh->faces.push_back({0, i - 1, i});
  }
  dsg->setMesh(mesh);

  const auto agent_key = dsg->getLayerKey(spark_dsg::DsgLayers::AGENTS);
  require(agent_key.has_value(), "default DSG has an agent layer");
  const auto agent_id = spark_dsg::NodeSymbol('a', step);
  auto agent = std::make_unique<spark_dsg::AgentNodeAttributes>(
      std::chrono::nanoseconds(stamp),
      Eigen::Quaterniond::Identity(),
      Eigen::Vector3d(static_cast<double>(step), 0.0, 0.0),
      agent_id);
  require(dsg->emplaceNode(agent_key->layer, agent_id, std::move(agent), 'a'),
          "agent node insertion succeeds");

  auto object = std::make_unique<ObjectAttrs>();
  object->position = Eigen::Vector3d(step, 0.25, 1.0);
  object->last_update_time_ns = stamp;
  object->is_active = false;
  object->is_predicted = false;
  object->name = "physical_10_step_" + std::to_string(step);
  object->color = spark_dsg::Color(1, 2, 3, 255);
  object->semantic_label = 75;
  object->bounding_box = spark_dsg::BoundingBox(
      Eigen::Vector3f(0.5f, 0.4f, 0.8f),
      Eigen::Vector3f(static_cast<float>(step), 0.25f, 1.0f));
  object->registered = true;
  object->world_R_object = Eigen::Quaterniond::Identity();
  object->first_observed_ns = {kBaseStamp};
  object->last_observed_ns = {std::numeric_limits<uint64_t>::max()};
  object->mesh = Mesh(true, true, true, true);
  object->mesh.resizeVertices(3);
  for (size_t i = 0; i < 3; ++i) {
    object->mesh.setPos(i, Eigen::Vector3f(0.01f * i, 0.02f * i, 0.0f));
    object->mesh.setColor(i, spark_dsg::Color(40 + i, 50 + i, 60 + i));
    object->mesh.setTimestamp(i, stamp);
    object->mesh.setFirstSeenTimestamp(i, kBaseStamp);
    object->mesh.setLabel(i, 75);
  }
  object->mesh.faces.push_back({0, 1, 2});
  object->trajectory_timestamps = {kBaseStamp, stamp};
  object->trajectory_positions = {
      Eigen::Vector3f::Zero(),
      Eigen::Vector3f(static_cast<float>(step), 0.25f, 1.0f)};
  object->dynamic_object_points = {
      {Eigen::Vector3f::Zero()},
      {Eigen::Vector3f(static_cast<float>(step), 0.25f, 1.0f)}};
  object->details["instance_id"] = {10};
  object->details["observation_first_stamp_ns"] = {kBaseStamp};
  object->details["observation_last_stamp_ns"] = {stamp};
  require(dsg->emplaceNode(spark_dsg::DsgLayers::OBJECTS,
                           spark_dsg::NodeSymbol('O', 10),
                           std::move(object)),
          "object node insertion succeeds");
  return dsg;
}

Dsg::Ptr makeLargeSnapshot(size_t num_vertices, TimeStamp stamp) {
  auto dsg = std::make_shared<Dsg>();
  // Deliberately omit first-seen stamps and disable incremental finalization in
  // the RSS fixture: this isolates snapshot retention from mesh sort scratch.
  auto mesh = std::make_shared<Mesh>(true, true, true, false);
  mesh->resizeVertices(num_vertices);
  for (size_t i = 0; i < num_vertices; ++i) {
    mesh->setPos(i,
                 Eigen::Vector3f(0.001f * static_cast<float>(i), 0.0f, 1.0f));
    mesh->setColor(i, spark_dsg::Color(1, 2, 3, 255));
    mesh->setTimestamp(i, stamp);
    mesh->setLabel(i, 3);
  }
  dsg->setMesh(mesh);
  return dsg;
}

uint64_t fingerprint(const Dsg& dsg) {
  std::vector<uint8_t> bytes;
  spark_dsg::io::binary::writeGraph(dsg, bytes, true);
  uint64_t hash = 1469598103934665603ULL;
  for (const auto byte : bytes) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  return hash;
}

void writeLegacyMap(const std::filesystem::path& path,
                    const SpatioTemporalMap& map,
                    const std::vector<Dsg::Ptr>& snapshots) {
  std::vector<uint8_t> bytes;
  spark_dsg::serialization::BinarySerializer serializer(&bytes);
  serializer.write(1);
  serializer.write(map.config.finalize_incrementally);
  serializer.write(map.stamps().size());
  serializer.write(map.stamps());
  serializer.write(map.earliest());
  serializer.write(map.latest());
  serializer.write(true);
  serializer.write(spark_dsg::io::FileHeader::current().serializeToBinary());
  for (const auto& dsg : snapshots) {
    std::vector<uint8_t> graph_bytes;
    spark_dsg::io::binary::writeGraph(*dsg, graph_bytes, true);
    serializer.write(graph_bytes);
  }
  std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  require(out.good(), "legacy reference map write succeeds");
}

std::vector<uint8_t> readFile(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::in | std::ios::binary);
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), {});
}

size_t currentRssKb() {
  std::ifstream status("/proc/self/status");
  std::string key;
  while (status >> key) {
    if (key == "VmRSS:") {
      size_t value = 0;
      std::string units;
      status >> value >> units;
      return value;
    }
    std::string remainder;
    std::getline(status, remainder);
  }
  return 0;
}

size_t countOpenFileDescriptors() {
  size_t result = 0;
  for (const auto& entry : std::filesystem::directory_iterator("/proc/self/fd")) {
    (void)entry;
    ++result;
  }
  return result;
}

struct MemoryMeasurement {
  size_t rss_kb = 0;
  uint64_t snapshot_bytes = 0;
  size_t resident_sources = 0;
};

MemoryMeasurement measureRetention(bool legacy) {
  constexpr size_t kSteps = 8;
  constexpr size_t kVerticesPerStep = 75'000;
  std::vector<Dsg::Ptr> retained;
  SpatioTemporalMap::Config config;
  config.finalize_incrementally = false;
  SpatioTemporalMap map(config);
  for (size_t step = 1; step <= kSteps; ++step) {
    auto dsg = makeLargeSnapshot(step * kVerticesPerStep, step * kBaseStamp);
    if (legacy) {
      retained.push_back(dsg);
    } else {
      map.update(dsg, step * kBaseStamp);
    }
    dsg.reset();
  }
  return {currentRssKb(),
          legacy ? 0 : map.snapshotStorageBytes(),
          legacy ? retained.size() : map.numResidentSourceSnapshots()};
}

MemoryMeasurement measureInChild(bool legacy) {
  int pipe_fds[2];
  require(::pipe(pipe_fds) == 0, "memory measurement pipe creation succeeds");
  const pid_t child = ::fork();
  require(child >= 0, "memory measurement fork succeeds");
  if (child == 0) {
    ::close(pipe_fds[0]);
    const auto result = measureRetention(legacy);
    const auto bytes = ::write(pipe_fds[1], &result, sizeof(result));
    ::close(pipe_fds[1]);
    ::_exit(bytes == static_cast<ssize_t>(sizeof(result)) ? 0 : 1);
  }

  ::close(pipe_fds[1]);
  MemoryMeasurement result;
  const auto bytes = ::read(pipe_fds[0], &result, sizeof(result));
  ::close(pipe_fds[0]);
  int status = 0;
  require(::waitpid(child, &status, 0) == child,
          "memory measurement child is reaped");
  require(bytes == static_cast<ssize_t>(sizeof(result)) &&
              WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "memory measurement child succeeds");
  return result;
}

void testExactTimelineAndLegacyCompatibility() {
  const auto unique = std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count());
  const auto root = std::filesystem::temp_directory_path();
  const auto streamed_path = root / ("spilled_map_" + unique + ".4dmap");
  const auto legacy_path = root / ("legacy_map_" + unique + ".4dmap");

  SpatioTemporalMap zero_time_map(SpatioTemporalMap::Config{});
  zero_time_map.update(makeRichSnapshot(0), 0);
  require(static_cast<bool>(zero_time_map.getDsgPtr(0)),
          "timestamp-zero query materializes a DSG instead of returning null");

  SpatioTemporalMap map(SpatioTemporalMap::Config{});
  std::vector<Dsg::Ptr> legacy_snapshots;
  for (size_t step = 0; step < 6; ++step) {
    auto dsg = makeRichSnapshot(step);
    map.update(dsg, (step + 1) * kBaseStamp);
    legacy_snapshots.push_back(std::move(dsg));
    require(map.numResidentSourceSnapshots() <= 1,
            "timeline retains at most one source graph in RAM");
  }
  require(map.numTimeSteps() == legacy_snapshots.size(),
          "every update remains in the timeline");
  require(map.snapshotStorageBytes() > 0,
          "lossless snapshot bytes are present in spill storage");

  std::vector<uint64_t> expected;
  for (const auto stamp : map.stamps()) {
    expected.push_back(fingerprint(*map.getDsgPtr(stamp)));
  }
  // Exercise source-cache eviction and backward/forward query order.
  for (auto it = map.stamps().rbegin(); it != map.stamps().rend(); ++it) {
    (void)map.getDsgPtr(*it);
    require(map.numResidentSourceSnapshots() == 1,
            "random access still uses a one-entry source cache");
  }

  require(map.save(streamed_path.string()), "streamed map save succeeds");
  writeLegacyMap(legacy_path, map, legacy_snapshots);
  require(readFile(streamed_path) == readFile(legacy_path),
          "streamed output is byte-compatible with legacy v1 serialization");

  auto loaded_streamed = SpatioTemporalMap::load(streamed_path.string());
  auto loaded_legacy = SpatioTemporalMap::load(legacy_path.string());
  require(loaded_streamed && loaded_legacy,
          "both streamed and legacy v1 maps load");
  require(loaded_streamed->stamps() == map.stamps() &&
              loaded_legacy->stamps() == map.stamps(),
          "snapshot order and timestamps survive load");
  require(loaded_streamed->numResidentSourceSnapshots() == 0 &&
              loaded_legacy->numResidentSourceSnapshots() == 0,
          "load does not eagerly materialize any graph");
  for (size_t i = 0; i < map.stamps().size(); ++i) {
    const auto stamp = map.stamps()[i];
    require(fingerprint(*loaded_streamed->getDsgPtr(stamp)) == expected[i],
            "streamed round trip preserves every DSG field");
    require(fingerprint(*loaded_legacy->getDsgPtr(stamp)) == expected[i],
            "legacy load preserves every DSG field");
  }

  // Copying shares immutable blobs. Replacing the copied terminal state must
  // not rewrite the original map's terminal snapshot.
  SpatioTemporalMap copied = map;
  const auto original_latest = fingerprint(*map.getDsgPtr(map.latest()));
  copied.update(makeRichSnapshot(99), copied.latest());
  require(fingerprint(*map.getDsgPtr(map.latest())) == original_latest,
          "terminal replacement in a copy leaves the original blob unchanged");
  require(fingerprint(*copied.getDsgPtr(copied.latest())) != original_latest,
          "copied map receives its own terminal replacement blob");

  std::error_code error;
  std::filesystem::remove(streamed_path, error);
  std::filesystem::remove(legacy_path, error);
}

void testBoundedResidentMemory() {
  const auto spilled = measureInChild(false);
  const auto legacy = measureInChild(true);
  require(spilled.resident_sources == 1,
          "spilled timeline has exactly one resident source snapshot");
  require(spilled.snapshot_bytes > 0,
          "spilled timeline retains nonzero lossless history bytes");
  require(legacy.resident_sources == 8,
          "legacy fixture retains every source snapshot");
  require(spilled.rss_kb < legacy.rss_kb,
          "spilled timeline RSS is lower than legacy cumulative retention");
  std::cout << "snapshot_storage_measurement spilled_rss_kb=" << spilled.rss_kb
            << " legacy_rss_kb=" << legacy.rss_kb
            << " spill_bytes=" << spilled.snapshot_bytes
            << " resident_sources=" << spilled.resident_sources << "\n";
}

void testBoundedFileDescriptors() {
  const pid_t child = ::fork();
  require(child >= 0, "fd-bound child creation succeeds");
  if (child == 0) {
    constexpr size_t kSteps = 192;
    const size_t baseline_fds = countOpenFileDescriptors();

    struct rlimit descriptor_limit;
    require(::getrlimit(RLIMIT_NOFILE, &descriptor_limit) == 0,
            "fd-bound child reads RLIMIT_NOFILE");
    rlim_t target_limit = static_cast<rlim_t>(baseline_fds + 16);
    if (descriptor_limit.rlim_max != RLIM_INFINITY) {
      target_limit = std::min(target_limit, descriptor_limit.rlim_max);
    }
    require(target_limit >= static_cast<rlim_t>(baseline_fds + 8),
            "fd-bound child has enough descriptor headroom");
    descriptor_limit.rlim_cur = target_limit;
    require(::setrlimit(RLIMIT_NOFILE, &descriptor_limit) == 0,
            "fd-bound child lowers RLIMIT_NOFILE");

    {
      const auto unique = std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count());
      const auto map_path = std::filesystem::temp_directory_path() /
                            ("fd_bounded_map_" + unique + ".4dmap");

      SpatioTemporalMap map(SpatioTemporalMap::Config{});
      for (size_t step = 0; step < kSteps; ++step) {
        map.update(makeRichSnapshot(step), (step + 1) * kBaseStamp);
      }
      require(countOpenFileDescriptors() <= baseline_fds + 1,
              "hundreds of snapshots use one spill-container descriptor");

      std::vector<uint64_t> expected;
      expected.reserve(kSteps);
      for (const auto stamp : map.stamps()) {
        expected.push_back(fingerprint(*map.getDsgPtr(stamp)));
      }

      require(map.save(map_path.string()),
              "fd-bounded timeline streams successfully");
      auto loaded = SpatioTemporalMap::load(map_path.string());
      require(loaded && loaded->stamps() == map.stamps(),
              "fd-bounded timeline loads every timestamp");
      require(countOpenFileDescriptors() <= baseline_fds + 2,
              "loaded timeline adds only one spill-container descriptor");
      for (size_t i = 0; i < kSteps; ++i) {
        require(fingerprint(*loaded->getDsgPtr(loaded->stamps()[i])) == expected[i],
                "fd-bounded round trip preserves every timeline step");
      }

      // A shallow map copy shares the immutable container. Replacing its last
      // snapshot appends a new extent and must not mutate the original extent.
      SpatioTemporalMap copied = map;
      const auto original_latest = fingerprint(*map.getDsgPtr(map.latest()));
      auto replacement = makeRichSnapshot(kSteps - 1);
      replacement->mesh()->setPos(
          0, replacement->mesh()->pos(0) + Eigen::Vector3f(42.0f, 0.0f, 0.0f));
      copied.update(replacement, copied.latest());
      require(fingerprint(*map.getDsgPtr(map.latest())) == original_latest,
              "fd-bounded copy replacement preserves the original extent");
      require(fingerprint(*copied.getDsgPtr(copied.latest())) != original_latest,
              "fd-bounded copy replacement receives a distinct extent");
      require(countOpenFileDescriptors() <= baseline_fds + 2,
              "copy and replacement do not allocate another descriptor");

      std::error_code error;
      std::filesystem::remove(map_path, error);
    }
    ::_exit(EXIT_SUCCESS);
  }

  int status = 0;
  require(::waitpid(child, &status, 0) == child,
          "fd-bound child is reaped");
  require(WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "timeline remains exact below a one-fd-per-snapshot descriptor limit");
}

}  // namespace

int main() {
  testExactTimelineAndLegacyCompatibility();
  testBoundedResidentMemory();
  testBoundedFileDescriptors();
  return EXIT_SUCCESS;
}
