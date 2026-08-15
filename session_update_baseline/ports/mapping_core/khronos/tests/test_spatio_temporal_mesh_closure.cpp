#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <spark_dsg/dynamic_scene_graph.h>

#include "khronos/spatio_temporal_map/spatio_temporal_map.h"

namespace {

using khronos::DynamicSceneGraph;
using khronos::SpatioTemporalMap;
using khronos::TimeStamp;
using spark_dsg::Color;
using spark_dsg::Mesh;

constexpr TimeStamp kFirst = 100;
constexpr TimeStamp kMiddle = 200;
constexpr TimeStamp kLatest = 300;

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << "\n";
    std::exit(EXIT_FAILURE);
  }
}

Mesh::Ptr makeMesh(bool colors, bool timestamps, bool labels) {
  auto mesh = std::make_shared<Mesh>(colors, timestamps, labels, true);
  mesh->resizeVertices(4);

  // Deliberately not time ordered. finalizeMesh() must reorder every enabled
  // per-vertex field with the same permutation and remap faces.
  const std::vector<TimeStamp> first_seen{kLatest, kFirst, kMiddle, kLatest};
  const std::vector<float> markers{30.0f, 10.0f, 20.0f, 31.0f};
  for (size_t i = 0; i < markers.size(); ++i) {
    mesh->setPos(i, Mesh::Pos(markers[i], markers[i] + 0.25f, markers[i] + 0.5f));
    mesh->setFirstSeenTimestamp(i, first_seen[i]);
    if (colors) {
      mesh->setColor(i,
                     Color(static_cast<uint8_t>(markers[i]),
                           static_cast<uint8_t>(markers[i] + 1),
                           static_cast<uint8_t>(markers[i] + 2)));
    }
    if (timestamps) {
      mesh->setTimestamp(i, 1'000 + static_cast<uint64_t>(markers[i]));
    }
    if (labels) {
      mesh->setLabel(i, 2'000 + static_cast<uint32_t>(markers[i]));
    }
  }
  mesh->faces.push_back({1, 2, 0});
  mesh->faces.push_back({1, 0, 3});
  return mesh;
}

DynamicSceneGraph::Ptr makeDsg(bool colors, bool timestamps, bool labels) {
  auto dsg = std::make_shared<DynamicSceneGraph>();
  dsg->setMesh(makeMesh(colors, timestamps, labels));
  return dsg;
}

SpatioTemporalMap makeMap(bool colors, bool timestamps, bool labels) {
  SpatioTemporalMap map(SpatioTemporalMap::Config{});
  auto first = makeDsg(colors, timestamps, labels);
  first->mesh()->resizeVertices(2);
  first->mesh()->faces.clear();
  map.update(first, kMiddle);
  map.update(makeDsg(colors, timestamps, labels), kLatest);
  return map;
}

void requireMeshEqual(const Mesh& actual,
                      const Mesh& expected,
                      const std::string& context) {
  require(actual.has_colors == expected.has_colors, context + ": color flag");
  require(actual.has_timestamps == expected.has_timestamps, context + ": timestamp flag");
  require(actual.has_labels == expected.has_labels, context + ": label flag");
  require(actual.has_first_seen_stamps == expected.has_first_seen_stamps,
          context + ": first-seen flag");
  require(actual.points == expected.points, context + ": positions");
  require(actual.colors == expected.colors, context + ": colors");
  require(actual.stamps == expected.stamps, context + ": timestamps");
  require(actual.first_seen_stamps == expected.first_seen_stamps,
          context + ": first-seen timestamps");
  require(actual.labels == expected.labels, context + ": labels");
  require(actual.faces == expected.faces, context + ": faces");
}

Mesh::Ptr queryDirectLatest(bool colors, bool timestamps, bool labels) {
  auto map = makeMap(colors, timestamps, labels);
  return map.getDsgPtr(kLatest)->mesh()->clone();
}

Mesh::Ptr querySteppedLatest(bool colors, bool timestamps, bool labels) {
  auto map = makeMap(colors, timestamps, labels);
  (void)map.getDsgPtr(kFirst);
  (void)map.getDsgPtr(kMiddle);
  return map.getDsgPtr(kLatest)->mesh()->clone();
}

void testEveryEnabledFieldCombination() {
  for (unsigned mask = 0; mask < 8; ++mask) {
    const bool colors = mask & 1;
    const bool timestamps = mask & 2;
    const bool labels = mask & 4;
    const auto direct = queryDirectLatest(colors, timestamps, labels);
    const auto stepped = querySteppedLatest(colors, timestamps, labels);
    requireMeshEqual(*stepped, *direct, "direct versus stepped mask=" + std::to_string(mask));
  }
}

void testLabelsSurviveFinalizeForwardSaveLoadAndReseed() {
  auto map = makeMap(true, true, true);
  const auto direct = map.getDsgPtr(kLatest)->mesh()->clone();

  require(direct->labels == Mesh::Labels({2010, 2020, 2030, 2031}),
          "finalize and direct forward preserve labels in first-seen order");
  require(direct->first_seen_stamps ==
              Mesh::Timestamps({kFirst, kMiddle, kLatest, kLatest}),
          "first-seen timestamps are sorted with vertices");
  require(direct->stamps == Mesh::Timestamps({1010, 1020, 1030, 1031}),
          "ordinary timestamps follow the same permutation");
  require(direct->faces == Mesh::Faces({Mesh::Face{0, 1, 2}, Mesh::Face{0, 2, 3}}),
          "faces are remapped to reordered vertices");

  const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto path = std::filesystem::temp_directory_path() /
                    ("khronos_mesh_closure_" + std::to_string(unique) + ".4dmap");
  require(map.save(path.string()), "map save succeeds");
  auto loaded = SpatioTemporalMap::load(path.string());
  std::error_code error;
  std::filesystem::remove(path, error);
  require(loaded != nullptr, "map load succeeds");
  const auto loaded_latest = loaded->getDsgPtr(kLatest);
  requireMeshEqual(*loaded_latest->mesh(), *direct, "save/load latest");

  // This is the same geometry hand-off used by a recurrent session: the prior
  // latest state becomes the sole seed snapshot of a new map.
  SpatioTemporalMap reseeded(SpatioTemporalMap::Config{});
  reseeded.update(loaded_latest->clone(), kLatest + 100);
  const auto reseeded_latest = reseeded.getDsgPtr(kLatest + 100);
  requireMeshEqual(*reseeded_latest->mesh(), *direct, "loaded latest reseed");
}

}  // namespace

int main() {
  testEveryEnabledFieldCombination();
  testLabelsSurviveFinalizeForwardSaveLoadAndReseed();
  return EXIT_SUCCESS;
}
