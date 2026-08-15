#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <hydra/common/global_info.h>
#include <hydra/input/camera.h>
#include <hydra/input/input_data.h>
#include <hydra/input/sensor_extrinsics.h>
#include <khronos/backend/change_detection/background/ray_background_change_detector.h>
#include <khronos/backend/change_detection/objects/ray_object_change_detector.h>
#include <khronos/backend/change_detection/physical_evidence_store.h>
#include <khronos/backend/change_detection/sequential_change_detector.h>
#include <khronos/backend/reconciliation/mesh/change_merger.h>
#include <khronos/backend/reconciliation/reconciler.h>
#include <khronos/backend/update_khronos_objects_functor.h>
#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <khronos/utils/khronos_attribute_utils.h>
#include <hydra/utils/pgmo_mesh_traits.h>
#include <kimera_pgmo/mesh_delta.h>
#include <kimera_pgmo/mesh_offset_info.h>
#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/mesh.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>

#include "session_update_baseline/runtime/session_state.h"

namespace {

using Dsg = spark_dsg::DynamicSceneGraph;
using ObjectAttrs = spark_dsg::KhronosObjectAttributes;
using Stamp = khronos::TimeStamp;

constexpr Stamp kInitialStamp = 1'000'000'000ULL;
constexpr Stamp kNewStamp = 10'000'000'000ULL;
constexpr Stamp kTerminalStamp = 20'000'000'000ULL;
constexpr int kObjectSemantic = 75;

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << "\n";
    std::exit(1);
  }
}

std::unique_ptr<ObjectAttrs> makeObject(size_t instance_id,
                                        float x,
                                        float marker,
                                        Stamp stamp) {
  auto attrs = std::make_unique<ObjectAttrs>();
  attrs->semantic_label = 75;
  attrs->details["instance_id"] = {instance_id};
  attrs->details["test_marker"] = {static_cast<size_t>(marker)};
  attrs->position = Eigen::Vector3d(x, 0.0, 1.0);
  attrs->bounding_box = khronos::BoundingBox(
      khronos::Point(0.2F, 0.2F, 0.2F), khronos::Point(x, 0.0F, 1.0F));
  attrs->first_observed_ns = {stamp};
  attrs->last_observed_ns = {stamp};
  attrs->mesh.resizeVertices(1);
  attrs->mesh.setPos(0, khronos::Point::Zero());
  khronos::setObservationBounds(*attrs, stamp, stamp);
  return attrs;
}

std::vector<khronos::Point> i7MixedCoveragePoints() {
  std::vector<khronos::Point> points;
  for (size_t i = 0; i < 10; ++i) {
    points.emplace_back(-2.0F + 0.2F * static_cast<float>(i), 0.0F, 2.0F);
  }
  return points;
}

std::unique_ptr<ObjectAttrs> makeMultiVertexObject(
    size_t instance_id,
    float marker,
    Stamp stamp,
    const std::vector<khronos::Point>& world_points) {
  require(!world_points.empty(), "multi-vertex object needs geometry");
  auto attrs = std::make_unique<ObjectAttrs>();
  attrs->semantic_label = kObjectSemantic;
  attrs->details["instance_id"] = {instance_id};
  attrs->details["test_marker"] = {static_cast<size_t>(marker)};
  attrs->bounding_box = khronos::BoundingBox(world_points);
  attrs->position = attrs->bounding_box.world_P_center.cast<double>();
  attrs->first_observed_ns = {stamp};
  attrs->last_observed_ns = {stamp};
  attrs->mesh = spark_dsg::Mesh(false, true, false, true);
  attrs->mesh.resizeVertices(world_points.size());
  for (size_t i = 0; i < world_points.size(); ++i) {
    attrs->mesh.setPos(i, world_points[i] - attrs->bounding_box.world_P_center);
    attrs->mesh.setFirstSeenTimestamp(i, stamp);
    attrs->mesh.setTimestamp(i, stamp);
  }
  khronos::setObservationBounds(*attrs, stamp, stamp);
  return attrs;
}

std::shared_ptr<hydra::Camera> makeEvidenceCamera() {
  hydra::Camera::Config config;
  config.min_range = 0.1;
  config.max_range = 20.0;
  config.width = 256;
  config.height = 64;
  config.cx = 128.0f;
  config.cy = 32.0f;
  config.fx = 10.0f;
  config.fy = 10.0f;
  config.extrinsics = hydra::ParamSensorExtrinsics::Config();
  return std::make_shared<hydra::Camera>(config, "hidden_change_evidence_camera");
}

cv::Point projectEvidencePixel(const khronos::Point& point) {
  const int u = static_cast<int>(std::lround(10.0F * point.x() / point.z() + 128.0F));
  const int v = static_cast<int>(std::lround(10.0F * point.y() / point.z() + 32.0F));
  return {u, v};
}

khronos::FrameData makeEvidenceFrame(const hydra::Sensor::ConstPtr& camera,
                                     Stamp stamp) {
  hydra::InputData input(camera);
  input.timestamp_ns = stamp;
  input.world_T_body = Eigen::Isometry3d::Identity();
  input.range_image = cv::Mat(64, 256, CV_32FC1, cv::Scalar(2.0F));
  input.depth_image = cv::Mat(64, 256, CV_32FC1, cv::Scalar(2.0F));
  input.label_image = cv::Mat(64, 256, CV_32SC1, cv::Scalar(1));
  input.color_image = cv::Mat(64, 256, CV_8UC3, cv::Scalar(0, 0, 0));
  khronos::FrameData frame(input);
  frame.instance_image = cv::Mat::zeros(64, 256, CV_32SC1);
  frame.dynamic_image = cv::Mat::zeros(64, 256, CV_32SC1);
  return frame;
}

khronos::PhysicalEvidenceStore::Ptr makeNewEvidenceStore(
    const hydra::Sensor::ConstPtr& camera) {
  auto store = std::make_shared<khronos::PhysicalEvidenceStore>();
  auto frame = makeEvidenceFrame(camera, kNewStamp);

  const auto i7_points = i7MixedCoveragePoints();
  for (size_t i = 0; i < i7_points.size(); ++i) {
    const auto pixel = projectEvidencePixel(i7_points[i]);
    require(pixel.x >= 0 && pixel.x < frame.instance_image.cols && pixel.y >= 0 &&
                pixel.y < frame.instance_image.rows,
            "I7 evidence projects inside the synthetic image");
    // Six typed I9 occluder samples and four reviewed-background samples give
    // 0.4 physical absence confidence: real coverage, but not enough to delete
    // I7. I9 deliberately never materializes as a terminal DSG object here.
    if (i < 6) {
      frame.instance_image.at<int>(pixel.y, pixel.x) = 9;
    }
  }

  // I21 is a legacy/unidentified object at an exactly matching old surface.
  // It is covered, but cannot be deleted from anonymous semantic evidence.
  const auto unknown_pixel = projectEvidencePixel(khronos::Point(8.0F, 0.0F, 1.0F));
  frame.dynamic_image.at<int>(unknown_pixel.y, unknown_pixel.x) = 1;

  require(store->ingest(frame), "new-session typed endpoint evidence is ingested");
  return store;
}

khronos::PhysicalEvidenceStore::Ptr makeBackgroundEvidenceStore(
    const hydra::Sensor::ConstPtr& camera,
    Stamp stamp) {
  auto store = std::make_shared<khronos::PhysicalEvidenceStore>();
  auto frame = makeEvidenceFrame(camera, stamp);
  require(store->ingest(frame), "typed background endpoint evidence is ingested");
  return store;
}

void addAgentPose(Dsg& graph, Stamp stamp, size_t index) {
  khronos::RayVerificator::Config ray_config;
  const auto key = graph.getLayerKey(spark_dsg::DsgLayers::AGENTS);
  require(key.has_value(), "graph has no AGENTS layer key");
  if (!graph.findLayer(key->layer, ray_config.prefix.key)) {
    graph.addLayer(key->layer, ray_config.prefix.key);
  }
  const auto pose_id = spark_dsg::NodeSymbol(ray_config.prefix.key, index);
  auto attrs = std::make_unique<spark_dsg::AgentNodeAttributes>(
      std::chrono::nanoseconds(stamp),
      Eigen::Quaterniond::Identity(),
      Eigen::Vector3d::Zero(),
      pose_id);
  require(graph.emplaceNode(
              key->layer, pose_id, std::move(attrs), ray_config.prefix.key),
          "could not insert agent pose");
}

void setBackground(Dsg& graph,
                   const std::vector<std::tuple<khronos::Point, Stamp, Stamp>>& vertices) {
  auto mesh = std::make_shared<spark_dsg::Mesh>(true, true, true, true);
  mesh->resizeVertices(vertices.size());
  for (size_t i = 0; i < vertices.size(); ++i) {
    mesh->setPos(i, std::get<0>(vertices[i]));
    mesh->setFirstSeenTimestamp(i, std::get<1>(vertices[i]));
    mesh->setTimestamp(i, std::get<2>(vertices[i]));
    mesh->setLabel(i, 1);
  }
  graph.setMesh(mesh);
}

Dsg::Ptr makeInitialObservation() {
  auto graph = std::make_shared<Dsg>();
  // I6 moves under the same physical identity. I20 is replaced by reviewed
  // background, I7 gets mixed background/I9-occluder coverage while I9 has not
  // materialized as a DSG object, I21 gets unidentified-object coverage, and
  // I22 remains wholly unobserved.
  require(graph->emplaceNode(khronos::DsgLayers::OBJECTS,
                             spark_dsg::NodeSymbol('O', 6),
                             makeObject(6, 0.0F, 6.0F, kInitialStamp)),
          "insert initial I6");
  require(graph->emplaceNode(khronos::DsgLayers::OBJECTS,
                             spark_dsg::NodeSymbol('O', 20),
                             makeObject(20, 4.0F, 20.0F, kInitialStamp)),
          "insert initial I20");
  require(graph->emplaceNode(khronos::DsgLayers::OBJECTS,
                             spark_dsg::NodeSymbol('O', 7),
                             makeMultiVertexObject(7,
                                                   7.0F,
                                                   kInitialStamp,
                                                   i7MixedCoveragePoints())),
          "insert initial I7");
  require(graph->emplaceNode(khronos::DsgLayers::OBJECTS,
                             spark_dsg::NodeSymbol('O', 21),
                             makeObject(21, 8.0F, 21.0F, kInitialStamp)),
          "insert initial unidentified-near I21");
  require(graph->emplaceNode(khronos::DsgLayers::OBJECTS,
                             spark_dsg::NodeSymbol('O', 22),
                             makeObject(22, 10.0F, 22.0F, kInitialStamp)),
          "insert initial unobserved I22");

  // This old background point will be cleared by a later ray through x=1.
  setBackground(*graph,
                {{khronos::Point(1.0F, 0.0F, 1.0F),
                  kInitialStamp,
                  kInitialStamp}});
  addAgentPose(*graph, kInitialStamp, 0);
  return graph;
}

khronos::Reconciler::Config makeReconcilerConfig() {
  khronos::Reconciler::Config config;
  config.time_estimates_conservative = false;
  config.allow_overestimation = true;
  config.merge_object_meshes = true;
  khronos::ChangeMerger::Config mesh;
  mesh.remove_objects_from_background = false;
  config.mesh_merger = mesh;
  return config;
}

void makeInitialCurrentState(Dsg& graph) {
  khronos::Changes changes;
  for (const auto& [node_id, node] :
       graph.getLayer(khronos::DsgLayers::OBJECTS).nodes()) {
    (void)node;
    changes.object_changes.emplace_back();
    changes.object_changes.back().node_id = node_id;
  }
  changes.background_changes.resize(
      graph.mesh()->numVertices(), khronos::ChangeState::kUnobserved);
  khronos::Reconciler(makeReconcilerConfig()).reconcile(
      graph, changes, kInitialStamp);

  for (const auto& [node_id, node] :
       graph.getLayer(khronos::DsgLayers::OBJECTS).nodes()) {
    (void)node_id;
    const auto& attrs = node->attributes<ObjectAttrs>();
    require(attrs.last_observed_ns.back() ==
                std::numeric_limits<Stamp>::max(),
            "initial current object did not receive an open presence interval");
    require(khronos::observationLastStamp(attrs) == kInitialStamp,
            "actual observation bound was lost during initial reconciliation");
  }
}

void appendNewObservations(Dsg& graph) {
  const auto old = graph.mesh();
  std::vector<std::tuple<khronos::Point, Stamp, Stamp>> vertices;
  for (size_t i = 0; i < old->numVertices(); ++i) {
    vertices.emplace_back(
        old->pos(i), old->firstSeenTimestamp(i), old->timestamp(i));
  }
  // A farther surface clears I6's old x=0 site, and a direct I6 segment at x=2
  // proves that the same logical object is current there. The point at x=4 is
  // deliberately within exact depth tolerance of old I20: the typed reviewed-
  // background endpoint is replacement evidence, never anonymous persistence.
  vertices.emplace_back(khronos::Point(0.0F, 0.0F, 2.0F),
                        kNewStamp,
                        kNewStamp);
  vertices.emplace_back(khronos::Point(1.0F, 0.0F, 2.0F),
                        kNewStamp,
                        kNewStamp);
  vertices.emplace_back(khronos::Point(2.0F, 0.0F, 2.0F),
                        kNewStamp,
                        kNewStamp);
  vertices.emplace_back(khronos::Point(4.0F, 0.0F, 1.0F),
                        kNewStamp,
                        kNewStamp);
  for (const auto& point : i7MixedCoveragePoints()) {
    vertices.emplace_back(point, kNewStamp, kNewStamp);
  }
  vertices.emplace_back(khronos::Point(8.0F, 0.0F, 1.0F),
                        kNewStamp,
                        kNewStamp);
  setBackground(graph, vertices);
  addAgentPose(graph, kNewStamp, 1);

  require(graph.emplaceNode(khronos::DsgLayers::OBJECTS,
                            spark_dsg::NodeSymbol('O', 106),
                            makeObject(6, 2.0F, 106.0F, kNewStamp)),
          "insert moved I6 terminal segment");
  require(graph.emplaceNode(khronos::DsgLayers::OBJECTS,
                            spark_dsg::NodeSymbol('O', 112),
                            makeObject(12, 6.0F, 112.0F, kNewStamp)),
          "insert new I12");
}

khronos::SequentialChangeDetector::Config makeDetectorConfig() {
  khronos::SequentialChangeDetector::Config config;
  config.ray_verificator.block_size = 0.25F;
  config.ray_verificator.radial_tolerance = 0.05F;
  config.ray_verificator.depth_tolerance = 0.05F;
  config.ray_verificator.ray_policy =
      khronos::RayVerificator::Config::RayPolicy::kAll;
  config.ray_verificator.active_window_duration = 0.0F;
  config.ray_change_detector.temporal_resolution = 1.0F;
  config.ray_change_detector.window_size = 1;
  config.ray_change_detector.use_relative_confidence = true;
  config.ray_change_detector.absence_confidence = 0.5F;
  config.ray_change_detector.presence_confidence = 0.5F;

  khronos::RayObjectChangeDetector::Config objects;
  objects.time_filtering_threshold = 0.1F;
  objects.query_subsampling = 1;
  config.objects = objects;
  khronos::RayBackgroundChangeDetector::Config background;
  background.time_filtering_threshold = 0.1F;
  config.background = background;
  return config;
}

khronos::ObjectChanges updateHidden(
    Dsg& graph,
    const khronos::PhysicalEvidenceStore::Ptr& evidence_store,
    Stamp stamp = kNewStamp) {
  // Production order: detect/reconcile each visibility segment first, then
  // reduce stable physical IDs to one authoritative current node.
  auto snapshot = graph.clone();
  khronos::SequentialChangeDetector detector(makeDetectorConfig());
  detector.setPhysicalEvidenceStore(evidence_store);
  detector.setDsg(snapshot);
  const auto& changes = detector.detectChanges({}, stamp, true);
  const auto result = changes.object_changes;
  khronos::Reconciler(makeReconcilerConfig()).reconcile(
      graph, changes, stamp);
  const auto merged =
      khronos::UpdateKhronosObjectsFunctor::canonicalizePhysicalObjects(graph);
  require(merged == 1,
          "physical visibility segments were not reduced after reconciliation");
  return result;
}

struct ObjectSummary {
  float center_x = 0.0F;
  float marker = 0.0F;
  Stamp observed_first = 0;
  Stamp observed_last = 0;
  Stamp presence_first = 0;
  Stamp presence_last = 0;
  std::vector<Stamp> presence_starts;
  std::vector<Stamp> presence_ends;
  std::vector<std::tuple<float, float, float>> world_geometry;

  bool operator==(const ObjectSummary& other) const {
    return center_x == other.center_x && marker == other.marker &&
           observed_first == other.observed_first &&
           observed_last == other.observed_last &&
           presence_first == other.presence_first &&
           presence_last == other.presence_last &&
           presence_starts == other.presence_starts &&
           presence_ends == other.presence_ends &&
           world_geometry == other.world_geometry;
  }
};

struct CurrentSummary {
  std::map<size_t, ObjectSummary> objects;
  std::vector<khronos::Point> background;
};

CurrentSummary summarizeCurrent(const Dsg& graph) {
  CurrentSummary result;
  const auto& objects = graph.getLayer(khronos::DsgLayers::OBJECTS);
  for (const auto& [node_id, node] : objects.nodes()) {
    (void)node_id;
    const auto& attrs = node->attributes<ObjectAttrs>();
    const auto physical_id =
        khronos::UpdateKhronosObjectsFunctor::physicalInstanceId(attrs);
    require(physical_id.has_value(), "current object lacks physical identity");
    require(!result.objects.count(*physical_id),
            "latest state has duplicate physical instance IDs");
    ObjectSummary summary;
    summary.center_x = attrs.bounding_box.world_P_center.x();
    const auto marker = attrs.details.find("test_marker");
    summary.marker = marker == attrs.details.end() || marker->second.empty()
                         ? -1.0F
                         : static_cast<float>(marker->second.front());
    summary.observed_first = khronos::observationFirstStamp(attrs);
    summary.observed_last = khronos::observationLastStamp(attrs);
    summary.presence_first = attrs.first_observed_ns.empty()
                                 ? 0
                                 : attrs.first_observed_ns.front();
    summary.presence_last = attrs.last_observed_ns.empty()
                                ? 0
                                : attrs.last_observed_ns.back();
    summary.presence_starts.assign(attrs.first_observed_ns.begin(),
                                   attrs.first_observed_ns.end());
    summary.presence_ends.assign(attrs.last_observed_ns.begin(),
                                 attrs.last_observed_ns.end());
    for (const auto& local_point : attrs.mesh.points) {
      const auto world_point = local_point + attrs.bounding_box.world_P_center;
      summary.world_geometry.emplace_back(
          world_point.x(), world_point.y(), world_point.z());
    }
    std::sort(summary.world_geometry.begin(), summary.world_geometry.end());
    result.objects.emplace(*physical_id, summary);
  }
  for (size_t i = 0; i < graph.mesh()->numVertices(); ++i) {
    result.background.push_back(graph.mesh()->pos(i));
  }
  std::sort(result.background.begin(), result.background.end(),
            [](const auto& lhs, const auto& rhs) {
              return std::tie(lhs.x(), lhs.y(), lhs.z()) <
                     std::tie(rhs.x(), rhs.y(), rhs.z());
            });
  return result;
}

void requireEquivalent(const CurrentSummary& lhs, const CurrentSummary& rhs) {
  require(lhs.objects == rhs.objects,
          "continuous D2 and serialized D3 object states differ");
  require(lhs.background.size() == rhs.background.size(),
          "continuous D2 and serialized D3 background sizes differ");
  for (size_t i = 0; i < lhs.background.size(); ++i) {
    require((lhs.background[i] - rhs.background[i]).norm() < 1.0e-6F,
            "continuous D2 and serialized D3 background geometry differs");
  }
}

const ObjectAttrs* findPhysicalObject(const Dsg& graph, size_t physical_id) {
  if (!graph.hasLayer(khronos::DsgLayers::OBJECTS)) {
    return nullptr;
  }
  const ObjectAttrs* result = nullptr;
  for (const auto& [unused_id, node] :
       graph.getLayer(khronos::DsgLayers::OBJECTS).nodes()) {
    (void)unused_id;
    const auto& attrs = node->attributes<ObjectAttrs>();
    if (khronos::UpdateKhronosObjectsFunctor::physicalInstanceId(attrs) !=
        std::optional<size_t>(physical_id)) {
      continue;
    }
    require(result == nullptr,
            "graph contains duplicate physical ID " +
                std::to_string(physical_id));
    result = &attrs;
  }
  return result;
}

const khronos::ObjectChange* findPhysicalChange(
    const Dsg& graph,
    const khronos::ObjectChanges& changes,
    size_t physical_id) {
  if (!graph.hasLayer(khronos::DsgLayers::OBJECTS)) {
    return nullptr;
  }
  for (const auto& [node_id, node] :
       graph.getLayer(khronos::DsgLayers::OBJECTS).nodes()) {
    const auto& attrs = node->attributes<ObjectAttrs>();
    if (khronos::UpdateKhronosObjectsFunctor::physicalInstanceId(attrs) !=
        std::optional<size_t>(physical_id)) {
      continue;
    }
    const auto change = changes.find(node_id);
    if (change != changes.end()) {
      return &*change;
    }
  }
  return nullptr;
}

void testMovedThenTerminalAbsent(const std::filesystem::path& output_dir,
                                 const hydra::Sensor::ConstPtr& camera) {
  auto graph = std::make_shared<Dsg>();
  require(graph->emplaceNode(khronos::DsgLayers::OBJECTS,
                             spark_dsg::NodeSymbol('O', 8),
                             makeObject(8, 0.0F, 8.0F, kInitialStamp)),
          "insert old I8 segment");
  // Every ray avoids I8's old x=0 surface, so this historical segment stays
  // unobserved/open. It must still be unable to overrule the newest segment's
  // finite right edge after the object moves.
  setBackground(*graph,
                {{khronos::Point(-1.0F, 0.0F, 1.0F),
                  kInitialStamp,
                  kInitialStamp}});
  addAgentPose(*graph, kInitialStamp, 0);
  makeInitialCurrentState(*graph);

  setBackground(
      *graph,
      {{khronos::Point(-1.0F, 0.0F, 1.0F),
        kInitialStamp,
        kInitialStamp},
       // The x=2,z=1 private surface lies exactly halfway along this
       // terminal ray, which is therefore true free-space absence evidence.
       {khronos::Point(4.0F, 0.0F, 2.0F),
        kTerminalStamp,
        kTerminalStamp}});
  addAgentPose(*graph, kNewStamp, 1);
  addAgentPose(*graph, kTerminalStamp, 2);
  require(graph->emplaceNode(khronos::DsgLayers::OBJECTS,
                             spark_dsg::NodeSymbol('O', 108),
                             makeObject(8, 2.0F, 108.0F, kNewStamp)),
          "insert moved I8 segment");

  const auto changes = updateHidden(
      *graph, makeBackgroundEvidenceStore(camera, kTerminalStamp), kTerminalStamp);
  const auto old_change = changes.find(spark_dsg::NodeSymbol('O', 8));
  const auto newest_change = changes.find(spark_dsg::NodeSymbol('O', 108));
  require(old_change != changes.end() && old_change->last_absent == 0 &&
              old_change->last_persistent == 0,
          "unobserved old I8 site acquired fabricated ray evidence");
  require(newest_change != changes.end() &&
              newest_change->last_absent == kTerminalStamp &&
              newest_change->last_persistent == 0,
          "new I8 site terminal free-space was not classified absent");

  const auto* terminal_attrs = findPhysicalObject(*graph, 8);
  const auto expected_right = (kNewStamp + kTerminalStamp) / 2;
  require(terminal_attrs != nullptr &&
              terminal_attrs->first_observed_ns ==
                  std::vector<Stamp>({kInitialStamp}) &&
              terminal_attrs->last_observed_ns ==
                  std::vector<Stamp>({expected_right}),
          "I8 canonical terminal interval is not newest-right authoritative");
  require(khronos::observationFirstStamp(*terminal_attrs) == kNewStamp &&
              khronos::observationLastStamp(*terminal_attrs) == kNewStamp,
          "I8 canonical direct observation provenance is not newest-segment authoritative");
  require(khronos::isPresent(*terminal_attrs, kInitialStamp) &&
              khronos::isPresent(*terminal_attrs, kNewStamp) &&
              khronos::isPresent(*terminal_attrs, expected_right - 1) &&
              !khronos::isPresent(*terminal_attrs, expected_right) &&
              !khronos::isPresent(*terminal_attrs, kTerminalStamp),
          "I8 presence transition is incorrect around its terminal right edge");

  khronos::SpatioTemporalMap terminal_map(
      khronos::SpatioTemporalMap::Config{});
  terminal_map.update(graph->clone(), kNewStamp);
  terminal_map.update(graph, kTerminalStamp);
  require(!summarizeCurrent(*terminal_map.getDsgPtr(kTerminalStamp))
               .objects.count(8),
          "old open I8 history swallowed newest terminal absence");

  const auto path = output_dir / "moved_then_terminal_absent.4dmap";
  std::filesystem::remove(path);
  require(terminal_map.save(path.string()),
          "failed to save moved-then-absent terminal map");
  auto loaded = khronos::SpatioTemporalMap::load(path.string());
  require(loaded != nullptr,
          "failed to reload moved-then-absent terminal map");
  const auto loaded_present = summarizeCurrent(*loaded->getDsgPtr(kNewStamp));
  require(loaded_present.objects.count(8) == 1 &&
              loaded_present.objects.at(8).presence_starts ==
                  std::vector<Stamp>({kInitialStamp}) &&
              loaded_present.objects.at(8).presence_ends ==
                  std::vector<Stamp>({expected_right}) &&
              loaded_present.objects.at(8).observed_first == kNewStamp &&
              loaded_present.objects.at(8).observed_last == kNewStamp,
          "I8 full interval/provenance changed across save/load");
  require(!summarizeCurrent(*loaded->getDsgPtr(kTerminalStamp))
               .objects.count(8),
          "loaded terminal state resurrected I8");
  const auto c_seed = session_update::runtime::latestSessionSeed(*loaded);
  require(!summarizeCurrent(*c_seed.dsg).objects.count(8),
          "moved-then-absent I8 was resurrected in C seed");
  std::filesystem::remove(path);
}

void requireReseededBackendSharesLiveMesh(const session_update::runtime::SessionSeed& seed) {
  auto private_graph = std::make_shared<Dsg>();
  auto unmerged_graph = std::make_shared<Dsg>();
  session_update::runtime::initializeHiddenChangeWorkingDsgPair(
      seed, *private_graph, *unmerged_graph);

  require(private_graph->mesh() == unmerged_graph->mesh(),
          "reseed split the backend's shared live mesh");
  const auto prior_vertices = private_graph->mesh()->numVertices();
  const auto prior_faces = private_graph->mesh()->numFaces();

  kimera_pgmo::traits::VertexTraits traits;
  traits.properties.has_color = true;
  traits.properties.has_stamp = true;
  traits.properties.has_label = true;
  traits.properties.has_first_seen_stamp = true;
  traits.color = {1, 2, 3, 255};
  traits.stamp = kNewStamp;
  traits.label = 42;
  traits.first_seen_stamp = kNewStamp;
  kimera_pgmo::MeshDelta delta({});
  delta.addVertex(khronos::Point(3.0F, 2.0F, 1.0F), traits);
  kimera_pgmo::MeshOffsetInfo offsets(
      prior_vertices, prior_vertices, prior_faces);
  delta.updateMesh(*private_graph->mesh(), offsets);

  require(private_graph->mesh()->numVertices() == prior_vertices + 1,
          "test MeshDelta did not append to the reseeded private mesh");
  require(unmerged_graph->mesh()->numVertices() == prior_vertices + 1,
          "MeshDelta applied to private mesh was invisible to change-detection graph");
  require(unmerged_graph->mesh()->label(prior_vertices) == 42,
          "MeshDelta label was not visible through the shared reseeded mesh");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: test_hidden_change_session_equivalence OUTPUT_DIR\n";
    return 2;
  }

  hydra::PipelineConfig global_config;
  global_config.store_visualization_details = true;
  global_config.label_space.total_labels = 256;
  global_config.label_space.object_labels.insert(kObjectSemantic);
  hydra::GlobalInfo::init(global_config);
  const auto evidence_camera = makeEvidenceCamera();
  require(hydra::GlobalInfo::instance().setSensor(evidence_camera),
          "synthetic evidence camera is registered");

  auto initial = makeInitialObservation();
  makeInitialCurrentState(*initial);

  // D2: the process remains alive and old observation provenance is still in
  // memory. The same hidden-change transition consumes the later observations.
  auto continuous = initial->clone();
  appendNewObservations(*continuous);
  const auto continuous_changes =
      updateHidden(*continuous, makeNewEvidenceStore(evidence_camera));

  const auto* i20_change =
      findPhysicalChange(*continuous, continuous_changes, 20);
  require(i20_change != nullptr,
          "old I20 has no physical change record");
  require(i20_change->last_absent == kNewStamp &&
              i20_change->last_persistent == 0,
          "typed background replacement did not remove I20");
  const auto* i7_change =
      findPhysicalChange(*continuous, continuous_changes, 7);
  require(i7_change != nullptr && i7_change->last_absent == 0 &&
              i7_change->last_persistent == 0,
          "mixed I9-occluder/background coverage incorrectly deleted I7");
  const auto* i21_change =
      findPhysicalChange(*continuous, continuous_changes, 21);
  require(i21_change != nullptr && i21_change->last_absent == 0 &&
              i21_change->last_persistent == 0,
          "unidentified near-surface evidence incorrectly deleted I21");
  const auto* old_i6_change =
      findPhysicalChange(*continuous, continuous_changes, 6);
  require(old_i6_change != nullptr &&
              old_i6_change->last_absent == kNewStamp &&
              old_i6_change->last_persistent == 0,
          "later direct same-ID I6 segment erased old-site absence evidence");

  // D3: serialize the exact same midpoint, start a fresh process-level working
  // state from latest(P_prev), append the exact same observations, and invoke
  // the exact same updateHidden() function. There is no cross-session algorithm.
  const std::filesystem::path output_dir(argv[1]);
  std::filesystem::create_directories(output_dir);
  const auto state_path = output_dir / "hidden_change_midpoint.4dmap";
  std::filesystem::remove(state_path);
  khronos::SpatioTemporalMap midpoint(khronos::SpatioTemporalMap::Config{});
  midpoint.update(initial, kInitialStamp);
  require(midpoint.save(state_path.string()), "failed to save midpoint state");
  auto loaded = khronos::SpatioTemporalMap::load(state_path.string());
  require(loaded && loaded->numTimeSteps() == 1,
          "failed to reload midpoint state");
  const auto seed = session_update::runtime::latestSessionSeed(*loaded);
  requireReseededBackendSharesLiveMesh(seed);
  auto restarted = std::make_shared<Dsg>();
  session_update::runtime::initializeHiddenChangeWorkingDsg(seed, *restarted);
  appendNewObservations(*restarted);
  const auto restarted_changes =
      updateHidden(*restarted, makeNewEvidenceStore(evidence_camera));
  const auto* restarted_i20 =
      findPhysicalChange(*restarted, restarted_changes, 20);
  require(restarted_i20 != nullptr &&
              restarted_i20->last_absent == kNewStamp &&
              restarted_i20->last_persistent == 0,
          "serialized D3 classified I20 differently from continuous D2: absent=" +
              (restarted_i20 == nullptr
                   ? std::string("missing")
                   : std::to_string(restarted_i20->last_absent)) +
              ", persistent=" +
              (restarted_i20 == nullptr
                   ? std::string("missing")
                   : std::to_string(restarted_i20->last_persistent)));
  const auto* restarted_i7 =
      findPhysicalChange(*restarted, restarted_changes, 7);
  const auto* restarted_i21 =
      findPhysicalChange(*restarted, restarted_changes, 21);
  require(restarted_i7 != nullptr && restarted_i7->last_absent == 0 &&
              restarted_i21 != nullptr && restarted_i21->last_absent == 0,
          "serialized D3 changed mixed/unknown coverage semantics");

  // Compare authoritative *current* materializations, not the raw temporal
  // nodes that deliberately retain absent history.
  khronos::SpatioTemporalMap d2_map(khronos::SpatioTemporalMap::Config{});
  d2_map.update(continuous, kNewStamp);
  khronos::SpatioTemporalMap d3_map(khronos::SpatioTemporalMap::Config{});
  d3_map.update(restarted, kNewStamp);
  const auto d2_current = d2_map.getDsgPtr(kNewStamp);
  const auto d3_current = d3_map.getDsgPtr(kNewStamp);
  const auto d2 = summarizeCurrent(*d2_current);
  const auto d3 = summarizeCurrent(*d3_current);
  requireEquivalent(d2, d3);
  require(d2.objects.count(6) && d2.objects.at(6).marker == 106.0F &&
              d2.objects.at(6).center_x == 2.0F,
          "moved I6 did not retain exactly its terminal direct materialization");
  require(!d2.objects.count(9),
          "frame-local active I9 was incorrectly materialized as a DSG object");
  require(d2.objects.count(7),
          "mixed typed coverage incorrectly deleted I7");
  require(d2.objects.count(21),
          "unknown legacy near evidence incorrectly deleted I21");
  require(d2.objects.count(22),
          "wholly unobserved I22 was incorrectly deleted");
  require(!d2.objects.count(20),
          "old I20 survived typed background-replacement evidence");
  require(d2.objects.count(12),
          "new I12 was not materialized in current state");
  require(d2.objects.at(12).presence_first == kNewStamp,
          "new physical I12 was incorrectly back-dated to time zero");
  require(d2.objects.at(12).presence_last ==
              std::numeric_limits<Stamp>::max(),
          "unobserved terminal I12 did not remain open/current");
  require(d2.background.size() == 15,
          "old background was not removed while new background was retained");

  // An absent terminal object must not leak through serialization into the
  // next recursive seed.
  const auto terminal_path = output_dir / "hidden_change_terminal.4dmap";
  std::filesystem::remove(terminal_path);
  require(d3_map.save(terminal_path.string()), "failed to save D3 terminal state");
  auto terminal_loaded = khronos::SpatioTemporalMap::load(terminal_path.string());
  require(terminal_loaded != nullptr, "failed to reload D3 terminal state");
  const auto c_seed = session_update::runtime::latestSessionSeed(*terminal_loaded);
  const auto c_summary = summarizeCurrent(*c_seed.dsg);
  requireEquivalent(d3, c_summary);
  require(!c_summary.objects.count(20),
          "absent I20 was resurrected in the C seed");
  require(c_summary.objects.count(6) && c_summary.objects.count(7) &&
              c_summary.objects.count(12) && c_summary.objects.count(21) &&
              c_summary.objects.count(22) && !c_summary.objects.count(9),
          "C seed lost a valid current physical object");

  testMovedThenTerminalAbsent(output_dir, evidence_camera);

  std::filesystem::remove(state_path);
  std::filesystem::remove(terminal_path);
  hydra::GlobalInfo::reset();
  std::cout << "hidden_change_session_equivalence_passed\n";
  return 0;
}
