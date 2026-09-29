// Unit checks of the session-end re-integration pieces:
//  - PresentTsdf reproduces Open3D 0.18 ScalableTSDFVolume on toy frames
//    (expectations measured with Open3D, wf_impl/design/offline_checks/
//    open3d_toy_expectations.json);
//  - TriangleGrid closest point and first hit equal brute force;
//  - FrameArchive frame packing and dump round trip;
//  - SessionRefusion on a scene with known answers: a previous surface face on
//    the measured plane stays, one in the plane's free space goes, one just
//    behind it (within the background truncation) goes, one far behind it
//    stays, and an object's face behind its own object's pixels goes; the
//    edited map, saved and loaded, is the next previous surface;
//  - a map with an empty or without a background mesh gets the present's
//    background faces;
//  - Track::stateStart: right after the last observed motion, else the first
//    sighting.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <set>

#include <khronos/active_window/data/track.h>
#include <khronos/backend/reconciliation/frame_archive.h>
#include <khronos/backend/reconciliation/present_tsdf.h>
#include <khronos/backend/reconciliation/triangle_grid.h>
#include <khronos/backend/reconciliation/session_refusion.h>
#include <khronos/backend/reconciliation/session_surface.h>
#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <khronos/utils/khronos_attribute_utils.h>
#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/node_symbol.h>

using namespace khronos;

namespace {

void require(bool value, const std::string& message) {
  if (!value) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(1);
  }
}

constexpr int W = 64, H = 48;
const PresentTsdf::Camera kCamera{W, H, 50.f, 50.f, 32.f, 24.f};

std::vector<float> rayNorm() {
  std::vector<float> m(W * H);
  for (int v = 0; v < H; ++v)
    for (int u = 0; u < W; ++u) {
      const float x = (u - 32.f) * (1.f / 50.f), y = (v - 24.f) * (1.f / 50.f);
      m[v * W + u] = std::sqrt(x * x + y * y + 1.f);
    }
  return m;
}

struct Mesh {
  std::vector<Eigen::Vector3f> V;
  std::vector<PresentTsdf::Face> F;
};

Mesh extract(const PresentTsdf& tsdf) {
  Mesh m;
  tsdf.extractMesh(m.V, m.F);
  return m;
}

std::vector<float> constant(float z) { return std::vector<float>(W * H, z); }

void testTsdf() {
  const auto mult = rayNorm();
  const Eigen::Isometry3d I = Eigen::Isometry3d::Identity();
  {
    PresentTsdf tsdf(0.02, 0.04, 2);
    tsdf.integrate(kCamera, I, constant(1.f), mult);
    const auto m = extract(tsdf);
    std::cout << "plane_z1 " << m.V.size() << " " << m.F.size() << '\n';
    require(m.V.size() == 2961 && m.F.size() == 5704, "plane z=1: 2961 vertices / 5704 faces");
    size_t toward = 0;
    for (const auto& f : m.F) {
      const Eigen::Vector3f n = (m.V[f[1]] - m.V[f[0]]).cross(m.V[f[2]] - m.V[f[0]]);
      toward += n.z() < 0;
    }
    require(toward == m.F.size(), "plane z=1: every face normal points to the camera");
    for (int k = 0; k < 3; ++k) tsdf.integrate(kCamera, I, constant(3.f), mult);
    const auto m2 = extract(tsdf);
    size_t near1 = 0;
    for (const auto& p : m2.V) near1 += std::abs(p.z() - 1.f) < 0.1f;
    std::cout << "plane_z1_then_3x_z3 " << m2.V.size() << " " << m2.F.size() << " " << near1 << '\n';
    require(m2.V.size() == 29128 && m2.F.size() == 57384 && near1 == 2961,
            "see-through frames leave the plane (29128 / 57384, 2961 near z=1)");
  }
  {
    PresentTsdf tsdf(0.02, 0.04, 2);
    tsdf.integrate(kCamera, I, constant(1.f), mult);
    for (int k = 0; k < 3; ++k) tsdf.integrate(kCamera, I, constant(1.2f), mult);
    const auto m = extract(tsdf);
    size_t near1 = 0;
    for (const auto& p : m.V) near1 += std::abs(p.z() - 1.f) < 0.05f;
    std::cout << "plane_z1_then_3x_z1p2 " << m.V.size() << " " << m.F.size() << " " << near1 << '\n';
    require(m.V.size() == 4332 && m.F.size() == 8400 && near1 == 0,
            "frames in the same unit remove the plane (4332 / 8400)");
  }
  {
    PresentTsdf tsdf(0.02, 0.04, 2);
    auto z = constant(0.f);
    for (int v = 1; v < H; v += 4)
      for (int u = 1; u < W; u += 4) z[v * W + u] = 1.f;
    tsdf.integrate(kCamera, I, z, mult);
    require(extract(tsdf).V.empty(), "only off-stride pixels: nothing integrated");
  }
  {
    PresentTsdf tsdf(0.02, 0.04, 2);
    auto z = constant(1.f);
    for (int v = 0; v < H; v += 4)
      for (int u = 0; u < W; u += 4) z[v * W + u] = 0.f;
    tsdf.integrate(kCamera, I, z, mult);
    require(extract(tsdf).V.empty(), "all but stride pixels: nothing integrated");
  }
  {
    PresentTsdf tsdf(0.02, 0.04, 2);
    auto z = constant(2.f);
    for (int v = 0; v < H; ++v)
      for (int u = 0; u < 40; ++u) z[v * W + u] = 1.f;
    for (int k = 0; k < 3; ++k) tsdf.integrate(kCamera, I, z, mult);
    const auto m = extract(tsdf);
    std::set<long> z15, z13;
    for (const auto& p : m.V) {
      if (std::abs(p.z() - 1.f) < 0.02f && std::abs(p.y()) < 0.05f) {
        if (std::abs(p.x() - 0.15f) < 1e-4f) z15.insert(std::lround(p.z() * 1e4));
        if (std::abs(p.x() - 0.13f) < 1e-4f) z13.insert(std::lround(p.z() * 1e4));
      }
    }
    std::cout << "step_edge_3x " << m.V.size() << " " << m.F.size() << '\n';
    require(m.V.size() == 6586 && m.F.size() == 12706, "depth step: 6586 / 12706");
    require(z15.size() == 1 && *z15.begin() == 10060, "rounded projection: x=0.15 crosses at 1.006");
    require(z13.size() == 1 && *z13.begin() == 10000, "x=0.13 crosses at 1.000");
  }
  {
    PresentTsdf tsdf(0.02, 0.04, 2);
    const double th = 20.0 * M_PI / 180.0;
    Eigen::Isometry3d c2w = Eigen::Isometry3d::Identity();
    c2w.linear() << std::cos(th), 0, std::sin(th), 0, 1, 0, -std::sin(th), 0, std::cos(th);
    c2w.translation() << 0.13, -0.07, 0.05;
    tsdf.integrate(kCamera, c2w, constant(1.5f), mult);
    const auto m = extract(tsdf);
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (const auto& p : m.V) centroid += p.cast<double>();
    centroid /= static_cast<double>(m.V.size());
    double area = 0.0;
    for (const auto& f : m.F) {
      area += 0.5 * (m.V[f[1]] - m.V[f[0]]).cast<double>().cross((m.V[f[2]] - m.V[f[0]]).cast<double>()).norm();
    }
    std::cout << "tilted " << m.V.size() << " " << m.F.size() << " " << centroid.transpose() << " "
              << area << '\n';
    require(m.V.size() == 8816 && m.F.size() == 17040, "tilted plane: 8816 / 17040");
    require((centroid - Eigen::Vector3d(0.61839, -0.08322, 1.46851)).norm() < 2e-5,
            "tilted plane centroid");
    require(std::abs(area - 2.64466) < 1e-4, "tilted plane area");
  }
}

void testGrid() {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> U(-1.f, 1.f), S(0.f, 0.08f);
  std::vector<Eigen::Vector3f> V;
  std::vector<TriangleGrid::Face> F;
  for (int i = 0; i < 3000; ++i) {
    const Eigen::Vector3f c(U(rng), U(rng), U(rng));
    const uint32_t b = static_cast<uint32_t>(V.size());
    for (int k = 0; k < 3; ++k) V.push_back(c + Eigen::Vector3f(S(rng), S(rng), S(rng)));
    F.push_back({b, b + 1, b + 2});
  }
  const TriangleGrid grid(V, F, nullptr, 0.08f);
  for (int i = 0; i < 2000; ++i) {
    const Eigen::Vector3f p(1.2f * U(rng), 1.2f * U(rng), 1.2f * U(rng));
    float best = std::numeric_limits<float>::infinity();
    for (const auto& f : F) {
      best = std::min(best, (closestPointOnTriangle(p, V[f[0]], V[f[1]], V[f[2]]) - p).norm());
    }
    const float r_max = 0.1f;
    float d = 0.f;
    Eigen::Vector3f c;
    uint32_t face = 0;
    const bool found = grid.closest(p, r_max, d, c, face);
    require(found == (best <= r_max), "grid closest: found iff within r_max");
    if (found) require(std::abs(d - best) < 1e-6f, "grid closest distance equals brute force");
    // First hit.
    Eigen::Vector3f dir(U(rng), U(rng), U(rng));
    dir.normalize();
    double best_t = std::numeric_limits<double>::infinity();
    for (const auto& f : F) {
      const Eigen::Vector3d a = V[f[0]].cast<double>(), e1 = V[f[1]].cast<double>() - a,
                            e2 = V[f[2]].cast<double>() - a, d = dir.cast<double>();
      const Eigen::Vector3d pv = d.cross(e2);
      const double det = e1.dot(pv);
      if (std::abs(det) < 1e-12) continue;
      const Eigen::Vector3d tv = p.cast<double>() - a;
      const double u = tv.dot(pv) / det;
      if (u < 0 || u > 1) continue;
      const Eigen::Vector3d qv = tv.cross(e1);
      const double v = d.dot(qv) / det;
      if (v < 0 || u + v > 1) continue;
      const double t = e2.dot(qv) / det;
      if (t > 0 && t < best_t) best_t = t;
    }
    float t = 0.f;
    const bool hit = grid.firstHit(p, dir, t, face);
    require(hit == std::isfinite(best_t), "grid first hit: hit iff brute force hits");
    if (hit) require(std::abs(t - best_t) < 1e-5, "grid first hit t equals brute force");
  }
}

void testArchive() {
  std::mt19937 rng(3);
  std::uniform_int_distribution<int> R(0, 65535), L(0, 5);
  const size_t n = W * H;
  std::vector<uint16_t> range(n);
  std::vector<FrameArchive::InstanceRun> runs;
  std::vector<uint16_t> ids(n);
  for (size_t p = 0; p < n; ++p) {
    range[p] = p % 7 ? static_cast<uint16_t>(R(rng)) : 0;
    ids[p] = static_cast<uint16_t>((p / 37) % 3 ? L(rng) : 0);
  }
  for (size_t p = 1; p <= n; ++p) {
    if (p == n || ids[p] != ids[p - 1]) runs.push_back({static_cast<uint32_t>(p), ids[p - 1]});
  }
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() << 1.0, 2.0, 3.0;
  std::vector<FrameArchive::Frame> frames;
  frames.push_back(FrameArchive::Frame::pack(42, pose, range, runs));
  std::vector<uint16_t> r2, i2;
  require(frames[0].decode(n, r2, i2) && r2 == range && i2 == ids, "frame pack / decode");
  const auto path = std::filesystem::temp_directory_path() / "test_session_refusion.kfa";
  FrameArchive::Camera camera;
  camera.width = W;
  camera.height = H;
  camera.fx = camera.fy = 50.f;
  require(FrameArchive::save(path.string(), frames, camera), "archive save");
  std::vector<FrameArchive::Frame> loaded;
  FrameArchive::Camera camera2;
  require(FrameArchive::load(path.string(), loaded, camera2) && loaded.size() == 1 &&
              camera2.sameAs(camera) && loaded[0].stamp == 42 &&
              loaded[0].world_T_sensor.translation() == pose.translation(),
          "archive load");
  require(loaded[0].decode(n, r2, i2) && r2 == range && i2 == ids, "loaded frame decode");
  std::filesystem::remove(path);
}

// A camera at the origin looking along +z sees the plane z = 2 in every
// frame (small sideways offsets); the pixels of the left third read object 7.
struct PlaneScene {
  FrameArchive::Camera camera;
  std::vector<FrameArchive::Frame> frames;
  PlaneScene() {
    const int w = 96, h = 72;
    camera.width = w;
    camera.height = h;
    camera.fx = camera.fy = 60.f;
    camera.cx = 48.f;
    camera.cy = 36.f;
    const size_t n = static_cast<size_t>(w) * h;
    for (int k = 0; k < 12; ++k) {
      Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
      pose.translation() << 0.01 * (k - 6), 0.005 * (k % 3), 0.0;
      std::vector<uint16_t> range(n);
      std::vector<FrameArchive::InstanceRun> runs;
      for (int v = 0; v < h; ++v) {
        for (int u = 0; u < w; ++u) {
          const double xn = (u - camera.cx) / camera.fx, yn = (v - camera.cy) / camera.fy;
          range[static_cast<size_t>(v) * w + u] = static_cast<uint16_t>(2000.0 * std::sqrt(xn * xn + yn * yn + 1.0));
        }
        const uint32_t row = static_cast<uint32_t>(v) * w;
        runs.push_back({row + w / 3, 7});
        runs.push_back({row + w, 0});
      }
      frames.push_back(FrameArchive::Frame::pack(1000 + k, pose, range, runs));
    }
  }
  SessionRefusion::Inputs inputs() const {
    SessionRefusion::Inputs in;
    in.frames = &frames;
    in.camera = camera;
    in.scales.background_voxel = 0.05f;
    in.scales.background_truncation = 0.15f;
    in.scales.object_voxel = 0.02f;
    in.final_stamp = 2000;
    return in;
  }
};

void addTriangle(spark_dsg::Mesh& mesh, const Eigen::Vector3f& c) {
  const size_t b = mesh.numVertices();
  mesh.resizeVertices(b + 3);
  mesh.setPos(b, c + Eigen::Vector3f(-0.02f, -0.02f, 0.f));
  mesh.setPos(b + 1, c + Eigen::Vector3f(0.02f, -0.02f, 0.f));
  mesh.setPos(b + 2, c + Eigen::Vector3f(0.f, 0.02f, 0.f));
  mesh.faces.push_back({b, b + 1, b + 2});
}

// Object 7 as a current node with one own face out of view.
void addObject7(spark_dsg::DynamicSceneGraph& dsg) {
  auto object = std::make_unique<KhronosObjectAttributes>();
  object->mesh = spark_dsg::Mesh(false, true, false, true);
  addTriangle(object->mesh, Eigen::Vector3f(0.f, 0.f, -3.f));
  object->bounding_box = spark_dsg::BoundingBox(Eigen::Vector3f(1.f, 1.f, 1.f), Eigen::Vector3f::Zero());
  object->details["instance_id"] = {7};
  require(dsg.emplaceNode(spark_dsg::DsgLayers::OBJECTS, spark_dsg::NodeSymbol('O', 7), std::move(object)),
          "object node inserted");
}

Eigen::Vector3f centroidOf(const SessionSurface& s, size_t f) {
  return (s.vertices[s.faces[f][0]] + s.vertices[s.faces[f][1]] + s.vertices[s.faces[f][2]]) / 3.f;
}

void testSurfaceUpdate() {
  const PlaneScene scene;
  // The final map: a background mesh and object 7, each with one own face out of view.
  spark_dsg::DynamicSceneGraph dsg;
  dsg.setMesh(std::make_shared<spark_dsg::Mesh>(false, true, false, true));
  addTriangle(*dsg.mesh(), Eigen::Vector3f(0.f, 0.f, -3.f));
  addObject7(dsg);
  // The previous surface: faces named by their centre.
  SessionSurface previous;
  std::map<int, Eigen::Vector3f> centre_of;
  auto addFace = [&](const Eigen::Vector3f& c, uint32_t identity, int name) {
    const auto b = static_cast<uint32_t>(previous.vertices.size());
    previous.vertices.push_back(c + Eigen::Vector3f(-0.02f, -0.02f, 0.f));
    previous.vertices.push_back(c + Eigen::Vector3f(0.02f, -0.02f, 0.f));
    previous.vertices.push_back(c + Eigen::Vector3f(0.f, 0.02f, 0.f));
    previous.faces.push_back({b, b + 1, b + 2});
    previous.identity.push_back(identity);
    previous.error.push_back(0.f);
    centre_of[name] = centroidOf(previous, previous.numFaces() - 1);
  };
  const float x_bg = 0.4f, x_obj = -0.8f;  // right: background pixels; left: object 7's pixels
  addFace(Eigen::Vector3f(x_bg, 0.f, 2.0f), 0, 11);    // on the plane: stays
  addFace(Eigen::Vector3f(x_bg, 0.2f, 1.5f), 0, 12);   // in its free space: goes
  addFace(Eigen::Vector3f(x_bg, -0.2f, 2.08f), 0, 13); // 8 cm behind, within T_bg: goes
  addFace(Eigen::Vector3f(x_bg, 0.3f, 2.4f), 0, 14);   // 40 cm behind: occluded, stays
  addFace(Eigen::Vector3f(x_obj, 0.f, 2.4f), 7, 15);   // object 7 behind its own pixels: goes
  addFace(Eigen::Vector3f(x_bg, -0.3f, 2.4f), 7, 16);  // object 7 behind background pixels: stays
  addFace(Eigen::Vector3f(x_bg, 0.1f, 2.0f), 9, 17);   // object without a current node: void
  require(previous.consistent(), "previous surface consistent");
  auto inputs = scene.inputs();
  inputs.previous = &previous;
  SessionRefusion::Config config;
  config.num_threads = 2;
  const SessionRefusion refusion(config);
  const auto result = refusion.apply(dsg, inputs);
  require(result.applied, "surface update applied");
  const SessionSurface surface = SessionSurface::fromDsg(dsg);
  require(surface.consistent(), "edited map surface consistent");
  require(result.surface_error.size() == surface.numFaces(), "one error per face of the edited map");
  std::set<int> kept;
  size_t present_bg = 0, present_obj = 0;
  for (size_t f = 0; f < surface.numFaces(); ++f) {
    const Eigen::Vector3f c = centroidOf(surface, f);
    int name = 0;
    for (const auto& [n, centre] : centre_of) {
      if ((c - centre).norm() < 1e-5f) name = n;
    }
    if (name) {
      kept.insert(name);
    } else if (std::abs(c.z() - 2.f) < 0.05f) {
      (surface.identity[f] == 7 ? present_obj : present_bg) += 1;
    }
  }
  std::cout << "surface_update kept_previous=";
  for (const int n : kept) std::cout << n << " ";
  std::cout << "present_bg=" << present_bg << " present_obj=" << present_obj << '\n';
  require(kept == std::set<int>({11, 14, 16}), "previous faces kept: on the plane and the occluded ones");
  require(present_bg > 0 && present_obj > 0, "present faces of the plane go to the background and to object 7");
  // The edited map is the state: saved and loaded, its latest snapshot's
  // surface is the next session's previous surface.
  const auto path = std::filesystem::temp_directory_path() / "test_session_refusion_map.4dmap.zpk";
  SpatioTemporalMap map{SpatioTemporalMap::Config()};
  map.update(dsg.clone(), 2000);
  require(map.save(path.string()), "map save");
  const auto loaded = SpatioTemporalMap::load(path.string());
  require(loaded && loaded->numTimeSteps() == 1, "map load");
  const SessionSurface next = SessionSurface::fromDsg(*loaded->rawDsg(0));
  require(next.faces == surface.faces && next.identity == surface.identity &&
              next.vertices.size() == surface.vertices.size(),
          "the saved map's surface is the next previous surface");
  bool same = true;
  for (size_t i = 0; i < next.vertices.size() && same; ++i) same = (next.vertices[i] - surface.vertices[i]).norm() < 1e-6f;
  require(same, "the saved map's surface vertices");
  const auto error_path = std::filesystem::temp_directory_path() / SessionSurface::kErrorFileName;
  require(SessionSurface::saveErrors(error_path.string(), result.surface_error), "errors save");
  SessionSurface with_errors = next;
  require(with_errors.loadErrors(error_path.string()) && with_errors.error == result.surface_error &&
              with_errors.consistent(),
          "the errors saved next to the map load onto its surface");
  SessionSurface other;
  other.faces.resize(3);
  require(!other.loadErrors(error_path.string()), "errors of another face count are refused");
  std::filesystem::remove(error_path);
  std::filesystem::remove(path);
}

// The background mesh empty or absent: the present's background faces still
// land in the background, added vertices carry the final stamp.
void testBackground() {
  const PlaneScene scene;
  for (const bool present_but_empty : {true, false}) {
    spark_dsg::DynamicSceneGraph dsg;
    if (present_but_empty) dsg.setMesh(std::make_shared<spark_dsg::Mesh>(false, true, false, true));
    addObject7(dsg);
    SessionRefusion::Config config;
    config.num_threads = 2;
    const SessionRefusion refusion(config);
    const auto result = refusion.apply(dsg, scene.inputs());
    require(result.applied, "update applied without a background surface");
    require(dsg.hasMesh() && dsg.mesh() && dsg.mesh()->numFaces() > 0, "background faces written");
    const auto& mesh = *dsg.mesh();
    const bool stamps_ok = !mesh.has_timestamps || mesh.stamps.size() == mesh.numVertices();
    const bool colors_ok = !mesh.has_colors || mesh.colors.size() == mesh.numVertices();
    const bool labels_ok = !mesh.has_labels || mesh.labels.size() == mesh.numVertices();
    require(stamps_ok && colors_ok && labels_ok, "background attributes sized to its vertices");
    if (mesh.has_timestamps) {
      require(std::all_of(mesh.stamps.begin(), mesh.stamps.end(), [](uint64_t t) { return t == 2000; }),
              "added background vertices carry the final stamp");
    }
    std::cout << "background " << (present_but_empty ? "empty" : "absent") << ": " << mesh.numFaces()
              << " faces\n";
  }
}

void testStateStart() {
  Track track;
  track.first_seen = 100;
  require(!track.motionEnd() && track.stateStart() == 100, "no motion: the first sighting");
  track.has_dynamic_history = true;
  track.last_motion_seen = 300;
  require(track.motionEnd() && *track.motionEnd() == 300 && track.stateStart() == 301,
          "after motion: right after the last motion");
  track.has_dynamic_history = false;
  require(track.stateStart() == 100, "motion bookkeeping cleared: the first sighting");
}

}  // namespace

int main() {
  testTsdf();
  testGrid();
  testArchive();
  testSurfaceUpdate();
  testBackground();
  testStateStart();
  std::cout << "test_session_refusion passed\n";
  return 0;
}
