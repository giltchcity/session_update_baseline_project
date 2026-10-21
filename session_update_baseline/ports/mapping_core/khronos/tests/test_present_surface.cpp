// Native integration, causality, evidence retirement and frame round trips.
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <set>

#include <nlohmann/json.hpp>
#include <session_core/surface/present_surface.h>
#include <hydra/input/camera.h>

#include <session_core/surface/frame_archive.h>
#include <session_core/surface/present_tsdf.h>
#include <session_core/surface/triangle_grid.h>

using namespace khronos;

namespace {

void require(bool value, const std::string& message) {
  if (!value) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(1);
  }
}

constexpr int W = 64, H = 48;
const PresentTsdf::Camera kCamera{W, H, 50.f, 50.f, 32.f, 24.f, .1f, 5.f};

std::vector<float> rayNorm() {
  std::vector<float> m(W * H);
  for (int v = 0; v < H; ++v)
    for (int u = 0; u < W; ++u) {
      const float x = (u - 32.f) * (1.f / 50.f), y = (v - 24.f) * (1.f / 50.f);
      m[v * W + u] = std::sqrt(x * x + y * y + 1.f);
    }
  return m;
}

hydra::VolumetricMap::Config mapConfig() {
  hydra::VolumetricMap::Config c;
  c.voxel_size=.05f;c.truncation_distance=.15f;
  return c;
}
hydra::ProjectiveIntegrator::Config integrationConfig() {
  hydra::ProjectiveIntegrator::Config c;c.num_threads=1;
  return c;
}
hydra::MeshIntegratorConfig meshConfig() {
  hydra::MeshIntegratorConfig c;c.integrator_threads=1;
  return c;
}
void testTsdf() {
  const auto mc=mapConfig();auto ic=integrationConfig();
  // Force saturation in this fixture; subtracting an old observation from a
  // capped mean is not equivalent to native integration without that frame.
  ic.max_weight=5.f;
  PresentTsdf persistent(mc,ic,meshConfig());
  hydra::VolumetricMap reference(mc);
  hydra::ProjectiveIntegrator native(ic);
  hydra::Camera::Config cc;
  cc.width=W;cc.height=H;cc.fx=cc.fy=50.f;cc.cx=32.f;cc.cy=24.f;
  cc.min_range=.1f;cc.max_range=5.f;cc.extrinsics=hydra::IdentitySensorExtrinsics::Config{};
  auto sensor=std::make_shared<hydra::Camera>(cc,"native_test");
  const auto norms=PresentTsdf::rayNorm(kCamera);
  const auto pose=Eigen::Isometry3d::Identity();
  std::vector<std::vector<float>> frames;
  for (int i=0;i<8;++i) {
    frames.emplace_back(W*H,1.f+.02f*(i%3));
    for (int y=0;y<H;++y) for (int x=40;x<W;++x) frames.back()[y*W+x]=2.f;
    for (int y=0;y<5;++y) for (int x=0;x<5;++x) frames.back()[y*W+x]=0.f;
  }
  const auto addNative=[&](const std::vector<float>& depth,hydra::VolumetricMap& map) {
    hydra::InputData data(sensor);data.timestamp_ns=0;data.world_T_body=pose;
    data.range_image=cv::Mat(H,W,CV_32FC1);data.min_range=INFINITY;data.max_range=0.f;
    for (size_t i=0;i<depth.size();++i) {
      const float r=depth[i]*norms[i];data.range_image.ptr<float>()[i]=r;
      if (r>0.f) {data.min_range=std::min(data.min_range,r);data.max_range=std::max(data.max_range,r);}
    }
    native.updateMap(data,map);
  };
  const auto compare=[&](const hydra::VolumetricMap& expected) {
    const auto& actual=persistent.map();
    require(actual.getTsdfLayer().numBlocks()==expected.getTsdfLayer().numBlocks(),"native allocation matches");
    for (const auto& block:expected.getTsdfLayer()) {
      const auto other=actual.getTsdfLayer().getBlockPtr(block.index);
      require(other!=nullptr,"native block exists");
      for (size_t i=0;i<block.numVoxels();++i) {
        const auto& a=block.getVoxel(i);const auto& c=other->getVoxel(i);
        require(a.weight==c.weight && a.distance==c.distance,"native voxel distance and capped weight match exactly");
      }
    }
  };
  for (const auto& depth:frames) {
    persistent.integrate(kCamera,pose,norms,{&depth,nullptr});addNative(depth,reference);
  }
  compare(reference);
  std::vector<Eigen::Vector3f> vertices;std::vector<PresentTsdf::Face> faces;
  persistent.extractMesh(vertices,faces);
  hydra::MeshIntegrator mesher(meshConfig());mesher.generateMesh(reference,false,true);
  size_t native_faces=0;for (const auto& block:reference.getMeshLayer()) native_faces+=block.faces.size();
  require(!faces.empty() && faces.size()==native_faces,"mesh extraction is native Hydra");
  persistent.invalidate(kCamera,pose,norms,{&frames[0],nullptr},{});
  persistent.rebuild(kCamera,norms,frames.size()-1,
    [&](size_t i,Eigen::Isometry3d& p,std::vector<float>& depth,std::vector<float>& limit) {
      p=pose;depth=frames[i+1];limit.clear();
    });
  hydra::VolumetricMap without_first(mc);
  for (size_t i=1;i<frames.size();++i) addNative(frames[i],without_first);
  compare(without_first);
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

// Exercise the actual online publication path, including a producer ahead of
// the snapshot clock. Buffered future frames must not alter an earlier map.
void testOnlineSurface() {
  FrameArchive::Camera camera;
  camera.width=W; camera.height=H; camera.fx=camera.fy=50.f;
  camera.cx=32.f; camera.cy=24.f; camera.min_range=.1f; camera.max_range=5.f;
  auto frame = [&](TimeStamp stamp, float depth, bool sparse=false) {
    std::vector<uint16_t> ranges(W*H, 0);
    const auto norms=rayNorm();
    for (int i=0;i<W*H;++i) if (!sparse || i==W+1) {
      ranges[i]=static_cast<uint16_t>(depth*norms[i]*1000.f);
    }
    return FrameArchive::Frame::pack(stamp,Eigen::Isometry3d::Identity(),ranges,{{W*H,7}});
  };
  auto graph = []() {
    auto dsg=std::make_shared<spark_dsg::DynamicSceneGraph>();
    dsg->setMesh(std::make_shared<spark_dsg::Mesh>(false,false,false,false));
    return dsg;
  };
  PresentSurface prefix(mapConfig(),integrationConfig(),meshConfig()),
                 buffered(mapConfig(),integrationConfig(),meshConfig());
  prefix.integrate(camera,frame(1000000000ULL,1.f));
  buffered.integrate(camera,frame(1000000000ULL,1.f));
  buffered.integrate(camera,frame(2000000000ULL,2.f));
  auto p=graph(),q=graph();
  const auto first=prefix.compose(*p,1000000000ULL,{}, {},false);
  const auto held=buffered.compose(*q,1000000000ULL,{}, {},false);
  require(nlohmann::json::parse(held.report_json)["frames"]==1,"future frame is not consumed");
  require(!p->mesh()->faces.empty() && p->mesh()->points==q->mesh()->points &&
          p->mesh()->faces==q->mesh()->faces,"future frames cannot affect an online snapshot");
  auto terminal=graph();
  buffered.compose(*terminal,1000000000ULL,{}, {},true);
  require(q->mesh()->points==terminal->mesh()->points && q->mesh()->faces==terminal->mesh()->faces,
          "terminal publication has no separate surface repair");
  auto next=graph();
  const auto second=buffered.compose(*next,2000000000ULL,{}, {},false);
  require(nlohmann::json::parse(second.report_json)["frames"]==2 &&
          next->mesh()->numVertices()>q->mesh()->numVertices(),"next snapshot consumes its new observation");

  // Retraction changes the accumulated volume itself, including already
  // published geometry; a newer observation at another place stays intact.
  PresentSurface moving(mapConfig(),integrationConfig(),meshConfig());
  moving.integrate(camera,frame(1000000000ULL,1.f));
  auto old_site=graph();
  moving.compose(*old_site,1000000000ULL,{}, {},false);
  require(!old_site->mesh()->faces.empty(),"old state initially has a surface");
  auto moved_frame=frame(2000000000ULL,2.f);
  moved_frame.world_T_sensor.translation().x()=10.;
  moving.integrate(camera,std::move(moved_frame));
  moving.compose(*old_site,2000000000ULL,{{7,2000000000ULL}}, {},false);
  require(!old_site->mesh()->faces.empty(),"new state remains after old contributions are removed");
  for (const auto& point:old_site->mesh()->points) {
    require(point.x()>5.f,"no old-state surface survives publication");
  }

  // Fill: an archived window face stays where a counted frame measured it (the
  // one valid pixel, (1,1) at 1 m), and only there.
  PresentSurface sparse(mapConfig(),integrationConfig(),meshConfig());
  sparse.integrate(camera,frame(1000000000ULL,1.f,true));
  auto observed=graph();
  const Eigen::Vector3f measured(-.625f,-.465f,1.f);
  observed->mesh()->points={{20.f,0.f,1.f},{21.f,0.f,1.f},{20.f,1.f,1.f},
                            measured,{-.615f,-.465f,1.f},{-.62f,-.455f,1.f}};
  observed->mesh()->faces={{0,1,2},{3,4,5}};
  sparse.compose(*observed,1000000000ULL,{}, {},false);
  bool filled=false;
  for (const auto& point:observed->mesh()->points) {
    require(point.x()<10.f,"an archived face no frame measured is not filled in");
    filled|=point==measured;
  }
  require(filled,"an archived face a counted frame measured is filled in");

  // One volume holds one surface per place: a shown face behind the present's
  // layer within the truncation, facing the same way, with no direct majority
  // (every reading lands in front of it) gives way; the same face facing the
  // other way, or one beyond the truncation, stays.
  PresentSurface wall(mapConfig(),integrationConfig(),meshConfig());
  wall.integrate(camera,frame(1000000000ULL,1.f));
  auto probe=graph();
  wall.compose(*probe,1000000000ULL,{}, {},false);
  const auto& pm=*probe->mesh();
  require(!pm.faces.empty(),"the wall has a present surface");
  Eigen::Vector3f wall_normal=Eigen::Vector3f::Zero();
  for (const auto& f:pm.faces) wall_normal+=(pm.pos(f[1])-pm.pos(f[0])).cross(pm.pos(f[2])-pm.pos(f[0]));
  PresentSurface::Surface shown;
  auto addShown=[&](float x,float z,bool same_facing) {
    const uint32_t b=static_cast<uint32_t>(shown.vertices.size());
    shown.vertices.insert(shown.vertices.end(),{{x,0.f,z},{x+.04f,0.f,z},{x,.04f,z}});
    // (b - a) x (c - a) points to +z; reversed winding to -z.
    const bool plus_z=(wall_normal.z()>0.f)==same_facing;
    shown.faces.push_back(plus_z ? std::array<uint32_t,3>{b,b+1,b+2} : std::array<uint32_t,3>{b,b+2,b+1});
    shown.face_physical.push_back(0);
  };
  addShown(-.2f,1.08f,true);   // behind the layer, facing the same way: gives way
  addShown(0.f,1.08f,false);   // facing the other way: stays
  addShown(.2f,1.2f,true);     // beyond the truncation: stays
  PresentSurface::Inputs with_memory;
  with_memory.shown=&shown;
  auto composed=graph();
  wall.compose(*composed,1000000000ULL,{},with_memory,false);
  auto shows=[&](float x,float z) {
    for (const auto& point:composed->mesh()->points) if (point==Eigen::Vector3f(x,0.f,z)) return true;
    return false;
  };
  require(!shows(-.2f,1.08f),"a same-facing shown face within the truncation behind the present gives way");
  require(shows(0.f,1.08f),"an opposite-facing shown face behind the present stays");
  require(shows(.2f,1.2f),"a shown face beyond the truncation stays");
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
  require(frames[0].stamp == 42 && frames[0].world_T_sensor.translation() == pose.translation(),
          "frame stamp and pose");
}

}  // namespace

int main() {
  hydra::PipelineConfig pipeline; hydra::GlobalInfo::init(pipeline);
  testTsdf();
  testGrid();
  testArchive();
  testOnlineSurface();
  std::cout << "test_present_surface passed\n";
  return 0;
}
