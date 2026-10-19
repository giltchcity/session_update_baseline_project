#include "session_core/surface/present_tsdf.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <set>
#include <thread>
#include <hydra/input/camera.h>
#include <hydra/input/sensor_utilities.h>

namespace khronos {
namespace {
struct IndexLess {
  bool operator()(const hydra::BlockIndex& a, const hydra::BlockIndex& b) const {
    for (int k=0;k<3;++k) { if (a[k]!=b[k]) return a[k]<b[k]; }
    return false;
  }
};
using Indices = std::set<hydra::BlockIndex, IndexLess>;

hydra::VolumetricMap::Config geometryConfig(hydra::VolumetricMap::Config config) {
  config.with_tracking = false;
  config.with_semantics = false;
  return config;
}

hydra::InputData input(const PresentTsdf::Camera& camera, const Eigen::Isometry3d& pose,
                       const std::vector<float>& norms, const PresentTsdf::Domain& domain) {
  hydra::Camera::Config config;
  config.width=camera.width; config.height=camera.height;
  config.fx=camera.fx; config.fy=camera.fy; config.cx=camera.cx; config.cy=camera.cy;
  config.min_range=camera.min_range; config.max_range=camera.max_range;
  config.extrinsics=hydra::IdentitySensorExtrinsics::Config{};
  hydra::InputData data(std::make_shared<hydra::Camera>(config, "present_surface"));
  data.timestamp_ns=0;
  data.world_T_body=pose;
  data.range_image=cv::Mat(camera.height,camera.width,CV_32FC1,cv::Scalar(0));
  data.min_range=std::numeric_limits<float>::infinity(); data.max_range=0.f;
  if (domain.depth_z) {
    for (size_t i=0;i<norms.size();++i) {
      const float r=(*domain.depth_z)[i]*norms[i];
      if (!std::isfinite(r) || r<=0.f) continue;
      data.range_image.ptr<float>()[i]=r;
      data.min_range=std::min(data.min_range,r);
      data.max_range=std::max(data.max_range,r);
    }
  }
  return data;
}

hydra::BlockIndices visible(const hydra::InputData& data, float size) {
  if (!(data.max_range>0.f)) return {};
  return hydra::findBlocksInViewFrustum(data.getSensor(),data.getSensorPose().cast<float>(),
                                       size,data.min_range,data.max_range);
}

// A native interpolated reading contributes free space only if each pixel
// participating in that reading allows it. This changes evidence eligibility;
// interpolation, distance, weight, saturation and extraction remain native.
bool allowed(const hydra::ProjectiveIntegrator::VoxelMeasurement& m, float distance,
             const PresentTsdf::Domain& domain, int width, float truncation) {
  if (!domain.free_limit || m.sdf<truncation) return true;
  const auto& w=m.interpolation_weights;
  const size_t p=static_cast<size_t>(w.v)*width+w.u;
  const auto legal=[&](size_t i) { return distance<(*domain.free_limit)[i]-truncation; };
  if (!w.use_bilinear) return legal(p);
  return (!(w.w0>0.f)||legal(p)) && (!(w.w1>0.f)||legal(p+width)) &&
         (!(w.w2>0.f)||legal(p+1)) && (!(w.w3>0.f)||legal(p+width+1));
}
}  // namespace

struct PresentTsdf::Impl { Indices dirty; };

PresentTsdf::PresentTsdf(const hydra::VolumetricMap::Config& map,
                         const hydra::ProjectiveIntegrator::Config& integrator,
                         const hydra::MeshIntegratorConfig& mesh)
    : impl_(std::make_unique<Impl>()), map_(geometryConfig(map)),
      integrator_(integrator), mesh_integrator_(mesh) {}
PresentTsdf::~PresentTsdf()=default;
size_t PresentTsdf::numUnits() const { return map_.getTsdfLayer().numBlocks(); }

void PresentTsdf::integrateBlocks(const hydra::InputData& data, const Domain& domain,
                                  const hydra::BlockIndices& blocks) {
  map_.allocateBlocks(blocks);
  const auto sensor_T_world=data.getSensorPose().cast<float>().inverse();
  std::atomic<size_t> next{0};
  const auto work=[&]() {
    size_t k;
    while ((k=next.fetch_add(1))<blocks.size()) {
      auto block=map_.getBlock(blocks[k]);
      bool updated=false;
      for (size_t i=0;i<block.tsdf->numVoxels();++i) {
        const auto p=sensor_T_world*block.tsdf->getVoxelPosition(i);
        const auto m=integrator_.getVoxelMeasurement(map_.config,data,{},p);
        if (!m.valid || !allowed(m,p.norm(),domain,data.range_image.cols,map_.config.truncation_distance)) continue;
        auto voxels=block.getVoxels(i);
        integrator_.updateVoxel(map_.config,data,m,voxels);
        updated=true;
      }
      if (updated) block.tsdf->setUpdated();
    }
  };
  std::vector<std::thread> workers;
  for (int i=1;i<integrator_.config.num_threads;++i) workers.emplace_back(work);
  work();
  for (auto& worker:workers) worker.join();
  for (const auto& index:blocks) {
    const auto block=map_.getTsdfLayer().getBlockPtr(index);
    if (!block->updated && std::none_of(block->begin(),block->end(),
                                      [](const auto& v){return v.weight>0.f;})) map_.removeBlock(index);
  }
}

void PresentTsdf::integrate(const Camera& camera, const Eigen::Isometry3d& pose,
                            const std::vector<float>& ray_norm, const Domain& domain) {
  const auto data=input(camera,pose,ray_norm,domain);
  integrateBlocks(data,domain,visible(data,map_.blockSize()));
}

void PresentTsdf::invalidate(const Camera& camera, const Eigen::Isometry3d& pose,
                             const std::vector<float>& ray_norm, const Domain& before,
                             const Domain& after) {
  const auto a=input(camera,pose,ray_norm,before),b=input(camera,pose,ray_norm,after);
  auto blocks=visible(a,map_.blockSize());
  const auto other=visible(b,map_.blockSize()); blocks.insert(blocks.end(),other.begin(),other.end());
  impl_->dirty.insert(blocks.begin(),blocks.end());
}

void PresentTsdf::rebuild(const Camera& camera, const std::vector<float>& ray_norm,
                          size_t frame_count, const ReadFrame& read) {
  if (impl_->dirty.empty()) return;
  for (const auto& index:impl_->dirty) map_.removeBlock(index);
  std::vector<float> depth,limit;
  Eigen::Isometry3d pose;
  for (size_t i=0;i<frame_count;++i) {
    read(i,pose,depth,limit);
    const Domain domain{&depth,limit.empty()?nullptr:&limit};
    const auto data=input(camera,pose,ray_norm,domain);
    auto blocks=visible(data,map_.blockSize());
    blocks.erase(std::remove_if(blocks.begin(),blocks.end(),[&](const auto& index) {
      return !impl_->dirty.count(index);
    }),blocks.end());
    integrateBlocks(data,domain,blocks);
  }
  impl_->dirty.clear();
}

void PresentTsdf::extractMesh(std::vector<Eigen::Vector3f>& vertices,std::vector<Face>& faces) {
  // A rebuilt block can also change the neighbouring block's boundary cubes.
  mesh_integrator_.generateMesh(map_,false,true);
  vertices.clear();faces.clear();
  auto indices=map_.getMeshLayer().allocatedBlockIndices();
  std::sort(indices.begin(),indices.end(),IndexLess{});
  for (const auto& index:indices) {
    const auto& block=map_.getMeshLayer().getBlock(index);
    const uint32_t offset=vertices.size();
    vertices.insert(vertices.end(),block.points.begin(),block.points.end());
    for (const auto& f:block.faces) faces.push_back({offset+static_cast<uint32_t>(f[0]),
      offset+static_cast<uint32_t>(f[1]),offset+static_cast<uint32_t>(f[2])});
  }
}

std::vector<float> PresentTsdf::rayNorm(const Camera& camera) {
  std::vector<float> result(static_cast<size_t>(camera.width)*camera.height);
  for (uint32_t v=0;v<camera.height;++v) for (uint32_t u=0;u<camera.width;++u) {
    const float x=(static_cast<float>(u)-camera.cx)/camera.fx;
    const float y=(static_cast<float>(v)-camera.cy)/camera.fy;
    result[static_cast<size_t>(v)*camera.width+u]=std::sqrt(x*x+y*y+1.f);
  }
  return result;
}

void PresentTsdf::depthFromRange(const Camera& camera,const std::vector<uint16_t>& range_mm,
                                 std::vector<float>& depth_z) {
  const auto norms=rayNorm(camera);
  depth_z.resize(norms.size());
  for (size_t i=0;i<norms.size();++i) depth_z[i]=range_mm[i]*.001f/norms[i];
}
}  // namespace khronos
