#include "session_core/surface/present_tsdf.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <map>
#include <optional>
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

hydra::Camera::Config sensorConfig(const PresentTsdf::Camera& camera) {
  hydra::Camera::Config config;
  config.width=camera.width; config.height=camera.height;
  config.fx=camera.fx; config.fy=camera.fy; config.cx=camera.cx; config.cy=camera.cy;
  config.min_range=camera.min_range; config.max_range=camera.max_range;
  config.extrinsics=hydra::IdentitySensorExtrinsics::Config{};
  return config;
}

hydra::InputData input(const PresentTsdf::Camera& camera, const Eigen::Isometry3d& pose,
                       const std::vector<float>& norms, const PresentTsdf::Domain& domain) {
  hydra::InputData data(std::make_shared<hydra::Camera>(sensorConfig(camera), "present_surface"));
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

// Dirty: the voxels a changed contribution reaches, per block, until rebuilt. Removed: blocks
// removed since the last extraction (their negative neighbours' boundary cubes read them).
struct PresentTsdf::Impl {
  std::map<hydra::BlockIndex, VoxelMask, IndexLess> dirty;
  Indices removed;
};

namespace {
bool isMarked(const PresentTsdf::VoxelMask& mask, size_t voxel) { return (mask[voxel >> 6] >> (voxel & 63)) & 1u; }
void mark(PresentTsdf::VoxelMask& mask, size_t voxel) { mask[voxel >> 6] |= uint64_t(1) << (voxel & 63); }
}  // namespace

PresentTsdf::PresentTsdf(const hydra::VolumetricMap::Config& map,
                         const hydra::ProjectiveIntegrator::Config& integrator,
                         const hydra::MeshIntegratorConfig& mesh)
    : impl_(std::make_unique<Impl>()), map_(geometryConfig(map)),
      integrator_(integrator), mesh_integrator_(mesh) {}
PresentTsdf::~PresentTsdf()=default;
size_t PresentTsdf::numUnits() const { return map_.getTsdfLayer().numBlocks(); }

bool PresentTsdf::foldVoxel(const hydra::BlockTuple& block, size_t i, const hydra::InputData& data,
                            const Eigen::Isometry3f& sensor_T_world, const Domain& domain) const {
  const auto p=sensor_T_world*block.tsdf->getVoxelPosition(i);
  const auto m=integrator_.getVoxelMeasurement(map_.config,data,{},p);
  if (!m.valid || !allowed(m,p.norm(),domain,data.range_image.cols,map_.config.truncation_distance)) return false;
  auto voxels=block.getVoxels(i);
  integrator_.updateVoxel(map_.config,data,m,voxels);
  return true;
}

PresentTsdf::Rebuilt PresentTsdf::fold(const Camera& camera, const std::vector<float>& ray_norm,
                                       const std::vector<size_t>& frames, const FramePose& pose,
                                       const ReadFrame& read, bool marked_only) {
  Rebuilt folded;
  const size_t batch=static_cast<size_t>(std::max(1,integrator_.config.num_threads));
  std::vector<std::vector<float>> depth(batch),limit(batch);
  std::vector<std::unique_ptr<hydra::InputData>> data(batch);
  std::vector<hydra::BlockIndices> blocks(batch);
  std::vector<Eigen::Isometry3f> sensor_T_world(batch);
  // Each reader also prepares its frame's input and the blocks it folds into.
  const auto prepare=[&](size_t j,size_t frame) {
    blocks[j].clear();
    if (!read(static_cast<int>(j),frame,depth[j],limit[j])) return;
    const Domain domain{&depth[j],limit[j].empty()?nullptr:&limit[j]};
    data[j]=std::make_unique<hydra::InputData>(input(camera,pose(frame),ray_norm,domain));
    sensor_T_world[j]=data[j]->getSensorPose().cast<float>().inverse();
    blocks[j]=visible(*data[j],map_.blockSize());
    if (marked_only) {
      blocks[j].erase(std::remove_if(blocks[j].begin(),blocks[j].end(),[&](const auto& index) {
        return !impl_->dirty.count(index);
      }),blocks[j].end());
    }
  };
  folded.frames=frames.size();
  for (size_t first=0;first<frames.size();first+=batch) {
    const size_t m=std::min(batch,frames.size()-first);
    const auto start=std::chrono::steady_clock::now();
    std::vector<std::thread> readers;
    for (size_t j=1;j<m;++j) readers.emplace_back([&,j]{prepare(j,frames[first+j]);});
    prepare(0,frames[first]);
    for (auto& reader:readers) reader.join();
    const auto read_done=std::chrono::steady_clock::now();
    std::map<hydra::BlockIndex,std::vector<size_t>,IndexLess> seen_by;
    for (size_t j=0;j<m;++j) for (const auto& index:blocks[j]) seen_by[index].push_back(j);
    hydra::BlockIndices indices;
    for (const auto& entry:seen_by) indices.push_back(entry.first);
    map_.allocateBlocks(indices);
    std::vector<const std::pair<const hydra::BlockIndex,std::vector<size_t>>*> work;
    for (const auto& entry:seen_by) work.push_back(&entry);
    std::atomic<size_t> next{0};
    const auto fold_blocks=[&]() {
      size_t k;
      while ((k=next.fetch_add(1))<work.size()) {
        const auto block=map_.getBlock(work[k]->first);
        const VoxelMask* mask=marked_only?&impl_->dirty.at(work[k]->first):nullptr;
        bool updated=false;
        for (const size_t j:work[k]->second) {
          const Domain domain{&depth[j],limit[j].empty()?nullptr:&limit[j]};
          if (!mask) {
            for (size_t i=0;i<block.tsdf->numVoxels();++i) updated|=foldVoxel(block,i,*data[j],sensor_T_world[j],domain);
            continue;
          }
          for (size_t w=0;w<mask->size();++w) {
            for (uint64_t bits=(*mask)[w];bits;bits&=bits-1) {
              foldVoxel(block,w*64+static_cast<size_t>(__builtin_ctzll(bits)),*data[j],sensor_T_world[j],domain);
            }
          }
        }
        if (updated) block.tsdf->setUpdated();
      }
    };
    std::vector<std::thread> workers;
    const size_t threads=std::min(batch,work.size());
    for (size_t i=1;i<threads;++i) workers.emplace_back(fold_blocks);
    fold_blocks();
    for (auto& worker:workers) worker.join();
    // A block integration allocated but nothing updated, without weight, is not kept.
    if (!marked_only) {
      for (const auto& index:indices) {
        const auto block=map_.getTsdfLayer().getBlockPtr(index);
        if (!block->updated && std::none_of(block->begin(),block->end(),[](const auto& v){return v.weight>0.f;})) {
          map_.removeBlock(index);
          impl_->removed.insert(index);
        }
      }
    }
    folded.read_s+=std::chrono::duration<double>(read_done-start).count();
    folded.integrate_s+=std::chrono::duration<double>(std::chrono::steady_clock::now()-read_done).count();
  }
  return folded;
}

void PresentTsdf::integrate(const Camera& camera, const std::vector<float>& ray_norm,
                            const std::vector<size_t>& frames, const FramePose& pose, const ReadFrame& read) {
  fold(camera,ray_norm,frames,pose,read,false);
}

PresentTsdf::Changes PresentTsdf::changedVoxels(const Camera& camera, const Eigen::Isometry3d& pose,
                                               const std::vector<float>& ray_norm, const Domain& before,
                                               const Domain& after, const Window& differs) const {
  // A voxel reads the pixel it projects to and that pixel's interpolation neighbours. Where a
  // pixel enters or leaves the domain, every voxel it updates changes (from the camera to the
  // truncation behind the farthest reading it interpolates with); where only its free limit
  // moves, the free updates between the two limits. Per pixel, the distances the changed updates
  // span, widened to its neighbours (kept per worker, reset where written).
  constexpr float kInf = std::numeric_limits<float>::infinity();
  const int W = static_cast<int>(camera.width), H = static_cast<int>(camera.height);
  const float T = map_.config.truncation_distance;
  const float behind = this->behind();
  const auto range = [&](const Domain& d, int u, int v) {
    const size_t p = static_cast<size_t>(v) * W + u;
    const float r = d.depth_z ? (*d.depth_z)[p] * ray_norm[p] : 0.f;
    return std::isfinite(r) && r > 0.f ? r : 0.f;
  };
  const auto limit = [&](const Domain& d, int u, int v) {
    return d.free_limit ? (*d.free_limit)[static_cast<size_t>(v) * W + u] : kInf;
  };
  thread_local std::vector<float> near, far;
  if (near.size() != static_cast<size_t>(W) * H) near.assign(static_cast<size_t>(W) * H, kInf), far.assign(near.size(), -kInf);
  float reach = 0.f, closest = kInf;
  int cu0 = W, cu1 = -1, cv0 = H, cv1 = -1;
  for (int v = std::max(0, differs.v0); v <= std::min(H - 1, differs.v1); ++v) {
    for (int u = std::max(0, differs.u0); u <= std::min(W - 1, differs.u1); ++u) {
      const float rb = range(before, u, v), ra = range(after, u, v);
      const float lb = limit(before, u, v), la = limit(after, u, v);
      if (rb == ra && (rb == 0.f || lb == la)) continue;
      float r = 0.f;
      for (int y = std::max(0, v - 1); y <= std::min(H - 1, v + 1); ++y) {
        for (int x = std::max(0, u - 1); x <= std::min(W - 1, u + 1); ++x) {
          r = std::max({r, range(before, x, y), range(after, x, y)});
        }
      }
      const float d0 = rb != ra ? 0.f : std::min(lb, la) - T;
      const float d1 = rb != ra ? r + behind : std::min(std::max(lb, la), r) - T;
      if (d1 < d0) continue;
      // The pixel and its neighbours: a voxel projecting next to it may read it.
      for (int y = std::max(0, v - 1); y <= std::min(H - 1, v + 1); ++y) {
        for (int x = std::max(0, u - 1); x <= std::min(W - 1, u + 1); ++x) {
          const size_t q = static_cast<size_t>(y) * W + x;
          near[q] = std::min(near[q], d0);
          far[q] = std::max(far[q], d1);
        }
      }
      reach = std::max(reach, d1), closest = std::min(closest, d0);
      cu0 = std::min(cu0, std::max(0, u - 1)), cu1 = std::max(cu1, std::min(W - 1, u + 1));
      cv0 = std::min(cv0, std::max(0, v - 1)), cv1 = std::max(cv1, std::min(H - 1, v + 1));
    }
  }
  // The blocks either domain integrates; a block only one of them reaches changes as a whole.
  // A voxel is in range only up to the frame's farthest reading: between the two domains'
  // farthest readings, every voxel's update changes.
  // Each domain's nearest and farthest reading (outside `differs` both read the same), and the
  // blocks it integrates: visible() of the input it builds.
  float nearest_a = kInf, farthest_a = 0.f, nearest_b = kInf, farthest_b = 0.f;
  for (int v = 0; v < H; ++v) {
    const bool row = v >= differs.v0 && v <= differs.v1;
    for (int u = 0; u < W; ++u) {
      const float ra = range(before, u, v);
      const float rb = row && u >= differs.u0 && u <= differs.u1 ? range(after, u, v) : ra;
      if (ra > 0.f) nearest_a = std::min(nearest_a, ra), farthest_a = std::max(farthest_a, ra);
      if (rb > 0.f) nearest_b = std::min(nearest_b, rb), farthest_b = std::max(farthest_b, rb);
    }
  }
  const hydra::Camera sensor(sensorConfig(camera), "present_surface");
  const auto blocks = [&](float nearest, float farthest) {
    return farthest > 0.f ? hydra::findBlocksInViewFrustum(sensor, pose.cast<float>(), map_.blockSize(), nearest, farthest)
                          : hydra::BlockIndices{};
  };
  const auto in_a = blocks(nearest_a, farthest_a), in_b = blocks(nearest_b, farthest_b);
  const Indices set_a(in_a.begin(), in_a.end()), set_b(in_b.begin(), in_b.end());
  const float farthest0 = std::min(farthest_a, farthest_b), farthest1 = std::max(farthest_a, farthest_b);
  Indices both = set_a;
  both.insert(set_b.begin(), set_b.end());
  const size_t num_voxels = static_cast<size_t>(map_.config.voxels_per_side) * map_.config.voxels_per_side *
                            map_.config.voxels_per_side;
  const VoxelMask all((num_voxels + 63) / 64, ~uint64_t(0));
  const Eigen::Isometry3f sensor_T_world = pose.cast<float>().inverse();
  const Eigen::Vector3f c = pose.translation().cast<float>();
  const float size = map_.blockSize();
  Changes changes;
  for (const auto& index : both) {
    if (!set_a.count(index) || !set_b.count(index)) {
      changes.emplace_back(index, all);
      continue;
    }
    // A block whose distances meet no changed span, nor the farthest readings', keeps every
    // voxel; so does one whose image misses every changed pixel.
    const Eigen::Vector3f lo = index.cast<float>() * size, hi = lo + Eigen::Vector3f::Constant(size);
    const float dn = (c.cwiseMax(lo).cwiseMin(hi) - c).norm();
    float df = 0.f, u0 = kInf, u1 = -kInf, v0 = kInf, v1 = -kInf;
    bool whole_image = false;  // a corner off the camera's front: any pixel
    for (int k = 0; k < 8; ++k) {
      const Eigen::Vector3f x =
          sensor_T_world * Eigen::Vector3f((k & 1) ? hi.x() : lo.x(), (k & 2) ? hi.y() : lo.y(), (k & 4) ? hi.z() : lo.z());
      df = std::max(df, x.norm());
      if (x.z() <= 0.f) {
        whole_image = true;
        continue;
      }
      const float u = x.x() / x.z() * camera.fx + camera.cx, v = x.y() / x.z() * camera.fy + camera.cy;
      u0 = std::min(u0, u), u1 = std::max(u1, u), v0 = std::min(v0, v), v1 = std::max(v1, v);
    }
    const bool between = farthest0 != farthest1 && dn <= farthest1 && df > farthest0;
    const bool seen = cu0 <= cu1 && dn <= reach && df >= closest &&
                      (whole_image || (u0 - 1.f <= cu1 && u1 + 1.f >= cu0 && v0 - 1.f <= cv1 && v1 + 1.f >= cv0));
    if (!between && !seen) continue;
    // Voxel positions from the map's block, or (not allocated yet) one made for them.
    const auto allocated = map_.getTsdfLayer().getBlockPtr(index);
    std::optional<hydra::TsdfBlock> made;
    const hydra::TsdfBlock& block =
        allocated ? *allocated : made.emplace(map_.config.voxel_size, map_.config.voxels_per_side, index);
    VoxelMask mask((num_voxels + 63) / 64, 0);
    bool any = false;
    for (size_t i = 0; i < num_voxels; ++i) {
      const Eigen::Vector3f x = sensor_T_world * block.getVoxelPosition(i);
      const float d = x.norm();
      if (between && d > farthest0 && d <= farthest1) {
        mark(mask, i), any = true;
        continue;
      }
      if (!seen || x.z() <= 0.f) continue;
      const int u = static_cast<int>(std::lround(x.x() / x.z() * camera.fx + camera.cx));
      const int v = static_cast<int>(std::lround(x.y() / x.z() * camera.fy + camera.cy));
      if (u < cu0 || u > cu1 || v < cv0 || v > cv1) continue;
      const size_t q = static_cast<size_t>(v) * W + u;
      if (near[q] <= d && d <= far[q]) mark(mask, i), any = true;
    }
    if (any) changes.emplace_back(index, std::move(mask));
  }
  for (int v = cv0; v <= cv1; ++v) {
    std::fill(near.begin() + static_cast<size_t>(v) * W + cu0, near.begin() + static_cast<size_t>(v) * W + cu1 + 1, kInf);
    std::fill(far.begin() + static_cast<size_t>(v) * W + cu0, far.begin() + static_cast<size_t>(v) * W + cu1 + 1, -kInf);
  }
  return changes;
}

void PresentTsdf::invalidate(const Changes& changes) {
  for (const auto& [index, mask] : changes) {
    auto& dirty = impl_->dirty[index];
    if (dirty.empty()) dirty.assign(mask.size(), 0);
    for (size_t w = 0; w < mask.size(); ++w) dirty[w] |= mask[w];
  }
}

float PresentTsdf::behind() const {
  const float extra=integrator_.config.extra_integration_distance;
  return map_.config.truncation_distance+(extra<0.f?-extra*map_.config.voxel_size:extra);
}

std::vector<Eigen::AlignedBox3f> PresentTsdf::marked() const {
  std::vector<Eigen::AlignedBox3f> boxes;
  const Eigen::Vector3f half=Eigen::Vector3f::Constant(0.5f*map_.config.voxel_size);
  for (const auto& [index,mask]:impl_->dirty) {
    const auto allocated=map_.getTsdfLayer().getBlockPtr(index);
    std::optional<hydra::TsdfBlock> made;
    const hydra::TsdfBlock& block=allocated?*allocated:made.emplace(map_.config.voxel_size,map_.config.voxels_per_side,index);
    Eigen::AlignedBox3f box;
    for (size_t w=0;w<mask.size();++w) {
      for (uint64_t bits=mask[w];bits;bits&=bits-1) box.extend(block.getVoxelPosition(w*64+static_cast<size_t>(__builtin_ctzll(bits))));
    }
    if (!box.isEmpty()) boxes.emplace_back(box.min()-half,box.max()+half);
  }
  return boxes;
}

PresentTsdf::Rebuilt PresentTsdf::rebuild(const Camera& camera, const std::vector<float>& ray_norm,
                                          const std::vector<size_t>& frames, const FramePose& pose,
                                          const ReadFrame& read) {
  if (impl_->dirty.empty()) return {};
  // A voxel is the fold of every frame's update in observation order: its dirty voxels start over;
  // the other voxels of a block keep theirs (voxels do not read each other).
  for (const auto& [index,mask]:impl_->dirty) {
    const auto block=map_.getTsdfLayer().getBlockPtr(index);
    if (!block) continue;
    for (size_t i=0;i<block->numVoxels();++i) if (isMarked(mask,i)) block->getVoxel(i)=hydra::TsdfVoxel();
  }
  // The frames' counted depths, folded into the marked voxels.
  const Rebuilt rebuilt=fold(camera,ray_norm,frames,pose,read,true);
  // A dirty block left without weight is one integration would not keep; the others mesh anew.
  for (const auto& [index,mask]:impl_->dirty) {
    const auto block=map_.getTsdfLayer().getBlockPtr(index);
    if (!block) continue;
    if (std::none_of(block->begin(),block->end(),[](const auto& v){return v.weight>0.f;})) {
      map_.removeBlock(index);
      impl_->removed.insert(index);
    } else {
      block->setUpdated();
    }
  }
  impl_->dirty.clear();
  return rebuilt;
}

void PresentTsdf::extractMesh(std::vector<Eigen::Vector3f>& vertices,std::vector<Face>& faces) {
  // A block's boundary cubes read its +x/+y/+z neighbours: a changed or removed block is
  // remeshed with the neighbours on its negative side; the other blocks keep their mesh.
  auto& tsdf=map_.getTsdfLayer();
  auto changed=tsdf.blockIndicesWithCondition(hydra::TsdfBlock::meshUpdated);
  changed.insert(changed.end(),impl_->removed.begin(),impl_->removed.end());
  for (const auto& index:changed) {
    for (int k=1;k<8;++k) {
      const hydra::BlockIndex neighbour=index-hydra::BlockIndex(k&1,(k>>1)&1,(k>>2)&1);
      const auto block=tsdf.getBlockPtr(neighbour);
      if (block) block->mesh_updated=true;
    }
  }
  impl_->removed.clear();
  mesh_integrator_.generateMesh(map_,true,true);
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

void PresentTsdf::depthFromRange(const std::vector<float>& ray_norm,const std::vector<uint16_t>& range_mm,
                                 std::vector<float>& depth_z) {
  depth_z.resize(ray_norm.size());
  for (size_t i=0;i<ray_norm.size();++i) depth_z[i]=range_mm[i]*.001f/ray_norm[i];
}
}  // namespace khronos
