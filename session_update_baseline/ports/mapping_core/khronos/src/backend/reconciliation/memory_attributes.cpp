#include "khronos/backend/reconciliation/memory_attributes.h"

#include <cstring>
#include <fstream>

#include <hydra/utils/nearest_neighbor_utilities.h>

namespace khronos {

namespace {
constexpr char kMagic[4] = {'K', 'M', 'A', '1'};
constexpr uint32_t kVersion = 1;
}  // namespace

bool writeMemoryAttributes(const std::string& path,
                           const std::vector<MemoryAttributeRecord>& records) {
  std::ofstream out(path, std::ios::binary);
  if (!out) return false;
  out.write(kMagic, 4);
  out.write(reinterpret_cast<const char*>(&kVersion), sizeof(kVersion));
  const uint64_t n = records.size();
  out.write(reinterpret_cast<const char*>(&n), sizeof(n));
  for (const auto& r : records) {
    char buf[28] = {0};
    const float f[5] = {r.x, r.y, r.z, r.q, r.s};
    std::memcpy(buf, f, sizeof(f));
    std::memcpy(buf + 20, &r.label, sizeof(uint32_t));
    std::memcpy(buf + 24, &r.layer, sizeof(uint8_t));
    out.write(buf, sizeof(buf));
  }
  return static_cast<bool>(out);
}

bool readMemoryAttributes(const std::string& path, std::vector<MemoryAttributeRecord>& records) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  char magic[4];
  uint32_t version = 0;
  uint64_t n = 0;
  if (!in.read(magic, 4) || std::memcmp(magic, kMagic, 4) != 0) return false;
  if (!in.read(reinterpret_cast<char*>(&version), sizeof(version)) || version != kVersion) {
    return false;
  }
  if (!in.read(reinterpret_cast<char*>(&n), sizeof(n))) return false;
  records.clear();
  records.reserve(n);
  for (uint64_t i = 0; i < n; ++i) {
    char buf[28];
    if (!in.read(buf, sizeof(buf))) return false;
    MemoryAttributeRecord r;
    float f[5];
    std::memcpy(f, buf, sizeof(f));
    r.x = f[0];
    r.y = f[1];
    r.z = f[2];
    r.q = f[3];
    r.s = f[4];
    std::memcpy(&r.label, buf + 20, sizeof(uint32_t));
    std::memcpy(&r.layer, buf + 24, sizeof(uint8_t));
    records.push_back(r);
  }
  return true;
}

MemoryAttributeLookup::MemoryAttributeLookup(std::vector<MemoryAttributeRecord> records)
    : records_(std::move(records)) {
  points_.reserve(records_.size());
  for (const auto& r : records_) points_.emplace_back(r.x, r.y, r.z);
  if (!points_.empty()) search_ = std::make_unique<hydra::PointNeighborSearch>(points_);
}

MemoryAttributeLookup::~MemoryAttributeLookup() = default;

const MemoryAttributeRecord* MemoryAttributeLookup::find(const Eigen::Vector3f& p,
                                                         float max_distance) const {
  if (!search_) return nullptr;
  float d_sq = 0.f;
  size_t idx = 0;
  if (!search_->search(p, d_sq, idx) || d_sq > max_distance * max_distance) return nullptr;
  return &records_[idx];
}

}  // namespace khronos
