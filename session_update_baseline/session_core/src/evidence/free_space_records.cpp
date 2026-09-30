#include "session_core/evidence/free_space_records.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "session_core/evidence/element_round.h"

namespace khronos {

uint64_t FreeSpaceRecords::keyOf(const Eigen::Vector3f& point) const {
  if (!(voxel_ > 0.f)) throw std::logic_error("Free-space records have no resolution");
  std::array<int64_t, 3> cell;
  for (size_t axis = 0; axis < 3; ++axis) {
    cell[axis] = static_cast<int64_t>(std::floor(point[axis] / voxel_));
  }
  return packElementCell(cell);
}

uint32_t FreeSpaceRecords::beginSession() { return ++num_sessions_; }

void FreeSpaceRecords::markFree(uint64_t key, TimeStamp stamp) {
  auto& records = voxels_[key];
  if (!records.empty() && records.back().session == num_sessions_) {
    auto& record = records.back();
    record.first = std::min(record.first, stamp);
    record.last = std::max(record.last, stamp);
    return;
  }
  records.push_back({num_sessions_, stamp, stamp});
}

void FreeSpaceRecords::merge(const FreeSpaceRecords& other) {
  for (const auto& [key, records] : other.voxels_) {
    for (const auto& record : records) {
      markFree(key, record.first);
      markFree(key, record.last);
    }
  }
}

bool FreeSpaceRecords::freeDuring(const Eigen::Vector3f& point, TimeStamp from, TimeStamp to) const {
  if (voxels_.empty()) return false;
  const auto found = voxels_.find(keyOf(point));
  if (found == voxels_.end()) return false;
  return std::any_of(found->second.begin(), found->second.end(),
                     [&](const Record& r) { return r.first <= to && r.last >= from; });
}

nlohmann::json FreeSpaceRecords::toJson() const {
  // One binary blob of fixed-width records: key, session, first, last.
  std::vector<uint8_t> bytes;
  const auto put = [&bytes](const auto& value) {
    const auto* raw = reinterpret_cast<const uint8_t*>(&value);
    bytes.insert(bytes.end(), raw, raw + sizeof(value));
  };
  std::vector<uint64_t> keys;
  keys.reserve(voxels_.size());
  for (const auto& entry : voxels_) keys.push_back(entry.first);
  std::sort(keys.begin(), keys.end());
  for (const auto key : keys) {
    for (const auto& record : voxels_.at(key)) {
      put(key);
      put(record.session);
      put(static_cast<uint64_t>(record.first));
      put(static_cast<uint64_t>(record.last));
    }
  }
  return nlohmann::json{{"voxel", voxel_},
                        {"sessions", num_sessions_},
                        {"records", nlohmann::json::binary(std::move(bytes))}};
}

FreeSpaceRecords FreeSpaceRecords::fromJson(const nlohmann::json& value) {
  FreeSpaceRecords result(value.at("voxel").get<float>());
  result.num_sessions_ = value.at("sessions").get<uint32_t>();
  const auto& blob = value.at("records");
  if (!blob.is_binary()) throw std::invalid_argument("Free-space records are not binary");
  const auto& bytes = blob.get_binary();
  constexpr size_t kRecord = sizeof(uint64_t) + sizeof(uint32_t) + 2 * sizeof(uint64_t);
  if (bytes.size() % kRecord != 0) throw std::invalid_argument("Truncated free-space records");
  for (size_t offset = 0; offset < bytes.size(); offset += kRecord) {
    uint64_t key, first, last;
    uint32_t session;
    const uint8_t* p = bytes.data() + offset;
    std::memcpy(&key, p, sizeof(key));
    std::memcpy(&session, p + sizeof(key), sizeof(session));
    std::memcpy(&first, p + sizeof(key) + sizeof(session), sizeof(first));
    std::memcpy(&last, p + sizeof(key) + sizeof(session) + sizeof(first), sizeof(last));
    if (session == 0 || session > result.num_sessions_ || last < first) {
      throw std::invalid_argument("Invalid free-space record");
    }
    result.voxels_[key].push_back({session, first, last});
  }
  return result;
}

}  // namespace khronos
