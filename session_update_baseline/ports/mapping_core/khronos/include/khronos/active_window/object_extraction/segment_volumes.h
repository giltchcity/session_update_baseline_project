#pragma once

#include <Eigen/Core>
#include <array>
#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <vector>

#include <hydra/reconstruction/volumetric_map.h>

namespace khronos {

/**
 * @brief The reconstruction volumes of extracted segments, from the extraction that built them to
 * the object reasoning that takes each into the state its segment belongs to. An object's
 * attributes carry the token (kSegmentVolumeDetail); the volume itself never leaves the process.
 */
class SegmentVolumes {
 public:
  using Volume = std::shared_ptr<hydra::VolumetricMap>;

  static size_t put(Volume volume) {
    auto& self = instance();
    std::lock_guard<std::mutex> lock(self.mutex_);
    const size_t token = self.next_++;
    self.volumes_[token] = std::move(volume);
    return token;
  }

  static Volume take(size_t token) {
    auto& self = instance();
    std::lock_guard<std::mutex> lock(self.mutex_);
    const auto it = self.volumes_.find(token);
    if (it == self.volumes_.end()) {
      return nullptr;
    }
    Volume volume = std::move(it->second);
    self.volumes_.erase(it);
    return volume;
  }

  //! A segment the extraction made no object of: its volume goes back to the background.
  static void giveBack(Volume volume) {
    if (!volume) return;
    auto& self = instance();
    std::lock_guard<std::mutex> lock(self.mutex_);
    self.returned_.push_back(std::move(volume));
  }

  static std::vector<Volume> takeReturned() {
    auto& self = instance();
    std::lock_guard<std::mutex> lock(self.mutex_);
    std::vector<Volume> returned;
    returned.swap(self.returned_);
    return returned;
  }

  //! The limits of the vote an object's surface is told by (the extraction's configuration).
  static void setVote(float min_confidence, float min_observations) {
    auto& self = instance();
    std::lock_guard<std::mutex> lock(self.mutex_);
    self.min_confidence_ = min_confidence;
    self.min_observations_ = min_observations;
  }

  //! A copy of an object's volume holding its surface only: a voxel at or behind the surface
  //! whose readings were not the object's by the configured share (or were too few to tell) is
  //! empty, exactly as the buffered reconstruction of an object empties it. The volume itself
  //! keeps every reading and vote (more frames may come, and votes of segments add up).
  static std::unique_ptr<hydra::VolumetricMap> voted(const hydra::VolumetricMap& volume) {
    float min_confidence = 0.f, min_observations = 0.f;
    {
      auto& self = instance();
      std::lock_guard<std::mutex> lock(self.mutex_);
      min_confidence = self.min_confidence_;
      min_observations = self.min_observations_;
    }
    auto copy = volume.clone();
    const auto* votes = copy->getSemanticLayer();
    if (!votes) return copy;
    for (auto& block : copy->getTsdfLayer()) {
      const auto vote_block = votes->getBlockPtr(block.index);
      for (size_t i = 0; i < block.numVoxels(); ++i) {
        auto& voxel = block.getVoxel(i);
        if (voxel.distance > 0.f) continue;
        float confidence = 0.f;   // no vote at all: not the object's
        if (vote_block) {
          const auto& vote = vote_block->getVoxel(i);
          if (!vote.empty) {
            const float total = vote.semantic_likelihoods(0) + vote.semantic_likelihoods(1);
            confidence = total < min_observations ? -1.f : vote.semantic_likelihoods(1) / total;
          }
        }
        if (confidence < min_confidence) voxel.distance = copy->config.truncation_distance;
      }
    }
    return copy;
  }

  //! The surfaces (world frame) of the volumes tracks hold before their first hand-over, by
  //! instance: shown with the background until handed over, as their pixels always were.
  struct Held {
    size_t instance = 0;
    std::vector<Eigen::Vector3f> vertices;
    std::vector<std::array<uint32_t, 3>> faces;
  };
  static void setHeld(std::vector<Held> held) {
    auto& self = instance();
    std::lock_guard<std::mutex> lock(self.mutex_);
    self.held_ = std::move(held);
  }

  static std::vector<Held> held() {
    auto& self = instance();
    std::lock_guard<std::mutex> lock(self.mutex_);
    return self.held_;
  }

 private:
  // Tokens of another run (a loaded map's details) name no volume of this one.
  SegmentVolumes() : next_(std::random_device{}()) { next_ = (next_ << 24) | 1u; }
  static SegmentVolumes& instance() {
    static SegmentVolumes self;
    return self;
  }
  std::mutex mutex_;
  size_t next_;
  std::map<size_t, Volume> volumes_;
  std::vector<Volume> returned_;
  std::vector<Held> held_;
  float min_confidence_ = 0.f, min_observations_ = 0.f;
};

}  // namespace khronos
