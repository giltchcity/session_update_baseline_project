#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <hydra/backend/backend_input.h>
#include <hydra/common/global_info.h>
#include <hydra/common/shared_module_state.h>
#include <hydra/utils/data_directory.h>

#include "khronos/backend/backend.h"

namespace {

using namespace std::chrono_literals;

class RecordingChangeDetector : public khronos::SequentialChangeDetector {
 public:
  explicit RecordingChangeDetector(const Config& config)
      : SequentialChangeDetector(config) {}

  khronos::RayVerificator::UpdateMode setDsg(
      std::shared_ptr<const khronos::DynamicSceneGraph>) override {
    return khronos::RayVerificator::UpdateMode::kIncremental;
  }

  const khronos::Changes& detectChanges(const khronos::RPGOMerges&,
                                        khronos::TimeStamp,
                                        bool had_loopclosure) override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      observations_.push_back(had_loopclosure);
      entered_ = true;
    }
    entered_cv_.notify_all();

    std::unique_lock<std::mutex> lock(mutex_);
    release_cv_.wait(lock, [this] { return !block_ || released_; });
    return changes_;
  }

  void setBlocked(bool block) {
    std::lock_guard<std::mutex> lock(mutex_);
    block_ = block;
    released_ = !block;
  }

  bool waitUntilEntered(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return entered_cv_.wait_for(lock, timeout, [this] { return entered_; });
  }

  void release() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      released_ = true;
    }
    release_cv_.notify_all();
  }

  std::vector<bool> observations() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return observations_;
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable entered_cv_;
  std::condition_variable release_cv_;
  bool block_ = false;
  bool entered_ = false;
  bool released_ = true;
  std::vector<bool> observations_;
  khronos::Changes changes_;
};

class TestBackend : public khronos::Backend {
 public:
  using Backend::Backend;

  RecordingChangeDetector* installRecordingDetector(bool blocked) {
    auto detector =
        std::make_unique<RecordingChangeDetector>(config.change_detection);
    detector->setBlocked(blocked);
    auto* result = detector.get();
    change_detector_ = std::move(detector);
    return result;
  }

  void armLoopClosure() {
    have_loopclosures_ = true;
    have_new_loopclosures_ = true;
  }

  khronos::LatestOnlyWorker::Stats workerStats() const {
    return change_detection_worker_->stats();
  }
};

khronos::Backend::Config makeConfig(int period) {
  khronos::Backend::Config config;
  config.verbosity = 0;
  config.pose_object_covariance = 0.1;
  config.object_merge_covariance = 1.0;
  config.pose_object_consistency_threshold = 0.2;
  config.fix_input_poses = false;
  config.add_places_to_deformation_graph = false;
  config.optimize_on_lc = true;
  config.optimize_on_new_merge = false;
  config.run_change_detection_every_n_frames = period;
  config.pgmo.odom_variance = 0.01;
  config.pgmo.lc_variance = 0.05;
  config.pgmo.prior_variance = 0.01;
  config.pgmo.mesh_edge_variance = 0.01;
  config.pgmo.pose_mesh_variance = 0.01;
  config.pgmo.place_mesh_variance = 0.01;
  config.pgmo.place_edge_variance = 1.0;
  config.pgmo.place_merge_variance = 1.0;
  config.pgmo.object_merge_variance = 1.0;
  config.pgmo.sg_loop_closure_variance = 0.1;
  return config;
}

struct BackendFixture {
  explicit BackendFixture(int period)
      : private_dsg(hydra::GlobalInfo::instance().createSharedDsg()),
        state(std::make_shared<hydra::SharedModuleState>()),
        backend(makeConfig(period), private_dsg, state) {
    state->backend_graph = hydra::GlobalInfo::instance().createSharedDsg();
    state->lcd_graph = hydra::GlobalInfo::instance().createSharedDsg();
  }

  hydra::BackendInput input(uint64_t sequence, uint64_t stamp) {
    state->backend_graph->sequence_number = sequence;
    hydra::BackendInput result;
    result.timestamp_ns = stamp;
    result.sequence_number = sequence;
    return result;
  }

  hydra::SharedDsgInfo::Ptr private_dsg;
  hydra::SharedModuleState::Ptr state;
  TestBackend backend;
};

bool testLoopClosureSurvivesOptimizeAndSameTickSave(
    const std::filesystem::path& output) {
  BackendFixture fixture(0);
  auto* detector = fixture.backend.installRecordingDetector(false);
  hydra::DataDirectory::Config directory_config;
  directory_config.overwrite = true;
  const hydra::DataDirectory directory(output, directory_config);

  std::atomic<bool> sink_entered{false};
  std::atomic<bool> sink_returned{false};
  fixture.backend.addSink(khronos::Backend::Sink::fromCallback(
      [&](uint64_t,
          const khronos::DynamicSceneGraph&,
          const kimera_pgmo::DeformationGraph&) {
        sink_entered = true;
        fixture.backend.saveFinalMap(directory);
        sink_returned = true;
      }));

  fixture.backend.armLoopClosure();
  fixture.backend.spinCallback(fixture.input(1, 1));
  fixture.backend.waitForChangeDetection();
  // Terminal finalization is synchronous but must preserve the same full
  // recomputation contract.
  fixture.backend.finishProcessing();

  const auto observed = detector->observations();
  const auto stats = fixture.backend.workerStats();
  const auto map_path = directory.path("final.4dmap");
  if (!sink_entered || !sink_returned ||
      observed != std::vector<bool>({true, true}) ||
      stats.requests != 1 || stats.executions != 1 ||
      !std::filesystem::is_regular_file(map_path) ||
      std::filesystem::file_size(map_path) == 0) {
    std::cerr << "loop-closure/save regression failed: sink_entered="
              << sink_entered << " sink_returned=" << sink_returned
              << " observations=" << observed.size()
              << " requests=" << stats.requests
              << " executions=" << stats.executions << " map='"
              << map_path << "'\n";
    return false;
  }

  return true;
}

bool testPeriodicAndLoopClosureRequestsCoalesceWithoutLosingTrue() {
  BackendFixture fixture(1);
  auto* detector = fixture.backend.installRecordingDetector(true);

  fixture.backend.spinCallback(fixture.input(1, 1));
  if (!detector->waitUntilEntered(2s)) {
    std::cerr << "periodic change detection did not start\n";
    detector->release();
    return false;
  }

  fixture.backend.armLoopClosure();
  fixture.backend.spinCallback(fixture.input(2, 2));
  fixture.backend.spinCallback(fixture.input(3, 3));
  detector->release();
  fixture.backend.waitForChangeDetection();

  const auto observed = detector->observations();
  const auto stats = fixture.backend.workerStats();
  if (observed != std::vector<bool>({false, true}) || stats.requests != 3 ||
      stats.executions != 2 || stats.coalesced_requests != 2) {
    std::cerr << "coalesced trigger regression failed: observations="
              << observed.size() << " requests=" << stats.requests
              << " executions=" << stats.executions
              << " coalesced=" << stats.coalesced_requests << "\n";
    return false;
  }

  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " OUTPUT_DIRECTORY\n";
    return EXIT_FAILURE;
  }

  hydra::PipelineConfig pipeline_config;
  hydra::GlobalInfo::init(pipeline_config);

  const bool save_ok =
      testLoopClosureSurvivesOptimizeAndSameTickSave(argv[1]);
  const bool coalescing_ok =
      testPeriodicAndLoopClosureRequestsCoalesceWithoutLosingTrue();
  hydra::GlobalInfo::reset();
  return save_ok && coalescing_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
