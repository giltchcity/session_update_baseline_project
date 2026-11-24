// The session backend's loop on Hydra's backend module: frontend outputs advance the scene graph and
// the mesh, every N updates a round runs on a clone of that moment's state, each round's snapshot
// goes into the 4D map (with the session's move trajectories), and the terminal round and the saved
// map end the session.
#include <chrono>
#include <filesystem>

#include <glog/logging.h>
#include <hydra/common/pipeline_queues.h>
#include <hydra/utils/timing_utilities.h>

#include "khronos/utils/khronos_attribute_utils.h"
#include "session_update_baseline/runtime/session_backend.h"

namespace session_update::runtime {
using namespace khronos;

void SessionBackend::start() { spin_thread_.reset(new std::thread(&SessionBackend::spinLoop, this)); }

void SessionBackend::spinLoop() {
  should_shutdown_ = false;
  bool should_shutdown = false;
  while (!should_shutdown) {
    auto& queue = hydra::PipelineQueues::instance().backend_queue;
    const bool has_data = queue.poll();
    if (hydra::GlobalInfo::instance().force_shutdown() || !has_data) {
      should_shutdown = should_shutdown_;
    }
    if (!has_data) continue;
    const auto input = queue.pop();
    spinCallback(*input);
  }
}

void SessionBackend::spinCallback(const hydra::BackendInput& input) {
  const uint64_t timestamp_ns = input.timestamp_ns;
  {
    // Serializes the backend state's advance and the clone a round is requested on. The sinks run
    // below, after the lock: a sink may wait for the round just requested.
    std::lock_guard<std::mutex> lock(mutex_);
    status_log_.emplace_back(hydra::BackendModuleStatus{});
    last_timestamp_received_ = timestamp_ns;
    last_sequence_number_ = input.sequence_number;
    if (config.run_change_detection_every_n_frames > 0) ++num_backend_updates_since_last_round_;

    hydra::timing::ScopedTimer timer("backend/update", last_timestamp_received_);
    updateFactorGraph(input);
    copyMeshDelta(input);
    updateFromLcdQueue();
    status_log_.back().total_loop_closures = num_loop_closures_;
    // The frontend's graph is read once every factor-graph packet of it has arrived.
    if (!updatePrivateDsg(timestamp_ns, false)) {
      CLOG(5) << "[SessionBackend] skipping input @" << timestamp_ns << ".";
      logStatus();
      return;
    }
    const bool had_loopclosure = have_new_loopclosures_;
    // Poses are given: the scene graph's mesh and objects follow the frontend (no optimization).
    updateDsgMesh(timestamp_ns);
    hydra::UpdateInfo::ConstPtr info(new hydra::UpdateInfo{timestamp_ns});
    dsg_updater_->callUpdateFunctions(timestamp_ns, info);
    // A round every n updates (and on a loop closure); negative: the terminal round alone.
    if (config.run_change_detection_every_n_frames >= 0) {
      const bool periodic = config.run_change_detection_every_n_frames > 0 &&
                            num_backend_updates_since_last_round_ >= config.run_change_detection_every_n_frames;
      if (had_loopclosure || periodic) {
        num_backend_updates_since_last_round_ = 0;
        requestRound(had_loopclosure);
      }
    }
    logStatus();
    have_new_loopclosures_ = false;
  }
  hydra::timing::ScopedTimer sink_timer("backend/sinks", timestamp_ns);
  Sink::callAll(sinks_, timestamp_ns, *private_dsg_->graph, *deformation_graph_);
}

void SessionBackend::requestRound(const bool had_loopclosure) {
  // The state of this request, taken now (the caller holds mutex_): the round of this time sees
  // exactly this time's backend state.
  const auto clone_start = std::chrono::steady_clock::now();
  Round round{unmerged_graph_->clone(), last_timestamp_received_, had_loopclosure};
  round_clone_time_ns_.fetch_add(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - clone_start).count(),
      std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> lock(rounds_mutex_);
    rounds_.push_back(std::move(round));
  }
  round_worker_->request();
}

void SessionBackend::waitForRounds() {
  if (round_worker_) round_worker_->waitUntilIdle();
}

void SessionBackend::finishProcessing() {
  std::lock_guard<std::mutex> finalize_lock(finalization_mutex_);
  if (final_processing_complete_) return;
  // The pipeline drained the active-window, frontend and backend queues before this call: join the
  // last requested round, then run the one terminal round synchronously.
  waitForRounds();
  std::lock_guard<std::mutex> lock(mutex_);
  updatePrivateDsg(last_timestamp_received_, true);
  if (config.run_change_detection_every_n_frames >= 0) {
    runRound(unmerged_graph_->clone(), last_timestamp_received_, /*finalize_pending=*/true);
    afterTerminalRound();
  } else {
    std::lock_guard<std::mutex> map_lock(map_mutex_);
    storeSnapshot(private_dsg_->graph->clone(), last_timestamp_received_);
  }
  const auto stats = round_worker_->stats();
  LOG(INFO) << "[BackendPerformance] round_unit=backend_update requests=" << stats.requests
            << " executions=" << stats.executions << " coalesced=" << stats.coalesced_requests
            << " clone_ms=" << round_clone_time_ns_.load() / 1.0e6
            << " compute_ms=" << round_compute_time_ns_.load() / 1.0e6;
  final_processing_complete_ = true;
}

namespace {

// How many of the samples (in order of time) are not later than the stamp.
size_t samplesUntil(const std::vector<MoveTrajectories::Sample>& samples, TimeStamp stamp) {
  size_t count = 0;
  while (count < samples.size() && samples[count].stamp <= stamp) ++count;
  return count;
}

// The samples of the moves a snapshot at `stamp` carries, as evaluation/chain_fill/motion_fill.cpp
// writes them: on the object node of the instance (details "instance_id"; the one present at the
// stamp, else the first), every sample not later than the stamp, after whatever trajectory the
// node has of its own. `carried` is how many samples the snapshot carries already, by instance.
bool writeMoveTrajectories(DynamicSceneGraph& dsg, TimeStamp stamp,
                           const std::map<int, std::vector<MoveTrajectories::Sample>>& moves,
                           std::map<int, size_t>& carried) {
  if (moves.empty() || !dsg.hasLayer(DsgLayers::OBJECTS)) return false;
  std::map<int, KhronosObjectAttributes*> owner;
  for (const auto& [id, node] : dsg.getLayer(DsgLayers::OBJECTS).nodes()) {
    auto* attrs = node->tryAttributes<KhronosObjectAttributes>();
    if (!attrs) continue;
    const auto instance = attrs->details.find("instance_id");
    if (instance == attrs->details.end() || instance->second.empty()) continue;
    const int instance_id = static_cast<int>(instance->second.front());
    if (!moves.count(instance_id)) continue;
    const auto found = owner.find(instance_id);
    if (found == owner.end() || (isPresent(*attrs, stamp) && !isPresent(*found->second, stamp))) {
      owner[instance_id] = attrs;
    }
  }
  bool changed = false;
  for (auto& [instance_id, attrs] : owner) {
    const auto& samples = moves.at(instance_id);
    const size_t count = samplesUntil(samples, stamp);
    size_t& had = carried[instance_id];
    if (count == had) continue;
    const size_t own = attrs->trajectory_timestamps.size() - std::min(had, attrs->trajectory_timestamps.size());
    attrs->trajectory_timestamps.resize(own);
    attrs->trajectory_positions.resize(std::min(own, attrs->trajectory_positions.size()));
    for (size_t i = 0; i < count; ++i) {
      attrs->trajectory_timestamps.push_back(samples[i].stamp);
      attrs->trajectory_positions.push_back(samples[i].position);
    }
    had = count;
    changed = true;
  }
  return changed;
}

}  // namespace

void SessionBackend::alignMoveSamples() {
  if (move_samples_.size() < map_.numTimeSteps()) {
    move_samples_.insert(move_samples_.begin(), map_.numTimeSteps() - move_samples_.size(), {});
  }
}

void SessionBackend::storeSnapshot(const DynamicSceneGraph::Ptr& dsg, TimeStamp stamp) {
  alignMoveSamples();
  // The moves whose end and time after lie before this snapshot are told by now. A move just told
  // belongs to the snapshots stored since it started as well: they are the latest few, written now
  // (a piece history stores again only from the first of them on).
  MoveTrajectories::resolve(stamp);
  snapshots_stored_anew_ += writeToldMoves(MoveTrajectories::moves());
  // A snapshot at the stamp of the last one takes its place (the terminal pass).
  if (map_.numTimeSteps() > 0 && map_.stamps().back() == stamp && !move_samples_.empty()) {
    move_samples_.back().clear();
  } else {
    move_samples_.emplace_back();
  }
  writeMoveTrajectories(*dsg, stamp, MoveTrajectories::moves(), move_samples_.back());
  map_.update(dsg, stamp);
}

size_t SessionBackend::writeToldMoves(const std::map<int, std::vector<MoveTrajectories::Sample>>& moves) {
  if (moves.empty() || map_.numTimeSteps() == 0 || move_samples_.size() != map_.numTimeSteps()) return 0;
  return map_.rewriteSnapshots(
      [&](size_t index, TimeStamp stamp) {
        for (const auto& [instance_id, of_instance] : moves) {
          const auto had = move_samples_[index].find(instance_id);
          if (samplesUntil(of_instance, stamp) != (had == move_samples_[index].end() ? 0 : had->second)) {
            return true;
          }
        }
        return false;
      },
      [&](DynamicSceneGraph& dsg, size_t index, TimeStamp stamp) {
        const bool changed = writeMoveTrajectories(dsg, stamp, moves, move_samples_[index]);
        // An instance without a node in the snapshot has nothing to carry there: settled.
        for (const auto& [instance_id, of_instance] : moves) {
          move_samples_[index][instance_id] = samplesUntil(of_instance, stamp);
        }
        return changed;
      });
}

void SessionBackend::fillMoveTrajectories() {
  alignMoveSamples();
  if (map_.numTimeSteps() == 0 || move_samples_.size() != map_.numTimeSteps()) return;
  const auto start = std::chrono::steady_clock::now();
  MoveTrajectories::resolve(map_.stamps().back());
  const auto moves = MoveTrajectories::moves();
  size_t samples = 0;
  for (const auto& [instance_id, of_instance] : moves) samples += of_instance.size();
  // A move told after a snapshot was stored belongs to that snapshot as well, up to the snapshot's
  // own time: the snapshots of the map are what a reader of the map sees.
  const size_t rewritten = writeToldMoves(moves);
  // The graph the mapping worked on: what trajectory it has is the tracker's own.
  size_t working_graph_samples = 0;
  if (private_dsg_ && private_dsg_->graph && private_dsg_->graph->hasLayer(DsgLayers::OBJECTS)) {
    for (const auto& [id, node] : private_dsg_->graph->getLayer(DsgLayers::OBJECTS).nodes()) {
      const auto* attrs = node->tryAttributes<KhronosObjectAttributes>();
      working_graph_samples += attrs ? attrs->trajectory_timestamps.size() : 0;
    }
  }
  size_t waiting = 0;
  for (const auto& [instance_id, end] : MoveTrajectories::waiting()) {
    (void)instance_id;
    (void)end;
    ++waiting;
  }
  LOG(INFO) << "MOVE_TRAJECTORY_FILL snapshots=" << map_.numTimeSteps() << " instances=" << moves.size()
            << " samples=" << samples << " snapshots_stored_anew=" << rewritten
            << " stored_anew_as_told=" << snapshots_stored_anew_ << " ends_waiting=" << waiting
            << " working_graph_trajectory_samples=" << working_graph_samples << " seconds="
            << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

void SessionBackend::save(const hydra::DataDirectory& log_setup) { saveFinalMap(log_setup); }

void SessionBackend::saveFinalMap(const hydra::DataDirectory& log_setup) {
  // A save may follow a round just requested: never serialize the map while a round is in flight.
  waitForRounds();
  const auto path = log_setup.path();
  std::lock_guard<std::mutex> lock(map_mutex_);
  if (map_.numTimeSteps() == 0) {
    LOG(ERROR) << "Refusing to save an empty 4D map. Stop/drain the pipeline and call "
                  "finishProcessing() before the terminal save.";
    return;
  }
  fillMoveTrajectories();
  if (map_.save(path / "final.4dmap.zpk")) {
    CLOG(1) << "Saved 4D map with " << map_.numTimeSteps() << " time steps to '" << path << "'.";
  }
  saveSessionState(log_setup);
}

}  // namespace session_update::runtime
