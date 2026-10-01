/** -----------------------------------------------------------------------------
 * Copyright (c) 2024 Massachusetts Institute of Technology.
 * All Rights Reserved.
 *
 * AUTHORS:      Lukas Schmid <lschmid@mit.edu>, Marcus Abate <mabate@mit.edu>,
 *               Yun Chang <yunchang@mit.edu>, Luca Carlone <lcarlone@mit.edu>
 * AFFILIATION:  MIT SPARK Lab, Massachusetts Institute of Technology
 * YEAR:         2024
 * SOURCE:       https://github.com/MIT-SPARK/Khronos
 * LICENSE:      BSD 3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * -------------------------------------------------------------------------- */

#include "khronos/backend/backend.h"
#include "session_core/surface/closed_object_background.h"
#include "session_core/adapters/evidence_round.h"
#include "session_core/evidence/element_measurement.h"
#include "session_core/evidence/frame_pair_sampler.h"
#include "session_core/evidence/observed_absence.h"
#include "session_core/runtime/session_bundle.h"
#include "session_core/surface/surface_sampling.h"
#include <chrono>
#include <functional>
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <fstream>
#include <malloc.h>
#include <glog/logging.h>
namespace khronos {
namespace {
// Return memory freed at session end (frame archive, stored evidence, volumes)
// to the system: glibc keeps freed heap pages of the arenas otherwise.
void releaseFreedMemory() { malloc_trim(0); }
}  // namespace


void Backend::setPhysicalEvidenceStore(PhysicalEvidenceStore::Ptr store) {
  physical_evidence_store_ = store;
  change_detector_->setPhysicalEvidenceStore(store);
  if (!session_extensions_enabled_ || !store) return;
  ensureErrorModel();
  // README principle 8: every newly stored frame feeds the online calibration of psi, from the
  // first frame of the session on.
  const auto verificator = change_detector_->getRayVerificator();
  if (!verificator) throw std::logic_error("Session evidence model is unavailable");
  const auto statistics = verificator->observedAbsenceModel().statistics();
  store->setIngestObserver(
      [statistics](const PhysicalEvidenceStore::Snapshot& snapshot, TimeStamp stamp) {
        try {
          double range = 0.0;
          accumulateFramePairs(snapshot, stamp, statistics->calibrator, statistics->zeta.load(), &range);
          if (range > 0.0) statistics->noteFrame(stamp, range);
        } catch (const std::exception& error) {
          LOG(WARNING) << "Online calibration skipped a frame: " << error.what();
        }
      });
}

namespace {
// The default range model of the appendix (a refusion_report.json of the same device and
// processing flow): its sigma_cm curve seeds sigma_s(rho, theta) and its depth scale the session
// scale; the outlier weights, variogram and alignment residual are estimated online (README (9v)).
model::RangeModel readDefaultRangeModel(const std::string& path, double& zeta) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Cannot read the default error model: " + path);
  const auto report = nlohmann::json::parse(in);
  // The object that holds the sigma_cm curve also holds the depth scale (sensor.depth_scale).
  const std::function<const nlohmann::json*(const nlohmann::json&)> find =
      [&find](const nlohmann::json& node) -> const nlohmann::json* {
        if (!node.is_object()) return nullptr;
        if (node.contains("sigma_cm") && node.at("sigma_cm").is_array()) return &node;
        for (const auto& item : node.items()) {
          if (const auto* found = find(item.value())) return found;
        }
        return nullptr;
      };
  const auto* sensor = find(report);
  if (!sensor) throw std::runtime_error("The default error model has no sigma_cm curve: " + path);
  using Calibrator = model::SensorCalibrator;
  model::RangeModel psi;
  psi.range_bin = Calibrator::kRangeBin;
  psi.num_range_bins = Calibrator::kRangeBins;
  psi.incidence_bin = 0.5 * 3.14159265358979323846 / static_cast<double>(Calibrator::kIncidenceBins);
  psi.num_incidence_bins = Calibrator::kIncidenceBins;
  psi.sigma_s.assign(psi.num_range_bins * psi.num_incidence_bins, 0.0);
  const auto& curve = sensor->at("sigma_cm");
  for (size_t b = 0; b < std::min<size_t>(curve.size(), psi.num_range_bins); ++b) {
    for (size_t t = 0; t < psi.num_incidence_bins; ++t) {
      psi.sigma_s[b * psi.num_incidence_bins + t] = curve.at(b).get<double>() / 100.0;
    }
  }
  zeta = sensor->value("depth_scale", report.value("depth_scale", 0.0));
  psi.zeta = zeta;
  return psi;
}

}  // namespace

void Backend::ensureErrorModel() {
  const auto verificator = change_detector_->getRayVerificator();
  if (!verificator) throw std::logic_error("Session evidence model is unavailable");
  auto& calibration = verificator->observedAbsenceModel();
  if (calibration.hasRangeModel()) return;
  // README section 8: without a previous state (and without the optional default file, which is
  // only an initial value of principle 8) psi is estimated from the session's own data, from the
  // first frame pairs on; until then no source is consumed.
  if (config.error_model_path.empty()) return;
  double zeta = 0.0;
  auto psi = readDefaultRangeModel(config.error_model_path, zeta);
  calibration.setInitialRangeModel(std::move(psi));
}

void Backend::setMapScales(const SessionRefusion::Scales& scales) { map_scales_ = scales; }

void Backend::setFrameArchive(FrameArchive::Ptr archive) { frame_archive_ = std::move(archive); }

void Backend::setShownMemory(SessionRefusion::Surface shown) {
  shown_memory_ = std::make_unique<SessionRefusion::Surface>(std::move(shown));
  inherited_current_keys_.clear();
  for (const auto id : persistent_objects_.trackedIds()) {
    const auto current = persistent_objects_.currentFragment(id);
    if (current) inherited_current_keys_.emplace(id, current->evidence_key);
  }
}

void Backend::setPreviousDepthScales(std::vector<float> scales) {
  previous_depth_scales_ = std::move(scales);
}

void Backend::setObjectSurfaceResolution(const float resolution) {
  if (session_extensions_enabled_) persistent_objects_.setMapResolution(resolution);
  object_surface_resolution_ = resolution;
}

void Backend::setConstructionHits(const double hits) {
  if (session_extensions_enabled_) persistent_objects_.setConstructionHits(hits);
}

size_t Backend::verifyCurrentObjectStates(const TimeStamp stamp) {
  const auto verificator = change_detector_->getRayVerificator();
  if (!verificator) {
    return 0;
  }
  // One frozen snapshot for the whole pass, so an asynchronous frame ingest cannot split a single
  // state decision across two store versions.
  const auto evidence = verificator->physicalEvidenceSnapshot();
  auto& calibration = verificator->observedAbsenceModel();
  const size_t closed = runEvidenceRound(persistent_objects_, calibration,
                                         evidence ? &*evidence : nullptr, stamp,
                                         object_surface_resolution_, frame_attribution_->frameInterval());
  // Delta_round of the write commitment (principle 5): the time from one round to the next.
  if (last_round_stamp_ > 0 && stamp > last_round_stamp_) {
    round_seconds_ = static_cast<double>(stamp - last_round_stamp_) * 1e-9;
  }
  last_round_stamp_ = stamp;
  // README (6m): what the session has measured so far predicts the next round.
  calibration.refreshRangeModel();
  publishAttribution(calibration.rangeModel(), calibration.sessionStart());
  return closed;
}

// README principle 5, (6b), (8): what the window's decisions read of the model.
void Backend::publishAttribution(const model::RangeModel& psi, TimeStamp session_start) {
  FrameAttribution::Snapshot snapshot;
  snapshot.closed_through = persistent_objects_.successionFloors();
  snapshot.hazards = persistent_objects_.motionPriors();
  const auto& prior = persistent_objects_.persistencePrior();
  for (const int cls : prior.classesWithExposure()) {
    const auto hazard = prior.classHazard(cls);
    if (hazard.valid()) snapshot.class_hazards[cls] = hazard;
  }
  snapshot.psi = psi;
  snapshot.rounds = std::make_shared<const model::RoundModel>(persistent_objects_.roundModel());
  snapshot.session_start = session_start;
  snapshot.round_seconds = round_seconds_;
  frame_attribution_->publish(std::move(snapshot));
}

void Backend::updateFinalMap() {
  if (!session_extensions_enabled_) return;
  std::lock_guard<std::mutex> map_lock(map_mutex_);
  session_terminal_ready_ = false;
  if (map_.numTimeSteps() == 0) {
    throw std::runtime_error("Terminal session map is empty");
  }
  // The terminal change-detection pass has just written the final snapshot:
  // the object reasoning's final state, which the next session reasons on.
  // Update a copy of it and replace that snapshot at the same timestamp; every
  // earlier snapshot and every change-detection input stays as it was.
  const size_t last = map_.numTimeSteps() - 1;
  const TimeStamp stamp = map_.stamps()[last];
  const auto final_dsg = map_.rawDsg(last);
  if (!final_dsg) throw std::runtime_error("Terminal session snapshot is unavailable");
  unconsolidated_final_ = final_dsg->clone();
  unconsolidated_stamp_ = stamp;
  // The terminal change detection was the last reader of the stored evidence
  // frames; release them before the update.
  if (physical_evidence_store_) physical_evidence_store_->clear();
  releaseFreedMemory();
  if (!config.refuse_final_map) {
    final_surface_error_.assign(SessionRefusion::fromDsg(*final_dsg).faces.size(),0.f);
    final_surface_records_ = {};
    session_terminal_ready_ = true;
    return;
  }
  if (!frame_archive_) throw std::runtime_error("Required session frame archive is unavailable");
  auto edited = final_dsg->clone();
  // A failed required estimator aborts terminal publication; the original snapshot
  // remains owned by map_ until the complete edited snapshot is available.
  refuseFinalMap(*edited, stamp);
  map_.update(edited, stamp);
  session_terminal_ready_ = true;
  // The frame archive and the update's volumes are gone now; hand the freed
  // memory back before the terminal save.
  releaseFreedMemory();
}

void Backend::refuseFinalMap(DynamicSceneGraph& edited, TimeStamp stamp) {
  const auto start = std::chrono::steady_clock::now();
  SessionRefusion::Inputs inputs;
  const auto frames = frame_archive_->release(&inputs.camera);
  if (frames.empty()) throw std::runtime_error("Required session frame archive is empty");
  inputs.frames = &frames;
  inputs.scales = map_scales_;
  inputs.final_stamp = stamp;
  inputs.shown = shown_memory_.get();
  inputs.previous_depth_scales = previous_depth_scales_;
  if (const char* dump = std::getenv("KHRONOS_REFUSION_DUMP")) inputs.dump_dir = dump;
  const auto verificator = change_detector_->getRayVerificator();
  if (!verificator) throw std::logic_error("Session evidence model is unavailable");
  auto& calibration = verificator->observedAbsenceModel();
  // README (8): measurement time domain and previous-surface ownership come from the same current
  // state, with separate time and identity meanings. t_L is the placement's presence begin, the
  // right end of the change interval of its predecessor (5f).
  for (const size_t id : persistent_objects_.trackedIds()) {
    const auto current = persistent_objects_.currentFragment(id);
    inputs.state_starts[id] =
        current ? std::optional<TimeStamp>(current->presence_begin) : std::nullopt;
    const auto previous = inherited_current_keys_.find(id);
    if (!current || previous == inherited_current_keys_.end() ||
        previous->second != current->evidence_key)
      inputs.replaced_states.insert(id);
  }
  // README (6m): the final estimate of the session's own data is the model of the session end.
  calibration.refreshRangeModel();
  inputs.psi = calibration.rangeModel();
  // README principle 7: the object resolution of the refusion is h_o = sqrt(12) sigma_eff(dt_f) of
  // the estimate just made, dt_f the interval between adjacent frames of the fused stream (the frame
  // archive: the median gap of its frames), with T = 2 h_o; before the pair scale exists the
  // configured one stays.
  {
    std::vector<double> gaps;
    for (size_t i = 1; i < frames.size(); ++i) {
      const double gap = (static_cast<double>(frames[i].stamp) - static_cast<double>(frames[i - 1].stamp)) * 1e-9;
      if (gap > 0.0) gaps.push_back(gap);
    }
    if (!gaps.empty()) {
      std::nth_element(gaps.begin(), gaps.begin() + gaps.size() / 2, gaps.end());
      if (const double h_o = inputs.psi.objectResolution(gaps[gaps.size() / 2]); h_o > 0.0) {
        inputs.scales.object_voxel = static_cast<float>(h_o);
        inputs.scales.object_truncation = static_cast<float>(2.0 * h_o);
      }
    }
  }
  inputs.rounds = &persistent_objects_.roundModel();
  inputs.construction_hits = persistent_objects_.constructionHits();
  // README principle 9: the committed-round histories of the elements are those of the start of
  // the session (the same data is not multiplied twice, README (5g)); the group of an element is
  // the class of its placement.
  for (const size_t id : persistent_objects_.trackedIds()) {
    const auto prior = persistent_objects_.startOfSessionPrior(id);
    if (prior) inputs.element_histories[id] = prior->histories;
    if (const auto current = persistent_objects_.currentFragment(id)) {
      inputs.identity_class[id] = current->semantic_label;
    }
  }
  // README principle 5: the smoothed estimate of the survival of each semantic class, from all the
  // events and exposure of the session, decides which unidentified readings the refusion fuses.
  {
    const auto& prior = persistent_objects_.persistencePrior();
    for (const int cls : prior.classesWithExposure()) {
      const auto hazard = prior.classHazard(cls);
      if (hazard.valid()) inputs.class_hazards[cls] = hazard;
    }
  }
  for (auto& surface : persistent_objects_.closedSurfaces()) {
    SessionRefusion::Inputs::ClosedSurface closed;
    closed.vertices = std::move(surface.vertices);
    closed.faces = std::move(surface.faces);
    closed.odds = surface.odds;
    inputs.closed_surfaces.push_back(std::move(closed));
  }
  const auto outcomes = calibration.elementOutcomes();
  inputs.fill_confirmed = outcomes.fill_confirmed;
  inputs.fill_total = outcomes.fill_total;

  SessionRefusion::Config refusion_config;
  refusion_config.num_threads = config.session_end_threads;
  const SessionRefusion refusion(refusion_config);
  auto refused = refusion.apply(edited, inputs);
  if (!refused.applied) throw std::runtime_error("Required session surface update did not complete");
  refusion_report_ = std::move(refused.report_json);
  session_depth_scale_ = refused.depth_scale;
  final_surface_error_ = std::move(refused.surface_error);
  final_surface_records_ = std::move(refused.surface_records);
  // README (15b): this session's estimate is the model of the next session; (12d) refits the
  // pair parameters on the session's memory elements and present surface; V_free gains this
  // session's free space; the completion decisions enter the statistics and the memory elements
  // that are hidden but not deleted travel with the evidence state.
  auto session_model = inputs.psi;
  if (refused.pair_fit_valid) {
    session_model.pi_dup = refused.pair_pi_dup;
    session_model.delta_s = refused.pair_delta_s;
    session_model.sigma_x = refused.pair_sigma_x;
  }
  calibration.setSessionModel(std::move(session_model));
  auto free_space = calibration.freeSpace();
  if (free_space.voxel() <= 0.f) free_space = FreeSpaceRecords(refused.free_space.voxel());
  free_space.beginSession();
  free_space.merge(refused.free_space);
  calibration.setFreeSpace(std::move(free_space));
  calibration.addElementOutcomes({refused.fill_confirmed, refused.fill_total});
  calibration.setHiddenRecords(refused.hidden.faces.empty() ? nlohmann::json()
                                                            : refused.hidden.toJson());
  LOG(INFO) << "[SessionRefusion] applied=" << refused.applied << " " << refused.summary
            << " elapsed_s="
            << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// README appendix (asynchronous access): a requested inference event completes before the next backend
// packet advances the source graph. Called after releasing the graph mutex.
void Backend::sessionCompleteUpdate() { waitForChangeDetection(); }

// README (6d): fix the sensor-time boundary before native and project queries.
void Backend::sessionBeforeDetect(TimeStamp stamp) {
  const auto verificator = change_detector_->getRayVerificator();
  if (!verificator) throw std::logic_error("Session evidence model is unavailable");
  verificator->setPhysicalEvidenceCutoff(stamp);
}

void Backend::sessionBeforeReconcile(
    const DynamicSceneGraph::Ptr& dsg, Changes& changes, TimeStamp stamp, bool finalize_pending) {
  persistent_objects_.ingestObjects(*dsg);
  // Object CURRENT states must face the same measurements the background mesh does. Before the
  // reconciler touches any mesh, while the ray index still matches the geometry it was built from.
  const size_t closed = verifyCurrentObjectStates(stamp);
  if (closed > 0) {
    CLOG(3) << "[Backend] Closed " << closed
            << " current object fragment(s) contradicted by later free-space evidence.";
  }
  if (finalize_pending) {
    persistent_objects_.finalizePendingAbsences(stamp);
  }
  if (dsg->hasMesh()) {
    const auto verificator = change_detector_->getRayVerificator();
    if (verificator) {
      markClosedObjectBackground(*dsg->mesh(), persistent_objects_, *verificator,
                                 object_surface_resolution_, stamp, changes.background_changes);
    }
  }
}
void Backend::sessionAfterReconcile(
    const DynamicSceneGraph::Ptr& dsg, TimeStamp stamp, bool finalize_pending) {
  // Change detection must see every visibility segment independently. In
  // particular, an old physical object can be cleared at its previous site
  // while a newer segment with the same stable ID materializes its current
  // geometry elsewhere. Folding the segments before the evidence transition
  // makes the old-site ray result incorrectly close the new current segment.
  // Reduce to one logical node per physical ID only after every segment has
  // been detected and reconciled.
  UpdateKhronosObjectsFunctor::canonicalizePhysicalObjects(*dsg, &persistent_objects_);
  {
    const auto verificator = change_detector_->getRayVerificator();
    if (verificator) {
      auto& calibration = verificator->observedAbsenceModel();
      publishAttribution(calibration.rangeModel(), calibration.sessionStart());
    }
  }
  if (finalize_pending) {
    // Canonicalization materializes resolved geometry. Re-measure that geometry
    // after the final settlement before publishing the terminal state.
    // Reconciliation changed mesh indices: rebuild once here rather than
    // querying the stale pre-reconciliation ray index.
    change_detector_->setDsg(dsg);
    const size_t terminal_closed = verifyCurrentObjectStates(stamp);
    persistent_objects_.finalizePendingAbsences(stamp);
    Changes terminal_changes;
    if (dsg->hasMesh()) {
      const auto verificator = change_detector_->getRayVerificator();
      if (verificator) {
        const auto background_closed = markClosedObjectBackground(
            *dsg->mesh(), persistent_objects_, *verificator,
            object_surface_resolution_, stamp, terminal_changes.background_changes);
        if (background_closed) {
          // No object changes: only the newly confirmed background removals.
          reconciler_->reconcile(*dsg, terminal_changes, stamp);
        }
      }
    }
    UpdateKhronosObjectsFunctor::canonicalizePhysicalObjects(*dsg, &persistent_objects_);
    LOG(INFO) << "TERMINAL_STATE_DRAIN stamp=" << stamp << " closed=" << terminal_closed;
  }
}
void Backend::prepareSessionSave(const hydra::DataDirectory& log_setup) {
  if (!session_terminal_ready_ || !unconsolidated_final_)
    throw std::logic_error("Complete terminal processing before publishing a session");
  session_io::beginBundle(log_setup.path());
}

void Backend::saveSessionState(const hydra::DataDirectory& log_setup, bool primary_saved) {
  if (!primary_saved) throw std::runtime_error("Required final map save failed");
  if (!session_terminal_ready_ || !unconsolidated_final_)
    throw std::runtime_error("Session terminal state was not prepared");
  const auto path = log_setup.path();
  std::vector<std::string> members{"final.4dmap.zpk","chain_state.4dmap.zpk",
      "shown_state.4dmap.zpk","registry_state.cbor","evidence_state.cbor","sensor_statistics.txt","depth_scales.txt"};
  const auto verificator = change_detector_->getRayVerificator();
  if (!verificator) throw std::logic_error("Session evidence model is unavailable");
  auto& absence = verificator->observedAbsenceModel();
  absence.setMotionState(frame_attribution_->motion().toJson());
  absence.save((path / "evidence_state.cbor").string(),unconsolidated_stamp_,
               persistent_objects_.liveEvidenceKeys());
  if (!absence.saveSensorStatistics((path / "sensor_statistics.txt").string())) {
    throw std::runtime_error("Failed to save absence sensor statistics");
  }
  if (unconsolidated_final_) {
      // The next session reasons on the object reasoning's final state and
      // shows the updated final map as its memory.
      SpatioTemporalMap chain(config.spatio_temporal_map);
      chain.update(unconsolidated_final_->clone(), unconsolidated_stamp_);
      if (!chain.save(path / "chain_state.4dmap.zpk")) {
        throw std::runtime_error("Failed to save session reasoning state");
      }
      persistent_objects_.saveCheckpoint((path / "registry_state.cbor").string(),
          (path / "chain_state.4dmap.zpk").string(), unconsolidated_stamp_, *unconsolidated_final_);
      {
        // What this session's final map shows: the next session's memory.
        const size_t last = map_.numTimeSteps() - 1;
        SpatioTemporalMap shown(config.spatio_temporal_map);
        shown.update(map_.rawDsg(last)->clone(), map_.stamps()[last]);
        if (!shown.save(path / "shown_state.4dmap.zpk")) {
          throw std::runtime_error("Failed to save shown session state");
        }
        auto surface = SessionRefusion::fromDsg(*map_.rawDsg(last));
        surface.face_error = final_surface_error_;
        surface.face_hits = final_surface_records_.hits;
        surface.face_through = final_surface_records_.through;
        surface.face_pending_hits = final_surface_records_.pending_hits;
        surface.face_pending_through = final_surface_records_.pending_through;
        SessionRefusion::saveSurfaceError((path / "surface_error.bin").string(), surface);
        members.push_back("surface_error.bin");
      }
      if (!refusion_report_.empty()) {
        std::ofstream report(path / "refusion_report.json");
        report.exceptions(std::ios::badbit | std::ios::failbit);
        report << refusion_report_ << "\n";
        report.close();
        members.push_back("refusion_report.json");
      }
      {
        // Calibration audit. Per-face source error drives surface correspondence.
        std::ofstream scales(path / "depth_scales.txt");
        scales.exceptions(std::ios::badbit | std::ios::failbit);
        for (const float s : previous_depth_scales_) scales << s << "\n";
        if (session_depth_scale_) scales << *session_depth_scale_ << "\n";
        scales.close();
      }
    }
  session_io::publishBundle(path, unconsolidated_stamp_, members);
}

}  // namespace khronos
