// The mapping session's part of every change-detection round (SessionBackend): object states,
// the present surface with the memory, the dated history and the saved session state.
#include "session_update_baseline/runtime/session_backend.h"

#include "khronos/backend/backend.h"
#include "khronos/active_window/online_front.h"
#include "session_core/adapters/physical_object_access.h"
#include "session_core/state/mechanism_ablation.h"
#include "session_core/surface/closed_object_background.h"
#include "session_core/surface/triangle_grid.h"
#include <algorithm>
#include <string>
#include <cmath>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <malloc.h>
#include <glog/logging.h>
#include "session_core/objects/physical_objects.h"
#include "session_core/objects/physical_presence.h"
namespace khronos {
namespace {
// Return memory freed at session end (stored evidence) to the system: glibc
// keeps freed heap pages of the arenas otherwise.
void releaseFreedMemory() { malloc_trim(0); }
}  // namespace
}  // namespace khronos

namespace session_update::runtime {
using namespace khronos;

void SessionBackend::connectSession(const SessionInputs& inputs) {
  physical_evidence_store_ = inputs.evidence;
  persistent_objects_.setMapResolution(inputs.object_surface_resolution);
  object_surface_resolution_ = inputs.object_surface_resolution;
  present_surface_ = inputs.present;
  persistent_objects_.setHighMobilitySemanticLabels(high_mobility_semantic_labels_);
}

void SessionBackend::setLoadedMemory(std::vector<Eigen::Vector3f> points,
                                     std::vector<std::array<uint32_t, 3>> faces) {
  loaded_memory_search_.reset();
  loaded_memory_ = std::move(points);
  loaded_memory_faces_ = std::move(faces);
  if (!loaded_memory_.empty()) {
    loaded_memory_search_ = std::make_unique<hydra::PointNeighborSearch>(loaded_memory_);
  }
  // A vertex of a round's snapshot is memory when it is a copy of a loaded vertex (P53).
  memory_copy_tolerance_ = PresentSurface::copyTolerance(loaded_memory_);
  LOG(INFO) << "[PresentSurface] memory copy tolerance m=" << memory_copy_tolerance_;
}

void SessionBackend::setShownMemory(PresentSurface::Surface shown) {
  shown_memory_ = std::make_unique<PresentSurface::Surface>(std::move(shown));
}

size_t SessionBackend::verifyCurrentObjectStates(const TimeStamp stamp) {
  if (!physical_evidence_store_) {
    return 0;
  }
  // One frozen snapshot for the whole pass, so an asynchronous frame ingest cannot split a single
  // state decision across two store versions.
  const RayVerificator::PhysicalEvidenceSnapshot evidence = physical_evidence_store_->snapshot();

  size_t closed = 0;
  for (const size_t id : persistent_objects_.trackedIds()) {
    const auto current = persistent_objects_.currentFragment(id);
    const auto session_current = persistent_objects_.sessionCurrentFragment(id);
    if ((!current || !current->geometry || current->geometry->numVertices() == 0) &&
        (!session_current || !session_current->geometry ||
         session_current->geometry->numVertices() == 0)) {
      continue;
    }

    // Surface evidence, not sparse-vertex evidence. Every sensor ray counts
    // once, no matter how many mesh samples it crosses.
    PersistentObjectState::SurfaceEvidence inherited_evidence;
    PersistentObjectState::SurfaceEvidence session_evidence;
    const auto copy_evidence =
        [&](PersistentObjectState::SurfaceEvidence& target,
            const RayVerificator::SurfaceEvidenceCounts& result, int state_slot,
            const uint64_t state_id) {
          target.latest_support_stamp = result.latest_support_stamp;
          target.absence_coverage_sufficient = result.absence_coverage_sufficient;
          target.support_rays = result.support_rays;
          target.contradiction_rays = result.contradiction_rays;
          target.surface_samples = result.surface_samples;
          target.supported_votes = result.supported_votes;
          target.free_space_votes = result.free_space_votes;
          target.replaced_by_other_votes = result.replaced_by_other_votes;
          target.replaced_by_background_votes =
              result.replaced_by_background_votes;
          target.occluded_votes = result.occluded_votes;
          target.unobserved_samples = result.unobserved_samples;
          target.reliable_in_view = result.reliable_in_view;
          target.reliable_seen_through = result.reliable_seen_through;
          target.reliable_samples = result.reliable_samples;
          const auto likelihood =
              physicalAbsenceLookLikelihood(&surface_evidence_, id, state_slot, stamp, state_id);
          target.measured_absence_log_ratio = likelihood.log_ratio;
          target.has_measured_absence_likelihood = likelihood.has_measurement;
          target.has_calibrated_absence_source = likelihood.calibrated_source;
        };
    const auto measure = [&](const PersistentObjectState::FragmentView& fragment,
                             const int state_slot) {
      bool projected = false;
      const double prior_log_odds = persistent_objects_.changePriorLogOdds(id, state_slot);
      auto counts = surface_evidence_.countCurrent(
          id, *fragment.geometry, *fragment.bbox, evidence,
          object_surface_resolution_,
          std::max(fragment.last_support_time, fragment.last_confirmed_support), stamp,
          nullptr, &projected, state_slot, fragment.birth_time, prior_log_odds, fragment.uid);
      LOG(INFO) << "STATE_EVIDENCE_WINDOW inst=" << id
                << " after=" << std::max(fragment.last_support_time, fragment.last_confirmed_support)
                << " latest_measured_support=" << counts.latest_support_stamp << " through=" << stamp
                << " projected=" << projected
                << " support=" << counts.support_rays
                << " contradiction=" << counts.contradiction_rays
                << " reliable=" << counts.reliable_samples
                << " reliable_in_view=" << counts.reliable_in_view
                << " reliable_seen_through=" << counts.reliable_seen_through
                << " absence_llr=" << counts.absence_llr << " prior_log_odds=" << prior_log_odds
                << " absent_samples=" << counts.contradicted_surface_samples
                << " total_samples=" << counts.surface_samples
                << " absence_coverage_sufficient=" << counts.absence_coverage_sufficient;
      return counts;
    };
    if (current && current->geometry && current->geometry->numVertices() > 0) {
      copy_evidence(inherited_evidence, measure(*current, 0), 0, current->uid);
    }
    if (session_current && session_current->geometry &&
        session_current->geometry->numVertices() > 0) {
      copy_evidence(session_evidence, measure(*session_current, 1), 1, session_current->uid);
    }
    // Per-slice six-class evidence ledger (STATE_SLICE): every change
    // detection round records what the RGB-D actually measured at the old
    // site, per surface sample. This is the ground-truth trace that answers
    // "did the map follow the real world, and when".
    LOG(INFO) << "STATE_SLICE inst=" << id
              << " stamp=" << stamp
              << " inherited_sup=" << inherited_evidence.support_rays
              << " inherited_con=" << inherited_evidence.contradiction_rays
              << " inherited_samples=" << inherited_evidence.surface_samples
              << " inherited_supported=" << inherited_evidence.supported_votes
              << " inherited_free=" << inherited_evidence.free_space_votes
              << " inherited_repl_other="
              << inherited_evidence.replaced_by_other_votes
              << " inherited_repl_bg="
              << inherited_evidence.replaced_by_background_votes
              << " inherited_occ=" << inherited_evidence.occluded_votes
              << " inherited_unobs=" << inherited_evidence.unobserved_samples
              << " session_sup=" << session_evidence.support_rays
              << " session_con=" << session_evidence.contradiction_rays
              << " session_samples=" << session_evidence.surface_samples
              << " cur_verts="
              << (current && current->geometry
                      ? current->geometry->numVertices()
                      : 0)
              << " candidate_verts="
              << (session_current && session_current->geometry
                      ? session_current->geometry->numVertices()
                      : 0);
    if (persistent_objects_.resolveCurrentEvidence(
            id, inherited_evidence, session_evidence, stamp)) {
      ++closed;
    }
  }
  return closed;
}

void SessionBackend::afterTerminalRound() {
  // The terminal change-detection pass has just written the final snapshot and
  // was the last reader of the stored evidence frames.
  if (physical_evidence_store_) physical_evidence_store_->clear();
  releaseFreedMemory();
}

void SessionBackend::runRound(DynamicSceneGraph::Ptr dsg, const TimeStamp stamp, const bool finalize_pending) {
  // One round at a time on the 4D map; the round's own state is the request's clone.
  std::lock_guard<std::mutex> lock(map_mutex_);
  mergeIdentities(dsg);
  decideStates(dsg, stamp, finalize_pending);
  composeRound(dsg, stamp, finalize_pending);
  storeSnapshot(dsg, stamp);
  showStatesFromStart(*dsg, stamp);
  CLOG(3) << "Session round completed: " << map_.numTimeSteps() << " time steps in 4D map.";
}

void SessionBackend::mergeIdentities(const DynamicSceneGraph::Ptr& dsg) {
  auto& aliases = PhysicalIdAliases::instance();
  if (!aliases.active()) return;
  for (const auto& [gone, keep] : aliases.takeNew()) {
    // the merge as registered (its ids may have merged further since: the layer folds them in order)
    persistent_objects_.mergeIdentity(gone, keep);
    if (present_surface_) present_surface_->mergeIdentity(gone, keep);
  }
  if (!dsg->hasLayer(DsgLayers::OBJECTS)) return;
  size_t relabelled = 0;
  for (const auto& [node_id, node] : dsg->getLayer(DsgLayers::OBJECTS).nodes()) {
    (void)node_id;
    auto* attrs = node->tryAttributes<KhronosObjectAttributes>();
    if (!attrs) continue;
    auto it = attrs->details.find("instance_id");
    if (it == attrs->details.end() || it->second.size() != 1 || it->second.front() == 0) continue;
    const size_t root = aliases.root(static_cast<size_t>(it->second.front()));
    if (root != static_cast<size_t>(it->second.front())) {
      it->second.front() = root;
      ++relabelled;
    }
  }
  if (relabelled) LOG(INFO) << "IDENTITY_RELABEL nodes=" << relabelled;
}

void SessionBackend::decideStates(const DynamicSceneGraph::Ptr& dsg, TimeStamp stamp, bool finalize_pending) {
  // One round is one observation event: every state decision until the next round reads the
  // persistence history resolved before it.
  persistent_objects_.beginObservationEvent(stamp);
  // Every CURRENT state faces the measurements the round's frames made of its surface.
  const size_t closed = verifyCurrentObjectStates(stamp);
  if (closed > 0) {
    CLOG(3) << "[Backend] Closed " << closed
            << " current object fragment(s) contradicted by later free-space evidence.";
  }
  if (finalize_pending) {
    persistent_objects_.finalizePendingAbsences(stamp);
  }
  estimatePhysicalPresence(*dsg, stamp);
  // A closed state's old-site background goes with it in the round's composition (composeRound).
}

void SessionBackend::estimatePhysicalPresence(DynamicSceneGraph& dsg, const TimeStamp stamp) {
  // Each round ends one look of the stored frames (the absence test's unit of evidence).
  if (round_ends_.empty() || stamp > round_ends_.back()) round_ends_.push_back(stamp);
  if (!dsg.hasLayer(DsgLayers::OBJECTS)) return;
  RayVerificator::PhysicalEvidenceSnapshot evidence;
  if (physical_evidence_store_) evidence = physical_evidence_store_->snapshot();
  size_t opened = 0, dated = 0;
  for (const auto& [node_id, node] : dsg.getLayer(DsgLayers::OBJECTS).nodes()) {
    (void)node_id;
    auto* attrs = node->tryAttributes<KhronosObjectAttributes>();
    if (!attrs) continue;
    const auto physical_id = session::physicalInstanceId(*attrs);
    if (!physical_id || attrs->first_observed_ns.empty() || attrs->last_observed_ns.empty()) continue;
    // The node's presence starts no later than the first stored frame that measured its identity:
    // the session's own first observation of it, which a node extracted later does not carry.
    TimeStamp first_seen = session::openPresence(*attrs);
    if (evidence) {
      const TimeStamp measured = evidence->firstMeasured(*physical_id);
      if (measured != 0 && measured < first_seen) first_seen = measured;
    }
    ++opened;
    const auto start =
        session::presenceStart(surface_evidence_, evidence, round_ends_, *physical_id, *attrs, first_seen, stamp);
    if (!start) continue;
    attrs->first_observed_ns = {*start};
    dated += *start > 0;
  }
  CLOG(3) << "[SessionBackend] presence of " << opened << " physical object node(s), " << dated
          << " dated after the session's start";
}
void SessionBackend::composeRound(const DynamicSceneGraph::Ptr& dsg, TimeStamp stamp, bool finalize_pending) {
  // The state decision sees every visibility segment independently: an old physical object can be
  // cleared at its previous site while a newer segment with the same stable ID materializes its
  // current geometry elsewhere, and folding the segments first would let the old site's evidence
  // close the new current segment. Reduce to one logical node per physical ID only after every
  // segment has been decided.
  session::canonicalizePhysicalObjects(*dsg, &persistent_objects_);
  if (finalize_pending) {
    // Canonicalization above ingests the last extractor segments. They did
    // not exist in the registry during the preceding state decision. Drain
    // them before the sole terminal snapshot, otherwise a late old segment
    // can remain CURRENT while a newer observed site waits for a next round
    // that will never happen (Synthetic A I69).
    const size_t terminal_closed = verifyCurrentObjectStates(stamp);
    persistent_objects_.finalizePendingAbsences(stamp);
    session::canonicalizePhysicalObjects(*dsg, &persistent_objects_);
    LOG(INFO) << "TERMINAL_STATE_DRAIN stamp=" << stamp << " closed=" << terminal_closed;
  }
  if (!present_surface_) return;
  // The legal contribution domain (README section 5): an identity whose
  // current state began in this session contributes from that state's first
  // sighting on (the registry's track_first_seen, else its birth).
  std::map<size_t, TimeStamp> state_starts;
  const TimeStamp session_start = present_surface_->firstStamp();
  for (const size_t id : persistent_objects_.trackedIds()) {
    const auto current = persistent_objects_.currentFragment(id);
    if (!current || current->birth_time < session_start) continue;
    state_starts[id] =
        current->track_first_seen > 0 ? current->track_first_seen : current->birth_time;
    if (finalize_pending) {
      LOG(INFO) << "[PresentSurface] current state began in this session: id=" << id
                << " birth=" << current->birth_time << " first_sighting=" << state_starts[id]
                << " (session start " << session_start << ")";
    }
  }
  // A CURRENT that is not the previous round's fragment (a successor, a new object, a return) and
  // began in this session is shown from its start: noted here, written once the snapshot is stored.
  // Its start is its birth, the first observation it kept (after a motion's cut), as the 3DGS map's
  // t_L; not the tracker's first sighting, which a watched move carries over from before the move.
  for (const size_t id : persistent_objects_.trackedIds()) {
    const auto current = persistent_objects_.currentFragment(id);
    const uint64_t uid = current ? current->uid : 0;
    uint64_t& last_uid = session_current_uid_[id];
    if (uid != 0 && uid != last_uid && current->birth_time >= session_start &&
        current->birth_time < stamp) {
      session_new_states_.emplace_back(id, current->birth_time);
    }
    last_uid = uid;
  }
  PresentSurface::Inputs inputs;
  inputs.shown = shown_memory_.get();
  // The state layer already knows which states closed. The shown memory and the snapshot's
  // own background read the same ownership, so a closed state's old-site faces follow its
  // death atomically instead of waiting for the per-face evidence to see through every pixel
  // (the object-level rule; every other face is decided by its votes, the element level).
  // (Ablation no_object_rule: neither the shown memory nor the background is hidden for a
  // closed state.)
  if (!mechanismAblation().no_object_rule) {
    inputs.closed_background = closedStateSurfaces(persistent_objects_, stamp);
    inputs.closed_background_cell = object_surface_resolution_;
    for (const auto& state : inputs.closed_background) {
      inputs.closed_memory.insert(inputs.closed_memory.end(), state.points.begin(),
                                  state.points.end());
    }
  }
  if (finalize_pending) {
    if (const char* dump = std::getenv("KHRONOS_REFUSION_DUMP")) inputs.dump_dir = dump;
  }
  // Without a shown state, memory is what the snapshot copies of the loaded
  // state (P53). DIAG in the terminal round (P53 data, no decision reads it):
  // distances of the snapshot's vertices to the loaded memory, to its nearest
  // vertex and to its surface (exact point-triangle distance up to 1 m), in 20
  // bins per decade from 1e-9 m. The grid cell only sets the search cost.
  std::unique_ptr<TriangleGrid> memory_surface;
  std::vector<std::atomic<uint64_t>> vertex_hist(201), surface_hist(201);
  const auto bin_of = [](const float d) {
    if (!(d > 0.f)) return size_t{0};
    return static_cast<size_t>(std::clamp(std::floor((std::log10(d) + 9.f) * 20.f), 0.f, 200.f));
  };
  if (loaded_memory_search_) {
    if (finalize_pending && !loaded_memory_faces_.empty()) {
      memory_surface = std::make_unique<TriangleGrid>(loaded_memory_, loaded_memory_faces_, nullptr,
                                                      std::max(object_surface_resolution_, 0.02f));
    }
    inputs.is_memory = [&, this](const Eigen::Vector3f& p) {
      float d_sq = 0.f;
      size_t idx = 0;
      const bool found = loaded_memory_search_->search(p, d_sq, idx);
      if (finalize_pending) {
        if (found) vertex_hist[bin_of(std::sqrt(d_sq))].fetch_add(1, std::memory_order_relaxed);
        if (memory_surface) {
          float d = 0.f;
          Eigen::Vector3f q;
          uint32_t face = 0;
          surface_hist[memory_surface->closest(p, 1.f, d, q, face) ? bin_of(d) : 200].fetch_add(
              1, std::memory_order_relaxed);
        }
      }
      return found && d_sq <= memory_copy_tolerance_ * memory_copy_tolerance_;
    };
  }
  auto round = present_surface_->compose(*dsg, stamp, state_starts, inputs, finalize_pending);
  // (Ablation no_object_rule: the composition rebuilt every current object's mesh from the present;
  // the surfaces of its closed states stay beside it.)
  if (mechanismAblation().no_object_rule && dsg->hasLayer(DsgLayers::OBJECTS)) {
    for (const auto& [node_id, node] : dsg->getLayer(DsgLayers::OBJECTS).nodes()) {
      auto* attrs = node->tryAttributes<KhronosObjectAttributes>();
      if (!attrs) continue;
      if (const auto physical = session::physicalInstanceId(*attrs)) {
        persistent_objects_.appendClosedGeometry(*physical, *attrs);
      }
    }
  }
  if (!finalize_pending) return;
  if (inputs.is_memory) {
    std::string v, s;
    for (size_t i = 0; i < vertex_hist.size(); ++i) {
      v += (i ? "," : "") + std::to_string(vertex_hist[i].load());
      s += (i ? "," : "") + std::to_string(surface_hist[i].load());
    }
    LOG(INFO) << "MEMORY_DISTANCE_DIAG bins_per_decade=20 from=1e-9 vertex=" << v << " surface=" << s;
  }
  refusion_report_ = std::move(round.report_json);
}
void SessionBackend::showStatesFromStart(const DynamicSceneGraph& dsg, TimeStamp stamp) {
  if (session_new_states_.empty()) return;
  std::vector<std::pair<size_t, TimeStamp>> pending;
  pending.swap(session_new_states_);
  if (!dsg.hasLayer(DsgLayers::OBJECTS)) return;
  // This round's node of each identity (one per physical id after canonicalization).
  std::map<size_t, std::pair<spark_dsg::NodeId, const KhronosObjectAttributes*>> shown_now;
  for (const auto& [node_id, node] : dsg.getLayer(DsgLayers::OBJECTS).nodes()) {
    const auto* attrs = node->tryAttributes<KhronosObjectAttributes>();
    if (!attrs) continue;
    if (const auto id = session_detail::getPhysicalInstanceId(*attrs)) shown_now[*id] = {node_id, attrs};
  }
  for (const auto& [id, start] : pending) {
    const auto it = shown_now.find(id);
    if (it == shown_now.end()) continue;
    const spark_dsg::NodeId node_now = it->second.first;
    const KhronosObjectAttributes& shown = *it->second.second;
    // In every snapshot stored since the state's start, the identity's node shows this state's
    // surface (its predecessor ends there); a snapshot without the identity gains its node, with
    // the trajectory known at that snapshot only.
    size_t amended = 0;
    map_.rewriteSnapshots([&](size_t, TimeStamp t) { return t >= start && t < stamp; },
                          [&](DynamicSceneGraph& past, size_t, TimeStamp t) {
      if (!past.hasLayer(DsgLayers::OBJECTS)) return false;
      std::vector<spark_dsg::NodeId> nodes;
      for (const auto& [node_id, node] : past.getLayer(DsgLayers::OBJECTS).nodes()) {
        const auto* attrs = node->tryAttributes<KhronosObjectAttributes>();
        if (attrs && session_detail::getPhysicalInstanceId(*attrs) == id) nodes.push_back(node_id);
      }
      if (nodes.empty()) {
        auto copy = shown.clone();
        auto& c = static_cast<KhronosObjectAttributes&>(*copy);
        // Every sample not later than the snapshot: the node's own, then the moves it carries (the
        // backend counts those per snapshot, as many as are not later than the snapshot).
        size_t known = 0;
        for (size_t s = 0; s < c.trajectory_timestamps.size(); ++s) {
          if (c.trajectory_timestamps[s] > t) continue;
          c.trajectory_timestamps[known] = c.trajectory_timestamps[s];
          if (s < c.trajectory_positions.size()) c.trajectory_positions[known] = c.trajectory_positions[s];
          ++known;
        }
        c.trajectory_timestamps.resize(known);
        c.trajectory_positions.resize(std::min(c.trajectory_positions.size(), known));
        const bool added = past.emplaceNode(DsgLayers::OBJECTS, node_now, std::move(copy));
        amended += added;
        return added;
      }
      for (size_t k = 0; k < nodes.size(); ++k) {
        auto copy = past.getNode(nodes[k]).attributes<KhronosObjectAttributes>().clone();
        auto& c = static_cast<KhronosObjectAttributes&>(*copy);
        if (k == 0) {
          c.mesh = shown.mesh;
          c.bounding_box = shown.bounding_box;
          c.position = shown.position;
          c.semantic_label = shown.semantic_label;
        } else {
          c.mesh = spark_dsg::Mesh(c.mesh.has_colors, c.mesh.has_timestamps, c.mesh.has_labels,
                                   c.mesh.has_first_seen_stamps);
        }
        past.setNodeAttributes(nodes[k], std::move(copy));
      }
      ++amended;
      return true;
    });
    LOG(INFO) << "STATE_START_HISTORY inst=" << id << " start=" << start << " round=" << stamp
              << " snapshots=" << amended;
  }
}

void SessionBackend::saveSessionState(const hydra::DataDirectory& log_setup) {
  const auto path = log_setup.path();
  saveAbsenceSensorStatistics((path / "sensor_statistics.txt").string());
    if (map_.numTimeSteps() > 0) {
      // One boundary surface: the consolidated final map is what the next
      // session reasons on and what it shows.
      const size_t final_index = map_.numTimeSteps() - 1;
      SpatioTemporalMap chain(config.spatio_temporal_map);
      chain.update(map_.rawDsg(final_index)->clone(), map_.stamps()[final_index]);
      if (!chain.save(path / "chain_state.4dmap.zpk")) {
        LOG(ERROR) << "Failed to save the chain state to '" << path << "'.";
      }
      {
        // What this session's final map shows: the next session's memory.
        const size_t last = map_.numTimeSteps() - 1;
        SpatioTemporalMap shown(config.spatio_temporal_map);
        shown.update(map_.rawDsg(last)->clone(), map_.stamps()[last]);
        if (!shown.save(path / "shown_state.4dmap.zpk")) {
          LOG(ERROR) << "Failed to save the shown state to '" << path << "'.";
        }
      }
      if (!refusion_report_.empty()) {
        std::ofstream(path / "refusion_report.json") << refusion_report_ << "\n";
      }
    }
}

}  // namespace session_update::runtime
