// Evaluation only, as chain_fill: the presence times of our 4D map of one run from the run's own records, for Khronos's
// official change metric (user 11-01: the map stores the correct 4D state). The single runs of real A/B/C and synthetic
// A/B keep almost every object present from the start and never ending (TESSE runs write their changes), while their
// snapshots and change records hold the changes:
//   closed: an object node shown (non-empty mesh) in a snapshot whose mesh the run empties in a later one (a state
//     closed with no successor) ends there, at the midpoint of the two snapshots (the run's rule for a closure,
//     persistent_object_state.cpp: the midpoint of last support and closure); shown again later, a new interval
//     begins at the midpoint of the snapshots around its return;
//   appeared: an object node first shown after the first snapshot whose place the run's change detection saw free
//     before (OBJECT_CHANGES.csv, the backend's ChangeState: first_absent before it was first shown) begins at the
//     official reconciler's estimate (reconciler.cpp estimatePresenceTimes, first seen = the snapshot it is first
//     shown in);
//   moved: the run keeps a moved object's node and counts its state closures (details "session_mobility_changes");
//     where the count rises while the node is shown and its surface now lies elsewhere (centroid displaced beyond the
//     ground truth's own state gate, eval/khronos4d_gt.py GATE: a listed move makes a new state only beyond it), the
//     earlier state stays as a record at the old place, ending at the move (the first observation at the new place,
//     details "observation_first_stamp_ns", else the midpoint of the two snapshots), and the node begins there (it
//     appeared at the new place). A rise without displacement (a state reopened in place) is no change; a removal
//     empties the mesh (closed above).
// Only default times (first 0, last never) are replaced, and each snapshot only from itself and earlier snapshots;
// objects with a trajectory (moving things) keep theirs. Meshes, presence decisions and the other layers are unchanged.
// usage: change_fill OUT.4dmap IN.4dmap OBJECT_CHANGES.csv
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>
#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <khronos/utils/khronos_attribute_utils.h>

namespace {
constexpr uint64_t kNever = std::numeric_limits<uint64_t>::max();
constexpr double kStateGate = 0.10;  // eval/khronos4d_gt.py GATE: the ground truth's new state beyond this displacement

struct Evidence {
  uint64_t first_absent = 0, first_persistent = 0;
};

struct Seen {  // a node in one snapshot
  bool shown = false;
  double count = -1;  // session_mobility_changes, -1 without it
  uint64_t observation_first = 0;
  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
};

struct Move {
  size_t before, at_snapshot;  // the old state's last snapshot shown, the snapshot shown at the new place
  uint64_t at;                 // the move's time
};

double detail(const khronos::KhronosObjectAttributes& attrs, const std::string& name, double fallback) {
  const auto it = attrs.details.find(name);
  return it != attrs.details.end() && !it->second.empty() ? it->second.front() : fallback;
}

// Presence intervals of a node from the snapshots [from, to] it is shown in: the first interval begins at `start`, a
// later one (shown again after an empty snapshot) at the midpoint around its return; an interval ends at the midpoint
// around the snapshot that empties it.
void intervals(const std::vector<std::optional<Seen>>& seen, const std::vector<uint64_t>& stamps, size_t from, size_t to,
               uint64_t start, std::vector<uint64_t>& first, std::vector<uint64_t>& last) {
  first.clear();
  last.clear();
  for (size_t k = from; k <= to; ++k) {
    const bool now = seen[k] && seen[k]->shown, before = k > from && seen[k - 1] && seen[k - 1]->shown;
    if (now && !before) {
      first.push_back(first.empty() ? start : (stamps[k - 1] + stamps[k]) / 2);
      last.push_back(kNever);
    } else if (!now && before && !last.empty()) {
      last.back() = (stamps[k - 1] + stamps[k]) / 2;
    }
  }
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: change_fill OUT.4dmap IN.4dmap OBJECT_CHANGES.csv\n";
    return 2;
  }
  std::map<spark_dsg::NodeId, Evidence> evidence;
  std::ifstream csv(argv[3]);
  std::string line;
  std::getline(csv, line);  // node_id,merged_id,first_absent,last_absent,first_persistent,last_persistent
  while (std::getline(csv, line)) {
    std::stringstream in(line);
    std::string f[6];
    for (auto& x : f) std::getline(in, x, ',');
    evidence[std::stoull(f[0])] = {std::stoull(f[2]), std::stoull(f[4])};
  }
  const auto map = khronos::SpatioTemporalMap::load(argv[2]);
  if (!map || !map->numTimeSteps()) {
    std::cerr << "cannot read " << argv[2] << "\n";
    return 3;
  }
  const auto& stamps = map->stamps();
  const size_t n = map->numTimeSteps();
  // Pass 1: each object node in each snapshot (shown, closure count, surface centroid); the moves.
  std::map<spark_dsg::NodeId, std::vector<std::optional<Seen>>> seen;
  for (size_t ix = 0; ix < n; ++ix) {
    const auto dsg = map->rawDsg(ix);
    if (!dsg->hasLayer(khronos::DsgLayers::OBJECTS)) continue;
    for (const auto& [id, node] : dsg->getLayer(khronos::DsgLayers::OBJECTS).nodes()) {
      const auto* attrs = node->tryAttributes<khronos::KhronosObjectAttributes>();
      if (!attrs || !attrs->trajectory_timestamps.empty()) continue;
      Seen s;
      s.shown = !attrs->mesh.points.empty();
      s.count = detail(*attrs, "session_mobility_changes", -1);
      s.observation_first = static_cast<uint64_t>(detail(*attrs, "observation_first_stamp_ns", 0));
      for (const auto& p : attrs->mesh.points) s.centroid += attrs->bounding_box.pointToWorldFrame(p).cast<double>();
      if (s.shown) s.centroid /= static_cast<double>(attrs->mesh.points.size());
      auto& v = seen[id];
      v.resize(n);
      v[ix] = s;
    }
  }
  std::map<spark_dsg::NodeId, std::vector<Move>> moves;
  for (const auto& [id, v] : seen) {
    std::optional<size_t> shown_before;
    double count = -1;
    for (size_t k = 0; k < n; ++k) {
      if (!v[k]) continue;
      if (v[k]->shown && shown_before && count >= 0 && v[k]->count > count &&
          (v[k]->centroid - v[*shown_before]->centroid).norm() > kStateGate) {
        const uint64_t mid = (stamps[k - 1] + stamps[k]) / 2, obs = v[k]->observation_first;
        moves[id].push_back({*shown_before, k, obs > stamps[*shown_before] && obs <= stamps[k] ? obs : mid});
      }
      count = v[k]->count;
      if (v[k]->shown) shown_before = k;
    }
  }
  // Pass 2: each snapshot with the states its own and earlier snapshots show.
  khronos::SpatioTemporalMap::Config config;
  config.finalize_incrementally = false;  // store the snapshots exactly as read
  khronos::SpatioTemporalMap out(config);
  std::map<std::pair<spark_dsg::NodeId, size_t>, khronos::KhronosObjectAttributes> old_states;  // (node, move) -> attrs
  std::map<spark_dsg::NodeId, bool> appeared, closed;
  for (size_t ix = 0; ix < n; ++ix) {
    auto dsg = map->rawDsg(ix)->clone();
    std::vector<khronos::KhronosObjectAttributes> records;  // the old places, as nodes
    if (dsg->hasLayer(khronos::DsgLayers::OBJECTS)) {
      for (const auto& [id, node] : dsg->getLayer(khronos::DsgLayers::OBJECTS).nodes()) {
        auto* attrs = node->tryAttributes<khronos::KhronosObjectAttributes>();
        if (!attrs || !attrs->trajectory_timestamps.empty() || !seen.count(id)) continue;
        if (attrs->first_observed_ns != std::vector<uint64_t>{0} || attrs->last_observed_ns != std::vector<uint64_t>{kNever}) continue;
        const auto& v = seen.at(id);
        const auto& m = moves.count(id) ? moves.at(id) : std::vector<Move>{};
        size_t from = 0;
        uint64_t start = 0;
        for (size_t j = 0; j < m.size(); ++j) {
          if (m[j].before == ix) old_states[{id, j}] = *attrs;  // the old place's last surface
          if (m[j].at_snapshot > ix) break;
          // The state before move j, from its own start to the move.
          auto record = old_states.at({id, j});
          intervals(v, stamps, from, m[j].before, start, record.first_observed_ns, record.last_observed_ns);
          if (record.last_observed_ns.empty()) continue;
          record.last_observed_ns.back() = std::min(record.last_observed_ns.back(), m[j].at);
          record.details.erase("instance_id");  // its own object: chain_fill must not fill the moved node from it
          records.push_back(std::move(record));
          from = m[j].at_snapshot;
          start = m[j].at;
        }
        std::vector<uint64_t> first, last;
        intervals(v, stamps, from, ix, start, first, last);
        if (first.empty()) continue;  // not shown up to here
        if (from == 0 && first.front() == 0) {  // the official reconciler on the run's evidence of a free place before
          size_t first_shown = 0;
          while (!(v[first_shown] && v[first_shown]->shown)) ++first_shown;
          const auto it = evidence.find(id);
          if (first_shown > 0 && it != evidence.end() && it->second.first_absent > 0 &&
              it->second.first_absent < stamps[first_shown]) {
            const bool persistent = it->second.first_persistent > 0 && it->second.first_persistent < stamps[first_shown] &&
                                    it->second.first_absent < it->second.first_persistent;
            first.front() = (it->second.first_absent + (persistent ? it->second.first_persistent : stamps[first_shown])) / 2;
          }
        }
        appeared[id] = appeared[id] || first.front() > 0;
        closed[id] = closed[id] || last.front() != kNever;
        attrs->first_observed_ns = first;
        attrs->last_observed_ns = last;
      }
      if (const auto objects = dsg->getLayerKey(khronos::DsgLayers::OBJECTS)) {
        for (size_t k = 0; k < records.size(); ++k) {
          dsg->emplaceNode(objects->layer, spark_dsg::NodeSymbol('h', k), records[k].clone());
        }
      }
    }
    out.update(dsg, stamps[ix]);
  }
  if (!out.save(argv[1])) {
    std::cerr << "cannot write " << argv[1] << "\n";
    return 4;
  }
  size_t n_moves = 0, n_appeared = 0, n_closed = 0;
  for (const auto& [id, m] : moves) n_moves += m.size();
  for (const auto& [id, a] : appeared) n_appeared += a;
  for (const auto& [id, c] : closed) n_closed += c;
  std::cout << "CHANGE_FILL snapshots=" << n << " nodes=" << seen.size() << " moves=" << n_moves << " appeared=" << n_appeared
            << " closed=" << n_closed << "\n";
  return 0;
}
