// Checks a 4D map kept as a piece history against the 4D map it is made from, through the map's own interface:
// converts MAP (usePieceHistory, block edge from the run's config.txt), saves and reloads it, then for every snapshot
// compares, between the two maps,
//   the stored snapshot: its serialization without the mesh (byte for byte), its mesh's vertex and face counts, and its
//     canonical pieces (PieceHistory::split: every vertex named by its record and rank, every face by its vertices),
//   the query at its stamp and the query halfway to the previous stamp: their counts and canonical pieces,
//   the snapshot cut to a plain 4D map (saveSnapshots) and read back: its canonical pieces.
// Prints the file sizes and the query times of both.
// usage: piece_map_check MAP.4dmap[.zpk] OUT.4dmap [CONFIG.txt]
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <khronos/spatio_temporal_map/piece_history.h>
#include <spark_dsg/serialization/graph_binary_serialization.h>
#include <khronos/spatio_temporal_map/spatio_temporal_map.h>

#include "block_edge.h"

namespace {

double seconds(std::chrono::steady_clock::time_point a) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
}

// The pieces in only one of a and b or different in them: count and the first few keys.
std::string difference(const khronos::PieceHistory::Pieces& a, const khronos::PieceHistory::Pieces& b) {
  size_t count = 0;
  std::string keys;
  auto note = [&](const std::string& key, const char* what) {
    if (++count <= 4) keys += " " + key + what;
  };
  for (const auto& [key, bytes] : a) {
    const auto it = b.find(key);
    if (it == b.end()) note(key, "(legacy only)");
    else if (it->second != bytes) note(key, "(differs)");
  }
  for (const auto& [key, bytes] : b) {
    if (!a.count(key)) note(key, "(pieces only)");
  }
  return std::to_string(count) + " pieces:" + keys;
}

std::string counts(const spark_dsg::DynamicSceneGraph& g) {
  return g.hasMesh() ? std::to_string(g.mesh()->numVertices()) + "/" + std::to_string(g.mesh()->numFaces()) : "-";
}

std::vector<uint8_t> withoutMesh(const spark_dsg::DynamicSceneGraph& g) {
  std::vector<uint8_t> bytes;
  spark_dsg::io::binary::writeGraph(g, bytes, false);
  return bytes;
}

using TimeStampList = std::vector<uint64_t>;

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "usage: piece_map_check MAP.4dmap[.zpk] OUT.4dmap [CONFIG.txt]\n";
    return 2;
  }
  const std::filesystem::path map_path(argv[1]), out_path(argv[2]);
  const auto config = argc > 3 ? std::filesystem::path(argv[3]) : map_path.parent_path() / "config.txt";
  const double edge = readBlockEdge(config);
  auto legacy = khronos::SpatioTemporalMap::load(map_path.string());
  auto convert = khronos::SpatioTemporalMap::load(map_path.string());
  if (!legacy || !convert || !legacy->numTimeSteps()) {
    std::cerr << "cannot read " << map_path << "\n";
    return 3;
  }
  auto t = std::chrono::steady_clock::now();
  convert->usePieceHistory(edge);
  const double convert_s = seconds(t);
  if (!convert->save(out_path.string())) return 4;
  convert.reset();
  const auto pieces = khronos::SpatioTemporalMap::load(out_path.string());
  if (!pieces || !pieces->usesPieceHistory() || pieces->stamps() != legacy->stamps() ||
      pieces->earliest() != legacy->earliest() || pieces->latest() != legacy->latest()) {
    std::cerr << "the reloaded piece map differs in its stamps or time range\n";
    return 5;
  }

  const auto& stamps = legacy->stamps();
  const size_t n = stamps.size();
  size_t stored_diff = 0, query_diff = 0, cut_diff = 0;
  double legacy_q = 0.0, pieces_q = 0.0, legacy_qmax = 0.0, pieces_qmax = 0.0;
  size_t queries = 0;
  // The queries, in time order: halfway to the previous stamp, which reads the snapshot from storage (timed), then at
  // the snapshot's own stamp, which replays forward from it.
  for (size_t i = 0; i < n; ++i) {
    const TimeStampList times{i ? stamps[i - 1] + (stamps[i] - stamps[i - 1]) / 2 : stamps[i], stamps[i]};
    for (size_t k = 0; k < times.size(); ++k) {
      const auto when = times[k];
      t = std::chrono::steady_clock::now();
      const auto& qa = legacy->getDsg(when);
      const double la = seconds(t);
      const auto a = khronos::PieceHistory::split(qa, edge);
      const auto ca = counts(qa);
      t = std::chrono::steady_clock::now();
      const auto& qb = pieces->getDsg(when);
      const double pb = seconds(t);
      const auto b = khronos::PieceHistory::split(qb, edge);
      const auto cb = counts(qb);
      ++queries;
      if (k == 0) {
        legacy_q += la, pieces_q += pb;
        legacy_qmax = std::max(legacy_qmax, la), pieces_qmax = std::max(pieces_qmax, pb);
      }
      if (a != b || ca != cb) {
        ++query_diff;
        std::cout << "query at " << when << " (snapshot " << i << ") differs: vertices/faces " << ca << " vs " << cb
                  << ", " << difference(a, b) << "\n";
      }
    }
  }
  // The same queries at the stamps in a scattered order (snapshot 37 k mod n, k = 0..n-1 when n is not a multiple of
  // 37): every query reads its snapshot without the one before it.
  double legacy_r = 0.0, pieces_r = 0.0, legacy_rmax = 0.0, pieces_rmax = 0.0;
  for (size_t k = 0; k < n; ++k) {
    const size_t i = (37 * k) % n;
    t = std::chrono::steady_clock::now();
    const auto& qa = legacy->getDsg(stamps[i]);
    const double la = seconds(t);
    const auto a = khronos::PieceHistory::split(qa, edge);
    const auto ca = counts(qa);
    t = std::chrono::steady_clock::now();
    const auto& qb = pieces->getDsg(stamps[i]);
    const double pb = seconds(t);
    const auto b = khronos::PieceHistory::split(qb, edge);
    legacy_r += la, pieces_r += pb;
    legacy_rmax = std::max(legacy_rmax, la), pieces_rmax = std::max(pieces_rmax, pb);
    ++queries;
    if (a != b || ca != counts(qb)) {
      ++query_diff;
      std::cout << "scattered query at " << stamps[i] << " (snapshot " << i << ") differs\n";
    }
  }
  const auto cut_path = out_path.string() + ".cut.4dmap";
  for (size_t i = 0; i < n; ++i) {
    const auto original = legacy->rawDsg(i);
    const auto restored = pieces->rawDsg(i);
    const auto reference = khronos::PieceHistory::split(*original, edge);
    const auto split = khronos::PieceHistory::split(*restored, edge);
    if (split != reference || counts(*original) != counts(*restored) || withoutMesh(*original) != withoutMesh(*restored)) {
      ++stored_diff;
      std::cout << "stored snapshot " << i << " differs: vertices/faces " << counts(*original) << " vs "
                << counts(*restored) << ", graph without mesh "
                << (withoutMesh(*original) == withoutMesh(*restored) ? "same" : "differs") << ", "
                << difference(reference, split) << "\n";
    }
    if (i + 1 == n) std::cout << "last snapshot: vertices/faces " << counts(*original) << ", pieces " << reference.size() << "\n";
    if (i % 7 == 0 || i + 1 == n) {
      if (!pieces->saveSnapshots({i}, cut_path)) return 6;
      const auto cut = khronos::SpatioTemporalMap::load(cut_path);
      if (!cut || cut->numTimeSteps() != 1 || cut->stamps()[0] != stamps[i] ||
          khronos::PieceHistory::split(*cut->rawDsg(0), edge) != reference) {
        ++cut_diff;
        std::cout << "cut of snapshot " << i << " differs\n";
      }
    }
  }
  std::filesystem::remove(cut_path);
  constexpr double kMB = 1024.0 * 1024.0;
  std::cout << std::fixed << std::setprecision(1) << "block edge " << edge << " m | snapshots " << n << " | legacy "
            << std::filesystem::file_size(map_path) / kMB << " MB | piece map " << std::filesystem::file_size(out_path) / kMB
            << " MB | converted in " << convert_s << " s | query reading a snapshot, in time order: legacy "
            << std::setprecision(3) << legacy_q / n << " s mean " << legacy_qmax << " max, pieces " << pieces_q / n
            << " s mean " << pieces_qmax << " max; scattered: legacy " << legacy_r / n << " s mean " << legacy_rmax
            << " max, pieces " << pieces_r / n << " s mean " << pieces_rmax << " max | differing: stored " << stored_diff << ", queries " << query_diff << " of "
            << queries << ", cuts " << cut_diff << "\n";
  return stored_diff + query_diff + cut_diff == 0 ? 0 : 1;
}
