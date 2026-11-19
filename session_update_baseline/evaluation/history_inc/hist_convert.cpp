// Converts the snapshots of a 4D map into a piece history (khronos::PieceHistory) and checks it: every snapshot
// restored from the written history has the same canonical pieces as the original snapshot. Prints the storage, the
// stored versions, and the time to restore each snapshot. The block edge is the run's own (config.txt next to the map).
// usage: hist_convert MAP.4dmap[.zpk] OUT.pieces [CONFIG.txt]
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include <khronos/spatio_temporal_map/piece_history.h>
#include <khronos/spatio_temporal_map/spatio_temporal_map.h>

#include "block_edge.h"

namespace {

double seconds(std::chrono::steady_clock::time_point a) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "usage: hist_convert MAP.4dmap[.zpk] OUT.pieces [CONFIG.txt]\n";
    return 2;
  }
  const std::filesystem::path map_path(argv[1]), out_path(argv[2]);
  const auto config = argc > 3 ? std::filesystem::path(argv[3]) : map_path.parent_path() / "config.txt";
  const double edge = readBlockEdge(config);
  const auto map = khronos::SpatioTemporalMap::load(map_path.string());
  if (!map || !map->numTimeSteps()) {
    std::cerr << "cannot read " << map_path << "\n";
    return 3;
  }
  const size_t n = map->numTimeSteps();

  auto t = std::chrono::steady_clock::now();
  khronos::PieceHistory history(edge);
  for (size_t i = 0; i < n; ++i) history.append(*map->rawDsg(i));
  const double append_s = seconds(t);
  {
    std::ofstream out(out_path, std::ios::binary);
    history.write(out);
  }
  std::ifstream in(out_path, std::ios::binary);
  const auto back = khronos::PieceHistory::read(in);

  size_t mismatched = 0;
  double restore_total = 0.0, restore_max = 0.0;
  for (size_t i = 0; i < n; ++i) {
    t = std::chrono::steady_clock::now();
    const auto restored = back->snapshot(i);
    const double s = seconds(t);
    restore_total += s;
    restore_max = std::max(restore_max, s);
    const auto a = khronos::PieceHistory::split(*restored, edge);
    const auto b = khronos::PieceHistory::split(*map->rawDsg(i), edge);
    if (a != b) {
      ++mismatched;
      for (const auto& [key, bytes] : b) {
        const auto it = a.find(key);
        if (it == a.end() || it->second != bytes) {
          std::cout << "MISMATCH snapshot " << i << " piece " << key << (it == a.end() ? " missing" : " differs") << "\n";
          break;
        }
      }
      if (a.size() != b.size()) std::cout << "MISMATCH snapshot " << i << " pieces " << a.size() << " vs " << b.size() << "\n";
    }
  }
  size_t full = 0, delta = 0, ended = 0;
  back->versionCounts(full, delta, ended);
  constexpr double kMB = 1024.0 * 1024.0;
  std::cout << std::fixed << std::setprecision(1) << "block edge " << edge << " m | snapshots " << n << " | map file "
            << std::filesystem::file_size(map_path) / kMB << " MB | piece history " << std::filesystem::file_size(out_path) / kMB
            << " MB | versions full " << full << ", delta " << delta << ", ended " << ended << " | appended in "
            << append_s << " s | restore " << std::setprecision(3) << restore_total / n << " s mean, " << restore_max
            << " s max | snapshots with different pieces " << mismatched << "\n";
  return mismatched == 0 ? 0 : 1;
}
