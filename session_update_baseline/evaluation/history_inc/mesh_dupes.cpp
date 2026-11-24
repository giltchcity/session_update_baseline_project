// Diagnostic: vertex multiplicity of a 4D map's stored meshes. For the chosen snapshots prints the vertices, faces,
// loose vertices, distinct vertex records, vertices sharing their record with another vertex, and vertices used by
// faces of more than one block (a face's block: its first vertex's, as PieceHistory::split assigns it).
// usage: mesh_dupes MAP CONFIG.txt INDEX[,INDEX...]
#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <khronos/spatio_temporal_map/spatio_temporal_map.h>

#include "block_edge.h"

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  const double edge = readBlockEdge(argv[2]);
  const auto map = khronos::SpatioTemporalMap::load(argv[1]);
  if (!map) return 3;
  std::stringstream list(argv[3]);
  for (std::string item; std::getline(list, item, ',');) {
    long long i = std::stoll(item);
    if (i < 0) i += static_cast<long long>(map->numTimeSteps());
    const auto g = map->rawDsg(static_cast<size_t>(i));
    const auto& m = *g->mesh();
    const size_t n = m.numVertices();
    auto record = [&](size_t v) {
      std::string r(reinterpret_cast<const char*>(m.points[v].data()), 12);
      if (m.has_colors) r += std::string{char(m.colors[v].r), char(m.colors[v].g), char(m.colors[v].b), char(m.colors[v].a)};
      if (m.has_timestamps) r.append(reinterpret_cast<const char*>(&m.stamps[v]), 8);
      if (m.has_first_seen_stamps) r.append(reinterpret_cast<const char*>(&m.first_seen_stamps[v]), 8);
      if (m.has_labels) r.append(reinterpret_cast<const char*>(&m.labels[v]), 4);
      return r;
    };
    auto block = [&](size_t v) {
      const auto& p = m.points[v];
      return std::to_string(long(std::floor(p.x() / edge))) + "," + std::to_string(long(std::floor(p.y() / edge))) +
             "," + std::to_string(long(std::floor(p.z() / edge)));
    };
    std::vector<uint8_t> used(n, 0);
    std::vector<std::set<std::string>> blocks_of(n);
    for (const auto& f : m.faces) {
      const auto b = block(f[0]);
      for (int k = 0; k < 3; ++k) used[f[k]] = 1, blocks_of[f[k]].insert(b);
    }
    std::unordered_map<std::string, std::vector<size_t>> copies;
    for (size_t v = 0; v < n; ++v) copies[record(v)].push_back(v);
    size_t loose = 0, shared_blocks = 0, dup_vertices = 0, dup_used_both = 0, dup_same_block = 0, dup_loose = 0;
    for (size_t v = 0; v < n; ++v) loose += !used[v], shared_blocks += blocks_of[v].size() > 1;
    for (const auto& [r, vs] : copies) {
      if (vs.size() < 2) continue;
      dup_vertices += vs.size();
      size_t used_count = 0;
      std::set<std::string> bl;
      for (const auto v : vs) used_count += used[v], dup_loose += !used[v], bl.insert(blocks_of[v].begin(), blocks_of[v].end());
      dup_used_both += used_count >= 2;
      size_t in_blocks = 0;
      for (const auto v : vs) in_blocks += blocks_of[v].size();
      dup_same_block += used_count >= 2 && bl.size() < in_blocks;  // two copies used by faces of one block
    }
    // Face order: runs of the same first-vertex block, distinct blocks, descents of the faces' newest first-seen time,
    // descents of the faces' smallest vertex index.
    size_t runs = 0, descents_time = 0, descents_index = 0;
    std::set<std::string> distinct_blocks;
    std::string previous_block;
    uint64_t previous_time = 0;
    size_t previous_index = 0;
    for (size_t k = 0; k < m.faces.size(); ++k) {
      const auto& f = m.faces[k];
      const auto b = block(f[0]);
      distinct_blocks.insert(b);
      if (k == 0 || b != previous_block) ++runs;
      const uint64_t t = std::max({m.first_seen_stamps[f[0]], m.first_seen_stamps[f[1]], m.first_seen_stamps[f[2]]});
      const size_t lowest = std::min({f[0], f[1], f[2]});
      if (k > 0 && t < previous_time) ++descents_time;
      if (k > 0 && lowest < previous_index) ++descents_index;
      previous_block = b, previous_time = t, previous_index = lowest;
    }
    std::cout << "snapshot " << i << " face order: block runs " << runs << " over " << distinct_blocks.size()
              << " blocks, newest-time descents " << descents_time << ", lowest-index descents " << descents_index << "\n";
    std::cout << "snapshot " << i << ": vertices " << n << ", faces " << m.faces.size() << ", loose " << loose
              << ", distinct records " << copies.size() << ", vertices sharing a record " << dup_vertices
              << " (loose " << dup_loose << "; records with >=2 used copies " << dup_used_both
              << ", of them with two copies in one block " << dup_same_block << "), vertices used in >1 block "
              << shared_blocks << "\n";
  }
  return 0;
}
