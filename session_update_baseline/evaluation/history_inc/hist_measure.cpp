// Read-only: how much of each snapshot of a 4D map changes from the snapshot before it, piece by piece, i.e. what
// an incremental history (a piece is stored again only when it changed) would store.
// Pieces: the mesh by the mapper's own voxel blocks (a face goes to the block of its first vertex, an unreferenced
// vertex to its own; block edge = voxel_size x voxels_per_side of the run's active-window volumetric map, read from
// the run's config.txt), every node by its attributes, every edge by its attributes, the graph metadata.
// A piece's bytes are canonical (the faces of a block sorted), so a block whose surface did not change is not
// stored again whatever order the mapper wrote the mesh in.
// usage: hist_measure MAP.4dmap[.zpk] [CONFIG.txt]   (default: config.txt in the map's directory)
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

#include <zstd.h>

#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/serialization/binary_conversions.h>
#include <spark_dsg/serialization/binary_serialization.h>
#include <spark_dsg/serialization/graph_binary_serialization.h>

namespace {

using Bytes = std::vector<uint8_t>;
using Pieces = std::unordered_map<std::string, Bytes>;
constexpr double kMB = 1024.0 * 1024.0;

double g_voxel = 0.0;  // the run's voxel size, read with the block edge

double readBlockEdge(const std::filesystem::path& config) {
  std::ifstream in(config);
  if (!in) throw std::runtime_error("cannot read " + config.string());
  const std::regex number(R"(:\s*([0-9.eE+-]+))");
  std::string line;
  bool inside = false;
  double voxel = -1.0;
  int per_side = -1;
  while (std::getline(in, line)) {
    if (line.find("volumetric_map") != std::string::npos) inside = true;
    if (!inside) continue;
    std::smatch m;
    if (line.find("voxel_size") != std::string::npos && std::regex_search(line, m, number)) voxel = std::stod(m[1]);
    if (line.find("voxels_per_side") != std::string::npos && std::regex_search(line, m, number)) per_side = std::stoi(m[1]);
    if (voxel > 0.0 && per_side > 0) { g_voxel = voxel; return voxel * per_side; }
  }
  throw std::runtime_error("no volumetric_map voxel_size and voxels_per_side in " + config.string());
}

std::string blockKey(char tag, const Eigen::Vector3f& p, double edge) {
  return std::string(1, tag) + std::to_string(static_cast<long>(std::floor(p.x() / edge))) + "," +
         std::to_string(static_cast<long>(std::floor(p.y() / edge))) + "," +
         std::to_string(static_cast<long>(std::floor(p.z() / edge)));
}

// Fixed-size records appended to one buffer per block, then sorted, so the block's bytes do not depend on the order.
Bytes sortedRecords(const Bytes& flat, size_t size) {
  const size_t n = flat.size() / size;
  std::vector<size_t> order(n);
  for (size_t i = 0; i < n; ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return std::memcmp(flat.data() + a * size, flat.data() + b * size, size) < 0;
  });
  Bytes out(flat.size());
  for (size_t i = 0; i < n; ++i) std::memcpy(out.data() + i * size, flat.data() + order[i] * size, size);
  return out;
}

// Diagnosis only (HIST_IGNORE=stamps,colors,positions,...): leave a vertex field out of the comparison, to see which
// field makes a block count as changed. Unset: every field, the exact history.
bool ignored(const char* field) {
  const char* list = std::getenv("HIST_IGNORE");
  return list && std::string(",") .append(list).append(",").find(std::string(",") + field + ",") != std::string::npos;
}

void meshPieces(const spark_dsg::Mesh& m, double edge, Pieces& out) {
  static const bool use_pos = !ignored("positions"), use_col = m.has_colors && !ignored("colors"),
                    use_st = m.has_timestamps && !ignored("stamps"),
                    use_fs = m.has_first_seen_stamps && !ignored("first_seen"), use_lab = m.has_labels && !ignored("labels");
  const size_t vsize = (use_pos ? 12 : 0) + (use_col ? 4 : 0) + (use_st ? 8 : 0) + (use_fs ? 8 : 0) + (use_lab ? 4 : 0);
  auto vertex = [&](size_t v, uint8_t* o) {
    if (use_pos) {
      // Diagnosis only (HIST_POS_QUANT=f): positions rounded to f voxels, to see how much of the change is sub-voxel.
      static const double q = std::getenv("HIST_POS_QUANT") ? std::atof(std::getenv("HIST_POS_QUANT")) * g_voxel : 0.0;
      if (q > 0.0) {
        const int32_t r[3] = {static_cast<int32_t>(std::lround(m.points[v].x() / q)),
                              static_cast<int32_t>(std::lround(m.points[v].y() / q)),
                              static_cast<int32_t>(std::lround(m.points[v].z() / q))};
        std::memcpy(o, r, 12);
      } else {
        std::memcpy(o, m.points[v].data(), 12);
      }
      o += 12;
    }
    if (use_col) {
      const uint8_t c[4] = {m.colors[v].r, m.colors[v].g, m.colors[v].b, m.colors[v].a};
      std::memcpy(o, c, 4);
      o += 4;
    }
    if (use_st) { std::memcpy(o, &m.stamps[v], 8); o += 8; }
    if (use_fs) { std::memcpy(o, &m.first_seen_stamps[v], 8); o += 8; }
    if (use_lab) { std::memcpy(o, &m.labels[v], 4); }
  };
  std::unordered_map<std::string, Bytes> faces, loose;
  std::vector<uint8_t> used(m.points.size(), 0);
  Bytes record(3 * vsize);
  for (const auto& f : m.faces) {
    for (int k = 0; k < 3; ++k) {
      vertex(f[k], record.data() + k * vsize);
      used[f[k]] = 1;
    }
    auto& b = faces[blockKey('f', m.points[f[0]], edge)];
    b.insert(b.end(), record.begin(), record.end());
  }
  Bytes single(vsize);
  for (size_t v = 0; v < m.points.size(); ++v) {
    if (used[v]) continue;
    vertex(v, single.data());
    auto& b = loose[blockKey('v', m.points[v], edge)];
    b.insert(b.end(), single.begin(), single.end());
  }
  for (auto& [key, flat] : faces) out[key] = sortedRecords(flat, 3 * vsize);
  for (auto& [key, flat] : loose) out[key] = sortedRecords(flat, vsize);
}

template <typename T>
Bytes binary(const T& value) {
  Bytes b;
  spark_dsg::serialization::BinarySerializer s(&b);
  spark_dsg::write_binary(s, value);
  return b;
}

void layerPieces(const std::string& name, const spark_dsg::SceneGraphLayer& layer, Pieces& out) {
  for (const auto& [id, node] : layer.nodes()) out["n" + name + "/" + std::to_string(id)] = binary(node->attributes());
  for (const auto& [key, e] : layer.edges())
    out["e" + name + "/" + std::to_string(e.source) + "-" + std::to_string(e.target)] = binary(*e.info);
}

Pieces pieces(const spark_dsg::DynamicSceneGraph& g, double edge) {
  Pieces out;
  for (const auto& [lid, layer] : g.layers()) layerPieces(std::to_string(lid), *layer, out);
  for (const auto& [lid, parts] : g.layer_partitions())
    for (const auto& [pid, layer] : parts)
      if (pid != 0) layerPieces(std::to_string(lid) + "p" + std::to_string(pid), *layer, out);
  for (const auto& [key, e] : g.interlayer_edges())
    out["x" + std::to_string(e.source) + "-" + std::to_string(e.target)] = binary(*e.info);
  const std::string meta = g.metadata.get().dump();
  out["meta"] = Bytes(meta.begin(), meta.end());
  if (g.hasMesh()) meshPieces(*g.mesh(), edge, out);
  return out;
}

size_t zstdSize(const Bytes& b) {
  Bytes dst(ZSTD_compressBound(b.size()));
  const size_t n = ZSTD_compress(dst.data(), dst.size(), b.data(), b.size(), 3);
  if (ZSTD_isError(n)) throw std::runtime_error(ZSTD_getErrorName(n));
  return n;
}

// A changed piece compressed against its own previous version (zstd prefix), or alone when it is new.
size_t zstdAgainst(ZSTD_CCtx* c, const Bytes& src, const Bytes* previous) {
  ZSTD_CCtx_reset(c, ZSTD_reset_session_only);
  ZSTD_CCtx_setParameter(c, ZSTD_c_compressionLevel, 3);
  if (previous) ZSTD_CCtx_refPrefix(c, previous->data(), previous->size());
  Bytes dst(ZSTD_compressBound(src.size()));
  const size_t n = ZSTD_compress2(c, dst.data(), dst.size(), src.data(), src.size());
  if (ZSTD_isError(n)) throw std::runtime_error(ZSTD_getErrorName(n));
  return n;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "usage: hist_measure MAP.4dmap[.zpk] [CONFIG.txt]\n";
    return 2;
  }
  const std::filesystem::path map_path(argv[1]);
  const std::filesystem::path config = argc > 2 ? std::filesystem::path(argv[2]) : map_path.parent_path() / "config.txt";
  const double edge = readBlockEdge(config);
  const auto map = khronos::SpatioTemporalMap::load(map_path.string());
  if (!map || !map->numTimeSteps()) {
    std::cerr << "cannot read " << map_path << "\n";
    return 3;
  }
  std::cout << "block edge " << edge << " m (" << config.string() << "), " << map->numTimeSteps() << " snapshots\n";
  std::cout << "index stamp_ns snapshot_MB pieces mesh_blocks changed_blocks changed_block_MB nodes changed_nodes "
               "changed_node_MB changed_edges increment_raw_MB increment_zstd_MB increment_prefix_MB\n";
  Pieces previous;
  double full = 0, inc_raw = 0, inc_zstd = 0, inc_prefix = 0, inc_bounded = 0, last_snapshot = 0;
  // Per piece: bytes of the deltas stored since its last full version. A piece is stored in full again once those
  // deltas would exceed one full version of it, so restoring any version decodes at most about twice its size.
  std::unordered_map<std::string, size_t> chain;
  ZSTD_CCtx* cctx = ZSTD_createCCtx();
  for (size_t i = 0; i < map->numTimeSteps(); ++i) {
    const auto dsg = map->rawDsg(i);
    Bytes whole;
    spark_dsg::io::binary::writeGraph(*dsg, whole, true);
    Pieces current = pieces(*dsg, edge);
    size_t blocks = 0, changed_blocks = 0, nodes = 0, changed_nodes = 0, changed_edges = 0;
    double block_bytes = 0, node_bytes = 0;
    Bytes increment;
    size_t prefix_bytes = 0, bounded_bytes = 0;
    std::vector<const std::string*> keys;
    for (const auto& [key, bytes] : current) keys.push_back(&key);
    std::sort(keys.begin(), keys.end(), [](const auto* a, const auto* b) { return *a < *b; });
    for (const std::string* key : keys) {
      const Bytes& bytes = current.at(*key);
      const char tag = (*key)[0];
      if (tag == 'f' || tag == 'v') ++blocks;
      if (tag == 'n') ++nodes;
      const auto it = previous.find(*key);
      if (it != previous.end() && it->second == bytes) continue;
      const size_t delta = zstdAgainst(cctx, bytes, it != previous.end() ? &it->second : nullptr);
      prefix_bytes += key->size() + delta;
      {
        const size_t alone = it == previous.end() ? delta : zstdAgainst(cctx, bytes, nullptr);
        size_t& since = chain[*key];
        if (it == previous.end() || since + delta > alone) { bounded_bytes += key->size() + alone; since = 0; }
        else { bounded_bytes += key->size() + delta; since += delta; }
      }
      increment.insert(increment.end(), key->begin(), key->end());
      increment.insert(increment.end(), bytes.begin(), bytes.end());
      if (tag == 'f' || tag == 'v') { ++changed_blocks; block_bytes += bytes.size(); }
      else if (tag == 'n') { ++changed_nodes; node_bytes += bytes.size(); }
      else if (tag == 'e' || tag == 'x') ++changed_edges;
    }
    for (const auto& [key, bytes] : previous)
      if (!current.count(key)) increment.insert(increment.end(), key.begin(), key.end());  // the piece ended here
    const size_t z = increment.empty() ? 0 : zstdSize(increment);
    full += whole.size();
    inc_raw += increment.size();
    inc_zstd += z;
    inc_prefix += prefix_bytes;
    inc_bounded += bounded_bytes;
    last_snapshot = whole.size();
    std::cout << i << " " << map->stamps()[i] << std::fixed << std::setprecision(3) << " " << whole.size() / kMB << " "
              << current.size() << " " << blocks << " " << changed_blocks << " " << block_bytes / kMB << " " << nodes
              << " " << changed_nodes << " " << node_bytes / kMB << " " << changed_edges << " "
              << increment.size() / kMB << " " << z / kMB << " " << prefix_bytes / kMB << "\n";
    previous = std::move(current);
  }
  std::cout << std::fixed << std::setprecision(1) << "TOTAL snapshots " << map->numTimeSteps()
            << " | every snapshot in full " << full / kMB << " MB | last snapshot " << last_snapshot / kMB
            << " MB | incremental raw " << inc_raw / kMB << " MB | incremental zstd " << inc_zstd / kMB
            << " MB | incremental, each piece against its previous version " << inc_prefix / kMB
            << " MB | same, with a full version when its deltas outgrow it " << inc_bounded / kMB << " MB\n";
  ZSTD_freeCCtx(cctx);
  return 0;
}
