#include "khronos/spatio_temporal_map/piece_history.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <istream>
#include <iterator>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <unordered_map>

#include <nlohmann/json.hpp>
#include <zstd.h>

#include <spark_dsg/edge_attributes.h>
#include <spark_dsg/mesh.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/serialization/attribute_registry.h>
#include <spark_dsg/serialization/attribute_serialization.h>
#include <spark_dsg/serialization/binary_conversions.h>
#include <spark_dsg/serialization/binary_serialization.h>
#include <spark_dsg/serialization/versioning.h>

namespace khronos {
namespace {

using Bytes = PieceHistory::Bytes;
using Pieces = PieceHistory::Pieces;
using PieceRefs = std::map<std::string, const Bytes*>;
using spark_dsg::DynamicSceneGraph;
using spark_dsg::serialization::BinaryDeserializer;
using spark_dsg::serialization::BinarySerializer;

constexpr char kMagic[] = "KHRONOS-PIECE-HISTORY-2";

template <typename T>
void put(Bytes& b, const T& v) {
  const auto* p = reinterpret_cast<const uint8_t*>(&v);
  b.insert(b.end(), p, p + sizeof(T));
}
template <typename T>
T get(const uint8_t*& p, const uint8_t* end) {
  if (p + sizeof(T) > end) throw std::runtime_error("piece history: truncated piece");
  T v;
  std::memcpy(&v, p, sizeof(T));
  p += sizeof(T);
  return v;
}
void putString(Bytes& b, const std::string& s) {
  put<uint32_t>(b, static_cast<uint32_t>(s.size()));
  b.insert(b.end(), s.begin(), s.end());
}
std::string getString(const uint8_t*& p, const uint8_t* end) {
  const auto n = get<uint32_t>(p, end);
  if (p + n > end) throw std::runtime_error("piece history: truncated string");
  std::string s(reinterpret_cast<const char*>(p), n);
  p += n;
  return s;
}

// --- mesh records: one vertex = position [, color][, stamp][, first seen][, label], as the mesh's own flags say ---
struct Layout {
  bool colors, stamps, first_seen, labels;
  size_t size() const { return 12 + (colors ? 4 : 0) + (stamps ? 8 : 0) + (first_seen ? 8 : 0) + (labels ? 4 : 0); }
  size_t firstSeenOffset() const { return 12 + (colors ? 4 : 0) + (stamps ? 8 : 0); }
};

// A vertex in the pieces is named by its record and its rank among the vertices with that record, in the mesh's
// order (4 bytes after the record). A mapper's mesh need not be welded: vertices with equal records are common, and the
// names keep how many vertices a mesh has and which of them each face uses.
constexpr size_t kRankSize = 4;

void writeVertex(const spark_dsg::Mesh& m, const Layout& l, size_t v, uint8_t* o) {
  std::memcpy(o, m.points[v].data(), 12);
  o += 12;
  if (l.colors) {
    const uint8_t c[4] = {m.colors[v].r, m.colors[v].g, m.colors[v].b, m.colors[v].a};
    std::memcpy(o, c, 4);
    o += 4;
  }
  if (l.stamps) { std::memcpy(o, &m.stamps[v], 8); o += 8; }
  if (l.first_seen) { std::memcpy(o, &m.first_seen_stamps[v], 8); o += 8; }
  if (l.labels) std::memcpy(o, &m.labels[v], 4);
}

void setVertex(spark_dsg::Mesh& m, const Layout& l, size_t v, const uint8_t* r) {
  std::memcpy(m.points[v].data(), r, 12);
  r += 12;
  if (l.colors) { m.colors[v] = spark_dsg::Color(r[0], r[1], r[2], r[3]); r += 4; }
  if (l.stamps) { std::memcpy(&m.stamps[v], r, 8); r += 8; }
  if (l.first_seen) { std::memcpy(&m.first_seen_stamps[v], r, 8); r += 8; }
  if (l.labels) std::memcpy(&m.labels[v], r, 4);
}

// Fixed-size byte records, each kept once (open addressing over pointers to them): a record's id is the order it was
// first met in. The records must outlive the table.
class RecordTable {
 public:
  RecordTable(size_t record_size, size_t capacity) : size_(record_size) {
    size_t slots = 64;
    while (slots < 2 * capacity) slots <<= 1;
    mask_ = slots - 1;
    slots_.assign(slots, kEmpty);
    records_.reserve(capacity);
  }
  uint32_t insert(const uint8_t* record) {
    for (size_t s = hash(record) & mask_;; s = (s + 1) & mask_) {
      const uint32_t id = slots_[s];
      if (id == kEmpty) {
        slots_[s] = static_cast<uint32_t>(records_.size());
        records_.push_back(record);
        return slots_[s];
      }
      if (std::memcmp(records_[id], record, size_) == 0) return id;
    }
  }
  size_t size() const { return records_.size(); }
  const uint8_t* operator[](size_t id) const { return records_[id]; }

 private:
  static constexpr uint32_t kEmpty = std::numeric_limits<uint32_t>::max();
  size_t hash(const uint8_t* p) const {
    uint64_t h = 0x9e3779b97f4a7c15ull;
    size_t i = 0;
    for (; i + 8 <= size_; i += 8) {
      uint64_t w;
      std::memcpy(&w, p + i, 8);
      h = (h ^ w) * 0xbf58476d1ce4e5b9ull;
      h ^= h >> 29;
    }
    if (i < size_) {
      uint64_t w = 0;
      std::memcpy(&w, p + i, size_ - i);
      h = (h ^ w) * 0x94d049bb133111ebull;
    }
    return static_cast<size_t>(h ^ (h >> 32));
  }
  size_t size_, mask_;
  std::vector<uint32_t> slots_;
  std::vector<const uint8_t*> records_;
};

// The named vertices in mesh order: by first-seen time, as finalizing a snapshot orders a mesh (a 4D query reveals the
// vertices seen by its time as a prefix, SpatioTemporalMap::moveMeshForward), the vertices sharing a record by rank.
// Stable counting sorts: by rank, then by the time in 13-bit digits.
std::vector<uint32_t> meshOrder(const RecordTable& names, const Layout& l) {
  const size_t n = names.size();
  const size_t vsize = l.size();
  std::vector<uint32_t> order(n), next(n);
  for (size_t i = 0; i < n; ++i) order[i] = static_cast<uint32_t>(i);
  auto pass = [&](auto digit, size_t buckets) {
    std::vector<uint32_t> start(buckets + 1, 0);
    for (const auto id : order) ++start[digit(id) + 1];
    for (size_t b = 1; b <= buckets; ++b) start[b] += start[b - 1];
    for (const auto id : order) next[start[digit(id)]++] = id;
    order.swap(next);
  };
  std::vector<uint32_t> rank(n);
  uint32_t max_rank = 0;
  for (size_t i = 0; i < n; ++i) {
    std::memcpy(&rank[i], names[i] + vsize, kRankSize);
    max_rank = std::max(max_rank, rank[i]);
  }
  if (max_rank > 0) pass([&](uint32_t id) { return rank[id]; }, size_t{max_rank} + 1);
  if (l.first_seen && n > 0) {
    std::vector<uint64_t> time(n);
    for (size_t i = 0; i < n; ++i) std::memcpy(&time[i], names[i] + l.firstSeenOffset(), 8);
    const auto [lo, hi] = std::minmax_element(time.begin(), time.end());
    const uint64_t base = *lo;
    const uint64_t range = *hi - *lo;
    for (unsigned shift = 0; shift < 64 && (range >> shift) != 0; shift += 13) {
      pass([&](uint32_t id) { return static_cast<uint32_t>(((time[id] - base) >> shift) & 8191u); }, 8192);
    }
  }
  return order;
}

// --- mesh blocks: the mapper's voxel blocks, by their integer coordinates ---
struct Block {
  long x, y, z;
  bool operator==(const Block& o) const { return x == o.x && y == o.y && z == o.z; }
};
struct BlockHash {
  size_t operator()(const Block& b) const {
    return static_cast<size_t>(b.x * 73856093L ^ b.y * 19349663L ^ b.z * 83492791L);
  }
};
Block blockOf(const Eigen::Vector3f& p, double edge) {
  return {static_cast<long>(std::floor(p.x() / edge)), static_cast<long>(std::floor(p.y() / edge)),
          static_cast<long>(std::floor(p.z() / edge))};
}
std::string blockKey(char tag, const Block& b) {
  return std::string(1, tag) + std::to_string(b.x) + "," + std::to_string(b.y) + "," + std::to_string(b.z);
}

// Fixed-size records sorted bytewise: a block's bytes do not depend on the order the mapper wrote its mesh in.
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

template <typename T>
Bytes attributes(const T& attrs) {
  Bytes b;
  BinarySerializer s(&b);
  spark_dsg::write_binary(s, attrs);
  return b;
}

// --- zstd: a version alone, or against the previous version of the same piece ---
Bytes compress(ZSTD_CCtx* c, const Bytes& src, const Bytes* previous) {
  ZSTD_CCtx_reset(c, ZSTD_reset_session_only);
  ZSTD_CCtx_setParameter(c, ZSTD_c_compressionLevel, 3);
  if (previous) ZSTD_CCtx_refPrefix(c, previous->data(), previous->size());
  Bytes dst(ZSTD_compressBound(src.size()));
  const size_t n = ZSTD_compress2(c, dst.data(), dst.size(), src.data(), src.size());
  if (ZSTD_isError(n)) throw std::runtime_error(std::string("piece history: zstd ") + ZSTD_getErrorName(n));
  dst.resize(n);
  return dst;
}

Bytes decompress(ZSTD_DCtx* d, const uint8_t* src, size_t length, size_t size, const Bytes* previous) {
  ZSTD_DCtx_reset(d, ZSTD_reset_session_only);
  if (previous) ZSTD_DCtx_refPrefix(d, previous->data(), previous->size());
  Bytes out(size);
  const size_t n = ZSTD_decompressDCtx(d, out.data(), out.size(), src, length);
  if (ZSTD_isError(n) || n != size) throw std::runtime_error("piece history: a stored version does not decode");
  return out;
}

using DCtxPtr = std::unique_ptr<ZSTD_DCtx, decltype(&ZSTD_freeDCtx)>;
DCtxPtr makeDCtx() { return DCtxPtr(ZSTD_createDCtx(), &ZSTD_freeDCtx); }

DynamicSceneGraph::Ptr assemble(const PieceRefs& pieces,
                                const Bytes& header,
                                const std::vector<std::string>& node_types,
                                const std::vector<std::string>& edge_types) {
  const auto file_header = spark_dsg::io::FileHeader::deserializeFromBinary(header);
  if (!file_header) throw std::runtime_error("piece history: invalid spark-dsg header");
  spark_dsg::io::GlobalInfo::ScopedInfo info(*file_header);
  const auto node_factory = spark_dsg::serialization::AttributeRegistry<spark_dsg::NodeAttributes>::fromNames(node_types);
  const auto edge_factory = spark_dsg::serialization::AttributeRegistry<spark_dsg::EdgeAttributes>::fromNames(edge_types);

  const Bytes& h = *pieces.at("h");
  const uint8_t* p = h.data();
  const uint8_t* end = h.data() + h.size();
  DynamicSceneGraph::LayerKeys keys(get<uint32_t>(p, end));
  for (auto& k : keys) {
    k.layer = get<int64_t>(p, end);
    k.partition = get<uint32_t>(p, end);
  }
  DynamicSceneGraph::LayerNames names;
  for (uint32_t n = get<uint32_t>(p, end); n > 0; --n) {
    const std::string name = getString(p, end);
    spark_dsg::LayerKey key;
    key.layer = get<int64_t>(p, end);
    key.partition = get<uint32_t>(p, end);
    names[name] = key;
  }
  auto graph = std::make_shared<DynamicSceneGraph>(keys, names);
  graph->metadata.set(nlohmann::json::parse(getString(p, end)));
  const bool has_mesh = get<uint8_t>(p, end);
  Layout layout{};
  bool timestamps = false;
  if (has_mesh) {
    layout.colors = get<uint8_t>(p, end);
    timestamps = get<uint8_t>(p, end);
    layout.labels = get<uint8_t>(p, end);
    layout.first_seen = get<uint8_t>(p, end);
    layout.stamps = timestamps;
  }

  for (auto it = pieces.lower_bound("n"); it != pieces.end() && it->first[0] == 'n'; ++it) {
    const uint8_t* q = it->second->data();
    const uint8_t* qend = q + it->second->size();
    const auto layer = get<int64_t>(q, qend);
    const auto partition = get<uint32_t>(q, qend);
    const auto id = get<uint64_t>(q, qend);
    BinaryDeserializer d(q, static_cast<size_t>(qend - q));
    auto attrs = spark_dsg::serialization::Visitor::from(node_factory, d);
    if (!attrs) throw std::runtime_error("piece history: unknown node attribute type");
    graph->emplaceNode(layer, id, std::move(attrs), partition);
  }
  for (auto it = pieces.lower_bound("e"); it != pieces.end() && it->first[0] == 'e'; ++it) {
    const uint8_t* q = it->second->data();
    const uint8_t* qend = q + it->second->size();
    const auto source = get<uint64_t>(q, qend);
    const auto target = get<uint64_t>(q, qend);
    BinaryDeserializer d(q, static_cast<size_t>(qend - q));
    auto attrs = spark_dsg::serialization::Visitor::from(edge_factory, d);
    graph->insertEdge(source, target, std::move(attrs));
  }
  if (has_mesh) {
    auto mesh = std::make_shared<spark_dsg::Mesh>(layout.colors, timestamps, layout.labels, layout.first_seen);
    const size_t nsize = layout.size() + kRankSize;
    std::vector<const Bytes*> face_pieces, loose_pieces;
    size_t corners = 0, loose = 0;
    for (auto it = pieces.lower_bound("f"); it != pieces.end() && it->first[0] == 'f'; ++it) {
      face_pieces.push_back(it->second);
      corners += it->second->size() / nsize;
    }
    for (auto it = pieces.lower_bound("v"); it != pieces.end() && it->first[0] == 'v'; ++it) {
      loose_pieces.push_back(it->second);
      loose += it->second->size() / nsize;
    }
    // Every vertex once, by name; the face corners as vertex ids.
    RecordTable vertices(nsize, corners + loose);
    std::vector<uint32_t> corner_ids;
    corner_ids.reserve(corners);
    for (const Bytes* b : face_pieces)
      for (size_t r = 0; r + nsize <= b->size(); r += nsize) corner_ids.push_back(vertices.insert(b->data() + r));
    for (const Bytes* b : loose_pieces)
      for (size_t r = 0; r + nsize <= b->size(); r += nsize) vertices.insert(b->data() + r);
    const auto order = meshOrder(vertices, layout);
    std::vector<size_t> position(order.size());
    mesh->resizeVertices(order.size());
    for (size_t i = 0; i < order.size(); ++i) {
      position[order[i]] = i;
      setVertex(*mesh, layout, i, vertices[order[i]]);
    }
    mesh->faces.resize(corner_ids.size() / 3);
    for (size_t f = 0; f < mesh->faces.size(); ++f) {
      mesh->faces[f] = {position[corner_ids[3 * f]], position[corner_ids[3 * f + 1]], position[corner_ids[3 * f + 2]]};
    }
    graph->setMesh(mesh);
  }
  return graph;
}

}  // namespace

PieceHistory::PieceHistory(double block_edge) : block_edge_(block_edge) {
  if (!(block_edge > 0.0)) throw std::invalid_argument("piece history: the block edge must be positive");
}

PieceHistory::Pieces PieceHistory::split(const DynamicSceneGraph& g, double edge) {
  Pieces out;
  const auto mesh = g.mesh();
  Bytes h;
  const auto keys = g.layer_keys();
  put<uint32_t>(h, static_cast<uint32_t>(keys.size()));
  for (const auto& k : keys) {
    put<int64_t>(h, k.layer);
    put<uint32_t>(h, k.partition);
  }
  const auto names = g.layer_names();
  put<uint32_t>(h, static_cast<uint32_t>(names.size()));
  for (const auto& [name, k] : names) {
    putString(h, name);
    put<int64_t>(h, k.layer);
    put<uint32_t>(h, k.partition);
  }
  putString(h, g.metadata.get().dump());
  put<uint8_t>(h, mesh != nullptr);
  if (mesh) {
    put<uint8_t>(h, mesh->has_colors);
    put<uint8_t>(h, mesh->has_timestamps);
    put<uint8_t>(h, mesh->has_labels);
    put<uint8_t>(h, mesh->has_first_seen_stamps);
  }
  out["h"] = std::move(h);

  for (const auto& [id, key] : g.node_lookup()) {
    Bytes b;
    put<int64_t>(b, key.layer);
    put<uint32_t>(b, key.partition);
    put<uint64_t>(b, id);
    const Bytes a = attributes(g.getNode(id).attributes());
    b.insert(b.end(), a.begin(), a.end());
    out["n" + std::to_string(id)] = std::move(b);
  }
  auto edge_piece = [&](const spark_dsg::SceneGraphEdge& e) {
    Bytes b;
    put<uint64_t>(b, e.source);
    put<uint64_t>(b, e.target);
    const Bytes a = attributes(*e.info);
    b.insert(b.end(), a.begin(), a.end());
    out["e" + std::to_string(e.source) + "-" + std::to_string(e.target)] = std::move(b);
  };
  for (const auto& [lid, layer] : g.layers())
    for (const auto& [k, e] : layer->edges()) edge_piece(e);
  for (const auto& [lid, parts] : g.layer_partitions())
    for (const auto& [pid, layer] : parts)
      for (const auto& [k, e] : layer->edges()) edge_piece(e);
  for (const auto& [k, e] : g.interlayer_edges()) edge_piece(e);

  if (mesh) {
    const Layout l{mesh->has_colors, mesh->has_timestamps, mesh->has_first_seen_stamps, mesh->has_labels};
    const size_t vsize = l.size();
    const size_t nsize = vsize + kRankSize;
    const size_t n = mesh->points.size();
    Bytes vertex_names(n * nsize);
    {
      RecordTable records(vsize, n);
      std::vector<uint32_t> copies;  // per record: the vertices met with it so far
      copies.reserve(n);
      for (size_t v = 0; v < n; ++v) {
        uint8_t* at = vertex_names.data() + v * nsize;
        writeVertex(*mesh, l, v, at);
        const uint32_t id = records.insert(at);
        if (id == copies.size()) copies.push_back(0);
        const uint32_t rank = copies[id]++;
        std::memcpy(at + vsize, &rank, kRankSize);
      }
    }
    std::unordered_map<Block, Bytes, BlockHash> faces, loose;
    std::vector<uint8_t> used(n, 0);
    for (const auto& f : mesh->faces) {
      if (f[0] >= n || f[1] >= n || f[2] >= n) throw std::runtime_error("piece history: a face uses a missing vertex");
      auto& b = faces[blockOf(mesh->points[f[0]], edge)];
      for (const size_t v : f) {
        b.insert(b.end(), vertex_names.begin() + v * nsize, vertex_names.begin() + (v + 1) * nsize);
        used[v] = 1;
      }
    }
    for (size_t v = 0; v < n; ++v) {
      if (used[v]) continue;
      auto& b = loose[blockOf(mesh->points[v], edge)];
      b.insert(b.end(), vertex_names.begin() + v * nsize, vertex_names.begin() + (v + 1) * nsize);
    }
    for (auto& [block, flat] : faces) out[blockKey('f', block)] = sortedRecords(flat, 3 * nsize);
    for (auto& [block, flat] : loose) out[blockKey('v', block)] = sortedRecords(flat, nsize);
  }
  return out;
}

void PieceHistory::append(const DynamicSceneGraph& graph) {
  if (header_.empty()) {
    header_ = spark_dsg::io::FileHeader::current().serializeToBinary();
    node_types_ = spark_dsg::serialization::AttributeRegistry<spark_dsg::NodeAttributes>::names();
    edge_types_ = spark_dsg::serialization::AttributeRegistry<spark_dsg::EdgeAttributes>::names();
  }
  undo_.clear();
  undo_data_ = data_.size();
  store(split(graph, block_edge_), static_cast<uint32_t>(count_));
  ++count_;
  undo_valid_ = true;
}

void PieceHistory::replaceLast(const DynamicSceneGraph& graph) {
  if (count_ == 0) throw std::logic_error("piece history: no snapshot to replace");
  if (!undo_valid_) {  // read from a file: rebuild without its last snapshot
    PieceHistory rebuilt(block_edge_);
    for (size_t i = 0; i + 1 < count_; ++i) rebuilt.append(*snapshot(i));
    rebuilt.append(graph);
    *this = std::move(rebuilt);
    return;
  }
  {
    // The last snapshot's versions are replaced: the decoded versions kept for queries may be among them.
    std::lock_guard<std::mutex> lock(decoded_.mutex);
    decoded_.pieces.clear();
  }
  for (auto it = undo_.rbegin(); it != undo_.rend(); ++it) {
    Piece& p = pieces_.at(it->key);
    p.versions.pop_back();
    p.last = std::move(it->last);
    p.since_full = it->since_full;
    if (!it->had_versions && p.versions.empty()) pieces_.erase(it->key);
  }
  undo_.clear();
  data_.resize(undo_data_);
  store(split(graph, block_edge_), static_cast<uint32_t>(count_ - 1));
  undo_valid_ = true;
}

void PieceHistory::truncate(size_t count) {
  if (count >= count_) return;
  {
    std::lock_guard<std::mutex> lock(decoded_.mutex);
    decoded_.pieces.clear();
  }
  // Versions are stored in the order of their snapshots: the bytes of the dropped ones are the tail of data_, from the
  // first of them on. A piece keeps its latest remaining version decoded (`last`), as after an append.
  uint64_t end = data_.size();
  for (auto it = pieces_.begin(); it != pieces_.end();) {
    Piece& p = it->second;
    const auto cut = std::find_if(p.versions.begin(), p.versions.end(),
                                  [count](const Version& v) { return v.index >= count; });
    if (cut != p.versions.end()) {
      end = std::min<uint64_t>(end, cut->offset);
      p.versions.erase(cut, p.versions.end());
      const bool alive = !p.versions.empty() && p.versions.back().kind != kEnded;
      p.last = alive ? decode(p, p.versions.size() - 1) : Bytes{};
      p.last_known = true;
      p.since_full = 0;
      for (auto v = p.versions.rbegin(); v != p.versions.rend() && v->kind == kDelta; ++v) p.since_full += v->length;
    }
    it = p.versions.empty() ? pieces_.erase(it) : std::next(it);
  }
  data_.resize(end);
  count_ = count;
  undo_.clear();
  undo_valid_ = false;
}

void PieceHistory::store(const Pieces& pieces, uint32_t index) {
  std::unique_ptr<ZSTD_CCtx, decltype(&ZSTD_freeCCtx)> c(ZSTD_createCCtx(), &ZSTD_freeCCtx);
  auto add = [&](Piece& p, uint8_t kind, const Bytes& stored, size_t size) {
    p.versions.push_back({index, kind, data_.size(), static_cast<uint32_t>(stored.size()), static_cast<uint32_t>(size)});
    data_.insert(data_.end(), stored.begin(), stored.end());
  };
  for (auto& [key, p] : pieces_) {  // pieces that ended with this snapshot
    if (p.versions.empty() || p.versions.back().kind == kEnded || pieces.count(key)) continue;
    undo_.push_back({key, p.last, p.since_full, true});
    add(p, kEnded, {}, 0);
    p.last.clear();
    p.last_known = true;
    p.since_full = 0;
  }
  for (const auto& [key, bytes] : pieces) {
    auto [it, inserted] = pieces_.try_emplace(key);
    Piece& p = it->second;
    const bool alive = !p.versions.empty() && p.versions.back().kind != kEnded;
    if (alive && !p.last_known) {  // read from a file: its latest version is decoded when first needed
      p.last = decode(p, p.versions.size() - 1);
      p.last_known = true;
    }
    if (alive && p.last == bytes) continue;
    undo_.push_back({key, p.last, p.since_full, !p.versions.empty()});
    const Bytes alone = compress(c.get(), bytes, nullptr);
    bool stored = false;
    if (alive) {
      const Bytes delta = compress(c.get(), bytes, &p.last);
      if (p.since_full + delta.size() <= alone.size()) {
        add(p, kDelta, delta, bytes.size());
        p.since_full += delta.size();
        stored = true;
      }
    }
    if (!stored) {
      add(p, kFull, alone, bytes.size());
      p.since_full = 0;
    }
    p.last = bytes;
  }
}

void PieceHistory::decodeInto(const Piece& piece, size_t version, Decoded& out, ZSTD_DCtx* d) const {
  if (out.version == version) return;
  size_t full = version;
  while (piece.versions[full].kind != kFull) {
    if (full == 0 || piece.versions[full].kind == kEnded) throw std::runtime_error("piece history: no full version");
    --full;
  }
  const size_t kept = out.version;
  out.version = Decoded::kNone;
  size_t next = full + 1;
  if (kept != Decoded::kNone && kept >= full && kept < version) {
    next = kept + 1;  // the kept version lies on the way: continue from it
  } else {
    const Version& f = piece.versions[full];
    out.bytes = decompress(d, data_.data() + f.offset, f.length, f.size, nullptr);
  }
  for (size_t k = next; k <= version; ++k) {
    const Version& v = piece.versions[k];
    out.bytes = decompress(d, data_.data() + v.offset, v.length, v.size, &out.bytes);
  }
  out.version = version;
}

PieceHistory::Bytes PieceHistory::decode(const Piece& piece, size_t version) const {
  const auto d = makeDCtx();
  Decoded out;
  decodeInto(piece, version, out, d.get());
  return std::move(out.bytes);
}

DynamicSceneGraph::Ptr PieceHistory::snapshot(size_t index) const {
  if (index >= count_) throw std::out_of_range("piece history: snapshot index out of range");
  std::lock_guard<std::mutex> lock(decoded_.mutex);
  const auto d = makeDCtx();
  PieceRefs at;
  for (const auto& [key, p] : pieces_) {
    const auto it = std::upper_bound(p.versions.begin(), p.versions.end(), index,
                                     [](size_t i, const Version& v) { return i < v.index; });
    if (it == p.versions.begin() || std::prev(it)->kind == kEnded) {
      decoded_.pieces.erase(key);
      continue;
    }
    Decoded& piece = decoded_.pieces[key];
    decodeInto(p, static_cast<size_t>(std::prev(it) - p.versions.begin()), piece, d.get());
    at.emplace(key, &piece.bytes);
  }
  return assemble(at, header_, node_types_, edge_types_);
}

void PieceHistory::versionCounts(size_t& full, size_t& delta, size_t& ended) const {
  full = delta = ended = 0;
  for (const auto& [key, p] : pieces_)
    for (const auto& v : p.versions) (v.kind == kFull ? full : v.kind == kDelta ? delta : ended)++;
}

void PieceHistory::write(std::ostream& out) const {
  Bytes b(kMagic, kMagic + sizeof(kMagic));
  put<double>(b, block_edge_);
  put<uint64_t>(b, count_);
  putString(b, std::string(header_.begin(), header_.end()));
  put<uint32_t>(b, static_cast<uint32_t>(node_types_.size()));
  for (const auto& s : node_types_) putString(b, s);
  put<uint32_t>(b, static_cast<uint32_t>(edge_types_.size()));
  for (const auto& s : edge_types_) putString(b, s);
  std::vector<const std::string*> keys;
  for (const auto& [key, p] : pieces_) keys.push_back(&key);
  std::sort(keys.begin(), keys.end(), [](const auto* a, const auto* z) { return *a < *z; });
  put<uint64_t>(b, keys.size());
  for (const std::string* key : keys) {
    const Piece& p = pieces_.at(*key);
    putString(b, *key);
    put<uint32_t>(b, static_cast<uint32_t>(p.versions.size()));
    for (const auto& v : p.versions) {
      put<uint32_t>(b, v.index);
      put<uint8_t>(b, v.kind);
      put<uint64_t>(b, v.offset);
      put<uint32_t>(b, v.length);
      put<uint32_t>(b, v.size);
    }
  }
  put<uint64_t>(b, data_.size());
  out.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
  out.write(reinterpret_cast<const char*>(data_.data()), static_cast<std::streamsize>(data_.size()));
  if (!out) throw std::runtime_error("piece history: write failed");
}

std::unique_ptr<PieceHistory> PieceHistory::read(std::istream& in) {
  auto exact = [&](void* dst, size_t n) {
    in.read(static_cast<char*>(dst), static_cast<std::streamsize>(n));
    if (!in) throw std::runtime_error("piece history: truncated file");
  };
  auto num = [&](auto& v) { exact(&v, sizeof(v)); };
  auto str = [&]() {
    uint32_t n;
    num(n);
    std::string s(n, '\0');
    exact(s.data(), n);
    return s;
  };
  char magic[sizeof(kMagic)];
  exact(magic, sizeof(magic));
  if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) throw std::runtime_error("piece history: not a piece history");
  double edge;
  num(edge);
  auto h = std::make_unique<PieceHistory>(edge);
  uint64_t count;
  num(count);
  h->count_ = count;
  const std::string header = str();
  h->header_.assign(header.begin(), header.end());
  uint32_t n;
  num(n);
  for (; n > 0; --n) h->node_types_.push_back(str());
  num(n);
  for (; n > 0; --n) h->edge_types_.push_back(str());
  uint64_t pieces;
  num(pieces);
  for (; pieces > 0; --pieces) {
    const std::string key = str();
    Piece& p = h->pieces_[key];
    uint32_t versions;
    num(versions);
    p.versions.resize(versions);
    for (auto& v : p.versions) {
      num(v.index);
      num(v.kind);
      num(v.offset);
      num(v.length);
      num(v.size);
    }
  }
  uint64_t size;
  num(size);
  h->data_.resize(size);
  exact(h->data_.data(), size);
  // Ready to append: the delta bytes since each live piece's last full version (its latest version is decoded when
  // an append first needs it).
  for (auto& [key, p] : h->pieces_) {
    p.last_known = p.versions.empty() || p.versions.back().kind == kEnded;
    for (auto it = p.versions.rbegin(); it != p.versions.rend() && it->kind == kDelta; ++it) p.since_full += it->length;
  }
  return h;
}

}  // namespace khronos
