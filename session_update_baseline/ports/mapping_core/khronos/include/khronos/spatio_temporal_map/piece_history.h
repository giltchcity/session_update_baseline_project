#pragma once

#include <cstdint>
#include <iosfwd>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <spark_dsg/dynamic_scene_graph.h>

struct ZSTD_DCtx_s;

namespace khronos {

/**
 * The history of a 4D map as versioned pieces. A snapshot is split into pieces: the mesh by the mapper's voxel blocks
 * (a face belongs to the block of its first vertex, a vertex no face uses to its own block), every node, every edge,
 * and one header (layers, metadata, mesh layout). A piece is stored again only when its canonical bytes change, as a
 * zstd delta against its own previous version; once the deltas stored since its last full version would outgrow a
 * full version, it is stored in full again. Restoring any version of a piece therefore decodes at most about twice
 * its size, and a snapshot is restored from the versions alive at its index, without replaying the history and
 * without periodic full snapshots. The mesh is kept exactly, unwelded vertices included: a vertex is named by its
 * values and its rank among the vertices with the same values, and a face by its vertices' names.
 * Restoring keeps each piece as decoded, so a later snapshot decodes only the pieces that changed since (one delta
 * each when queried in time order).
 */
class PieceHistory {
 public:
  using Bytes = std::vector<uint8_t>;
  using Pieces = std::map<std::string, Bytes>;

  /** block_edge: the edge of the mapper's voxel blocks (voxel size x voxels per side). */
  explicit PieceHistory(double block_edge);

  /** Append a snapshot. */
  void append(const spark_dsg::DynamicSceneGraph& graph);
  /** Replace the last snapshot (a terminal pass at the same stamp). */
  void replaceLast(const spark_dsg::DynamicSceneGraph& graph);
  /**
   * Keep the first `count` snapshots only: the versions of the later ones and their bytes are dropped, and the next
   * append stores snapshot `count` (snapshots stored anew from there on, e.g. with what was learnt about them later).
   */
  void truncate(size_t count);
  /** The snapshot at `index`, as it was appended. */
  spark_dsg::DynamicSceneGraph::Ptr snapshot(size_t index) const;

  size_t size() const { return count_; }
  double blockEdge() const { return block_edge_; }
  /** Bytes of all stored piece versions. */
  uint64_t storedBytes() const { return data_.size(); }
  /** Stored versions: full, delta, ended. */
  void versionCounts(size_t& full, size_t& delta, size_t& ended) const;

  void write(std::ostream& out) const;
  static std::unique_ptr<PieceHistory> read(std::istream& in);

  /** The canonical pieces of a graph; two graphs with equal pieces are the same snapshot. */
  static Pieces split(const spark_dsg::DynamicSceneGraph& graph, double block_edge);

 private:
  enum Kind : uint8_t { kFull = 0, kDelta = 1, kEnded = 2 };
  struct Version {
    uint32_t index = 0;   // the first snapshot it belongs to
    uint8_t kind = kFull;
    uint64_t offset = 0;  // in data_
    uint32_t length = 0;  // stored (compressed) bytes
    uint32_t size = 0;    // decoded bytes
  };
  struct Piece {
    std::vector<Version> versions;
    Bytes last;                 // the latest version, decoded (for the next delta)
    bool last_known = true;     // false after read(): `last` is decoded when an append first needs it
    uint64_t since_full = 0;    // delta bytes stored since its last full version
  };
  struct Undo {                 // what appending the last snapshot changed, to replace it
    std::string key;
    Bytes last;
    uint64_t since_full = 0;
    bool had_versions = false;
  };

  struct Decoded {              // a version of a piece, decoded
    static constexpr size_t kNone = static_cast<size_t>(-1);
    size_t version = kNone;
    Bytes bytes;
  };
  struct DecodeCache {          // the pieces of the last restored snapshot; a copy starts empty
    DecodeCache() = default;
    DecodeCache(const DecodeCache&) {}
    DecodeCache& operator=(const DecodeCache&) {
      std::lock_guard<std::mutex> lock(mutex);
      pieces.clear();
      return *this;
    }
    std::mutex mutex;
    std::unordered_map<std::string, Decoded> pieces;
  };

  void store(const Pieces& pieces, uint32_t index);
  // Decodes `version` of the piece into `out`, from `out`'s version when that lies on the way.
  void decodeInto(const Piece& piece, size_t version, Decoded& out, ZSTD_DCtx_s* d) const;
  Bytes decode(const Piece& piece, size_t version) const;

  double block_edge_;
  size_t count_ = 0;
  std::unordered_map<std::string, Piece> pieces_;
  Bytes data_;
  std::vector<Undo> undo_;
  uint64_t undo_data_ = 0;                 // data_ size before the last snapshot was stored
  bool undo_valid_ = false;                // undo_ describes the last snapshot (not after read())
  Bytes header_;                           // spark-dsg file header the attributes were written with
  std::vector<std::string> node_types_;    // attribute registries the attributes were written with
  std::vector<std::string> edge_types_;
  mutable DecodeCache decoded_;
};

}  // namespace khronos
