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

#include "khronos/spatio_temporal_map/spatio_temporal_map.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#include <system_error>
#include <unordered_set>

#include <unistd.h>
#include <fcntl.h>
#include <openssl/evp.h>
#include <nlohmann/json.hpp>
#include <zstd.h>

#include <config_utilities/config.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <spark_dsg/serialization/binary_serialization.h>
#include <spark_dsg/serialization/graph_binary_serialization.h>
#include <spark_dsg/serialization/versioning.h>

#include "khronos/utils/khronos_attribute_utils.h"

namespace khronos {

struct SnapshotSpillDirectory
    : public std::enable_shared_from_this<SnapshotSpillDirectory> {
  SnapshotSpillDirectory(std::filesystem::path root_path, int descriptor)
      : root(std::move(root_path)), fd(descriptor) {}

  ~SnapshotSpillDirectory() {
    if (fd >= 0) {
      ::close(fd);
    }
    if (root.empty()) return;  // read-only archive backing, not an owned temporary directory
    std::error_code error;
    std::filesystem::remove_all(root, error);
    if (error) {
      LOG(WARNING) << "Could not clean 4D-map spill directory '" << root
                   << "': " << error.message();
    }
  }

  std::shared_ptr<SnapshotBlob> createBlob(uint64_t size);

  const std::filesystem::path root;
  const int fd;

 private:
  std::mutex allocation_mutex;
  uint64_t next_offset = 0;
};

struct SnapshotBlob {
  SnapshotBlob(std::shared_ptr<SnapshotSpillDirectory> backing_store,
               uint64_t byte_offset,
               uint64_t byte_count)
      : store(std::move(backing_store)), offset(byte_offset), stored_size(byte_count), size(byte_count) {}

  SnapshotBlob(const SnapshotBlob&) = delete;
  SnapshotBlob& operator=(const SnapshotBlob&) = delete;

  const std::shared_ptr<SnapshotSpillDirectory> store;
  const uint64_t offset;
  const uint64_t stored_size;
  uint64_t size;  // decoded graph bytes; fixed before publication
  bool compressed = false;
  std::shared_ptr<SnapshotBlob> base;
  size_t depth = 0;
  std::string sha256;
};

std::shared_ptr<SnapshotBlob> SnapshotSpillDirectory::createBlob(uint64_t size) {
  std::lock_guard<std::mutex> lock(allocation_mutex);
  constexpr uint64_t kMaxFileOffset =
      static_cast<uint64_t>(std::numeric_limits<off_t>::max());
  if (next_offset > kMaxFileOffset || size > kMaxFileOffset - next_offset) {
    throw std::runtime_error("4D-map snapshot spill container offset overflow");
  }
  const uint64_t offset = next_offset;
  next_offset += size;
  return std::make_shared<SnapshotBlob>(shared_from_this(), offset, size);
}

namespace {

std::shared_ptr<SnapshotSpillDirectory> makeSpillDirectory() {
  const auto temp_root = std::filesystem::temp_directory_path();
  std::string path_template =
      (temp_root / "session_update_4dmap_XXXXXX").string();
  std::vector<char> writable(path_template.begin(), path_template.end());
  writable.push_back('\0');
  const char* created = ::mkdtemp(writable.data());
  if (!created) {
    throw std::runtime_error("Could not create an owned 4D-map spill directory under " +
                             temp_root.string());
  }

  const std::filesystem::path root(created);
  std::string container_template = (root / "snapshots_XXXXXX").string();
  std::vector<char> container_path(container_template.begin(),
                                   container_template.end());
  container_path.push_back('\0');
  const int fd = ::mkstemp(container_path.data());
  if (fd < 0) {
    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
    throw std::runtime_error("Could not create a 4D-map snapshot spill container");
  }

  // Keep one descriptor for the complete store but remove its directory entry
  // immediately. The kernel then reclaims the container after either normal
  // destruction or process termination, including SIGKILL.
  if (::unlink(container_path.data()) != 0) {
    const int saved_errno = errno;
    ::close(fd);
    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
    throw std::runtime_error(
        "Could not unlink a 4D-map snapshot spill container: errno=" +
        std::to_string(saved_errno));
  }

  try {
    return std::make_shared<SnapshotSpillDirectory>(root, fd);
  } catch (...) {
    ::close(fd);
    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
    throw;
  }
}

template <typename T>
T readLittleEndian(std::istream& in) {
  static_assert(std::is_integral_v<T>);
  using Unsigned = std::make_unsigned_t<T>;
  Unsigned result = 0;
  for (size_t i = 0; i < sizeof(T); ++i) {
    const int byte = in.get();
    if (byte == std::char_traits<char>::eof()) {
      throw std::runtime_error("Unexpected end of 4D-map stream");
    }
    result |= static_cast<Unsigned>(static_cast<uint8_t>(byte)) << (8 * i);
  }
  T value;
  std::memcpy(&value, &result, sizeof(T));
  return value;
}

void expectPackType(std::istream& in,
                    spark_dsg::serialization::PackType expected,
                    const char* field) {
  const int encoded = in.get();
  if (encoded == std::char_traits<char>::eof()) {
    throw std::runtime_error(std::string("Unexpected end while reading ") + field);
  }
  if (static_cast<uint8_t>(encoded) != static_cast<uint8_t>(expected)) {
    throw std::runtime_error(std::string("Invalid pack type while reading ") + field);
  }
}

template <typename T>
T readPackedIntegral(std::istream& in, const char* field) {
  expectPackType(in,
                 spark_dsg::serialization::PackTypeLookup::value<T>(),
                 field);
  return readLittleEndian<T>(in);
}

bool readPackedBool(std::istream& in, const char* field) {
  const int encoded = in.get();
  if (encoded == static_cast<int>(spark_dsg::serialization::PackType::TRUE)) {
    return true;
  }
  if (encoded == static_cast<int>(spark_dsg::serialization::PackType::FALSE)) {
    return false;
  }
  throw std::runtime_error(std::string("Invalid bool while reading ") + field);
}

uint32_t readPackedArrayLength(std::istream& in, const char* field) {
  expectPackType(in, spark_dsg::serialization::PackType::ARR32, field);
  return readLittleEndian<uint32_t>(in);
}

void pwriteAll(int fd,
               const uint8_t* data,
               size_t size,
               uint64_t file_offset,
               const char* context) {
  size_t written = 0;
  while (written < size) {
    const ssize_t bytes =
        ::pwrite(fd, data + written, size - written, file_offset + written);
    if (bytes < 0 && errno == EINTR) {
      continue;
    }
    if (bytes <= 0) {
      throw std::runtime_error(std::string("Could not write ") + context);
    }
    written += static_cast<size_t>(bytes);
  }
}

void preadAll(int fd,
              uint8_t* data,
              size_t size,
              uint64_t file_offset,
              const char* context) {
  size_t read = 0;
  while (read < size) {
    const ssize_t bytes =
        ::pread(fd, data + read, size - read, file_offset + read);
    if (bytes < 0 && errno == EINTR) {
      continue;
    }
    if (bytes <= 0) {
      throw std::runtime_error(std::string("Could not read ") + context);
    }
    read += static_cast<size_t>(bytes);
  }
}

template <typename T>
std::vector<T> readPackedIntegralVector(std::istream& in, const char* field) {
  const auto count = readPackedArrayLength(in, field);
  std::vector<T> result;
  result.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    result.push_back(readPackedIntegral<T>(in, field));
  }
  return result;
}


constexpr char kZpkMagic[] = "4DMAPZPK\0\1";
constexpr size_t kZpkMagicSize = 10;
constexpr char kZpkTrailerMagic[] = "ZPKEND01";
constexpr size_t kZpkTrailerSize = 24;
constexpr size_t kSnapshotKeyframe = 10;

void checkZstd(size_t code) {
  if (ZSTD_isError(code)) {
    throw std::runtime_error(std::string("4D-map zstd: ") + ZSTD_getErrorName(code));
  }
}

std::string hexBytes(const uint8_t* data, size_t size) {
  constexpr char digits[] = "0123456789abcdef";
  std::string output(2 * size, '0');
  for (size_t i = 0; i < size; ++i) {
    output[2 * i] = digits[data[i] >> 4];
    output[2 * i + 1] = digits[data[i] & 15];
  }
  return output;
}

class Sha256 {
 public:
  Sha256() : context_(EVP_MD_CTX_new(), &EVP_MD_CTX_free) {
    if (!context_ || EVP_DigestInit_ex(context_.get(), EVP_sha256(), nullptr) != 1) {
      throw std::runtime_error("Could not initialize 4D-map SHA256");
    }
  }
  void update(const void* data, size_t size) {
    if (EVP_DigestUpdate(context_.get(), data, size) != 1) {
      throw std::runtime_error("Could not update 4D-map SHA256");
    }
  }
  std::string finish() {
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int count = 0;
    if (EVP_DigestFinal_ex(context_.get(), digest, &count) != 1) {
      throw std::runtime_error("Could not finalize 4D-map SHA256");
    }
    return hexBytes(digest, count);
  }
 private:
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context_;
};

std::string hashBytes(const std::vector<uint8_t>& bytes) {
  Sha256 digest;
  digest.update(bytes.data(), bytes.size());
  return digest.finish();
}

std::vector<uint8_t> snapshotHeader(uint64_t size) {
  if (size > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("A DSG snapshot exceeds the v1 4D-map array limit");
  }
  std::vector<uint8_t> header;
  spark_dsg::serialization::BinarySerializer serializer(&header);
  serializer.startFixedArray(static_cast<size_t>(size));
  return header;
}

std::vector<uint8_t> readStoredSnapshot(const SnapshotBlob& blob) {
  std::vector<uint8_t> result(blob.stored_size);
  preadAll(blob.store->fd, result.data(), result.size(), blob.offset,
           "compressed DSG snapshot");
  return result;
}

std::vector<uint8_t> compressSnapshot(
    const std::vector<uint8_t>& bytes,
    const std::vector<uint8_t>* prefix) {
  std::unique_ptr<ZSTD_CCtx, decltype(&ZSTD_freeCCtx)> context(
      ZSTD_createCCtx(), &ZSTD_freeCCtx);
  if (!context) throw std::runtime_error("Could not allocate 4D-map zstd compressor");
  checkZstd(ZSTD_CCtx_setParameter(context.get(), ZSTD_c_compressionLevel, 3));
  checkZstd(ZSTD_CCtx_setParameter(context.get(), ZSTD_c_checksumFlag, 1));
  if (prefix) {
    // Raw-prefix codec compatible with zstd --patch-from. Span both graphs so
    // unchanged bytes can match anywhere in the previous graph, not just its tail.
    const uint64_t span = static_cast<uint64_t>(bytes.size()) + prefix->size();
    int window_log = 10;
    while (window_log < 31 && (uint64_t{1} << window_log) < span) ++window_log;
    checkZstd(ZSTD_CCtx_setParameter(context.get(), ZSTD_c_windowLog, window_log));
    if (span >= (uint64_t{1} << 23)) {
      checkZstd(ZSTD_CCtx_setParameter(context.get(), ZSTD_c_enableLongDistanceMatching, 1));
    }
    checkZstd(ZSTD_CCtx_refPrefix(context.get(), prefix->data(), prefix->size()));
  }
  std::vector<uint8_t> encoded(ZSTD_compressBound(bytes.size()));
  const size_t size = ZSTD_compress2(context.get(), encoded.data(), encoded.size(),
                                     bytes.data(), bytes.size());
  checkZstd(size);
  encoded.resize(size);
  return encoded;
}

std::vector<uint8_t> decodeSnapshot(const SnapshotBlob& blob) {
  std::vector<const SnapshotBlob*> chain;
  const SnapshotBlob* cursor = &blob;
  while (cursor) {
    if (chain.size() >= 1024) {
      throw std::runtime_error("4D-map compressed snapshot prefix chain is too long");
    }
    chain.push_back(cursor);
    cursor = cursor->base.get();
  }
  std::vector<uint8_t> decoded;
  for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
    const auto& entry = **it;
    auto encoded = readStoredSnapshot(entry);
    if (!entry.compressed) {
      decoded = std::move(encoded);
    } else {
      std::unique_ptr<ZSTD_DCtx, decltype(&ZSTD_freeDCtx)> context(
          ZSTD_createDCtx(), &ZSTD_freeDCtx);
      if (!context) throw std::runtime_error("Could not allocate 4D-map zstd decoder");
      checkZstd(ZSTD_DCtx_setParameter(context.get(), ZSTD_d_windowLogMax, 31));
      if (entry.base) {
        checkZstd(ZSTD_DCtx_refPrefix(context.get(), decoded.data(), decoded.size()));
      }
      const auto frame_size = ZSTD_getFrameContentSize(encoded.data(), encoded.size());
      if (frame_size != ZSTD_CONTENTSIZE_UNKNOWN && frame_size != entry.size) {
        throw std::runtime_error("4D-map zstd frame size disagrees with its manifest");
      }
      if (frame_size == ZSTD_CONTENTSIZE_ERROR) {
        throw std::runtime_error("Invalid 4D-map zstd frame");
      }
      std::vector<uint8_t> next(entry.size);
      const size_t size = ZSTD_decompressDCtx(context.get(), next.data(), next.size(),
                                              encoded.data(), encoded.size());
      checkZstd(size);
      if (size != entry.size) {
        throw std::runtime_error("4D-map snapshot decoded length mismatch");
      }
      decoded = std::move(next);
    }
    if (!entry.sha256.empty() && hashBytes(decoded) != entry.sha256) {
      throw std::runtime_error("4D-map snapshot SHA256 mismatch");
    }
  }
  return decoded;
}

std::shared_ptr<SnapshotBlob> storeSnapshotBytes(
    const std::shared_ptr<SnapshotSpillDirectory>& store,
    const std::vector<uint8_t>& bytes,
    const std::shared_ptr<SnapshotBlob>& base = nullptr) {
  std::vector<uint8_t> prefix;
  if (base) prefix = decodeSnapshot(*base);
  auto encoded = compressSnapshot(bytes, base ? &prefix : nullptr);
  auto blob = store->createBlob(encoded.size());
  blob->size = bytes.size();
  blob->compressed = true;
  blob->base = base;
  blob->depth = base ? base->depth + 1 : 0;
  blob->sha256 = hashBytes(bytes);
  pwriteAll(blob->store->fd, encoded.data(), encoded.size(), blob->offset,
            "compressed DSG snapshot");
  return blob;
}

void hashTypedSnapshot(Sha256& digest, const std::vector<uint8_t>& bytes) {
  const auto header = snapshotHeader(bytes.size());
  digest.update(header.data(), header.size());
  constexpr size_t kChunkSize = 1u << 20;
  std::vector<uint8_t> typed(2 * std::min(bytes.size(), kChunkSize));
  for (size_t offset = 0; offset < bytes.size(); offset += kChunkSize) {
    const auto count = std::min(bytes.size() - offset, kChunkSize);
    for (size_t i = 0; i < count; ++i) {
      typed[2 * i] = 0xcc;
      typed[2 * i + 1] = bytes[offset + i];
    }
    digest.update(typed.data(), 2 * count);
  }
}

void verifyEncodedSnapshot(const std::vector<uint8_t>& encoded,
                           const std::vector<uint8_t>& expected,
                           const std::vector<uint8_t>* prefix) {
  std::unique_ptr<ZSTD_DCtx, decltype(&ZSTD_freeDCtx)> context(
      ZSTD_createDCtx(), &ZSTD_freeDCtx);
  if (!context) throw std::runtime_error("Could not allocate snapshot verification decoder");
  checkZstd(ZSTD_DCtx_setParameter(context.get(), ZSTD_d_windowLogMax, 31));
  if (prefix) checkZstd(ZSTD_DCtx_refPrefix(context.get(), prefix->data(), prefix->size()));
  ZSTD_inBuffer input{encoded.data(), encoded.size(), 0};
  std::vector<uint8_t> chunk(1u << 20);
  size_t offset = 0;
  size_t remaining = 1;
  while (remaining) {
    ZSTD_outBuffer output{chunk.data(), chunk.size(), 0};
    const auto previous_input = input.pos;
    remaining = ZSTD_decompressStream(context.get(), &output, &input);
    checkZstd(remaining);
    if (offset > expected.size() || output.pos > expected.size() - offset ||
        !std::equal(chunk.begin(), chunk.begin() + output.pos, expected.begin() + offset)) {
      throw std::runtime_error("Direct delta snapshot verification failed");
    }
    offset += output.pos;
    if (remaining && output.pos == 0 && input.pos == previous_input) {
      throw std::runtime_error("Truncated direct delta snapshot");
    }
  }
  if (offset != expected.size() || input.pos != input.size) {
    throw std::runtime_error("Direct delta snapshot framing mismatch");
  }
}

std::vector<uint8_t> readPackedByteVector(std::istream& in, const char* field) {
  const auto count = readPackedArrayLength(in, field);
  std::vector<uint8_t> result(count);
  for (uint32_t i = 0; i < count; ++i) {
    expectPackType(in, spark_dsg::serialization::PackType::UINT8, field);
    result[i] = readLittleEndian<uint8_t>(in);
  }
  return result;
}

std::shared_ptr<SnapshotBlob> readPackedByteBlob(
    std::istream& in,
    const std::shared_ptr<SnapshotSpillDirectory>& directory,
    const char* field,
    const std::shared_ptr<SnapshotBlob>& base = nullptr) {
  if (!directory) throw std::runtime_error("4D-map loader has no spill directory");
  const uint32_t count = readPackedArrayLength(in, field);
  std::vector<uint8_t> bytes(count);
  constexpr size_t kChunkElements = 1u << 20;
  std::vector<uint8_t> typed(2 * std::min<size_t>(count, kChunkElements));
  for (size_t offset = 0; offset < count; offset += kChunkElements) {
    const auto length = std::min<size_t>(count - offset, kChunkElements);
    in.read(reinterpret_cast<char*>(typed.data()), 2 * length);
    if (!in) throw std::runtime_error("Truncated typed 4D-map snapshot");
    for (size_t i = 0; i < length; ++i) {
      if (typed[2 * i] != 0xcc) throw std::runtime_error("Invalid typed 4D-map byte");
      bytes[offset + i] = typed[2 * i + 1];
    }
  }
  return storeSnapshotBytes(directory, bytes, base);
}

void writePackedByteVector(std::ostream& out, const SnapshotBlob& blob) {
  const auto header = snapshotHeader(blob.size);
  out.write(reinterpret_cast<const char*>(header.data()), header.size());
  const auto raw = decodeSnapshot(blob);
  constexpr size_t kChunkElements = 1u << 20;
  std::vector<uint8_t> encoded(2 * std::min(raw.size(), kChunkElements));
  for (size_t offset = 0; offset < raw.size(); offset += kChunkElements) {
    const size_t count = std::min(raw.size() - offset, kChunkElements);
    for (size_t i = 0; i < count; ++i) {
      encoded[2 * i] = 0xcc;
      encoded[2 * i + 1] = raw[offset + i];
    }
    out.write(reinterpret_cast<const char*>(encoded.data()), 2 * count);
    if (!out) throw std::runtime_error("Failed while streaming a 4D-map snapshot");
  }
}

void trimTrajectoryHistory(KhronosObjectAttributes& attrs, TimeStamp target_time) {
  const size_t available = trajectoryHistorySize(attrs);
  attrs.trajectory_timestamps.resize(available);
  attrs.trajectory_positions.resize(available);
  const auto end = attrs.trajectory_timestamps.begin() + available;
  const auto iter = std::upper_bound(
      attrs.trajectory_timestamps.begin(), end, target_time);
  const size_t keep = iter - attrs.trajectory_timestamps.begin();
  attrs.trajectory_timestamps.resize(keep);
  attrs.trajectory_positions.resize(keep);
  if (!attrs.dynamic_object_points.empty()) {
    // Empty entries explicitly represent history samples for which detailed
    // visualization points were not stored.
    attrs.dynamic_object_points.resize(keep);
  }
}

}  // namespace

void declare_config(SpatioTemporalMap::Config& config) {
  using namespace config;
  name("SpatioTemporalMap");
  field(config.verbosity, "verbosity");
  field(config.finalize_incrementally, "finalize_incrementally");
}

SpatioTemporalMap::SpatioTemporalMap(const Config& config)
    : config(config::checkValid(config)),
      spill_directory_(makeSpillDirectory()),
      serialization_header_(
          spark_dsg::io::FileHeader::current().serializeToBinary()),
      finalized_(config.finalize_incrementally) {}

void SpatioTemporalMap::copyMembers(const SpatioTemporalMap& other) {
  const_cast<Config&>(config) = other.config;
  stamps_ = other.stamps_;
  snapshots_ = other.snapshots_;
  spill_directory_ = other.spill_directory_;
  serialization_header_ = other.serialization_header_;
  source_dsg_cache_idx_ = other.source_dsg_cache_idx_;
  source_dsg_cache_ = other.source_dsg_cache_;
  earliest_ = other.earliest_;
  latest_ = other.latest_;
  resetQueryCache();
  finalized_ = other.finalized_;
}

void SpatioTemporalMap::moveMembers(SpatioTemporalMap&& other) {
  const_cast<Config&>(config) = std::move(other.config);
  stamps_ = std::move(other.stamps_);
  snapshots_ = std::move(other.snapshots_);
  spill_directory_ = std::move(other.spill_directory_);
  serialization_header_ = std::move(other.serialization_header_);
  source_dsg_cache_idx_ = other.source_dsg_cache_idx_;
  source_dsg_cache_ = std::move(other.source_dsg_cache_);
  earliest_ = other.earliest_;
  latest_ = other.latest_;
  current_time_ = other.current_time_;
  previous_time_ = other.previous_time_;
  current_dsg_idx_ = other.current_dsg_idx_;
  current_dsg_ = std::move(other.current_dsg_);
  finalized_ = other.finalized_;
}

SpatioTemporalMap::SpatioTemporalMap(const SpatioTemporalMap& other) { copyMembers(other); }

SpatioTemporalMap& SpatioTemporalMap::operator=(const SpatioTemporalMap& other) {
  copyMembers(other);
  return *this;
}

SpatioTemporalMap::SpatioTemporalMap(SpatioTemporalMap&& other) noexcept {
  moveMembers(std::move(other));
}

SpatioTemporalMap& SpatioTemporalMap::operator=(SpatioTemporalMap&& other) noexcept {
  moveMembers(std::move(other));
  return *this;
}

std::shared_ptr<SnapshotBlob> SpatioTemporalMap::spillSnapshot(
    const DynamicSceneGraph& dsg,
    const std::shared_ptr<SnapshotBlob>& base) const {
  if (!spill_directory_) {
    throw std::runtime_error("4D-map snapshot store has no spill directory");
  }

  std::vector<uint8_t> buffer;
  spark_dsg::io::binary::writeGraph(dsg, buffer, true);
  if (buffer.size() > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(
        "A DSG snapshot exceeds the v1 4D-map serialization limit");
  }

  return storeSnapshotBytes(spill_directory_, buffer, base);
}

DynamicSceneGraph::Ptr SpatioTemporalMap::sourceDsg(size_t index) const {
  if (index >= snapshots_.size()) {
    throw std::out_of_range("4D-map snapshot index is out of range");
  }
  if (source_dsg_cache_ && source_dsg_cache_idx_ == index) {
    return source_dsg_cache_;
  }

  const auto& blob = snapshots_[index];
  if (!blob) {
    throw std::runtime_error("4D-map snapshot storage contains an empty entry");
  }
  auto buffer = decodeSnapshot(*blob);

  const auto header =
      spark_dsg::io::FileHeader::deserializeFromBinary(serialization_header_);
  if (!header) {
    throw std::runtime_error("Stored spark-dsg serialization header is invalid");
  }
  spark_dsg::io::GlobalInfo::ScopedInfo info(*header);
  auto loaded = spark_dsg::io::binary::readGraph(buffer);
  if (!loaded) {
    throw std::runtime_error("Could not decode a spilled DSG snapshot");
  }

  source_dsg_cache_idx_ = index;
  source_dsg_cache_ = std::move(loaded);
  return source_dsg_cache_;
}

void SpatioTemporalMap::setSnapshot(size_t index,
                                    const DynamicSceneGraph::Ptr& dsg) {
  if (!dsg) {
    throw std::invalid_argument("Cannot store a null DSG snapshot");
  }
  if (index >= snapshots_.size()) {
    throw std::out_of_range("Cannot replace an out-of-range DSG snapshot");
  }

  // Snapshot files are immutable. A replacement receives a fresh path so maps
  // copied with the documented shallow-copy semantics cannot overwrite each
  // other's timeline.
  const auto base = index % kSnapshotKeyframe == 0
                        ? nullptr : snapshots_[index - 1];
  snapshots_[index] = spillSnapshot(*dsg, base);
  source_dsg_cache_idx_ = index;
  source_dsg_cache_ = dsg;
}

void SpatioTemporalMap::resetQueryCache() {
  current_time_ = 0;
  previous_time_ = 0;
  current_dsg_idx_ = std::numeric_limits<size_t>::max();
  current_dsg_.reset();
}

uintmax_t SpatioTemporalMap::snapshotStorageBytes() const {
  uintmax_t total = 0;
  std::unordered_set<const SnapshotBlob*> counted;
  for (const auto& snapshot : snapshots_) {
    const SnapshotBlob* blob = snapshot.get();
    while (blob && counted.insert(blob).second) {
      total += blob->stored_size;
      blob = blob->base.get();
    }
  }
  return total;
}

void SpatioTemporalMap::update(const DynamicSceneGraph::Ptr& dsg, TimeStamp stamp) {
  if (!dsg) {
    throw std::invalid_argument("SpatioTemporalMap::update received a null DSG");
  }
  if (!stamps_.empty() && stamp < stamps_.back()) {
    throw std::invalid_argument(
        "SpatioTemporalMap updates must be timestamp ordered: received " +
        std::to_string(stamp) + " after " + std::to_string(stamps_.back()));
  }

  if (config.finalize_incrementally) {
    finalizeDsg(*dsg);
  } else {
    finalized_ = false;
  }

  if (!stamps_.empty() && stamp == stamps_.back()) {
    // A terminal change-detection pass intentionally revisits the last input
    // timestamp. Replace that state so the reconciled snapshot is authoritative;
    // appending a second/raw state at the same time makes latest ambiguous.
    setSnapshot(snapshots_.size() - 1, dsg);
  } else {
    stamps_.push_back(stamp);
    snapshots_.push_back({});
    setSnapshot(snapshots_.size() - 1, dsg);
  }

  // A query may have populated the mutable lookup cache. Any state update
  // invalidates it, including replacement at the same timestamp.
  resetQueryCache();
  
  if (stamp < earliest_) {
    earliest_ = stamp;
  }
  if (stamp > latest_) {
    latest_ = stamp;
  }
}

void SpatioTemporalMap::finalize() {
  if (finalized_) {
    return;
  }

  // Process all DSGs that have not yet been finalized.
  for (size_t i = 0; i < snapshots_.size(); ++i) {
    auto dsg = sourceDsg(i);
    finalizeDsg(*dsg);
    setSnapshot(i, dsg);
  }
  finalized_ = true;
}

void SpatioTemporalMap::finalizeDsg(DynamicSceneGraph& dsg) {
  finalizeMesh(*dsg.mesh());
  updateTimingInfo(dsg);
}

const DynamicSceneGraph& SpatioTemporalMap::getDsg(TimeStamp robot_time) {
  return *getDsgPtr(robot_time);
}

DynamicSceneGraph::Ptr SpatioTemporalMap::getDsgPtr(TimeStamp robot_time) {
  if (stamps_.empty()) {
    throw std::runtime_error("Cannot query an empty SpatioTemporalMap");
  }
  if (!finalized_) {
    finalize();
  }
  current_time_ = std::clamp(robot_time, earliest_, latest_);

  // If no change in time no change in DSG is needed.
  if (current_dsg_ && previous_time_ == current_time_) {
    return current_dsg_;
  }

  // Find the next closest DSG as data source.
  size_t new_dsg_idx =
      std::lower_bound(stamps_.begin(), stamps_.end(), current_time_) - stamps_.begin();
  new_dsg_idx = std::min(new_dsg_idx, stamps_.size() - 1);

  // If the source changes we need to reset the DSG.
  if (current_dsg_idx_ != new_dsg_idx) {
    current_dsg_idx_ = new_dsg_idx;
    current_dsg_ = sourceDsg(current_dsg_idx_)->clone();

    // Trim mesh and other components
    trimDsgToTime(previous_time_);
  }

  // Update the DSG based on robot time.
  if (current_time_ < previous_time_) {
    moveMeshBackward();
    moveAgentBackward();
    moveObjectsBackward();
    moveDynamicObjectAttributesBackward();
  } else if (current_time_ > previous_time_) {
    moveMeshForward();
    moveAgentForward();
    moveObjectsForward();
    moveDynamicObjectAttributesForward();
  }

  // Update time tracking.
  previous_time_ = current_time_;
  return current_dsg_;
}

DynamicSceneGraph::Ptr SpatioTemporalMap::rawDsg(size_t idx) const {
  if (idx >= snapshots_.size()) {
    throw std::out_of_range("rawDsg index out of range");
  }
  return sourceDsg(idx);
}

void SpatioTemporalMap::moveMeshForward() {
  if (!current_dsg_->hasMesh()) {
    return;
  }

  auto& mesh = *current_dsg_->mesh();
  const auto source = sourceDsg(current_dsg_idx_);
  const auto& src_mesh = *source->mesh();

  if (src_mesh.first_seen_stamps.empty()) {
    return;
  }

  // Find how many vertices should be visible at current_time
  const auto vertex_it = std::upper_bound(src_mesh.first_seen_stamps.begin(),
                                          src_mesh.first_seen_stamps.end(),
                                          current_time_,
                                          std::less<uint64_t>());

  const size_t target_vertices = vertex_it - src_mesh.first_seen_stamps.begin();
  const size_t current_vertices = mesh.numVertices();
  
  // Only proceed if we need to add more vertices
  if (target_vertices <= current_vertices) {
    return;
  }

  // Copy vertex data from source mesh
  mesh.resizeVertices(target_vertices);
  for (size_t i = current_vertices; i < target_vertices; ++i) {
    mesh.setPos(i, src_mesh.pos(i));
    if (mesh.has_colors) {
      mesh.setColor(i, src_mesh.color(i));
    }
    if (mesh.has_timestamps) {
      mesh.setTimestamp(i, src_mesh.timestamp(i));
    }
    if (mesh.has_labels) {
      mesh.setLabel(i, src_mesh.label(i));
    }
    if (mesh.has_first_seen_stamps) {
      mesh.setFirstSeenTimestamp(i, src_mesh.firstSeenTimestamp(i));
    }
  }
  
  updateMeshFaces(mesh, src_mesh);
}

void SpatioTemporalMap::moveAgentForward() {
  // Add nodes starting from the current one.
  const auto source = sourceDsg(current_dsg_idx_);
  const auto src_layer = source->findLayer(
      source->getLayerKey(DsgLayers::AGENTS)->layer, robot_prefix_.key);
  if (!src_layer) {
    return;
  }

  for (const auto& [node_id, node] : src_layer->nodes()) {
    if (current_dsg_->hasNode(node_id)) {
      continue;
    }

    const auto& attrs = node->attributes<spark_dsg::AgentNodeAttributes>();
    if (static_cast<size_t>(attrs.timestamp.count()) > current_time_) {
      return;
    }

    current_dsg_->emplaceNode(node->layer, node_id, attrs.clone());
  }
}

void SpatioTemporalMap::moveObjectsForward() {
  if (!current_dsg_->hasLayer(DsgLayers::OBJECTS)) {
    return;
  }

  const auto source = sourceDsg(current_dsg_idx_);
  const auto& src_layer = source->getLayer(DsgLayers::OBJECTS);

  // Get source mesh for timing information
  if (!source->hasMesh()) {
    LOG(WARNING) << "[moveObjectsForward] No mesh available for object timing";
    return;
  }
  const auto& src_mesh = *source->mesh();

  std::vector<NodeId> objects_to_add;
  std::vector<NodeId> objects_to_remove;

  for (const auto& [id, node] : src_layer.nodes()) {
    const auto& attrs = node->attributes<KhronosObjectAttributes>();
    const bool visible = getObjectEffectiveTime(attrs, src_mesh) <= current_time_ &&
                         (isPresent(attrs, current_time_) ||
                          attrs.mesh.numVertices() > 0);
    if (visible && !current_dsg_->hasNode(id)) {
      objects_to_add.push_back(id);
    } else if (!visible && current_dsg_->hasNode(id)) {
      objects_to_remove.push_back(id);
    }
  }

  for (const auto& id : objects_to_remove) {
    current_dsg_->removeNode(id);
  }

  for (const auto& id : objects_to_add) {
    const auto& node = src_layer.nodes().at(id);
    const auto& attrs = node->attributes<KhronosObjectAttributes>();
    auto new_attrs = attrs.clone();
    // NOTE(lschmid): Clear dynamic attrs, these will be updated separately.
    auto& new_khronos_attrs = reinterpret_cast<KhronosObjectAttributes&>(*new_attrs);
    new_khronos_attrs.trajectory_timestamps.clear();
    new_khronos_attrs.trajectory_positions.clear();
    new_khronos_attrs.dynamic_object_points.clear();
    current_dsg_->emplaceNode(src_layer.id.layer, id, std::move(new_attrs));
  }
}

void SpatioTemporalMap::moveDynamicObjectAttributesForward() {
  if (!current_dsg_->hasLayer(DsgLayers::OBJECTS)) {
    return;
  }

  const auto source = sourceDsg(current_dsg_idx_);
  for (const auto& [id, node] : current_dsg_->getLayer(DsgLayers::OBJECTS).nodes()) {
    auto& attrs = node->attributes<KhronosObjectAttributes>();
    const auto& src_attrs =
        source->getNode(id).attributes<KhronosObjectAttributes>();
    const size_t source_count = trajectoryHistorySize(src_attrs);
    const size_t current_count = trajectoryHistorySize(attrs);
    attrs.trajectory_timestamps.resize(current_count);
    attrs.trajectory_positions.resize(current_count);
    if (source_count == current_count) {
      if (!attrs.dynamic_object_points.empty()) {
        attrs.dynamic_object_points.resize(current_count);
      }
      continue;
    }

    const auto it = std::upper_bound(src_attrs.trajectory_timestamps.begin(),
                                     src_attrs.trajectory_timestamps.begin() + source_count,
                                     current_time_);
    const size_t num_old = current_count;
    const size_t num_new = it - src_attrs.trajectory_timestamps.begin();
    if (num_new <= num_old) {
      continue;
    }
    attrs.trajectory_timestamps.insert(attrs.trajectory_timestamps.end(),
                                       src_attrs.trajectory_timestamps.begin() + num_old,
                                       src_attrs.trajectory_timestamps.begin() + num_new);
    attrs.trajectory_positions.insert(attrs.trajectory_positions.end(),
                                      src_attrs.trajectory_positions.begin() + num_old,
                                      src_attrs.trajectory_positions.begin() + num_new);
    if (!src_attrs.dynamic_object_points.empty() || !attrs.dynamic_object_points.empty()) {
      attrs.dynamic_object_points.resize(num_old);
      for (size_t i = num_old; i < num_new; ++i) {
        if (i < src_attrs.dynamic_object_points.size()) {
          attrs.dynamic_object_points.push_back(src_attrs.dynamic_object_points[i]);
        } else {
          attrs.dynamic_object_points.emplace_back();
        }
      }
    }
  }
}

void SpatioTemporalMap::moveMeshBackward() {
  if (!current_dsg_->hasMesh()) {
    return;
  }

  auto& mesh = *current_dsg_->mesh();
  const auto source = sourceDsg(current_dsg_idx_);
  const auto& src_mesh = *source->mesh();
  
  if (mesh.first_seen_stamps.empty()) {
    return;
  }

  // Prune all vertices that are newer than the robot time.
  // NOTE(lschmid): Vertices and faces are sorted by timestamp in pre-processing.

  // Visibility includes observations at the query timestamp, matching forward
  // replay and trimDsgToTime. lower_bound deleted those vertices, so a backward
  // query returned a different mesh than a forward query of the same time.
  const auto vertex_it = std::upper_bound(mesh.first_seen_stamps.begin(),
                                          mesh.first_seen_stamps.end(),
                                          current_time_,
                                          std::less<uint64_t>());
  const size_t new_vertices = vertex_it - mesh.first_seen_stamps.begin();
  mesh.resizeVertices(new_vertices);
  
  updateMeshFaces(mesh, src_mesh);
}

void SpatioTemporalMap::moveAgentBackward() {
  const auto agent_layer = current_dsg_->findLayer(
      current_dsg_->getLayerKey(DsgLayers::AGENTS)->layer, robot_prefix_.key);
  if (!agent_layer) {
    return;
  }

  // Remove all nodes that are newer than the robot time.
  std::unordered_set<NodeId> nodes_to_remove;
  for (const auto& [node_id, node] : agent_layer->nodes()) {
    if (static_cast<size_t>(node->attributes<spark_dsg::AgentNodeAttributes>().timestamp.count()) >
        current_time_) {
      nodes_to_remove.insert(node_id);
    } else {
      // NOTE(lschmid): The agent nodes are ordered by timestamp.
      break;
    }
  }

  for (const auto& node_id : nodes_to_remove) {
    current_dsg_->removeNode(node_id);
  }
}

void SpatioTemporalMap::moveObjectsBackward() {
  if (!current_dsg_->hasLayer(DsgLayers::OBJECTS)) {
    return;
  }

  // Get source mesh for timing information
  const auto source = sourceDsg(current_dsg_idx_);
  if (!source->hasMesh()) {
    LOG(WARNING) << "[moveObjectsBackward] No mesh available for object timing";
    return;
  }
  const auto& src_mesh = *source->mesh();

  const auto& src_layer = source->getLayer(DsgLayers::OBJECTS);
  std::vector<NodeId> nodes_to_add;
  std::vector<NodeId> nodes_to_remove;

  for (const auto& [id, node] : src_layer.nodes()) {
    const auto& attrs = node->attributes<KhronosObjectAttributes>();
    const bool visible = getObjectEffectiveTime(attrs, src_mesh) <= current_time_ &&
                         (isPresent(attrs, current_time_) ||
                          attrs.mesh.numVertices() > 0);
    if (visible && !current_dsg_->hasNode(id)) {
      nodes_to_add.push_back(id);
    } else if (!visible && current_dsg_->hasNode(id)) {
      nodes_to_remove.push_back(id);
    }
  }

  for (const auto& node_id : nodes_to_remove) {
    current_dsg_->removeNode(node_id);
  }

  for (const auto& id : nodes_to_add) {
    const auto& node = src_layer.nodes().at(id);
    auto new_attrs = node->attributes<KhronosObjectAttributes>().clone();
    auto& attrs = reinterpret_cast<KhronosObjectAttributes&>(*new_attrs);
    attrs.trajectory_timestamps.clear();
    attrs.trajectory_positions.clear();
    attrs.dynamic_object_points.clear();
    current_dsg_->emplaceNode(src_layer.id.layer, id, std::move(new_attrs));
  }
}

void SpatioTemporalMap::trimDsgToTime(TimeStamp target_time) {
  if (!current_dsg_) {
    LOG(WARNING) << "[SpatioTemporalMap] No current_dsg_ in trimDsgToTime";
    return;
  }

  // Note: Object trimming is now handled separately in DSG switching
  // to maintain incremental reveal continuity

  if (current_dsg_->hasMesh()) {
    auto& mesh = *current_dsg_->mesh();
    const auto source = sourceDsg(current_dsg_idx_);
    const auto& src_mesh = *source->mesh();
    
    if (!src_mesh.first_seen_stamps.empty()) {
      const auto vertex_it = std::upper_bound(src_mesh.first_seen_stamps.begin(),
                                              src_mesh.first_seen_stamps.end(),
                                              target_time,
                                              std::less<uint64_t>());
      const size_t num_vertices = vertex_it - src_mesh.first_seen_stamps.begin();      
      mesh.resizeVertices(num_vertices);
    } else {
      LOG(WARNING) << "[SpatioTemporalMap] Mesh first_seen_stamps is empty or not available! "
                      "Cannot perform time-based trimming. Keeping all vertices.";
    }
    
    updateMeshFaces(mesh, src_mesh);
  }
  
  const auto agent_layer = current_dsg_->findLayer(
      current_dsg_->getLayerKey(DsgLayers::AGENTS)->layer, robot_prefix_.key);
  if (agent_layer) {
    std::unordered_set<NodeId> nodes_to_remove;
    for (const auto& [node_id, node] : agent_layer->nodes()) {
      const auto& attrs = node->attributes<spark_dsg::AgentNodeAttributes>();
      if (static_cast<size_t>(attrs.timestamp.count()) > target_time) {
        nodes_to_remove.insert(node_id);
      }
    }
    for (const auto& node_id : nodes_to_remove) {
      current_dsg_->removeNode(node_id);
    }
  }

  // Objects appear based on their effective timestamps
  if (current_dsg_->hasLayer(DsgLayers::OBJECTS)) {
    const auto& objects_layer = current_dsg_->getLayer(DsgLayers::OBJECTS);
    const auto& mesh = current_dsg_->mesh();

    std::vector<NodeId> objects_to_remove;
    size_t visible_count = 0;
    size_t total_count = 0;

    for (const auto& [id, node] : objects_layer.nodes()) {
      total_count++;
      const auto& attrs = node->attributes<KhronosObjectAttributes>();

      // Get the effective time when this object should appear
      uint64_t effective_time = getObjectEffectiveTime(attrs, *mesh);

      // A current-time DSG contains only objects whose estimated presence interval covers the
      // query. Historical nodes remain serialized in their source time step, but a confirmed
      // disappearance must not leak into the current map or be re-seeded into the next session.
      // A node that still materializes CURRENT geometry (non-empty mesh) is never hidden by the
      // estimate: during a relocation transition the reducer keeps the old mesh until the new
      // site is promoted, and hiding that node is what made objects flicker in and out of the
      // timeline view. Only nodes without current geometry follow the presence estimate.
      if (effective_time > target_time ||
          (!isPresent(attrs, target_time) && attrs.mesh.numVertices() == 0)) {
        objects_to_remove.push_back(id);
      } else {
        visible_count++;
      }
    }
    for (const auto& id : objects_to_remove) {
      current_dsg_->removeNode(id);
    }

    // A source snapshot contains complete history known at that snapshot. Crop
    // retained nodes to the query target before incremental replay, otherwise a
    // source switch leaks future trajectory samples into earlier queries.
    for (const auto& [id, node] :
         current_dsg_->getLayer(DsgLayers::OBJECTS).nodes()) {
      (void)id;
      auto& attrs = node->attributes<KhronosObjectAttributes>();
      trimTrajectoryHistory(attrs, target_time);
    }
  }
}

void SpatioTemporalMap::moveDynamicObjectAttributesBackward() {
  if (!current_dsg_->hasLayer(DsgLayers::OBJECTS)) {
    return;
  }

  for (const auto& [id, node] : current_dsg_->getLayer(DsgLayers::OBJECTS).nodes()) {
    (void)id;
    auto& attrs = node->attributes<KhronosObjectAttributes>();
    if (!hasTrajectoryHistory(attrs)) {
      continue;
    }

    if (attrs.trajectory_timestamps.back() < current_time_) {
      continue;
    }
    trimTrajectoryHistory(attrs, current_time_);
  }
}

void SpatioTemporalMap::updateTimingInfo(const DynamicSceneGraph& dsg) {
  // Compute the presence time based on the mesh.
  if (!dsg.hasMesh() || dsg.mesh()->numVertices() == 0) {
    return;
  }
  
  if (!dsg.mesh()->first_seen_stamps.empty()) {
    earliest_ = std::min(earliest_, dsg.mesh()->first_seen_stamps.front());
    latest_ = std::max(latest_, dsg.mesh()->first_seen_stamps.back());
  }
}

void SpatioTemporalMap::finalizeMesh(Mesh& mesh) {
  if (mesh.numVertices() == 0) {
    return;
  }
  
  // Sort by stamps if available
  if (!mesh.first_seen_stamps.empty() && mesh.first_seen_stamps.size() == mesh.numVertices()) {
    const auto sorted_indices = sortIndices(mesh.first_seen_stamps);

    // Sort the mesh vertices and indices for easier addition and removal.
    const Mesh old_mesh = mesh;
    std::unordered_map<size_t, size_t> old_to_new;
    old_to_new.reserve(mesh.numVertices());

    for (size_t i = 0; i < mesh.numVertices(); ++i) {
      mesh.setPos(i, old_mesh.pos(sorted_indices[i]));
      if (mesh.has_colors) {
        mesh.setColor(i, old_mesh.color(sorted_indices[i]));
      }
      if (mesh.has_timestamps) {
        mesh.setTimestamp(i, old_mesh.timestamp(sorted_indices[i]));
      }
      if (mesh.has_labels) {
        mesh.setLabel(i, old_mesh.label(sorted_indices[i]));
      }
      if (mesh.has_first_seen_stamps) {
        mesh.setFirstSeenTimestamp(i, old_mesh.firstSeenTimestamp(sorted_indices[i]));
      }
      old_to_new[sorted_indices[i]] = i;
    }

    // Update the face indices of the old mesh to the new indices.
    for (auto& face : mesh.faces) {
      for (auto& vertex : face) {
        vertex = old_to_new[vertex];
      }
    }
  }
}

bool SpatioTemporalMap::saveZpk(std::string filepath, size_t keyframe) const {
  if (std::filesystem::path(filepath).extension() != ".zpk") {
    if (std::filesystem::path(filepath).extension().empty()) filepath += kExtension;
    filepath += ".zpk";
  }
  static std::atomic<uint64_t> sequence{0};
  const std::filesystem::path destination(filepath);
  const auto temporary = destination.string() + ".tmp." + std::to_string(::getpid()) +
                         "." + std::to_string(sequence.fetch_add(1));
  std::ofstream out;
  try {
    if (keyframe == 0 || keyframe > 1024) {
      throw std::invalid_argument("4D-map keyframe interval must be in [1, 1024]");
    }
    if (snapshots_.size() != stamps_.size()) {
      throw std::runtime_error("4D-map stamps and snapshots have different sizes");
    }
    out.open(temporary, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("Could not create direct delta map");
    out.write(kZpkMagic, kZpkMagicSize);

    std::vector<uint8_t> metadata;
    spark_dsg::serialization::BinarySerializer serializer(&metadata);
    serializer.write(kSerializationVersion);
    serializer.write(config.finalize_incrementally);
    serializer.write(stamps_.size());
    serializer.write(stamps_);
    serializer.write(earliest_);
    serializer.write(latest_);
    serializer.write(finalized_);
    serializer.write(serialization_header_);
    const uint64_t prefix_offset = static_cast<uint64_t>(out.tellp());
    out.write(reinterpret_cast<const char*>(metadata.data()), metadata.size());

    Sha256 source_hash;
    source_hash.update(metadata.data(), metadata.size());
    uint64_t source_size = metadata.size();
    nlohmann::json entries = nlohmann::json::array();
    std::vector<uint8_t> previous;
    for (size_t i = 0; i < snapshots_.size(); ++i) {
      const auto& blob = snapshots_[i];
      if (!blob) throw std::runtime_error("4D-map contains a null snapshot");
      auto current = decodeSnapshot(*blob);
      const bool key = i % keyframe == 0;
      const bool reusable = blob->compressed &&
          (key ? !blob->base : blob->base == snapshots_[i - 1]);
      auto compressed = reusable ? readStoredSnapshot(*blob)
                                 : compressSnapshot(current, key ? nullptr : &previous);
      // Verify the exact bytes written against the output chain, including
      // the copy-on-write case where a stored delta still references an old base.
      verifyEncodedSnapshot(compressed, current, key ? nullptr : &previous);
      const auto header = snapshotHeader(current.size());
      hashTypedSnapshot(source_hash, current);
      source_size += header.size() + 2 * static_cast<uint64_t>(current.size());
      entries.push_back({
          {"header", hexBytes(header.data(), header.size())},
          {"decoded_len", current.size()}, {"kind", key ? "key" : "delta"},
          {"offset", static_cast<uint64_t>(out.tellp())},
          {"length", compressed.size()}, {"sha256", hashBytes(current)}});
      out.write(reinterpret_cast<const char*>(compressed.data()), compressed.size());
      if (!out) throw std::runtime_error("Could not write direct delta snapshot");
      previous = std::move(current);
    }
    auto source_name = destination.filename().string();
    source_name.resize(source_name.size() - 4);  // strip .zpk
    const auto now = std::time(nullptr);
    std::tm utc{};
    ::gmtime_r(&now, &utc);
    std::ostringstream created;
    created << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    const nlohmann::json manifest = {
        {"format", "4dmap-zpk"}, {"format_version", 1},
        {"source", {{"name", source_name}, {"size", source_size},
                    {"sha256", source_hash.finish()}, {"n", stamps_.size()},
                    {"stamps", stamps_}}},
        {"zstd", std::string("libzstd ") + ZSTD_versionString()},
        {"level", 3}, {"keyframe", keyframe},
        {"prefix", {{"offset", prefix_offset}, {"length", metadata.size()}}},
        {"snapshots", std::move(entries)}, {"created", created.str()}};
    const auto json = manifest.dump();
    const uint64_t manifest_offset = static_cast<uint64_t>(out.tellp());
    out.write(json.data(), json.size());
    const auto write_u64 = [&out](uint64_t value) {
      for (size_t i = 0; i < sizeof(value); ++i) {
        out.put(static_cast<char>((value >> (8 * i)) & 0xff));
      }
    };
    write_u64(manifest_offset);
    write_u64(json.size());
    out.write(kZpkTrailerMagic, 8);
    out.flush();
    if (!out) throw std::runtime_error("Could not flush direct delta map");
    out.close();
    if (!out) throw std::runtime_error("Could not close direct delta map");
    std::filesystem::rename(temporary, destination);
    LOG(INFO) << "Saved direct delta 4D map " << destination << " snapshots="
              << stamps_.size() << " compressed_bytes=" << std::filesystem::file_size(destination)
              << " virtual_v1_bytes=" << source_size;
    return true;
  } catch (const std::exception& error) {
    out.close();
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    LOG(ERROR) << "Could not save direct delta 4D map to " << filepath << ": " << error.what();
    return false;
  }
}

std::unique_ptr<SpatioTemporalMap> SpatioTemporalMap::loadZpk(std::string filepath) {
  try {
    std::ifstream in(filepath, std::ios::binary);
    if (!in) throw std::runtime_error("Could not open 4D-map archive");
    in.seekg(0, std::ios::end);
    const auto end = in.tellg();
    if (end < static_cast<std::streamoff>(kZpkMagicSize + kZpkTrailerSize)) {
      throw std::runtime_error("Truncated 4D-map archive");
    }
    const uint64_t file_size = static_cast<uint64_t>(end);
    in.seekg(0);
    char magic[kZpkMagicSize];
    in.read(magic, sizeof(magic));
    if (!in || std::memcmp(magic, kZpkMagic, sizeof(magic))) {
      throw std::runtime_error("Invalid 4D-map archive magic");
    }
    in.seekg(file_size - kZpkTrailerSize);
    const auto manifest_offset = readLittleEndian<uint64_t>(in);
    const auto manifest_size = readLittleEndian<uint64_t>(in);
    char trailer[8];
    in.read(trailer, sizeof(trailer));
    if (!in || std::memcmp(trailer, kZpkTrailerMagic, sizeof(trailer)) ||
        manifest_offset < kZpkMagicSize ||
        manifest_offset > file_size - kZpkTrailerSize ||
        manifest_size != file_size - kZpkTrailerSize - manifest_offset ||
        manifest_size > (uint64_t{1} << 26)) {
      throw std::runtime_error("Invalid 4D-map archive footer bounds");
    }
    const auto read_region = [&in, manifest_offset](uint64_t offset, uint64_t size) {
      if (offset < kZpkMagicSize || offset > manifest_offset ||
          size > manifest_offset - offset) {
        throw std::runtime_error("Invalid 4D-map archive entry bounds");
      }
      std::vector<uint8_t> bytes(size);
      in.seekg(offset);
      in.read(reinterpret_cast<char*>(bytes.data()), size);
      if (!in) throw std::runtime_error("Truncated 4D-map archive entry");
      return bytes;
    };
    std::string json(manifest_size, '\0');
    in.seekg(manifest_offset);
    in.read(json.data(), json.size());
    if (!in) throw std::runtime_error("Truncated 4D-map manifest");
    const auto manifest = nlohmann::json::parse(json);
    if (manifest.at("format") != "4dmap-zpk" || manifest.at("format_version") != 1) {
      throw std::runtime_error("Unsupported 4D-map archive version");
    }
    const auto uint_field = [](const nlohmann::json& value) -> uint64_t {
      if (!value.is_number_unsigned()) {
        throw std::runtime_error("Invalid unsigned 4D-map manifest field");
      }
      return value.get<uint64_t>();
    };
    const auto prefix_offset = uint_field(manifest.at("prefix").at("offset"));
    const auto prefix_size = uint_field(manifest.at("prefix").at("length"));
    if (prefix_size > (uint64_t{1} << 26)) {
      throw std::runtime_error("4D-map metadata prefix is too large");
    }
    const auto prefix = read_region(prefix_offset, prefix_size);
    std::istringstream metadata(
        std::string(reinterpret_cast<const char*>(prefix.data()), prefix.size()), std::ios::binary);
    if (readPackedIntegral<int>(metadata, "serialization version") != kSerializationVersion) {
      throw std::runtime_error("Unsupported embedded 4D-map version");
    }
    Config config;
    config.finalize_incrementally = readPackedBool(metadata, "finalize_incrementally");
    auto result = std::make_unique<SpatioTemporalMap>(config);
    const auto count = readPackedIntegral<size_t>(metadata, "snapshot count");
    result->stamps_ = readPackedIntegralVector<TimeStamp>(metadata, "snapshot timestamps");
    result->earliest_ = readPackedIntegral<TimeStamp>(metadata, "earliest timestamp");
    result->latest_ = readPackedIntegral<TimeStamp>(metadata, "latest timestamp");
    result->finalized_ = readPackedBool(metadata, "finalized");
    result->serialization_header_ = readPackedByteVector(metadata, "spark-dsg header");
    const auto header =
        spark_dsg::io::FileHeader::deserializeFromBinary(result->serialization_header_);
    if (!header) throw std::runtime_error("Invalid embedded spark-dsg header");
    spark_dsg::io::checkCompatibility(*header);
    if (metadata.peek() != std::char_traits<char>::eof() ||
        count != result->stamps_.size() || !manifest.at("snapshots").is_array() ||
        count != manifest.at("snapshots").size() ||
        count != uint_field(manifest.at("source").at("n")) ||
        result->stamps_ != manifest.at("source").at("stamps").get<std::vector<TimeStamp>>() ||
        !std::is_sorted(result->stamps_.begin(), result->stamps_.end())) {
      throw std::runtime_error("Inconsistent 4D-map archive timeline metadata");
    }
    const int fd = ::open(filepath.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) throw std::runtime_error("Could not retain 4D-map archive descriptor");
    std::shared_ptr<SnapshotSpillDirectory> archive;
    try {
      archive = std::make_shared<SnapshotSpillDirectory>(std::filesystem::path{}, fd);
    } catch (...) {
      ::close(fd);
      throw;
    }
    uint64_t virtual_size = prefix.size();
    uint64_t previous_end = prefix_offset + prefix_size;
    for (size_t i = 0; i < count; ++i) {
      const auto& entry = manifest.at("snapshots").at(i);
      const auto offset = uint_field(entry.at("offset"));
      const auto length = uint_field(entry.at("length"));
      const auto decoded = uint_field(entry.at("decoded_len"));
      const std::string kind = entry.at("kind").get<std::string>();
      const std::string digest = entry.at("sha256").get<std::string>();
      const bool valid_digest = digest.size() == 64 &&
          digest.find_first_not_of("0123456789abcdef") == std::string::npos;
      if (offset != previous_end || offset > manifest_offset ||
          length == 0 || length > manifest_offset - offset ||
          decoded > std::numeric_limits<uint32_t>::max() ||
          (kind != "key" && kind != "delta") || (i == 0 && kind != "key") ||
          !valid_digest) {
        throw std::runtime_error("Invalid 4D-map compressed snapshot entry");
      }
      const auto typed_header = snapshotHeader(decoded);
      if (entry.at("header") != hexBytes(typed_header.data(), typed_header.size())) {
        throw std::runtime_error("4D-map snapshot array header mismatch");
      }
      // Check the zstd-declared decoded size before trusting an allocation size
      // supplied by the JSON manifest. Existing packers always include it.
      const auto frame_header = read_region(offset, std::min<uint64_t>(length, 18));
      const auto declared = ZSTD_getFrameContentSize(frame_header.data(), frame_header.size());
      if (declared != decoded) {
        throw std::runtime_error("4D-map snapshot length disagrees with zstd frame header");
      }
      auto blob = std::make_shared<SnapshotBlob>(archive, offset, length);
      blob->size = decoded;
      blob->compressed = true;
      blob->sha256 = digest;
      if (kind == "delta") {
        blob->base = result->snapshots_.back();
        blob->depth = blob->base->depth + 1;
        if (blob->depth >= 1024) throw std::runtime_error("4D-map prefix chain is too long");
      }
      result->snapshots_.push_back(std::move(blob));
      virtual_size += typed_header.size() + 2 * decoded;
      previous_end = offset + length;
    }
    if (previous_end != manifest_offset ||
        virtual_size != uint_field(manifest.at("source").at("size"))) {
      throw std::runtime_error("4D-map archive source length mismatch");
    }
    if (result->earliest_ == std::numeric_limits<TimeStamp>::max() && count) {
      result->earliest_ = result->stamps_.front();
    }
    if (result->latest_ == 0 && count) result->latest_ = result->stamps_.back();
    // Compatible historical graph formats are normalized one snapshot at a
    // time into compressed storage; current archives stay entirely lazy.
    const auto current_header = spark_dsg::io::FileHeader::current();
    if (header->project_name != current_header.project_name || header->version != current_header.version) {
      for (size_t i = 0; i < count; ++i) {
        const auto graph = result->sourceDsg(i);
        result->snapshots_[i] = result->spillSnapshot(*graph);
      }
      result->serialization_header_ = current_header.serializeToBinary();
      result->source_dsg_cache_.reset();
      result->source_dsg_cache_idx_ = std::numeric_limits<size_t>::max();
    }
    return result;
  } catch (const std::exception& error) {
    LOG(ERROR) << "Could not load delta 4D map from " << filepath << ": " << error.what();
    return nullptr;
  }
}

bool SpatioTemporalMap::save(std::string filepath) const {
  if (std::filesystem::path(filepath).extension() == ".zpk") {
    return saveZpk(std::move(filepath));
  }
  // Fix extension if needed.
  if (filepath.find('.') == std::string::npos) {
    filepath += kExtension;
  }

  static std::atomic<uint64_t> save_id{0};
  const std::filesystem::path destination(filepath);
  const std::filesystem::path temporary =
      destination.string() + ".tmp." + std::to_string(::getpid()) + "." +
      std::to_string(save_id.fetch_add(1, std::memory_order_relaxed));

  // Setup file. Publish only after every snapshot has been streamed so a
  // failed save can never masquerade as a complete state artifact.
  std::ofstream out(temporary, std::ios::out | std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    LOG(ERROR) << "Could not open file " << temporary << " for writing.";
    return false;
  }

  try {
    // The metadata and typed-byte graph framing are byte-for-byte compatible
    // with serialization version 1. Only the construction strategy changed:
    // metadata is small and each graph is copied directly from its spill blob.
    std::vector<uint8_t> metadata;
    spark_dsg::serialization::BinarySerializer serializer(&metadata);
    serializer.write(kSerializationVersion);
    serializer.write(config.finalize_incrementally);
    serializer.write(stamps_.size());
    serializer.write(stamps_);
    serializer.write(earliest_);
    serializer.write(latest_);
    serializer.write(finalized_);
    serializer.write(serialization_header_);
    out.write(reinterpret_cast<const char*>(metadata.data()), metadata.size());

    if (snapshots_.size() != stamps_.size()) {
      throw std::runtime_error("4D-map stamps and snapshots have different sizes");
    }
    for (const auto& blob : snapshots_) {
      if (!blob) {
        throw std::runtime_error("4D-map snapshot storage contains an empty entry");
      }
      writePackedByteVector(out, *blob);
    }

    out.flush();
    if (!out.good()) {
      throw std::runtime_error("Failed while flushing the streamed 4D map");
    }
    out.close();

    std::error_code rename_error;
    std::filesystem::rename(temporary, destination, rename_error);
    if (rename_error) {
      throw std::runtime_error("Could not publish streamed 4D map: " +
                               rename_error.message());
    }
    return true;
  } catch (const std::exception& error) {
    out.close();
    std::error_code cleanup_error;
    std::filesystem::remove(temporary, cleanup_error);
    LOG(ERROR) << "Could not save 4D map to " << filepath << ": " << error.what();
    return false;
  }
}

std::unique_ptr<SpatioTemporalMap> SpatioTemporalMap::load(std::string filepath) {
  // Fix extension if needed.
  if (filepath.find('.') == std::string::npos) {
    filepath = filepath + kExtension;
  }

  // Setup file.
  std::ifstream in(filepath, std::ios::in | std::ios::binary);
  if (!in.is_open()) {
    LOG(ERROR) << "Could not open file " << filepath << " for reading.";
    return nullptr;
  }

  // Peek without consuming or seeking: legacy raw maps can arrive through a
  // FIFO. Only archives need a seekable file for their manifest and index.
  if (in.peek() == kZpkMagic[0]) {
    in.close();
    return loadZpk(std::move(filepath));
  }

  try {
    // Version-1 files use a small typed metadata prefix followed by one fixed
    // array of typed uint8 values per graph. Parse that stream incrementally;
    // never allocate a vector the size of the complete .4dmap.
    const int version = readPackedIntegral<int>(in, "serialization version");
    if (version != kSerializationVersion) {
      throw std::runtime_error("Unsupported 4D-map serialization version: " +
                               std::to_string(version));
    }

    Config config;
    config.finalize_incrementally =
        readPackedBool(in, "finalize_incrementally");
    auto result = std::make_unique<SpatioTemporalMap>(config);
    const size_t num_dsgs = readPackedIntegral<size_t>(in, "snapshot count");
    result->stamps_ =
        readPackedIntegralVector<TimeStamp>(in, "snapshot timestamps");
    result->earliest_ = readPackedIntegral<TimeStamp>(in, "earliest timestamp");
    result->latest_ = readPackedIntegral<TimeStamp>(in, "latest timestamp");
    result->finalized_ = readPackedBool(in, "finalized flag");

    const auto header_buffer =
        readPackedByteVector(in, "spark-dsg file header");
    const auto header =
        spark_dsg::io::FileHeader::deserializeFromBinary(header_buffer);
    if (!header) {
      throw std::runtime_error("Invalid spark-dsg file header in 4D map");
    }
    spark_dsg::io::checkCompatibility(*header);

    if (num_dsgs != result->stamps_.size()) {
      throw std::runtime_error(
          "4D-map snapshot count does not match timestamp count");
    }

    const auto current_header = spark_dsg::io::FileHeader::current();
    const bool already_current =
        header->project_name == current_header.project_name &&
        header->version == current_header.version;
    result->snapshots_.reserve(num_dsgs);
    if (already_current) {
      // Fast path: retain the exact graph bytes in anonymous spill files. No
      // graph or whole-map allocation is required during load.
      result->serialization_header_ = header_buffer;
      for (size_t i = 0; i < num_dsgs; ++i) {
        result->snapshots_.push_back(readPackedByteBlob(
            in, result->spill_directory_, "DSG snapshot",
            i % kSnapshotKeyframe == 0 ? nullptr : result->snapshots_.back()));
      }
    } else {
      // Compatible legacy graph encodings are transcoded one snapshot at a
      // time. This preserves bounded memory and prevents a store from mixing
      // snapshots that require different deserialization headers.
      result->serialization_header_ = current_header.serializeToBinary();
      spark_dsg::io::GlobalInfo::ScopedInfo info(*header);
      for (size_t i = 0; i < num_dsgs; ++i) {
        auto bytes = readPackedByteVector(in, "legacy DSG snapshot");
        auto dsg = spark_dsg::io::binary::readGraph(bytes);
        if (!dsg) {
          throw std::runtime_error("Could not decode a legacy DSG snapshot");
        }
        bytes.clear();
        bytes.shrink_to_fit();
        result->snapshots_.push_back(result->spillSnapshot(*dsg));
      }
    }

    // Fix timing info if it wasn't properly saved.
    if (result->earliest_ == std::numeric_limits<TimeStamp>::max() &&
        !result->stamps_.empty()) {
      LOG(WARNING) << "Fixing invalid earliest timestamp in loaded map";
      result->earliest_ =
          *std::min_element(result->stamps_.begin(), result->stamps_.end());
    }
    if (result->latest_ == 0 && !result->stamps_.empty()) {
      LOG(WARNING) << "Fixing invalid latest timestamp in loaded map";
      result->latest_ =
          *std::max_element(result->stamps_.begin(), result->stamps_.end());
    }
    return result;
  } catch (const std::exception& error) {
    LOG(ERROR) << "Could not load 4D map from " << filepath << ": " << error.what();
    return nullptr;
  }
}

std::vector<size_t> sortIndices(const std::vector<uint64_t>& values) {
  std::vector<size_t> idx(values.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(
      idx.begin(), idx.end(), [&values](size_t i1, size_t i2) { return values[i1] < values[i2]; });
  return idx;
}

void SpatioTemporalMap::updateMeshFaces(Mesh& mesh, const Mesh& src_mesh) const {
  const size_t current_num_vertices = mesh.numVertices();
  mesh.faces.clear();
  for (const auto& face : src_mesh.faces) {
    bool all_vertices_present = true;
    for (const auto& vertex_idx : face) {
      if (vertex_idx >= current_num_vertices) {
        all_vertices_present = false;
        break;
      }
    }
    if (all_vertices_present) {
      mesh.faces.push_back(face);
    }
  }
}

uint64_t SpatioTemporalMap::getObjectEffectiveTime(const KhronosObjectAttributes& attrs,
                                                    const Mesh& mesh) const {
  // Presence semantics define zero as "present from the beginning". Object
  // labels are excluded from the global/background TSDF, so an object with a
  // zero start time must not be made contingent on finding nearby background
  // vertices. Only legacy objects with no presence interval at all use the
  // mesh-based fallback below.
  if (!attrs.first_observed_ns.empty()) {
    return attrs.first_observed_ns.front();
  }

  // Legacy object with no explicit timestamp: estimate from nearby mesh.
  if (!mesh.has_first_seen_stamps || mesh.first_seen_stamps.empty()) {
    // No mesh timing info available, hide until we have mesh data
    return std::numeric_limits<uint64_t>::max();
  }

  const auto& bbox = attrs.bounding_box;
  uint64_t min_time = std::numeric_limits<uint64_t>::max();

  // Check mesh vertices within/near the bounding box
  const float margin = 0.5f;  // meters - look slightly beyond bbox
  // Calculate bounding box corners
  Eigen::Vector3f min_corner = bbox.world_P_center - bbox.dimensions * 0.5f;
  Eigen::Vector3f max_corner = bbox.world_P_center + bbox.dimensions * 0.5f;

  for (size_t i = 0; i < mesh.numVertices() && i < mesh.first_seen_stamps.size(); ++i) {
    const auto& pos = mesh.pos(i);
    // Check if vertex is near the object
    if (pos.x() >= min_corner.x() - margin && pos.x() <= max_corner.x() + margin &&
        pos.y() >= min_corner.y() - margin && pos.y() <= max_corner.y() + margin &&
        pos.z() >= min_corner.z() - margin && pos.z() <= max_corner.z() + margin) {
      if (mesh.first_seen_stamps[i] < min_time && mesh.first_seen_stamps[i] > 0) {
        min_time = mesh.first_seen_stamps[i];
      }
    }
  }

  // Return the earliest time mesh vertices appeared near this object
  // If no vertices found, object remains hidden (max timestamp)
  return min_time;
}

}  // namespace khronos
