#include <khronos/spatio_temporal_map/spatio_temporal_map.h>

#include <fstream>
#include <array>
#include <vector>

#include <glog/logging.h>
#include <spark_dsg/serialization/binary_serialization.h>
#include <spark_dsg/serialization/graph_binary_serialization.h>
#include <spark_dsg/serialization/versioning.h>

namespace khronos {

bool SpatioTemporalMap::save(std::string filepath) const {
  if (filepath.find('.') == std::string::npos) {
    filepath += kExtension;
  }

  std::ofstream out(filepath, std::ios::out | std::ios::binary);
  if (!out.is_open()) {
    LOG(ERROR) << "Could not open file " << filepath << " for writing.";
    return false;
  }

  // Preserve the official binary layout while avoiding the original full-file
  // memory buffer. Metadata is small; each DSG is serialized, written, and
  // released before the next DSG is processed.
  std::vector<uint8_t> metadata;
  spark_dsg::serialization::BinarySerializer serializer(&metadata);
  serializer.write(kSerializationVersion);
  serializer.write(config.finalize_incrementally);
  serializer.write(stamps_.size());
  serializer.write(stamps_);
  serializer.write(earliest_);
  serializer.write(latest_);
  serializer.write(finalized_);
  serializer.write(spark_dsg::io::FileHeader::current().serializeToBinary());
  out.write(reinterpret_cast<const char*>(metadata.data()), metadata.size());

  for (const auto& dsg : dsgs_) {
    std::vector<uint8_t> dsg_buffer;
    spark_dsg::io::binary::writeGraph(*dsg, dsg_buffer, true);

    std::vector<uint8_t> array_header;
    spark_dsg::serialization::BinarySerializer header_serializer(&array_header);
    header_serializer.startFixedArray(dsg_buffer.size());
    out.write(reinterpret_cast<const char*>(array_header.data()), array_header.size());

    // BinarySerializer encodes every uint8_t as [UINT8 type, value]. Emit the
    // same byte sequence in bounded chunks instead of appending it to the full
    // map buffer.
    std::vector<uint8_t> encoded_sample;
    spark_dsg::serialization::BinarySerializer sample_serializer(&encoded_sample);
    sample_serializer.write(uint8_t{0});
    const uint8_t uint8_type = encoded_sample.at(0);
    constexpr std::size_t kChunkElements = 1U << 20;
    std::vector<uint8_t> encoded_chunk;
    encoded_chunk.resize(2 * std::min(kChunkElements, dsg_buffer.size()));
    for (std::size_t begin = 0; begin < dsg_buffer.size(); begin += kChunkElements) {
      const auto count = std::min(kChunkElements, dsg_buffer.size() - begin);
      encoded_chunk.resize(2 * count);
      for (std::size_t i = 0; i < count; ++i) {
        encoded_chunk[2 * i] = uint8_type;
        encoded_chunk[2 * i + 1] = dsg_buffer[begin + i];
      }
      out.write(reinterpret_cast<const char*>(encoded_chunk.data()), encoded_chunk.size());
    }
    if (!out.good()) {
      LOG(ERROR) << "Failed while streaming map to " << filepath << ".";
      return false;
    }
  }

  out.flush();
  return out.good();
}

}  // namespace khronos
