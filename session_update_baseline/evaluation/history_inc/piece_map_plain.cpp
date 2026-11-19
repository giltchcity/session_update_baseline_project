// Any 4D map, a piece-history map included, as a plain serialization-version-1 4D map with the map's own time range and
// flags (SpatioTemporalMap::savePlain), for the tools built before the piece format that read a whole map
// (eval/chain_fill's chain_fill, change_fill, motion_fill, timeline_export; the official Khronos evaluator). OUT may be
// a pipe.
// usage: piece_map_plain MAP OUT.4dmap
#include <iostream>

#include <khronos/spatio_temporal_map/spatio_temporal_map.h>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: piece_map_plain MAP OUT.4dmap\n";
    return 2;
  }
  const auto map = khronos::SpatioTemporalMap::load(argv[1]);
  if (!map) {
    std::cerr << "cannot read " << argv[1] << "\n";
    return 3;
  }
  return map->savePlain(argv[2]) ? 0 : 4;
}
