// Chosen snapshots of any 4D map (a piece-history map included) as a plain 4dmap v1 file, the way
// eval_fast/vendor/pack/zpk_snapshot.py cuts a keyframe/delta archive: the map's own flags, the chosen stamps, earliest
// and latest the first and last chosen stamp (SpatioTemporalMap::saveSnapshots). Tools that read one moment at a time
// read this file unchanged. --stamps prints the snapshot stamps, one per line.
// usage: piece_map_cut MAP OUT.4dmap INDEX[,INDEX...]   |   piece_map_cut MAP --stamps
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <khronos/spatio_temporal_map/spatio_temporal_map.h>

int main(int argc, char** argv) {
  if (argc < 3 || (std::string(argv[2]) != "--stamps" && argc != 4)) {
    std::cerr << "usage: piece_map_cut MAP OUT.4dmap INDEX[,INDEX...] | piece_map_cut MAP --stamps\n";
    return 2;
  }
  const auto map = khronos::SpatioTemporalMap::load(argv[1]);
  if (!map) {
    std::cerr << "cannot read " << argv[1] << "\n";
    return 3;
  }
  if (std::string(argv[2]) == "--stamps") {
    for (const auto stamp : map->stamps()) std::cout << stamp << "\n";
    return 0;
  }
  std::vector<size_t> indices;
  std::stringstream list(argv[3]);
  for (std::string item; std::getline(list, item, ',');) {
    const long long i = std::stoll(item);
    indices.push_back(static_cast<size_t>(i < 0 ? static_cast<long long>(map->numTimeSteps()) + i : i));
  }
  return map->saveSnapshots(indices, argv[2]) ? 0 : 4;
}
