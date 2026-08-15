#include <iostream>
#include <string>

#include <khronos/spatio_temporal_map/spatio_temporal_map.h>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: stream_4dmap_copy INPUT OUTPUT\n";
    return 2;
  }

  auto map = khronos::SpatioTemporalMap::load(argv[1]);
  if (!map) {
    std::cerr << "failed to load input map\n";
    return 3;
  }
  std::cout << "loaded_time_steps=" << map->numTimeSteps() << "\n";
  if (map->numTimeSteps() > 0) {
    std::cout << "first_stamp_ns=" << map->stamps().front() << "\n";
    std::cout << "latest_stamp_ns=" << map->stamps().back() << "\n";
  }
  if (!map->save(argv[2])) {
    std::cerr << "failed to save output map\n";
    return 4;
  }
  auto check = khronos::SpatioTemporalMap::load(argv[2]);
  if (!check || check->numTimeSteps() != map->numTimeSteps()) {
    std::cerr << "streamed map validation failed\n";
    return 5;
  }
  std::cout << "validated_time_steps=" << check->numTimeSteps() << "\n";
  return 0;
}
