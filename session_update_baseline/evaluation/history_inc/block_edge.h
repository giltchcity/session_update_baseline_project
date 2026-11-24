// The edge of a voxel block of a run (volumetric_map voxel_size x voxels_per_side), read from the run's config.txt.
#pragma once
#include <filesystem>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>

inline double readBlockEdge(const std::filesystem::path& config) {
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
    if (voxel > 0.0 && per_side > 0) return voxel * per_side;
  }
  throw std::runtime_error("no volumetric_map voxel_size and voxels_per_side in " + config.string());
}
