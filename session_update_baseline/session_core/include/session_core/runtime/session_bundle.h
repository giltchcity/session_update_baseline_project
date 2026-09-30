#pragma once
#include <cstdint>
#include <filesystem>
#include <string>
#include <set>
#include <vector>
namespace khronos::session_io {
std::string fileSha256(const std::filesystem::path& path);
// Claim a fresh output directory before writing any member. The durable marker
// remains after failure or success; a published directory must never be reused.
void beginBundle(const std::filesystem::path& directory);
void publishBundle(const std::filesystem::path& directory, uint64_t stamp,
                   const std::vector<std::string>& files);
// Returns false for a legacy map; a present but invalid bundle throws.
bool verifyBundle(const std::filesystem::path& input, std::set<std::string>* members = nullptr,
                  uint64_t* stamp = nullptr);
}  // namespace khronos::session_io
