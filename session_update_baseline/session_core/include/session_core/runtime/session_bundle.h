#pragma once
#include <cstdint>
#include <filesystem>
#include <string>
#include <set>
#include <vector>
namespace khronos::session_io {
// Claim a fresh output directory before writing any member. The durable marker
// remains after failure or success; a published directory must never be reused.
void beginBundle(const std::filesystem::path& directory);
// README appendix: the record lists each member with its length, the final time and the return
// status of the save; no file hash is recorded.
void publishBundle(const std::filesystem::path& directory, uint64_t stamp,
                   const std::vector<std::string>& files);
// Returns false for an old-format map; a present but invalid bundle throws.
bool verifyBundle(const std::filesystem::path& input, std::set<std::string>* members = nullptr,
                  uint64_t* stamp = nullptr);
}  // namespace khronos::session_io
