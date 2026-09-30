#include "session_core/runtime/session_bundle.h"

#include <array>
#include <fstream>
#include <iomanip>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>

namespace khronos::session_io {
namespace {
using Json = nlohmann::json;
constexpr const char* kStarted = "session_bundle.started";
constexpr const char* kManifest = "session_bundle.json";
bool entryExists(const std::filesystem::path& path) {
  return std::filesystem::exists(std::filesystem::symlink_status(path));
}
void synchronize(const std::filesystem::path& path, bool directory=false) {
  const int fd=::open(path.c_str(),O_RDONLY | (directory ? O_DIRECTORY : 0));
  if (fd<0) throw std::system_error(errno,std::generic_category(),"Opening session output for synchronization");
  const int result=::fsync(fd);
  const int error=errno;
  ::close(fd);
  if (result!=0) throw std::system_error(error,std::generic_category(),"Synchronizing session output");
}
void checkName(const std::string& name) {
  if (name.empty() || name=="." || name==".." ||
      std::filesystem::path(name).filename().string()!=name)
    throw std::invalid_argument("Invalid bundle member name");
}
const std::set<std::string> required{
    "final.4dmap.zpk","chain_state.4dmap.zpk","shown_state.4dmap.zpk",
    "registry_state.cbor", "evidence_state.cbor", "surface_error.bin",
    "sensor_statistics.txt", "depth_scales.txt"};
}

void beginBundle(const std::filesystem::path& directory) {
  if (!std::filesystem::is_directory(directory))
    throw std::invalid_argument("Session output directory does not exist");
  if (entryExists(directory/kManifest))
    throw std::runtime_error("Cannot overwrite a completed session bundle");
  for (const auto& name:required) {
    if (entryExists(directory/name))
      throw std::runtime_error("Cannot overwrite an existing session output: "+name);
  }
  // O_EXCL serializes all publishers using this protocol, including failed runs.
  // Keep the marker on every error so readers cannot mistake a partial run for legacy.
  const auto marker=directory/kStarted;
  const int fd=::open(marker.c_str(),O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,0644);
  if (fd<0) throw std::system_error(errno,std::generic_category(),"Claiming session output directory");
  const int result=::fsync(fd);
  const int error=errno;
  const int closed=::close(fd);
  const int close_error=errno;
  if (result!=0) throw std::system_error(error,std::generic_category(),"Synchronizing session publication marker");
  if (closed!=0) throw std::system_error(close_error,std::generic_category(),"Closing session publication marker");
  synchronize(directory,true);
}

std::string fileSha256(const std::filesystem::path& path) {
  std::ifstream in(path,std::ios::binary);
  if (!in) throw std::runtime_error("Cannot read session output: "+path.string());
  std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> hash(EVP_MD_CTX_new(),EVP_MD_CTX_free);
  if (!hash || EVP_DigestInit_ex(hash.get(),EVP_sha256(),nullptr)!=1)
    throw std::runtime_error("Cannot initialize session digest");
  std::array<char,65536> buffer;
  while (in) {
    in.read(buffer.data(),buffer.size());
    if (EVP_DigestUpdate(hash.get(),buffer.data(),in.gcount())!=1)
      throw std::runtime_error("Cannot update session digest");
  }
  if (!in.eof()) throw std::runtime_error("Cannot finish session output read");
  std::array<unsigned char,EVP_MAX_MD_SIZE> digest;
  unsigned size=0;
  if (EVP_DigestFinal_ex(hash.get(),digest.data(),&size)!=1)
    throw std::runtime_error("Cannot finish session digest");
  std::ostringstream out;
  for (unsigned i=0;i<size;++i)
    out<<std::hex<<std::setw(2)<<std::setfill('0')<<static_cast<unsigned>(digest[i]);
  return out.str();
}

void publishBundle(const std::filesystem::path& directory, uint64_t stamp,
                   const std::vector<std::string>& files) {
  if (!std::filesystem::is_regular_file(directory/kStarted))
    throw std::runtime_error("Session output directory was not claimed before writing");
  if (entryExists(directory/kManifest))
    throw std::runtime_error("Cannot overwrite a completed session bundle");
  Json members=Json::object();
  for (const auto& name:files) {
    checkName(name);
    if (members.contains(name)) throw std::invalid_argument("Duplicate bundle member");
    const auto path=directory/name;
    if (!std::filesystem::is_regular_file(path))
      throw std::runtime_error("Missing required session output: "+path.string());
    synchronize(path);
    members[name]={{"bytes",std::filesystem::file_size(path)},{"sha256",fileSha256(path)}};
  }
  for (const auto& name:required)
    if (!members.contains(name)) throw std::invalid_argument("Incomplete session output set");
  const Json bundle{{"schema",1},{"stamp",stamp},{"files",std::move(members)}};
  const auto temporary=directory/"session_bundle.json.tmp";
  std::ofstream out(temporary,std::ios::trunc);
  out.exceptions(std::ios::badbit | std::ios::failbit);
  out<<bundle.dump(2)<<'\n'; out.close();
  synchronize(temporary);
  std::filesystem::rename(temporary,directory/kManifest);
  synchronize(directory,true);
}

bool verifyBundle(const std::filesystem::path& input, std::set<std::string>* members,
                  uint64_t* stamp) {
  if (members) members->clear();
  if (stamp) *stamp=0;
  const auto directory=input.parent_path();
  const auto manifest=directory/kManifest;
  if (!entryExists(manifest)) {
    // A legacy map must already exist before checking the publication marker.
    // Otherwise a reader could classify an empty directory as legacy, then open
    // the first final map written by a publisher that started in between.
    auto legacy_input=input;
    if (input.string().find('.')==std::string::npos) legacy_input += ".4dmap";
    if (!std::filesystem::exists(legacy_input))
      throw std::runtime_error("Requested session map does not exist: "+legacy_input.string());
    if (entryExists(directory/kStarted) || entryExists(directory/"registry_state.cbor") ||
        entryExists(directory/"evidence_state.cbor"))
      throw std::runtime_error("Session output has no completed bundle publication");
    return false;
  }
  std::ifstream in(manifest);
  if (!in) throw std::runtime_error("Cannot read session bundle");
  const auto bundle=Json::parse(in);
  if (bundle.at("schema").get<unsigned>()!=1) throw std::invalid_argument("Unsupported session bundle schema");
  if (!bundle.at("stamp").is_number_unsigned())
    throw std::invalid_argument("Invalid session bundle timestamp");
  const auto bundle_stamp=bundle.at("stamp").get<uint64_t>();
  const auto& files=bundle.at("files");
  if (!files.is_object()) throw std::invalid_argument("Invalid bundle file table");
  for (const auto& name:required)
    if (!files.contains(name)) throw std::invalid_argument("Incomplete session bundle");
  const auto name=input.filename().string();
  if ((name!="final.4dmap.zpk" && name!="chain_state.4dmap.zpk" && name!="shown_state.4dmap.zpk") ||
      !files.contains(name)) throw std::invalid_argument("Requested map is not a member of this session bundle");
  for (auto it=files.begin();it!=files.end();++it) {
    checkName(it.key());
    const auto path=directory/it.key();
    if (!std::filesystem::is_regular_file(path) ||
        std::filesystem::file_size(path)!=it.value().at("bytes").get<uintmax_t>() ||
        fileSha256(path)!=it.value().at("sha256").get<std::string>())
      throw std::runtime_error("Session bundle member mismatch: "+it.key());
    if (members) members->insert(it.key());
  }
  if (stamp) *stamp=bundle_stamp;
  return true;
}
}  // namespace khronos::session_io
