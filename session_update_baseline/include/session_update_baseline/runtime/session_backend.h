#pragma once

#include <string>

#include <hydra/common/shared_module_state.h>
#include <khronos/backend/backend.h>

namespace session_update::runtime {

class SessionBackend : public khronos::Backend {
 public:
  struct Config : public khronos::Backend::Config {
    std::string prior_map;
    std::string prior_seed_map;
  };

  SessionBackend(const Config& config,
                 const hydra::SharedDsgInfo::Ptr& dsg,
                 const hydra::SharedModuleState::Ptr& state);

  void save(const hydra::DataDirectory& log_setup) override;

 private:
  void loadPriorMap(const std::string& history_path, const std::string& seed_path);

  inline static const auto registration_ = config::RegistrationWithConfig<
      hydra::BackendModule,
      SessionBackend,
      Config,
      hydra::SharedDsgInfo::Ptr,
      hydra::SharedModuleState::Ptr>("SessionBackend");
};

void declare_config(SessionBackend::Config& config);

}  // namespace session_update::runtime
