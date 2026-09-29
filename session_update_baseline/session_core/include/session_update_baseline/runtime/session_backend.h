#pragma once

#include <string>

#include <hydra/common/shared_module_state.h>
#include <khronos/backend/backend.h>

namespace session_update::runtime {

class SessionBackend : public khronos::Backend {
 public:
  struct Config : public khronos::Backend::Config {
    // The one recursive state input, produced by a previous run_session call.
    std::string input_state;
  };

  SessionBackend(const Config& config,
                 const hydra::SharedDsgInfo::Ptr& dsg,
                 const hydra::SharedModuleState::Ptr& state);

 private:
  void loadInputState(const std::string& state_path);

  inline static const auto registration_ = config::RegistrationWithConfig<
      hydra::BackendModule,
      SessionBackend,
      Config,
      hydra::SharedDsgInfo::Ptr,
      hydra::SharedModuleState::Ptr>("SessionBackend");
};

void declare_config(SessionBackend::Config& config);

}  // namespace session_update::runtime
