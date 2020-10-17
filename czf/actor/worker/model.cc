#include <torch/utils.h>
// #include <nvrtc.h>

#include "model.h"
#include "utils/config.h"

namespace czf::actor::worker {

void ModelManager::prepare_nvrtc() {
  /*nvrtcProgram prog;
  nvrtcCreateProgram(&prog, " ", "dddddd", 0, nullptr, nullptr);
  nvrtcDestroyProgram(&prog);*/
}

void ModelManager::resize(size_t size) {
  torch::init_num_threads();
  torch::set_num_threads(BuildOption::kTorchNumIntraThread);
  torch::set_num_interop_threads(BuildOption::kTorchNumInterThread);
  models_.resize(size);
  forward_devices_.clear();
  for (size_t i = 0; i < size; ++i) {
    forward_devices_.emplace_back(torch::kCUDA, i);
  }
}

void ModelManager::load(const std::string& path) {
  size_t new_version = version_switch_ ^ 1;  // NOLINT
  size_t num_devices = forward_devices_.size();
  for (size_t i = 0; i < num_devices; ++i) {
    models_[i][new_version] =
        std::make_shared<Model>(torch::jit::load(path, forward_devices_[i]));
  }
  version_switch_ = new_version;
}

std::tuple<torch::Device, ModelPtr> ModelManager::get() {
  size_t num_devices = forward_devices_.size();
  size_t idx = (device_switch_++) % num_devices;
  if (idx == 0) {
    device_switch_ -= num_devices;
  }
  return {forward_devices_[idx], models_[idx][version_switch_]};
}

}  // namespace czf::actor::worker
