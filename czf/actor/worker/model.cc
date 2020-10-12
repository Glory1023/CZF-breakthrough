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
  torch::set_num_threads(BuildOption::TorchNumIntraThread);
  torch::set_num_interop_threads(BuildOption::TorchNumInterThread);
  models_.resize(size);
}

void ModelManager::load(const std::string &path) {
  int version = version_switch_ ^ 1;
  size_t num_devices = models_.size();
  for (size_t i = 0; i < num_devices; ++i) {
    models_[i][version] = std::make_shared<Model>(
        torch::jit::load(path, torch::Device(torch::kCUDA, i)));  // NOLINT
  }
  version_switch_ ^= 1;
}

std::tuple<torch::Device, ModelPtr> ModelManager::get() {
  size_t num_devices = models_.size();
  int idx = (device_switch_++) % num_devices;
  if (idx == 0) {
    device_switch_ -= num_devices;
  }
  return {torch::Device(torch::kCUDA, idx), models_[idx][version_switch_]};
}

}  // namespace czf::actor::worker
