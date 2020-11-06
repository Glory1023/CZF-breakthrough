#include "model.h"

#include <torch/utils.h>

#include "utils/config.h"

namespace czf::actor::worker {

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

void ModelManager::load_from_file(const std::string &path) {
  size_t num_devices = forward_devices_.size();
  for (size_t i = 0; i < num_devices; ++i) {
    auto model =
        std::make_shared<Model>(torch::jit::load(path, forward_devices_[i]));
    std::atomic_store(&models_[i], model);
  }
}

std::tuple<torch::Device, ModelPtr> ModelManager::get() {
  size_t num_devices = forward_devices_.size();
  size_t idx = (device_switch_++) % num_devices;
  if (idx == 0) {
    device_switch_ -= num_devices;
  }
  return {forward_devices_[idx], models_[idx]};
}

}  // namespace czf::actor::worker
