#include "model.h"

#include <torch/utils.h>

#include <sstream>

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

void ModelManager::load_from_bytes(const std::string &bytes) {
  size_t num_devices = forward_devices_.size();
  for (size_t i = 0; i < num_devices; ++i) {
    std::istringstream model_stream(bytes);
    auto model = std::make_shared<Model>(
        torch::jit::load(model_stream, forward_devices_[i]));
    std::atomic_store(&models_[i], model);
  }
}

void ModelManager::load_from_file(const std::string &filename) {
  size_t num_devices = forward_devices_.size();
  for (size_t i = 0; i < num_devices; ++i) {
    auto model = std::make_shared<Model>(
        torch::jit::load(filename, forward_devices_[i]));
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
