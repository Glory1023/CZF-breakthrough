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
  const auto next_toggle = toggle_ ^ 1;  // NOLINT
  size_t num_devices = forward_devices_.size();
  for (size_t i = 0; i < num_devices; ++i) {
    std::istringstream model_stream(bytes);
    models_[i][next_toggle] = std::make_shared<Model>(
        torch::jit::load(model_stream, forward_devices_[i]));
  }
  toggle_ ^= 1;
}

void ModelManager::load_from_file(const std::string &filename) {
  const auto next_toggle = toggle_ ^ 1;  // NOLINT
  size_t num_devices = forward_devices_.size();
  for (size_t i = 0; i < num_devices; ++i) {
    models_[i][next_toggle] = std::make_shared<Model>(
        torch::jit::load(filename, forward_devices_[i]));
  }
  toggle_ ^= 1;
}

std::tuple<torch::Device, ModelPtr> ModelManager::get() {
  size_t num_devices = forward_devices_.size();
  size_t idx = (device_switch_++) % num_devices;
  if (idx == 0) {
    device_switch_ -= num_devices;
  }
  return {forward_devices_[idx], models_[idx][toggle_]};
}

}  // namespace czf::actor::worker
