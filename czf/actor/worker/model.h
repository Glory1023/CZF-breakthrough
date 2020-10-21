#pragma once
#include <torch/script.h>

#include <array>
#include <atomic>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace czf::actor::worker {

using Model = torch::jit::script::Module;  ///< the type of a model
using ModelPtr = std::shared_ptr<Model>;   ///< the type of a pointer to model

class ModelManager {
 public:
  /** initialize NVRTC (used by PyTorch) */
  static void prepare_nvrtc();
  /** resize the number of model */
  void resize(size_t);
  /** load a model from the path */
  void load(const std::string &);
  /** get the pointer to a model */
  std::tuple<torch::Device, ModelPtr> get();

 private:
  std::vector<torch::Device> forward_devices_;
  // models_[device][version]
  std::vector<std::array<ModelPtr, 2>> models_;
  std::atomic_size_t version_switch_{0u}, device_switch_{0u};
};

}  // namespace czf::actor::worker
