#pragma once
#include <torch/script.h>

#include <atomic>
#include <memory>
#include <tuple>
#include <vector>

namespace czf::actor::worker {

using Model = torch::jit::script::Module;
using ModelPtr = std::shared_ptr<Model>;

class ModelManager {
 public:
  /** Initialize NVRTC (used by PyTorch) */
  static void prepare_nvrtc();
  /** Resize the number of model */
  void resize(size_t);
  /** Load a model from the path */
  void load(const std::string &);
  /** Get the pointer to a model */
  std::tuple<torch::Device, ModelPtr> get();

 private:
  // models_[device][version]
  std::vector<std::array<ModelPtr, 2>> models_;
  std::atomic_int version_switch_{0}, device_switch_{0};
};

}  // namespace czf::actor::worker
