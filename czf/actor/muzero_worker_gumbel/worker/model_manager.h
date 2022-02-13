#pragma once
#include <torch/script.h>

#include <array>
#include <atomic>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace czf::actor::muzero_worker_gumbel::worker {

using Model = torch::jit::script::Module;  ///< the type of a model
using ModelPtr = std::shared_ptr<Model>;   ///< the type of a pointer to model

class ModelManager {
 public:
  /** resize the number of model */
  void resize(size_t);
  /** load a model from bytes */
  void load_from_bytes(const std::string &);
  /** load a model from the path */
  void load_from_file(const std::string &);
  /** get the pointer to a model */
  std::tuple<torch::Device, ModelPtr> get();

 private:
  /** models on forward devices */
  std::vector<std::array<ModelPtr, 2>> models_;
  /** all devices to forward */
  std::vector<torch::Device> forward_devices_;
  /** the latest device index */
  std::atomic_size_t device_switch_{0U};
  /** the latest model pointer */
  std::atomic_size_t toggle_{0U};
};

}  // namespace czf::actor::muzero_worker_gumbel::worker
