#pragma once
#include <torch/script.h>

#include <array>
#include <vector>

#include "utils/config.h"

namespace czf::actor::mcts {
using State = torch::Tensor;

struct ForwardInfo {
  State state;
  torch::Scalar action;
};

struct ForwardResult {
  State state;
  std::array<float, GameOption::ActionDim> policy;
  float value, reward;
};

class Model {
 public:
  static size_t get_cuda_device_count();

  void load(const std::string &, size_t);

  void resize_batch(size_t);

  void prepare_forward(size_t, const ForwardInfo &);

  void forward();

  ForwardResult get_result(size_t) const;

 private:
  size_t device_id_;
  torch::jit::script::Module model_;
  torch::Tensor state_, action_;
  std::vector<ForwardResult> output_;
};
}  // namespace czf::actor::mcts
