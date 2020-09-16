#include "model.hpp"

#include <torch/cuda.h>
#include <torch/script.h>
#include <torch/utils.h>

namespace Mcts {

size_t Model::get_cuda_device_count() { return torch::cuda::device_count(); }

void Model::load(const std::string &filename, size_t device_id) {
  torch::init_num_threads();
  torch::set_num_threads(BuildOption::TorchNumIntraThread);
  torch::set_num_interop_threads(BuildOption::TorchNumInterThread);
  // prepare_nvrtc();
  device_id_ = device_id;
  model_ = torch::jit::load(filename, torch::Device(torch::kCUDA, device_id));
  std::cerr << "> Load model on CUDA#" << device_id << std::endl;
}

void Model::resize_batch(size_t batch_size) {
  const auto bsize = static_cast<long>(batch_size);
  state_ = torch::empty({bsize, 1, 1});
  action_ = torch::empty({bsize, 1});
  output_.resize(batch_size);
}

void Model::prepare_forward(size_t batch_index, const ForwardInfo &info) {
  state_[batch_index] = info.state;
  action_[batch_index] = info.action;
}

void Model::forward() {
  auto state = state_.to(torch::Device(torch::kCUDA, device_id_));
  auto action = action_.to(torch::Device(torch::kCUDA, device_id_));
  // (r, s, p, v) = model(s', a)
  const auto &outputs = model_.forward({state, action}).toTuple()->elements();
  const auto &r_tensor = outputs[0].toTensor().to(torch::kCPU);
  const auto &r_view = r_tensor.accessor<float, 2>();
  const auto &s_tensor = outputs[1].toTensor().to(torch::kCPU);
  const auto &p_tensor = outputs[2].toTensor().to(torch::kCPU);
  const auto &p_view = p_tensor.accessor<float, 2>();
  const auto &v_tensor = outputs[3].toTensor().to(torch::kCPU);
  const auto &v_view = v_tensor.accessor<float, 2>();
  // store (r, s, p[], v)
  const auto batch_size = output_.size();
  for (size_t i = 0; i < batch_size; ++i) {
    auto &output = output_[i];
    output.reward = r_view[0][0];
    output.state = s_tensor[i];
    for (size_t j = 0; j < GameOption::ActionDim; ++j) {
      output.policy[j] = p_view[i][j];
    }
    output.value = v_view[0][0];
  }
}

ForwardResult Model::get_result(size_t batch_index) const {
  return output_[batch_index];
}

}  // namespace Mcts