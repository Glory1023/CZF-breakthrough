#pragma once

namespace czf::actor {

namespace BuildOption {
const constexpr int TorchNumIntraThread = 1;  // number of PyTorch intra-threads
const constexpr int TorchNumInterThread = 1;  // number of PyTorch inter-threads
const constexpr float FloatEps = 1e-9;        // floating point epsilon
}  // namespace BuildOption

struct GameInfo {
  std::vector<int64_t> observation_shape;  // root observation shape
  std::vector<int64_t> state_shape;        // internal state shape
  std::vector<int32_t> all_actions;        // all possible actions
  size_t num_actions;                      // number of all possible actions
  bool is_two_player = true;               // single-player or two-player
};

struct JobOption {
  size_t batch_size = 200u,    // GPU batch size
      simulation_count = 800u  // Mcts simulation counts
      ;
};

struct MctsOption {
  float C_PUCT = 1.25F,          // pUCT constant
      dirichlet_alpha = .03F,    // Dir(a)
      dirichlet_epsilon = .25F,  // (1 - eps) * p + eps * Dir(a);
      discount = 1.F             // reward discount factor
      ;
};

}  // namespace czf::actor
