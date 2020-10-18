#pragma once
#include <pcg_random.hpp>
#include <vector>

namespace czf::actor {

using RNG_t = pcg32;

namespace BuildOption {
const constexpr int kTorchNumIntraThread = 1;  // PyTorch intra-threads
const constexpr int kTorchNumInterThread = 1;  // PyTorch inter-threads
const constexpr float kFloatEps = 1e-9;        // floating point epsilon
}  // namespace BuildOption

struct GameInfo {
  std::vector<int64_t> observation_shape;  // root observation shape
  std::vector<int64_t> state_shape;        // internal state shape
  std::vector<int32_t> all_actions;        // all possible actions
  size_t num_actions;                      // number of all possible actions
  bool is_two_player;                      // single-player or two-player
};

struct JobOption {
  uint64_t seed;              // random number seed
  size_t timeout_us = 1000u,  // GPU wait max timeout (microseconds)
      batch_size,             // GPU max batch size
      simulation_count        // Mcts simulation counts
      ;
};

struct MctsOption {
  // TODO: TreeInfo min & max value
  // float tree_min_value = std::numeric_limits<float>::max(),
  //      tree_max_value = std::numeric_limits<float>::lowest();
  // Mcts
  float C_PUCT,           // pUCT constant
      dirichlet_alpha,    // Dir(alpha)
      dirichlet_epsilon,  // (1 - eps) * p + eps * Dir(a);
      discount = 1.F      // discount factor of the return
      ;
};

}  // namespace czf::actor
