#pragma once
#include <pcg_random.hpp>
#include <vector>

namespace czf::actor::gumbel_muzero_worker {

using RNG_t = pcg32;               ///< the type for the random number generator
using Seed_t = RNG_t::state_type;  ///< the type of random seed

namespace BuildOption {
static constexpr int kTorchNumIntraThread = 1;    ///< PyTorch intra-threads
static constexpr int kTorchNumInterThread = 1;    ///< PyTorch inter-threads
static constexpr size_t kDefaultTimeout = 1000U;  ///< default timeout: 1000ms
}  // namespace BuildOption

struct GameInfo {
  std::vector<int64_t> observation_shape;    ///< root observation shape
  std::vector<int64_t> state_shape;          ///< internal state shape
  std::vector<int32_t> all_actions;          ///< all possible actions
  size_t num_actions;                        ///< number of all possible actions
  std::vector<int32_t> all_chance_outcomes;  ///< all possible chance outcomes
  size_t num_chance_outcomes;                ///< number of all possible chance outcomes
  bool is_two_player;                        ///< single-player or two-player
  bool is_stochastic;                        ///< whether is stochastis game
};

struct WorkerOption {
  Seed_t seed;                                      ///< random number seed
  size_t timeout_us = BuildOption::kDefaultTimeout; /**< GPU wait max timeout
                                                       (microseconds)*/
  size_t batch_size;                                ///< GPU max batch size
};

struct TreeOption {
  size_t simulation_count;           ///< simulation counts
  float tree_min_value,              ///< default min q value on tree
      tree_max_value,                ///< default max q value on tree
      c_puct,                        ///< pUCT constant
      dirichlet_alpha,               ///< Dir(alpha)
      dirichlet_epsilon,             ///< (1 - eps) * p + eps * Dir(a);
      discount;                      ///< discount factor of the return
  size_t gumbel_sampled_actions;     ///< number of sampled actions
  float gumbel_c_visit,              ///< gumbel muzero transformation constant
      gumbel_c_scale;                ///< gumbel muzero transformation constant
  bool gumbel_use_noise,             ///< whether to use gumbel noise
      gumbel_use_best_action_value,  ///< whether to use the value of best action for root value
      gumbel_use_simple_loss;        ///< whether to use simple loss
};

enum class GpuWorkerType {
  kWorkerRoot,
  kWorkerState,
  kWorkerAfterstate,
};

}  // namespace czf::actor::gumbel_muzero_worker
