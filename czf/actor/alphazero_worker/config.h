#pragma once
#include <pcg_random.hpp>
#include <vector>

namespace czf::actor::alphazero_worker {

using RNG_t = pcg32;               ///< the type for the random number generator
using Seed_t = RNG_t::state_type;  ///< the type of random seed

namespace BuildOption {
static constexpr int kTorchNumIntraThread = 1;    ///< PyTorch intra-threads
static constexpr int kTorchNumInterThread = 1;    ///< PyTorch inter-threads
static constexpr size_t kDefaultTimeout = 1000U;  ///< default timeout: 1000ms
}  // namespace BuildOption

struct WorkerOption {
  Seed_t seed;                                      ///< random number seed
  size_t timeout_us = BuildOption::kDefaultTimeout; /**< GPU wait max timeout
                                                       (microseconds)*/
  size_t batch_size;                                ///< GPU max batch size
  size_t num_sampled_transformations;
};

struct TreeOption {
  size_t simulation_count;  ///< simulation counts
  float c_puct,             ///< pUCT constant
      dirichlet_alpha,      ///< Dir(alpha)
      dirichlet_epsilon;    ///< (1 - eps) * p + eps * Dir(a);
};

}  // namespace czf::actor::alphazero_worker
