#pragma once

#include <random>
#include <tuple>
#include <vector>

#include "czf/env/czf_env/game.h"
#include "node.h"
#include "tree.h"

namespace czf::actor::alphazero_worker {

class Job {
 public:
  enum Step { kEnqueue, kSelect, kEvaluate, kUpdate, kDone, kDequeue };

  Job();
  void preprocess();
  void select(std::mt19937& rng);
  void update(std::mt19937& rng);
  void postprocess(size_t num_actions);

  Step next_step;

  // select
  czf::env::czf_env::StatePtr leaf_state;
  std::vector<float> leaf_observation;
  // previous player, current player, node ptr
  std::vector<std::tuple<czf::env::czf_env::Player, czf::env::czf_env::Player, Node*>>
      selection_path;
  // evaluate
  std::vector<float> leaf_policy;
  std::vector<float> leaf_returns;
  // update
  Tree tree;
  // play
  czf::env::czf_env::StatePtr root_state;
  // report
  std::string job_str;
};

}  // namespace czf::actor::alphazero_worker