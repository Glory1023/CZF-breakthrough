#pragma once

#include <random>
#include <tuple>
#include <vector>

#include "config.h"
#include "czf/env/czf_env/game.h"

namespace czf::actor::alphazero_worker {

class Node {
 public:
  Node();
  std::tuple<czf::env::czf_env::Action, Node*> select(std::mt19937& rng,
                                                      const TreeOption& tree_option) const;
  void expand(const std::vector<czf::env::czf_env::Action>& legal_actions);
  void reset();

  ~Node() = default;

  int num_visits;
  float parent_player_value_sum;
  float current_player_value_sum;
  // prior, action, child
  std::vector<std::tuple<float, czf::env::czf_env::Action, Node>> children;
};

}  // namespace czf::actor::alphazero_worker
