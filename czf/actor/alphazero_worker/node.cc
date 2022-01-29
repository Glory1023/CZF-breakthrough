#include "node.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#include "config.h"

namespace czf::actor::alphazero_worker {

Node::Node() : num_visits(0), parent_player_value_sum(0.0F), current_player_value_sum(0.0F) {}

std::tuple<czf::env::czf_env::Action, Node*> Node::select(std::mt19937& rng,
                                                          const TreeOption& tree_option) const {
  const auto& C_PUCT = tree_option.c_puct;
  const auto& EPS = 1e-3;

  int num_candidates = 0;
  czf::env::czf_env::Action selected_action;
  Node* selected_node = nullptr;
  float best_score = std::numeric_limits<float>::lowest();

  const float DEFAULT_Q = current_player_value_sum / num_visits;

  for (const auto& [p, action, child] : children) {
    const float q =
        child.num_visits != 0 ? child.parent_player_value_sum / child.num_visits : DEFAULT_Q;
    const float u = C_PUCT * p * std::sqrt(num_visits) / (1 + child.num_visits);
    const float score = q + u;

    if (score >= best_score + EPS) {
      num_candidates = 1;
      best_score = score;
      selected_action = action;
      selected_node = const_cast<Node*>(&child);
    } else if (score > best_score - EPS) {
      ++num_candidates;
      if (std::uniform_int_distribution<int>{0, num_candidates - 1}(rng) == 0) {
        selected_action = action;
        selected_node = const_cast<Node*>(&child);
      }
    }
  }

  return {selected_action, selected_node};
}

void Node::expand(const std::vector<czf::env::czf_env::Action>& legal_actions) {
  children.reserve(legal_actions.size());
  for (const auto& action : legal_actions) children.emplace_back(0.0F, action, Node{});
}

void Node::reset() {
  num_visits = 0;
  parent_player_value_sum = 0.0F;
  current_player_value_sum = 0.0F;
  children.clear();
}

}  // namespace czf::actor::alphazero_worker
