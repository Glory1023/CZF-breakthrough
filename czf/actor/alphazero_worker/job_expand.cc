#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "czf/env/czf_env/game.h"
#include "job.h"
#include "node.h"
#include "tree.h"

namespace czf::actor::alphazero_worker {

void Job::print_mcts_results(const int top_n) const {
  std::vector<int> children_index;
  const std::vector<std::tuple<float, czf::env::czf_env::Action, Node>>& children =
      tree.root_node.children;
  for (int i = 0; i < children.size(); i++) {
    children_index.push_back(i);
  }
  std::sort(children_index.begin(), children_index.end(), [this](int a, int b) {
    return std::get<2>(tree.root_node.children[a]).num_visits >
           std::get<2>(tree.root_node.children[b]).num_visits;
  });
  int rank = std::min(top_n, (int)children.size());
  std::cerr << std::right << std::setw(6) << "ID" << std::setw(10) << "Action" << std::setw(14)
            << "Prior Policy" << std::setw(12) << "MCTS Count" << std::setw(13) << "MCTS Policy"
            << std::setw(20) << "MCTS Winrate[-1~1]" << std::endl;
  for (int i = 0; i < rank; i++) {
    std::cerr << std::right << std::fixed << std::setprecision(6) << std::setw(6)
              << std::get<1>(children[children_index[i]]) << std::setw(10)
              << engine->game->action_to_string(std::get<1>(children[children_index[i]]))
              << std::setw(14) << std::get<0>(children[children_index[i]]) << std::setw(12)
              << std::get<2>(children[children_index[i]]).num_visits << std::setw(13)
              << (double)std::get<2>(children[children_index[i]]).num_visits /
                     (tree.root_node.num_visits - 1)
              << std::setw(20)
              << std::get<2>(children[children_index[i]]).parent_player_value_sum /
                     std::get<2>(children[children_index[i]]).num_visits
              << std::endl;
  }
}

void Job::dump_mcts() const {}

}  // namespace czf::actor::alphazero_worker