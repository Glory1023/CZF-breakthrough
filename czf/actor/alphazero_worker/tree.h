#pragma once

#include "config.h"
#include "node.h"

namespace czf::actor::alphazero_worker {

class Job;

class Tree {
 public:
  Tree() = default;
  int num_simulations() const;
  void add_dirichlet_noise(std::mt19937& rng);
  void reset();
  void set_option(const TreeOption& tree_option);
  ~Tree() = default;

  Node root_node;
  TreeOption tree_option;
};

}  // namespace czf::actor::alphazero_worker