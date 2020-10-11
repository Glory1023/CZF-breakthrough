#pragma once
#include <array>
#include <list>
#include <memory>

#include "utils/config.h"
#include "utils/model.h"
#include "utils/random.h"

namespace czf::actor::mcts {

using PRNG = czf::actor::utils::random::Xorshift;
using Action_t = int32_t;

struct PolicyResult {
  size_t total_visits = 0u;
  std::array<size_t, GameOption::ActionDim> visits;

  Action_t get_best_move() const;
  Action_t get_softmax_move(PRNG &rng) const;
};

struct TreeInfo {
  float min_value = std::numeric_limits<float>::max(),
        max_value = std::numeric_limits<float>::lowest();

  void update(float);
};

struct MctsInfo {
  size_t visits = 0;
  float sqrt_visits = 0.f;
  float value = 0.f, reward = 0.f;
  std::array<float, GameOption::ActionDim> policy;

  float get_normalized_value(const TreeInfo &) const;
  float update(float);
};

class Node;

struct NodeInfo {
  Action_t action;
  Node *parent = nullptr;
  std::unique_ptr<std::vector<Node>> children = nullptr;
  size_t children_size;
  std::vector<Action_t> legal_actions;

  bool has_children() const;
  void expand();
};

class Node {
 public:
  bool has_children() const;
  Node *get_parent() const;
  float get_value() const;
  const ForwardInfo &get_forward_info() const;

 public:
  void construct_root(State, std::vector<Action_t>);

 public:
  Node *select_child(const TreeInfo &, PRNG &) const;
  void expand_children();
  void expand_dirichlet(PRNG &);
  void set_forward_result(ForwardResult &);
  float update(float);

 private:
  void add_dirichlet_noise(PRNG &);

  ForwardInfo forward_info_;
  NodeInfo node_info_;
  MctsInfo mcts_info_;
};

class Tree {
 public:
  void before_forward(PRNG &);
  void after_forward(PRNG &);

 public:
  void construct_root(State, std::vector<Action_t>);
  const ForwardInfo &get_forward_info() const;
  void set_forward_result(ForwardResult &);

 private:
  Node tree_, *current_node_;
  TreeInfo tree_info_;
};

}  // namespace czf::actor::mcts
