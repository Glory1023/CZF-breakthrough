#pragma once
#include "config.hpp"
#include "model.hpp"
#include "random.hpp"
#include <array>
#include <list>
#include <memory>

namespace Mcts {
using PRNG = ::Utils::Random::Xorshift;

struct PolicyResult {
  size_t total_visits = 0u;
  std::array<size_t, GameOption::ActionDim> visits;

  size_t get_best_move() const;
  size_t get_softmax_move(PRNG &rng) const;
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
  size_t action;
  Node *parent = nullptr;
  std::unique_ptr<std::list<Node>> children = nullptr;
  std::vector<size_t> legal_actions;

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
  const ForwardInfo &before_forward(PRNG &);
  void after_forward(ForwardResult, PRNG &);

private:
  Node tree_, *current_node_;
  TreeInfo tree_info_;
};

class TreeManager {
public:
  void resize_batch();
  void run();

private:
  std::vector<Model> models_;
  std::vector<Tree> trees_;
  PRNG rng_;
};

} // namespace Mcts