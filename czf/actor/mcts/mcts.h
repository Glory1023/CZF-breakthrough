#pragma once
#include <memory>
#include <unordered_map>
#include <vector>

#include "utils/config.h"
#include "utils/random.h"

namespace czf::actor::mcts {

using PRNG = czf::actor::utils::random::Xorshift;
using State_t = std::vector<float>;
using Action_t = int32_t;
using Policy_t = std::vector<float>;

struct MctsOption {
  float C_PUCT = 1.25F,          // pUCT constant
      dirichlet_alpha = .03F,    // Dir(a)
      dirichlet_epsilon = .25F,  // (1 - eps) * p + eps * Dir(a);
      discount = 1.F             // reward discount factor
      ;
};

struct TreeInfo {
  float min_value = std::numeric_limits<float>::max(),
        max_value = std::numeric_limits<float>::lowest();

  /** update the min & max value on tree */
  void update(float);
};

struct TreeResult {
  size_t total_visits;                          // simulation counts
  std::unordered_map<Action_t, size_t> visits;  // child visit counts
};

struct ForwardInfo {
  State_t state;
  Action_t action;
};

struct ForwardResult {
  State_t state;
  Policy_t policy;
  float value, reward;
};

struct MctsInfo {
  size_t visits = 0;
  float sqrt_visits = 0.F;
  float value = 0.F, reward = 0.F;
  Policy_t policy;

  /** Get value normalized by `TreeInfo` */
  float get_normalized_value(const TreeInfo &) const;
  /** Mcts update helper function */
  float update(float);
};

class Node;

struct NodeInfo {
  std::vector<Node> children;
  bool has_selected = false;

  /** Check if the node select child */
  bool can_select_child() const;
  /** Mcts expansion helper function */
  void expand(const std::vector<Action_t> &);
};

class Node {
 public:
  /** Check if the node can select child */
  bool can_select_child() const;
  /** Get the value */
  float get_value() const;
  /** Get visit counts */
  size_t get_visits() const;
  /** Get children visit counts */
  std::unordered_map<Action_t, size_t> get_children_visits() const;
  /** Set the action */
  void set_action(size_t);

 public:
  /** Select a child according to the pUCT score */
  Node *select_child(const TreeInfo &, PRNG &);
  /** Expand children according to legal actions */
  void expand_children(const std::vector<Action_t> &);
  /** Add Dirichlet noise to the policy (only applies to the root node) */
  void add_dirichlet_noise(PRNG &);
  /** Get the forward state */
  const State_t &get_forward_state() const;
  /** Get the forward action */
  const Action_t &get_forward_action() const;
  /** Set the forward result */
  void set_forward_result(ForwardResult);
  /** Set the value */
  float update(float);

 private:
  ForwardInfo forward_info_;
  NodeInfo node_info_;
  MctsInfo mcts_info_;
};

class Tree {
 public:
  static MctsOption option;

 public:
  /** Mcts selection & expansion */
  void before_forward(PRNG &, const std::vector<Action_t> &);
  /** Mcts update */
  void after_forward();

 public:
  /** Expand root legal actions */
  void expand_root(const std::vector<Action_t> &);
  /** Add Dirichlet noise to the the root node */
  void add_dirichlet_noise(PRNG &);
  /** Get the root simulation counts */
  size_t get_root_visits() const;
  /** Get the forward input */
  ForwardInfo get_forward_input() const;
  /** Set the forward result */
  void set_forward_result(ForwardResult);
  /** Get the tree result */
  TreeResult get_tree_result();

 private:
  Node tree_, *current_node_ = &tree_;
  TreeInfo tree_info_;
  std::vector<Node *> selection_path_;
};

}  // namespace czf::actor::mcts
