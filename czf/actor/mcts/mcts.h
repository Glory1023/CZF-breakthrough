#pragma once
#include <memory>
#include <unordered_map>
#include <vector>

#include "utils/config.h"

namespace czf::actor::mcts {

using RNG_t = czf::actor::RNG_t;
using State_t = std::vector<float>;
using Action_t = int32_t;
using Policy_t = std::vector<float>;

struct TreeInfo {
  /** min & max q value on tree */
  float min_value = std::numeric_limits<float>::max(),
        max_value = std::numeric_limits<float>::lowest();

  /** update the min & max value on tree */
  void update(float);
};

struct TreeResult {
  size_t total_visits;                          // simulation counts
  std::unordered_map<Action_t, size_t> visits;  // child visit counts
  float value                                   // root forward value
      ;
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
  size_t action_index,  // action index in the parent policy (usually equals to
                        // the forward action)
      visits = 0;       // visit counts
  float sqrt_visits = 0.F;  // square root of visit count
  float reward = 0.F,       // reward of the dynamics
      value = 0.F,          // Mcts Q-value
      forward_value = 0.F   // forward value
      ;
  Policy_t policy;

  /** get value normalized by `TreeInfo` */
  float get_normalized_value(const TreeInfo &) const;
  /** Mcts update helper function */
  float update(bool, float);
};

class Node;

struct NodeInfo {
  std::vector<Node> children;  // children of the node
  bool has_selected = false,   // check if the node has been expanded (selected)
      is_root_player = true    // check if the same player as the root node
      ;

  /** check if the node can select child */
  bool can_select_child() const;
  /** Mcts expansion helper function */
  void expand(const std::vector<Action_t> &);
};

class Node {
 public:
  /** set player and action */
  void set_player_and_action(bool, Action_t);

 public:
  /** check if the node can select child */
  bool can_select_child() const;
  /** select a child according to the pUCT score */
  Node *select_child(const TreeInfo &, RNG_t &);
  /** expand children according to legal actions */
  void expand_children(const std::vector<Action_t> &);
  /** normalize policy by legal actions (only applies to the root node) */
  void normalize_policy();
  /** add Dirichlet noise to the policy (only applies to the root node) */
  void add_dirichlet_noise(RNG_t &);
  /** get the forward state */
  const State_t &get_forward_state() const;
  /** get the forward action */
  const Action_t &get_forward_action() const;
  /** set the forward result */
  void set_forward_result(ForwardResult);
  /** get the forward value */
  float get_forward_value() const;
  /** update the Mcts Q-value */
  float update(float);
  /** get visit counts */
  size_t get_visits() const;
  /** get children visit counts */
  std::unordered_map<Action_t, size_t> get_children_visits() const;

 private:
  ForwardInfo forward_info_;
  NodeInfo node_info_;
  MctsInfo mcts_info_;
};

class Tree {
 public:
  /** Mcts selection & expansion */
  void before_forward(RNG_t &, const std::vector<Action_t> &);
  /** Mcts update */
  void after_forward();

 public:
  /** select and expand root legal actions */
  void expand_root(const std::vector<Action_t> &);
  /** normalize root policy by legal actions */
  void normalize_root_policy();
  /** add Dirichlet noise to the the root node */
  void add_dirichlet_noise_to_root(RNG_t &);
  /** get the forward input of current node */
  ForwardInfo get_forward_input() const;
  /** set the forward result to current node */
  void set_forward_result(ForwardResult);
  /** get the root simulation counts */
  size_t get_root_visits() const;
  /** get the tree result */
  TreeResult get_tree_result();

 private:
  Node tree_,                  // the root node
      *current_node_ = &tree_  // current node visitor
      ;
  /** Mcts selection path (from root to current node)*/
  std::vector<Node *> selection_path_;
  TreeInfo tree_info_;
};

}  // namespace czf::actor::mcts
