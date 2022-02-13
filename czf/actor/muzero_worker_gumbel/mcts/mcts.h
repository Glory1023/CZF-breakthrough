#pragma once
#include <memory>
#include <unordered_map>
#include <vector>

#include "utils/config.h"

namespace czf::actor::muzero_worker_gumbel::mcts {

using RNG_t = czf::actor::muzero_worker_gumbel::RNG_t;  ///< the type for the random
                                                        ///< number generator
using State_t = std::vector<float>;                     ///< the type of a state
using Action_t = int32_t;                               ///< the type of an action
using Policy_t = std::vector<float>;                    ///< the type of a policy

struct TreeInfo {
  float min_value,  ///< min q value on tree
      max_value;    ///< max q value on tree

  /** update the min & max value on tree */
  void update(float);
  /** get value normalized by `TreeInfo` */
  float get_normalized_value(float) const;
};

struct TreeResult {
  // size_t total_visits;                                  ///< simulation counts
  // std::unordered_map<Action_t, size_t> visits;          ///< child visit counts

  size_t selected_action;                               ///< best action by plannig with gumbel
  std::unordered_map<Action_t, float> improved_policy;  ///< improved policy
  float value;                                          ///< root forward value
};

struct ForwardInfo {
  /** \f$f(\textbf{observation})\f$ or \f$g(\textbf{state}, \text{action})\f$ */
  State_t state;
  /** \f$g(\text{state}, \textbf{action})\f$ */
  Action_t action;
};

struct ForwardResult {
  /// \f$\textbf{next_state}, \text{reward} = g(\text{state}, \text{action})\f$
  State_t state;
  /** \f$\textbf{policy}, \text{value} =f(\text{state})\f$ */
  Policy_t policy;
  float value,  ///< \f$\text{policy}, \textbf{value} =f(\text{state})\f$
      reward;   /**< \f$\text{next_state}, \textbf{reward} = g(\text{state},
                   \text{action})\f$ */
};

struct MctsInfo {
  size_t action_index,      /**< action index in the parent policy (usually equals to
                               the forward action)*/
      visits = 0;           ///< visit counts
  float sqrt_visits = 0.F,  ///< square root of visit count
      reward,               ///< reward of the dynamics
      value = 0.F,          ///< Mcts Q-value
      forward_value;        ///< forward value
  Policy_t policy;          ///< policy of children
  Policy_t gumbel_noise;

  /** Mcts update helper function */
  float update(float, bool, bool, float);
};

class Node;

struct NodeInfo {
  /// children of the node
  std::vector<Node> children;
  /// check if the current player to play is the same as the root player to play
  bool is_root_player = true;

  /** check if the node can select child */
  bool can_select_child() const;
  /** Mcts expansion helper function */
  void expand(const std::vector<Action_t> &, bool);
};

class Node {
 public:
  /** check if the node can select child */
  bool can_select_child() const;
  /** select a child according to the pUCT score */
  Node *select_child(const TreeInfo &, const TreeOption &, bool) const;
  /** set player and action */
  void set_player_and_action(bool, Action_t);
  /** expand children according to legal actions */
  void expand_children(const std::vector<Action_t> &, bool);
  /** normalize policy by legal actions (only applies to the root node) */
  void normalize_policy();
  /** add Dirichlet noise to the policy (only applies to the root node) */
  void add_dirichlet_noise(RNG_t &, const TreeOption &);
  /** get the forward state */
  const State_t &get_forward_state() const;
  /** get the forward action */
  const Action_t &get_forward_action() const;
  /** set the forward result */
  void set_forward_result(ForwardResult);
  /** get the forward value */
  float get_forward_value() const;
  /** get if root player */
  bool is_root_player() const;
  /** update the Mcts Q-value */
  float update(float, bool, float);
  /** get visit counts */
  size_t get_visits() const;
  /** get children visit counts */
  std::unordered_map<Action_t, size_t> get_children_visits() const;
  /** get Q-value */
  float get_q_value() const;

  size_t get_children_size() const;
  void set_gumbel_noise(bool, size_t, RNG_t &);
  std::vector<size_t> get_top_actions(const std::vector<size_t> &, size_t, bool, const TreeInfo &,
                                      const TreeOption &, bool) const;
  Node *gumbel_select_child(size_t) const;
  size_t get_selected_action(size_t) const;
  std::unordered_map<Action_t, float> get_improved_policy(const TreeInfo &, const TreeOption &,
                                                          bool) const;
  float calculate_transformed_q_value(const TreeOption &, float) const;

 private:
  ForwardInfo forward_info_;
  NodeInfo node_info_;
  MctsInfo mcts_info_;
};

class Tree {
 public:
  /** Mcts selection & expansion */
  void before_forward(const std::vector<Action_t> &);
  /** Mcts update */
  bool after_forward(RNG_t &);

 public:
  /** set the tree option and game info */
  void set_option(const TreeOption &, bool);
  /** select and expand root legal actions */
  void expand_root(const std::vector<Action_t> &);
  /** get the forward input of current node */
  ForwardInfo get_forward_input() const;
  /** set the forward result to current node */
  void set_forward_result(ForwardResult);
  /** get the tree result */
  TreeResult get_tree_result() const;

  void gumbel_init(size_t, RNG_t &);
  size_t gumbel_search(const std::vector<Action_t> &);

 private:
  /** get the root simulation counts */
  size_t get_root_visits() const;

 private:
  Node tree_,                   ///< the root node
      *current_node_ = &tree_;  ///< current node visitor
  /** selection path (from root to current node)*/
  std::vector<Node *> selection_path_;
  /** tree information */
  TreeInfo tree_info_;
  /** (const) tree option */
  TreeOption tree_option_;
  /** (const) tree of single-player or two-player */
  bool is_two_player_;

  std::vector<size_t> top_actions_;
  size_t current_iter_this_phase_, used_simulations_, simulations_this_phase_;
};

}  // namespace czf::actor::muzero_worker_gumbel::mcts
