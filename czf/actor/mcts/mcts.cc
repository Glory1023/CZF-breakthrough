#include "mcts/mcts.h"

#include <iostream>
#include <memory>
#include <random>

#include "utils/config.h"

namespace czf::actor::mcts {

void TreeInfo::update(float value) {
  min_value = std::min(value, min_value);
  max_value = std::max(value, max_value);
}

float TreeInfo::get_normalized_value(float value) const {
  return (max_value > min_value) ? (value - min_value) / (max_value - min_value)
                                 : value;
}

float MctsInfo::update(float z, bool is_root_player, bool is_two_player,
                       float discount) {
  const auto w = is_two_player && is_root_player ? -z : z;
  ++visits;
  sqrt_visits = std::sqrt(visits);
  value += (w - value) / visits;
  return is_two_player ? z : reward + discount * z;
}

bool NodeInfo::can_select_child() const { return !children.empty(); }

void NodeInfo::expand(const std::vector<Action_t> &legal_actions,
                      const bool is_two_player) {
  const auto size = legal_actions.size();
  const auto child_player = is_two_player ? !is_root_player : is_root_player;
  children.resize(size);
  for (size_t i = 0; i < size; ++i) {
    children[i].set_player_and_action(child_player, legal_actions[i]);
  }
}

bool Node::can_select_child() const { return node_info_.can_select_child(); }

Node *Node::select_child(const TreeInfo &tree_info,
                         const TreeOption &tree_option,
                         bool is_two_player) const {
  // init value
  float init_value = 0.F;
  // if (is_two_player) {
    float value_sum = 0.F;
    size_t num_selected = 0U;
    for (const auto &child : node_info_.children) {
      if (child.can_select_child()) {
        ++num_selected;
      value_sum += is_two_player
                       ? child.mcts_info_.value
                       : child.mcts_info_.reward +
                             tree_option.discount * child.mcts_info_.value;
      }
    }
    if (num_selected > 0) {
      init_value = value_sum / static_cast<float>(num_selected + 1U);
    }
  // }
  // selection
  float selected_score = std::numeric_limits<float>::lowest();
  Node *selected_child = nullptr;
  for (const auto &child : node_info_.children) {
    // calculate pUCT score
    const float child_value =
        child.mcts_info_.visits > 0
            ? (is_two_player
                   ? child.mcts_info_.value
                   : tree_info.get_normalized_value(child.mcts_info_.reward +
                                                    tree_option.discount *
                                                        child.mcts_info_.value))
            : (is_two_player ? init_value
                             : tree_info.get_normalized_value(init_value));
    const float score =
        child_value + tree_option.c_puct *
                          mcts_info_.policy[child.mcts_info_.action_index] *
                          mcts_info_.sqrt_visits /
                          static_cast<float>(1 + child.mcts_info_.visits);
    // argmax
    if (score > selected_score) {
      selected_score = score;
      selected_child = const_cast<Node *>(&child);  // NOLINT
    }
  }
  return selected_child;
}

void Node::set_player_and_action(bool player, Action_t action) {
  node_info_.is_root_player = player;
  forward_info_.action = action;
  mcts_info_.action_index = static_cast<size_t>(action);
}

void Node::expand_children(const std::vector<Action_t> &legal_actions,
                           bool is_two_player) {
  node_info_.expand(legal_actions, is_two_player);
}

void Node::normalize_policy() {
  float sum = 0.F;
  for (const auto &child : node_info_.children) {
    sum += mcts_info_.policy[child.mcts_info_.action_index];
  }
  for (const auto &child : node_info_.children) {
    mcts_info_.policy[child.mcts_info_.action_index] /= sum;
  }
}

void Node::add_dirichlet_noise(RNG_t &rng, const TreeOption &tree_option) {
  size_t size = node_info_.children.size();
  std::vector<float> noise(size);
  std::gamma_distribution<float> gamma(tree_option.dirichlet_alpha);
  float sum = 0.F;
  for (size_t i = 0; i < size; ++i) {
    noise[i] = gamma(rng);
    sum += noise[i];
  }
  if (sum < std::numeric_limits<float>::min()) {
    return;
  }
  const auto eps = tree_option.dirichlet_epsilon;
  size_t i = 0;
  for (const auto &child : node_info_.children) {
    mcts_info_.policy[child.mcts_info_.action_index] =
        (1 - eps) * mcts_info_.policy[child.mcts_info_.action_index] +
        eps * (noise[i++] / sum);
  }
}

const State_t &Node::get_forward_state() const { return forward_info_.state; }

const Action_t &Node::get_forward_action() const {
  return forward_info_.action;
}

void Node::set_forward_result(ForwardResult result) {
  // r, s = g(s', a)
  forward_info_.state = std::move(result.state);
  mcts_info_.reward = result.reward;
  // p, v = f(s)
  mcts_info_.policy = std::move(result.policy);
  mcts_info_.forward_value = result.value;
}

float Node::get_forward_value() const { return mcts_info_.forward_value; }

bool Node::is_root_player() const { return node_info_.is_root_player; }

float Node::update(float z, bool is_two_player, float discount) {
  return mcts_info_.update(z, node_info_.is_root_player, is_two_player,
                           discount);
}

size_t Node::get_visits() const { return mcts_info_.visits; }

std::unordered_map<Action_t, size_t> Node::get_children_visits() const {
  std::unordered_map<Action_t, size_t> visits;
  for (const auto &child : node_info_.children) {
    visits[child.forward_info_.action] = child.mcts_info_.visits;
  }
  return visits;
}

float Node::get_q_value() const { return mcts_info_.value; }

void Tree::before_forward(const std::vector<Action_t> &all_actions) {
  // selection
  auto *node = &tree_;
  selection_path_.clear();
  selection_path_.emplace_back(node);
  while (node->can_select_child()) {
    node = node->select_child(tree_info_, tree_option_, is_two_player_);
    selection_path_.emplace_back(node);
  }
  current_node_ = node;
  // expansion
  node->expand_children(all_actions, is_two_player_);
}

bool Tree::after_forward(RNG_t &rng) {
  if (selection_path_.size() == 1) {
    tree_.normalize_policy();
    tree_.add_dirichlet_noise(rng, tree_option_);
  }
  // update
  auto z = current_node_->get_forward_value();
  tree_info_.update(z);
  const auto end = std::rend(selection_path_);
  for (auto it = std::rbegin(selection_path_); it != end; ++it) {
    z = (*it)->update(z, is_two_player_, tree_option_.discount);
    tree_info_.update(z);
  }
  return get_root_visits() >= tree_option_.simulation_count;
}

void Tree::set_option(const TreeOption &tree_option, bool is_two_player) {
  tree_option_ = tree_option;
  tree_info_.min_value = tree_option.tree_min_value;
  tree_info_.max_value = tree_option.tree_max_value;
  is_two_player_ = is_two_player;
}

void Tree::expand_root(const std::vector<Action_t> &legal_actions) {
  selection_path_.emplace_back(&tree_);
  tree_.expand_children(legal_actions, is_two_player_);
}

ForwardInfo Tree::get_forward_input() const {
  Node *parent = selection_path_.size() > 1
                     ? selection_path_[selection_path_.size() - 2]
                     : current_node_;
  return {parent->get_forward_state(), current_node_->get_forward_action()};
}

void Tree::set_forward_result(ForwardResult result) {
  current_node_->set_forward_result(std::move(result));
}

TreeResult Tree::get_tree_result() const {
  auto value = tree_.get_q_value();
  return {get_root_visits(), tree_.get_children_visits(),
          is_two_player_ ? -value : value};
}

size_t Tree::get_root_visits() const {
  // the first root expansion is not counted
  return tree_.get_visits() - 1;
}

}  // namespace czf::actor::mcts
