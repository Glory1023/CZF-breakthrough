#include "mcts/mcts.h"

#include <iostream>
#include <memory>
#include <random>

#include "utils/config.h"
#include "worker/worker.h"

namespace czf::actor::mcts {

using czf::actor::worker::WorkerManager;

void TreeInfo::update(float value) {
  min_value = std::min(value, min_value);
  max_value = std::max(value, max_value);
}

float MctsInfo::get_normalized_value(const TreeInfo &tree_info) const {
  const auto &maxv = tree_info.max_value;
  const auto &minv = tree_info.min_value;
  return (maxv >= minv) ? value : (value - minv) / (maxv - minv);
}

float MctsInfo::update(bool same, float z) {
  const auto w = same ? z : -z;
  ++visits;
  sqrt_visits = std::sqrt(visits);
  value += (w - value) / visits;
  return z;
  // return reward + WorkerManager::mcts_option.discount * z;
}

bool NodeInfo::can_select_child() const { return !children.empty(); }

void NodeInfo::expand(const std::vector<Action_t> &legal_actions) {
  const auto size = legal_actions.size();
  const auto child_player =
      WorkerManager::game_info.is_two_player ? !is_root_player : is_root_player;
  children.resize(size);
  for (size_t i = 0; i < size; ++i) {
    children[i].set_player_and_action(child_player, legal_actions[i]);
  }
}

void Node::set_player_and_action(bool player, Action_t action) {
  node_info_.is_root_player = player;
  forward_info_.action = action;
  mcts_info_.action_index = static_cast<size_t>(action);
}

bool Node::can_select_child() const { return node_info_.can_select_child(); }

Node *Node::select_child(const TreeInfo & /*tree_info*/) const {
  // init value
  float value_sum = 0.F;
  size_t num_selected = 0U;
  for (const auto &child : node_info_.children) {
    if (child.can_select_child()) {
      ++num_selected;
      value_sum += child.mcts_info_.value;
    }
  }
  const auto init_value =
      num_selected > 0 ? value_sum / static_cast<float>(num_selected + 1U)
                       : 0.F;
  // selection
  float selected_score = std::numeric_limits<float>::lowest();
  Node *selected_child = nullptr;
  for (const auto &child : node_info_.children) {
    // calculate pUCT score
    const float child_value =
        child.mcts_info_.visits > 0 ? child.mcts_info_.value : init_value;
    // child.mcts_info_.visits > 0 ?
    // child.mcts_info_.get_normalized_value(tree_info) : 0.F;
    const float score =
        child_value + WorkerManager::mcts_option.C_PUCT *
                          mcts_info_.policy[child.mcts_info_.action_index] *
                          mcts_info_.sqrt_visits /
                          static_cast<float>(1 + child.mcts_info_.visits);
    // argmax
    if (score > selected_score) {
      selected_score = score;
      selected_child = const_cast<Node *>(&child);
    }
  }
  return selected_child;
}

void Node::expand_children(const std::vector<Action_t> &legal_actions) {
  node_info_.expand(legal_actions);
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

void Node::add_dirichlet_noise(RNG_t &rng) {
  size_t size = node_info_.children.size();
  std::vector<float> noise(size);
  std::gamma_distribution<float> gamma(
      WorkerManager::mcts_option.dirichlet_alpha);
  float sum = 0.F;
  for (size_t i = 0; i < size; ++i) {
    noise[i] = gamma(rng);
    sum += noise[i];
  }
  if (sum < std::numeric_limits<float>::min()) {
    return;
  }
  const auto eps = WorkerManager::mcts_option.dirichlet_epsilon;
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

float Node::update(float z) {
  return mcts_info_.update(node_info_.is_root_player, z);
}

size_t Node::get_visits() const { return mcts_info_.visits; }

std::unordered_map<Action_t, size_t> Node::get_children_visits() const {
  std::unordered_map<Action_t, size_t> visits;
  for (const auto &child : node_info_.children) {
    visits[child.forward_info_.action] = child.mcts_info_.visits;
  }
  return visits;
}

void Tree::before_forward(const std::vector<Action_t> &all_actions) {
  // selection
  auto *node = &tree_;
  selection_path_.clear();
  selection_path_.emplace_back(node);
  while (node->can_select_child()) {
    node = node->select_child(tree_info_);
    selection_path_.emplace_back(node);
  }
  current_node_ = node;
  // expansion
  node->expand_children(all_actions);
}

void Tree::after_forward(RNG_t &rng) {
  if (selection_path_.size() == 1) {
    tree_.normalize_policy();
    tree_.add_dirichlet_noise(rng);
  }
  // update
  auto z = current_node_->get_forward_value();
  z = (current_node_->is_root_player()) ? -z : z;
  tree_info_.update(z);
  const auto end = std::rend(selection_path_);
  for (auto it = std::rbegin(selection_path_); it != end; ++it) {
    z = (*it)->update(z);
    tree_info_.update(z);
  }
}

void Tree::expand_root(const std::vector<Action_t> &legal_actions) {
  selection_path_.emplace_back(&tree_);
  tree_.expand_children(legal_actions);
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

size_t Tree::get_root_visits() const {
  // the first root expansion is not counted
  return tree_.get_visits() - 1;
}

TreeResult Tree::get_tree_result() {
  return {get_root_visits(), tree_.get_children_visits(),
          tree_.get_forward_value()};
}

}  // namespace czf::actor::mcts
