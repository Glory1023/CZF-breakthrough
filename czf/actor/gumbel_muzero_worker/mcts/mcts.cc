#include "mcts/mcts.h"

#include <iostream>
#include <memory>
#include <random>

#include "utils/config.h"

namespace czf::actor::gumbel_muzero_worker::mcts {

void TreeInfo::update(float value) {
  min_value = std::min(value, min_value);
  max_value = std::max(value, max_value);
}

float TreeInfo::get_normalized_value(float value) const {
  return (max_value > min_value) ? (value - min_value) / (max_value - min_value) : value;
}

float MctsInfo::update(float z, bool is_root_player, bool is_two_player, float discount) {
  const auto w = is_two_player && is_root_player ? -z : z;
  ++visits;
  sqrt_visits = std::sqrt(visits);
  value += (w - value) / visits;
  return is_two_player ? z : reward + discount * z;
}

bool NodeInfo::can_select_child() const { return !children.empty(); }

void NodeInfo::expand(const std::vector<Action_t> &actions, const bool is_two_player,
                      const bool is_stochastic) {
  const auto size = actions.size();
  const auto child_player = is_two_player ? !is_root_player : is_root_player;
  const auto child_is_chance_node = is_stochastic && !is_chance_node;
  children.resize(size);
  for (size_t i = 0; i < size; ++i) {
    children[i].set_player_and_action(child_player, actions[i], child_is_chance_node);
  }
}

bool Node::can_select_child() const { return node_info_.can_select_child(); }

Node *Node::select_child(const TreeInfo &tree_info, const TreeOption &tree_option,
                         bool is_two_player) const {
  // calculate init value for unvisited actions
  float init_value = 0.F;
  float value_sum = 0.F;
  size_t num_selected = 0U;
  for (const auto &child : node_info_.children) {
    if (child.can_select_child()) {
      ++num_selected;
      value_sum +=
          is_two_player
              ? child.mcts_info_.value
              : tree_info.get_normalized_value(child.mcts_info_.reward +
                                               tree_option.discount * child.mcts_info_.value);
    }
  }
  if (num_selected > 0) {
    init_value = value_sum / static_cast<float>(num_selected + 1U);
  }

  // selection
  float selected_score = std::numeric_limits<float>::lowest();
  Node *selected_child = nullptr;
  for (const auto &child : node_info_.children) {
    float score;
    if (is_chance_node()) {
      // quasi-random sampling
      score = mcts_info_.policy[child.mcts_info_.action_index] /
              static_cast<float>(1 + child.mcts_info_.visits);
    } else {
      // calculate pUCT score
      const float child_value =
          child.mcts_info_.visits > 0
              ? (is_two_player ? child.mcts_info_.value
                               : tree_info.get_normalized_value(child.mcts_info_.reward +
                                                                tree_option.discount *
                                                                    child.mcts_info_.value))
              : init_value;
      score = child_value + tree_option.c_puct * mcts_info_.policy[child.mcts_info_.action_index] *
                                mcts_info_.sqrt_visits /
                                static_cast<float>(1 + child.mcts_info_.visits);
    }
    // argmax
    if (score > selected_score) {
      selected_score = score;
      selected_child = const_cast<Node *>(&child);  // NOLINT
    }
  }
  return selected_child;
}

void Node::set_player_and_action(bool player, Action_t action, bool is_chance_node) {
  node_info_.is_root_player = player;
  node_info_.is_chance_node = is_chance_node;
  forward_info_.action = action;
  mcts_info_.action_index = static_cast<size_t>(action);
}

void Node::expand_children(const std::vector<Action_t> &actions, bool is_two_player,
                           bool is_stochastic) {
  node_info_.expand(actions, is_two_player, is_stochastic);
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
        (1 - eps) * mcts_info_.policy[child.mcts_info_.action_index] + eps * (noise[i++] / sum);
  }
}

const State_t &Node::get_forward_state() const { return forward_info_.state; }

const Action_t &Node::get_forward_action() const { return forward_info_.action; }

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

bool Node::is_chance_node() const { return node_info_.is_chance_node; }

float Node::update(float z, bool is_two_player, float discount) {
  return mcts_info_.update(z, node_info_.is_root_player, is_two_player, discount);
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

size_t Node::get_children_size() const { return node_info_.children.size(); }

void Node::set_gumbel_noise(bool use_noise, size_t num_actions, RNG_t &rng) {
  if (!use_noise) {
    mcts_info_.gumbel_noise.clear();
    mcts_info_.gumbel_noise.resize(num_actions, 0.F);
    return;
  }
  std::extreme_value_distribution<double> gumbel_distribution(0.0, 1.0);
  mcts_info_.gumbel_noise.clear();
  mcts_info_.gumbel_noise.resize(num_actions);
  for (const auto &child : node_info_.children) {
    mcts_info_.gumbel_noise[child.mcts_info_.action_index] = gumbel_distribution(rng);
  }
}

std::vector<size_t> Node::get_top_actions(const std::vector<size_t> &origin_top_actions,
                                          size_t num_top_actions, bool use_value,
                                          const TreeInfo &tree_info, const TreeOption &tree_option,
                                          bool is_two_player) const {
  std::vector<std::pair<float, size_t>> vec;
  for (const auto &index : origin_top_actions) {
    const auto &child = node_info_.children.at(index);
    float g = mcts_info_.gumbel_noise[child.mcts_info_.action_index];
    float logit = std::log(mcts_info_.policy[child.mcts_info_.action_index]);
    float compared_value = g + logit;
    if (use_value) {
      float q_value = child.mcts_info_.visits > 0
                          ? (is_two_player ? child.mcts_info_.value
                                           : tree_info.get_normalized_value(
                                                 child.mcts_info_.reward +
                                                 tree_option.discount * child.mcts_info_.value))
                          : 0.F;
      compared_value += get_transformed_q_value(tree_option, q_value);
    }
    vec.push_back(std::make_pair(compared_value, index));
  }
  std::sort(vec.rbegin(), vec.rend());

  std::vector<size_t> top_actions;
  for (size_t i = 0; i < num_top_actions; ++i) {
    top_actions.push_back(vec[i].second);
  }
  return top_actions;
}

Node *Node::gumbel_select_child(size_t selected_index) const {
  // selection
  const auto &child = node_info_.children.at(selected_index);
  Node *selected_child = const_cast<Node *>(&child);
  return selected_child;
}

size_t Node::get_best_action(size_t index) const {
  return node_info_.children.at(index).mcts_info_.action_index;
}

float Node::get_best_action_value(size_t index, const TreeOption &tree_option,
                                  bool is_two_player) const {
  const auto &child = node_info_.children.at(index);
  float child_value =
      is_two_player ? child.mcts_info_.value
                    : (child.mcts_info_.reward + tree_option.discount * child.mcts_info_.value);
  return child_value;
}

std::unordered_map<Action_t, float> Node::get_improved_policy(const TreeInfo &tree_info,
                                                              const TreeOption &tree_option,
                                                              bool is_two_player,
                                                              size_t best_action) const {
  std::unordered_map<Action_t, float> improved_policy;

  if (tree_option.gumbel_use_simple_loss) {
    improved_policy[best_action] = 1.F;
    return improved_policy;
  }

  std::vector<std::pair<Action_t, float>> improved_policy_logits;
  float max_logit = std::numeric_limits<float>::lowest();
  float exp_sum = 0.F;
  // root forward value
  float forward_value = tree_info.get_normalized_value(get_forward_value());

  for (const auto &child : node_info_.children) {
    // calculate improved policy
    float policy_logit = std::log(mcts_info_.policy[child.mcts_info_.action_index]);
    float completed_q_value =
        child.mcts_info_.visits > 0
            ? (is_two_player
                   ? child.mcts_info_.value
                   : tree_info.get_normalized_value(child.mcts_info_.reward +
                                                    tree_option.discount * child.mcts_info_.value))
            : forward_value;
    float transformed_q_value = get_transformed_q_value(tree_option, completed_q_value);
    float improved_policy_logit = policy_logit + transformed_q_value;
    max_logit = std::max(max_logit, improved_policy_logit);
    improved_policy_logits.push_back(
        std::make_pair(child.mcts_info_.action_index, improved_policy_logit));
  }
  for (auto &logit : improved_policy_logits) {
    logit.second -= max_logit;
    logit.second = std::exp(logit.second);
    exp_sum += logit.second;
  }
  for (const auto &logit : improved_policy_logits) {
    improved_policy[logit.first] = logit.second / exp_sum;
  }
  return improved_policy;
}

float Node::get_transformed_q_value(const TreeOption &tree_option, float q_value) const {
  std::unordered_map<Action_t, size_t> visits = get_children_visits();
  float max_visit = static_cast<float>(
      std::max_element(visits.begin(), visits.end(),
                       [](const std::pair<Action_t, size_t> &p1,
                          const std::pair<Action_t, size_t> &p2) { return p1.second < p2.second; })
          ->second);
  return (tree_option.gumbel_c_visit + max_visit) * tree_option.gumbel_c_scale * q_value;
}

bool Tree::before_forward(const std::vector<Action_t> &all_actions,
                          const std::vector<Action_t> &all_chance_outcomes) {
  // selection
  // auto *node = &tree_;
  // selection_path_.clear();
  // selection_path_.emplace_back(node);
  auto *node = current_node_;
  while (node->can_select_child()) {
    node = node->select_child(tree_info_, tree_option_, is_two_player_);
    selection_path_.emplace_back(node);
  }
  current_node_ = node;
  // expansion
  if (node->is_chance_node()) {
    node->expand_children(all_chance_outcomes, is_two_player_, is_stochastic_);
  } else {
    node->expand_children(all_actions, is_two_player_, is_stochastic_);
  }
  return node->is_chance_node();
}

bool Tree::after_forward(RNG_t &rng) {
  if (selection_path_.size() == 1) {
    // normalize policy for legal actions only
    tree_.normalize_policy();
    // tree_.add_dirichlet_noise(rng, tree_option_);
  }
  // update
  auto z = current_node_->get_forward_value();
  if (is_two_player_ && !current_node_->is_root_player()) {
    z = -z;
  }
  tree_info_.update(z);
  const auto end = std::rend(selection_path_);
  for (auto it = std::rbegin(selection_path_); it != end; ++it) {
    z = (*it)->update(z, is_two_player_, tree_option_.discount);
    tree_info_.update(z);
  }
  return get_root_visits() >= tree_option_.simulation_count;
}

void Tree::set_option(const TreeOption &tree_option, bool is_two_player, bool is_stochastic) {
  tree_option_ = tree_option;
  tree_info_.min_value = tree_option.tree_min_value;
  tree_info_.max_value = tree_option.tree_max_value;
  is_two_player_ = is_two_player;
  is_stochastic_ = is_stochastic;
}

void Tree::expand_root(const std::vector<Action_t> &legal_actions) {
  selection_path_.emplace_back(&tree_);
  tree_.expand_children(legal_actions, is_two_player_, is_stochastic_);
}

ForwardInfo Tree::get_forward_input() const {
  Node *parent =
      selection_path_.size() > 1 ? selection_path_[selection_path_.size() - 2] : current_node_;
  return {parent->get_forward_state(), current_node_->get_forward_action()};
}

void Tree::set_forward_result(ForwardResult result) {
  current_node_->set_forward_result(std::move(result));
}

TreeResult Tree::get_tree_result() const {
  auto root_value = tree_.get_q_value();
  auto best_action = tree_.get_best_action(top_actions_[0]);
  auto best_action_value =
      tree_.get_best_action_value(top_actions_[0], tree_option_, is_two_player_);
  auto value = tree_option_.gumbel_use_best_action_value ? best_action_value : root_value;
  return {best_action,
          tree_.get_improved_policy(tree_info_, tree_option_, is_two_player_, best_action),
          is_two_player_ ? -value : value};
}

size_t Tree::get_root_visits() const {
  // the first root expansion is not counted
  return tree_.get_visits() - 1;
}

void Tree::gumbel_init(size_t num_actions, RNG_t &rng) {
  size_t n = tree_option_.simulation_count;
  size_t m = tree_option_.gumbel_sampled_actions;
  size_t num_legal_actions = tree_.get_children_size();

  tree_.set_gumbel_noise(tree_option_.gumbel_use_noise, num_actions, rng);
  top_actions_.clear();
  for (size_t i = 0; i < num_legal_actions; ++i) top_actions_.push_back(i);

  if (n <= m) {
    // without sequential halving
    m = std::min(m, num_legal_actions);
    top_actions_ =
        tree_.get_top_actions(top_actions_, m, false, tree_info_, tree_option_, is_two_player_);
    current_iter_this_phase_ = 0;
  } else {
    // with sequential halving
    m = std::min(m, num_legal_actions);
    top_actions_ =
        tree_.get_top_actions(top_actions_, m, false, tree_info_, tree_option_, is_two_player_);
    current_iter_this_phase_ = 0;
    used_simulations_ = 0;
    if (m == 1) {
      simulations_this_phase_ = n;
    } else {
      size_t num = std::log2(m);
      if (m != (1 << num)) ++num;
      simulations_this_phase_ = n / (num * m);
    }
  }
}

size_t Tree::gumbel_search(const std::vector<Action_t> &all_actions,
                           const std::vector<Action_t> &all_chance_outcomes) {
  size_t n = tree_option_.simulation_count;
  size_t m = tree_option_.gumbel_sampled_actions;

  if (n <= m) {
    // without sequential halving
    if (current_iter_this_phase_ < top_actions_.size()) {
      auto *node = &tree_;
      selection_path_.clear();
      selection_path_.emplace_back(node);
      node = node->gumbel_select_child(top_actions_[current_iter_this_phase_]);
      selection_path_.emplace_back(node);
      current_node_ = node;
      if (node->is_chance_node()) {
        node->expand_children(all_chance_outcomes, is_two_player_, is_stochastic_);
      } else {
        node->expand_children(all_actions, is_two_player_, is_stochastic_);
      }
      ++current_iter_this_phase_;
      return 1;
    } else {
      top_actions_ =
          tree_.get_top_actions(top_actions_, 1, true, tree_info_, tree_option_, is_two_player_);
      return 0;
    }
  } else {
    // with sequential halving
    size_t k = top_actions_.size();
    if (current_iter_this_phase_ < simulations_this_phase_ * k) {
      auto *node = &tree_;
      selection_path_.clear();
      selection_path_.emplace_back(node);
      node = node->gumbel_select_child(
          top_actions_[current_iter_this_phase_ / simulations_this_phase_]);
      selection_path_.emplace_back(node);
      current_node_ = node;
      ++current_iter_this_phase_;
      return 2;
    } else {
      used_simulations_ += simulations_this_phase_ * k;
      k = (k + 1) / 2;
      top_actions_ =
          tree_.get_top_actions(top_actions_, k, true, tree_info_, tree_option_, is_two_player_);
      if (k == 1) {
        return 0;
      } else {
        size_t num = std::log2(k);
        if (k != (1 << num)) ++num;
        simulations_this_phase_ = (n - used_simulations_) / (num * k);
        current_iter_this_phase_ = 0;
        return 3;
      }
    }
  }
  return 0;
}

}  // namespace czf::actor::gumbel_muzero_worker::mcts
