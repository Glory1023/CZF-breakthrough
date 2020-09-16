#include "mcts.hpp"

#include <memory>

#include "config.hpp"

namespace Mcts {

void TreeInfo::update(float value) {
  min_value = std::min(value, min_value);
  max_value = std::max(value, max_value);
}

float MctsInfo::get_normalized_value(const TreeInfo &tree_info) const {
  const auto &maxv = tree_info.max_value, &minv = tree_info.min_value;
  return (maxv > minv) ? value : (value - minv) / (maxv - minv);
}

float MctsInfo::update(float z) {
  ++visits;
  sqrt_visits = std::sqrt(visits);
  value += (z - value) / visits;
  // parent_->value_ += (z - cparent_->value_) / cparent_->visits_;
  // return cparent_->reward_ + MctsOption::Discount * z;
  return z;
}

bool NodeInfo::has_children() const { return children != nullptr; }

void NodeInfo::expand() {
  // TODO: legal actions
  children = std::make_unique<std::list<Node>>();
  children->resize(GameOption::ActionDim);
}

bool Node::has_children() const { return node_info_.has_children(); }

Node *Node::get_parent() const { return node_info_.parent; }

float Node::get_value() const { return mcts_info_.value; }

const ForwardInfo &Node::get_forward_info() const { return forward_info_; }

Node *Node::select_child(const TreeInfo &tree_info, PRNG &rng) const {
  float selected_score = -2.f;
  size_t selected_count = 1;
  Node *selected_child = nullptr;
  for (auto &child : *node_info_.children) {
    // calculate pUCT score
    float score = child.mcts_info_.get_normalized_value(tree_info) +
                  MctsOption::C_PUCT *
                      mcts_info_.policy[child.node_info_.action] *
                      mcts_info_.sqrt_visits / (1 + child.mcts_info_.visits);
    // argmax
    if (score > (selected_score - BuildOption::FloatEps)) {
      if (score > (selected_score + BuildOption::FloatEps)) {
        // select the child with the current max score
        selected_score = score;
        selected_child = &child;
        selected_count = 1;
      } else {
        // select one child with same scores
        ++selected_count;
        if ((rng() % selected_count) == 0) {
          selected_child = &child;
        }
      }
    }
  }
  return selected_child;
}

void Node::expand_children() { node_info_.expand(); }

void Node::set_forward_result(ForwardResult &result) {
  // r, s = g(s', a)
  forward_info_.state = result.state;
  mcts_info_.reward = result.reward;
  // p, v = f(s)
  mcts_info_.policy = result.policy;
  mcts_info_.value = result.value;
}

float Node::update(float z) { return mcts_info_.update(z); }

const ForwardInfo &Tree::before_forward(PRNG &rng) {
  // selection
  auto node = &tree_;
  while (node->has_children()) {
    node = node->select_child(tree_info_, rng);
  }
  current_node_ = node;
  return node->get_forward_info();
}

void Tree::after_forward(ForwardResult result, PRNG &rng) {
  // expansion
  auto node = current_node_;
  node->set_forward_result(result);
  node->expand_children();
  node->expand_dirichlet(rng);
  // update
  auto z = node->get_value();
  do {
    tree_info_.update(z);
    z = node->update(z);
    node = node->get_parent();
  } while (node != nullptr);
}

void TreeManager::run() {
  const auto batch_size = trees_.size();
  const auto num_device = models_.size();
  for (size_t i = 0; i < batch_size; ++i) {
    auto &tree = trees_[i];
    auto &model = models_[i % num_device];
    const auto &info = tree.before_forward(rng_);
    model.prepare_forward(i, info);
  }
  for (size_t i = 0; i < num_device; ++i) {
    models_[i].forward();
  }
  for (size_t i = 0; i < batch_size; ++i) {
    auto &tree = trees_[i];
    auto &model = models_[i % num_device];
    tree.after_forward(model.get_result(i), rng_);
  }
}
}  // namespace Mcts