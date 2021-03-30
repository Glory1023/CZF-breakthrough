#include "job.h"

#include "config.h"
#include "czf.pb.h"
#include "node.h"
#include "tree.h"

namespace czf::actor::alphazero_worker {

Job::Job() : next_step(Job::Step::kEnqueue) {}

void Job::preprocess() {
  czf::pb::Job job_pb;
  job_pb.ParseFromString(job_str);
  const auto& state = job_pb.payload().state();
  const auto& option = state.tree_option();
  TreeOption tree_option{static_cast<size_t>(option.simulation_count()),
                         option.c_puct(), option.dirichlet_alpha(),
                         option.dirichlet_epsilon()};
  tree.set_option(tree_option);
  // root_state = engine->game->deserialize_state(state.serialized_state());
  job_pb.mutable_payload()->mutable_state()->clear_observation_tensor();
  job_pb.SerializeToString(&job_str);
  next_step = Job::Step::kSelect;
}

void Job::select(std::mt19937& rng) {
  auto* leaf_node = &tree.root_node;
  leaf_state = root_state->clone();

  czf::env::czf_env::Player previous_player = leaf_state->current_player();

  selection_path.clear();
  selection_path.emplace_back(previous_player, previous_player, leaf_node);

  czf::env::czf_env::Action action;
  while (!leaf_node->children.empty()) {
    std::tie(action, leaf_node) = leaf_node->select(rng, tree.tree_option);
    leaf_state->apply_action(action);
    czf::env::czf_env::Player current_player = leaf_state->current_player();
    selection_path.emplace_back(previous_player, current_player, leaf_node);
    previous_player = current_player;
  }

  if (leaf_state->is_terminal()) {
    leaf_policy.clear();
    leaf_returns = leaf_state->rewards();
    next_step = Job::Step::kUpdate;
  } else {
    leaf_observation = leaf_state->observation_tensor();
    next_step = Job::Step::kEvaluate;
  }
}

void Job::update(std::mt19937& rng) {
  for (const auto& [parent_player, current_player, node] : selection_path) {
    ++node->num_visits;
    node->parent_player_value_sum += leaf_returns[parent_player];
    node->current_player_value_sum += leaf_returns[current_player];
  }

  if (!leaf_policy.empty()) {
    auto& [parent_player, current_player, leaf_node] = selection_path.back();
    const auto legal_actions = leaf_state->legal_actions();
    leaf_node->expand(legal_actions);
    // extract legal action policy and normalize
    float policy_sum = 0.0F;
    for (auto& [p, action, child] : leaf_node->children) {
      p = leaf_policy[action];
      policy_sum += p;
    }
    for (auto& [p, action, child] : leaf_node->children) p /= policy_sum;
    // first simulation -> add dirichlet noise to root policy
    if (tree.num_simulations() == 1) tree.add_dirichlet_noise(rng);
  }

  if (tree.root_node.num_visits >= tree.tree_option.simulation_count) {
    next_step = Job::Step::kDone;
  } else {
    next_step = Job::Step::kSelect;
  }
}

void Job::postprocess(size_t num_actions) {
  // calculate mcts policy with visit count
  const auto& root_node = tree.root_node;
  std::vector<float> mcts_policy(num_actions, 0.0F);
  for (const auto& [p, action, child] : root_node.children) {
    mcts_policy[action] =
        static_cast<float>(child.num_visits) / (root_node.num_visits - 1);
  }

  czf::pb::Job job_pb;
  job_pb.ParseFromString(job_str);
  auto* evaluation =
      job_pb.mutable_payload()->mutable_state()->mutable_evaluation();

  evaluation->set_value(root_node.current_player_value_sum /
                        root_node.num_visits);
  for (const auto& p : mcts_policy) evaluation->add_policy(p);
  job_pb.SerializeToString(&job_str);

  next_step = Job::Step::kDequeue;
}

}  // namespace czf::actor::alphazero_worker