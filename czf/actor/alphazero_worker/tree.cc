#include "tree.h"

#include "node.h"

namespace czf::actor::alphazero_worker {

int Tree::num_simulations() const { return root_node.num_visits; }

void Tree::add_dirichlet_noise(std::mt19937& rng) {
  auto& children = root_node.children;
  const auto& ALPHA = tree_option.dirichlet_alpha;
  const auto& EPSILON = tree_option.dirichlet_epsilon;
  std::gamma_distribution<float> gamma(ALPHA, 1.0F);

  float noise_sum = 0.0F;
  std::vector<float> noise(children.size());
  for (int i = 0; i < noise.size(); ++i) {
    noise[i] = gamma(rng);
    noise_sum += noise[i];
  }

  if (noise_sum < std::numeric_limits<float>::min()) return;

  for (int i = 0; i < children.size(); ++i) {
    noise[i] /= noise_sum;
    auto& [p, action, child] = children[i];
    p = p * (1.0F - EPSILON) + noise[i] * EPSILON;
  }
}

void Tree::reset() { root_node.reset(); }

void Tree::set_option(const TreeOption& tree_option) { this->tree_option = tree_option; }

}  // namespace czf::actor::alphazero_worker