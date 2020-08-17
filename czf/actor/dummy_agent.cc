#include <iostream>
#include <random>

#include "third_party/czf_env/czf_env/czf_env.h"
using namespace std;

int main() {
  std::random_device rd;
  std::mt19937 gen(rd());

  auto game = czf_env::load_game("tic_tac_toe");

  auto state = game->new_initial_state();
  while (!state->is_terminal()) {
    auto legal_actions = state->legal_actions();
    uniform_int_distribution<int> distribution(0, legal_actions.size() - 1);
    auto action_index = distribution(gen);
    state->apply_action(legal_actions[action_index]);
    cout << state->serialize() << endl;
  }
  return 0;
}